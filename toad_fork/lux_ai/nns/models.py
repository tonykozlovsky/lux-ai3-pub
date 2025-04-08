import os
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple, Union

import gym
import numpy as np
import torch
import torch.nn.functional as F
from click.core import batch
from torch import nn
import tempfile
import uuid
import math
from .transformer import CrossAttentionTransformer, TransformerWithPosEmbed

from .conv_blocks import ResidualBlockNoMask


from ..torchbeast.profiler import ScopedProfiler

from ..lux_gym.reward_spaces import RewardSpec
from .in_blocks import DictInputLayer



import torch
import torch.nn as nn
from typing import Tuple, Optional, List

import torch
import torch.nn as nn

def build_2d_sincos_position_embedding(
        height: int,
        width: int,
        embed_dim: int
) -> torch.Tensor:
    """
    Returns a [1, embed_dim, height, width] tensor of fixed 2D sinusoidal embeddings.
    """
    pe = torch.zeros(embed_dim, height, width)  # [C, H, W]

    # Each dimension of the embedding corresponds to a different frequency of sin/cos
    # half_dim: how many channels we dedicate to 'y' vs 'x'
    half_dim = embed_dim // 2

    # Frequencies follow the original formula from "Attention Is All You Need":
    # freq = 1 / (10000^(2i/dim))  but we apply it in a 2D manner
    # We'll assign half for y-based sin/cos, half for x-based sin/cos
    y_positions = torch.arange(0, height).unsqueeze(1)  # shape [H, 1]
    x_positions = torch.arange(0, width).unsqueeze(1)   # shape [W, 1]

    div_term = torch.exp(
        torch.arange(0, half_dim, 2, dtype=torch.float32)
        * (-math.log(10000.0) / half_dim)
    )  # shape: [half_dim/2] if half_dim is even

    # 1) Encode Y positions with sin/cos
    # We'll broadcast over x, so each row is repeated across all columns
    #   y_positions: [H,1]
    #   div_term: [half_dim/2]
    # We alternate sin/cos in different embedding channels
    for i in range(0, half_dim, 2):
        # i -> i+1 = sin/cos pair
        idx = i // 2
        pe[i, :, :]   = torch.sin(y_positions * div_term[idx])  # shape => [H,1], broadcast -> [H,W]
        pe[i+1, :, :] = torch.cos(y_positions * div_term[idx])

    # 2) Encode X positions with sin/cos in the remaining half of channels
    for i in range(0, half_dim, 2):
        idx = i // 2
        # offset = half_dim means we start filling after y-channels
        pe[half_dim + i, :, :]   = torch.sin(x_positions * div_term[idx])  # shape => [W,1], broadcast
        pe[half_dim + i + 1, :, :] = torch.cos(x_positions * div_term[idx])

    # If embed_dim is odd, we have 1 leftover channel (you can handle it in your own way)
    # For simplicity, we won't handle odd embed_dim here.

    # We want shape [1, C, H, W]
    pe = pe.unsqueeze(0)  # => [1, embed_dim, H, W]
    return pe

class SpatialTransformer(nn.Module):
    """
    A single transformer block operating on [B, C, H, W] data,
    including a learnable 2D positional embedding of shape [1, C, H, W].

    Steps:
      1) Add learnable positional embedding: x = x + pos_embed
      2) Flatten (H, W) -> sequence dimension
      3) Multi-head self-attention in that sequence dimension
      4) Residual + LayerNorm
      5) FeedForward
      6) Residual + LayerNorm
      7) Reshape back to [B, C, H, W]
    """
    def __init__(
            self,
            hidden_dim: int,
            height: int,
            width: int,
            num_heads: int = 4,
            ff_multiplier: int = 4
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.height = height
        self.width = width

        # A learnable 2D positional embedding
        # shape [1, C=hidden_dim, H, W]
        #self.pos_embed = nn.Parameter(torch.zeros(
        #    1, hidden_dim, height, width
        #))

        pe_tensor = build_2d_sincos_position_embedding(
            height, width, hidden_dim
        )
        # (2) Store as a buffer so it's not trainable
        self.register_buffer("pos_embed", pe_tensor)  # no gradient

        # Multihead self-attention
        # uses shape [B, seq_len, embed_dim] if batch_first=True
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # A small feedforward sub-block
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_multiplier * hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_multiplier * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, hidden_dim, H, W]
        returns: same shape [B, hidden_dim, H, W]
        """
        B, C, H, W = x.shape
        assert C == self.hidden_dim, \
            f"Expected channel dim={self.hidden_dim}, got {C}"
        assert H == self.height and W == self.width, \
            f"Expected H,W=({self.height},{self.width}), got ({H},{W})"

        # (1) Add learnable positional embedding
        x = x + self.pos_embed  # broadcast over batch dimension

        # (2) Flatten [B, C, H, W] => [B, H*W, C]
        x_seq = x.permute(0, 2, 3, 1).contiguous().reshape(B, H*W, C)

        # (3) Multi-head self-attention
        # attn_out: shape [B, seq_len, C]
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)  # self-attn

        # (4) Residual + LayerNorm
        x_seq = x_seq + attn_out
        x_seq = self.norm1(x_seq)

        # (5) FeedForward
        ff_out = self.ff(x_seq)

        # (6) Residual + LayerNorm
        x_seq = x_seq + ff_out
        x_seq = self.norm2(x_seq)

        # (7) Reshape back => [B, C, H, W]
        x_out = x_seq.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return x_out


class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell:
      - x: input of shape [B, C_in, H, W]
      - h_prev, c_prev: previous hidden and cell states, each [B, C_hidden, H, W]
      - returns next hidden and cell states of same shape
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 kernel_size: Tuple[int, int], bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias

        # Convolution for input+hidden -> 4 gates
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self,
                x: torch.Tensor,
                h_prev: torch.Tensor,
                c_prev: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:      [B, C_in, H, W]
        h_prev: [B, C_hidden, H, W]
        c_prev: [B, C_hidden, H, W]

        returns:
          h_next: [B, C_hidden, H, W]
          c_next: [B, C_hidden, H, W]
        """
        combined = torch.cat([x, h_prev], dim=1)  # [B, C_in + C_hidden, H, W]
        gates = self.conv(combined)               # [B, 4*C_hidden, H, W]

        # Split along channel dimension: i, f, o, g
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    A multi-layer ConvLSTM:
      - input of shape [B, T, C_in, H, W] if batch_first=True
      - hidden_dim is the number of channels in each layer's hidden state
      - returns either (all hidden states over T, final hidden/cell for each layer)
        or (final hidden state only, final hidden/cell).
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 kernel_size: Tuple[int, int],
                 num_layers: int = 1,
                 bias: bool = True,
                 batch_first: bool = True,
                 return_sequence: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.return_sequence = return_sequence

        # Stacked ConvLSTMCells
        cells = []
        for i in range(num_layers):
            cell_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cells.append(ConvLSTMCell(
                input_dim=cell_input_dim,
                hidden_dim=self.hidden_dim,
                kernel_size=self.kernel_size,
                bias=self.bias
            ))
        self.layers = nn.ModuleList(cells)

    def forward(self,
                x: torch.Tensor,
                hidden_states_h,
                hidden_states_c: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
                done=None,
                one_player=None) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        """
        x:
          - if batch_first=True:  shape [B, T, C_in, H, W]
          - otherwise:            shape [T, B, C_in, H, W]

        hidden_states:
          - optional (h_list, c_list), each is a list of length self.num_layers
          - h_list[i], c_list[i] have shape [B, hidden_dim, H, W]

        returns:
          output:
            - if return_sequence=True: shape [B, T, hidden_dim, H, W]
            - else: shape [B, hidden_dim, H, W]  (the final top-layer hidden state)
          (h_list, c_list):
            - final hidden and cell states for each layer,
              lists of length num_layers, each [B, hidden_dim, H, W].
        """
        if not self.batch_first:
            # Convert to [B, T, C_in, H, W]
            x = x.permute(1, 0, 2, 3, 4)
            if done is not None:
                done = done.permute(1, 0)
                if one_player is None:
                    done = done.repeat_interleave(2, dim=0)

        B, T, _, H, W = x.shape

        # Initialize hidden states if not provided
        if hidden_states_h is None:
            h_list = [
                torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
                for _ in range(self.num_layers)
            ]
            c_list = [
                torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype)
                for _ in range(self.num_layers)
            ]
        else:
            h_list, c_list = ([hidden_states_h], [hidden_states_c])
            assert len(h_list) == self.num_layers
            assert len(c_list) == self.num_layers

        # Optionally store every top-layer hidden state
        if self.return_sequence:
            outputs_h = []
            outputs_c = []
        else:
            outputs_h = None
            outputs_c = None

        # Process time steps
        for tstep in range(T):
            x_t = x[:, tstep, ...]  # [B, C_in, H, W]
            done_mask = done[:, tstep, ...] if done is not None else None
            # Pass through stacked LSTM layers
            if done_mask is not None:
                done_mask = done_mask.view(-1, 1, 1, 1)  # Convert to (B, 1, 1, 1)
                done_mask = done_mask.expand(B, self.hidden_dim, H, W)
                done_mask = done_mask.float()
                keep_mask = (1. - done_mask)

                for layer_idx, cell in enumerate(self.layers):
                    h_list[layer_idx] = h_list[layer_idx] * keep_mask
                    c_list[layer_idx] = c_list[layer_idx] * keep_mask

                    h_list[layer_idx], c_list[layer_idx] = cell(
                        x_t, h_list[layer_idx], c_list[layer_idx]
                    )
                    # The output of this layer is the input to the next
                    x_t = h_list[layer_idx]
            else:
                for layer_idx, cell in enumerate(self.layers):
                    h_list[layer_idx], c_list[layer_idx] = cell(
                        x_t, h_list[layer_idx], c_list[layer_idx]
                    )
                    # The output of this layer is the input to the next
                    x_t = h_list[layer_idx]

            # Collect top-layer output if we want the full sequence
            if self.return_sequence:
                outputs_h.append(h_list[-1])  # top layer hidden state
                outputs_c.append(c_list[-1])  # top layer hidden state

        # Stack or get final
        if self.return_sequence:
            # [B, T, hidden_dim, H, W]
            outputs_h = torch.stack(outputs_h, dim=1)
            outputs_c = torch.stack(outputs_c, dim=1)
            return outputs_h, outputs_c
        else:
            # Just the final top-layer hidden state
            outputs_h = h_list[-1]

        return outputs_h, h_list[0], c_list[0]



class Sliser(nn.Module):
    def __init__(
            self,
    ):
        super(Sliser, self).__init__()

    def forward(self, x, units_masks, one_player):
        batch_size, num_channels, map_height, map_width = x.shape
        n = 15
        M = 16
        H, W = map_height, map_width

        # Coordinates are shape [B, M]
        x_cord = units_masks['x_cord']
        y_cord = units_masks['y_cord']

        if one_player is not None:
            x_cord = x_cord[:, 16 * one_player:16 *(one_player + 1)]
            y_cord = y_cord[:, 16 * one_player:16 *(one_player + 1)]

        # Repeat each batch item M times
        # Result: [B*M, C, H, W]
        fm_expanded = x.unsqueeze(1).expand(-1, M, -1, -1, -1)
        fm_expanded = fm_expanded.reshape(batch_size * M, num_channels, H, W)

        # Flatten the coords in the same order: shape [B*M]
        xs_flat = x_cord.reshape(-1)
        ys_flat = y_cord.reshape(-1)

        # Build the base grid from [-1..1], size n×n
        linspace_1d = torch.linspace(-1, 1, steps=n, device=x.device)
        grid_y, grid_x = torch.meshgrid(linspace_1d, linspace_1d, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1)         # (n, n, 2)
        base_grid = base_grid.unsqueeze(0).expand(batch_size * M, n, n, 2)

        # Convert (x,y) from [0..W-1]/[0..H-1] -> [-1..1]
        # so pixel i => normalized => (i/(W-1))*2 -1
        xs_norm = (xs_flat / (W - 1)) * 2 - 1
        ys_norm = (ys_flat / (H - 1)) * 2 - 1
        xs_norm = xs_norm.view(-1, 1, 1)
        ys_norm = ys_norm.view(-1, 1, 1)

        # Use (n-1)/(W-1), ensures the patch spans exactly +/- 7 pixels from the center
        scale_x = (n - 1) / (W - 1)  # e.g. 14/23 ~ 0.6087
        scale_y = (n - 1) / (H - 1)  # e.g. 14/23 ~ 0.6087

        final_grid = torch.zeros_like(base_grid)
        final_grid[..., 0] = base_grid[..., 0] * scale_x + ys_norm
        final_grid[..., 1] = base_grid[..., 1] * scale_y + xs_norm

        # For visualization only
        in_bounds = (
                (final_grid[..., 0] >= -1) & (final_grid[..., 0] <= 1) &
                (final_grid[..., 1] >= -1) & (final_grid[..., 1] <= 1)
        ).float()

        # Sample patches
        patches = F.grid_sample(
            fm_expanded,
            final_grid,
            mode='nearest',       # exact nearest pixel
            padding_mode='zeros',
            align_corners=True    # consistent with our -1..1 corner definition
        )

        # Concatenate a mask channel
        in_bounds = in_bounds.unsqueeze(1)
        patches_with_mask = torch.cat([patches, in_bounds], dim=1)

        return patches_with_mask

class Predictor(nn.Module):
    def __init__(
            self,
            in_channels: int,
            flags
    ):
        super(Predictor, self).__init__()

        self.one_prediction_head = flags.one_prediction_head

        if self.one_prediction_head:
            self.prediction = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=3,
                    kernel_size=(3, 3),
                    padding='same',
                )
            )
        else:
            self.position = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=1,
                    kernel_size=(3, 3),
                    padding='same',
                )
            )
            self.near_position = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=1,
                    kernel_size=(3, 3),
                    padding='same',
                )
            )
            self.sensor_mask = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=1,
                    kernel_size=(3, 3),
                    padding='same',
                )
            )




    def forward(
            self,
            x: torch.Tensor,
    ):
        if self.one_prediction_head:
            return self.prediction(x)
        else:
            return torch.cat([self.position(x), self.near_position(x), self.sensor_mask(x)], dim=1)

class DictSeparateActorResnBinary(nn.Module):
    def __init__(
            self,
            in_channels: int,
            action_space: gym.spaces.Dict,
            flags,
    ):
        super(DictSeparateActorResnBinary, self).__init__()
        if not all([isinstance(space, gym.spaces.MultiDiscrete) for space in action_space.spaces.values()]):
            act_space_types = {key: type(space) for key, space in action_space.spaces.items()}
            raise ValueError(f"All action spaces must be MultiDiscrete. Found: {act_space_types}")
        if not all([len(space.shape) == 3 for space in action_space.spaces.values()]):
            act_space_ndims = {key: space.shape for key, space in action_space.spaces.items()}
            raise ValueError(f"All action spaces must have 3 dimensions. Found: {act_space_ndims}")
        if not all([space.nvec.min() == space.nvec.max() for space in action_space.spaces.values()]):
            act_space_n_acts = {key: np.unique(space.nvec) for key, space in action_space.spaces.items()}
            raise ValueError(f"Each action space must have the same number of actions throughout the space. "
                             f"Found: {act_space_n_acts}")

        self.enable_per_unit_resnet = flags.enable_per_unit_resnet

        self.n_actions = {
            key: space.nvec.max() for key, space in action_space.spaces.items()
        }
        self.action_plane_shapes = {
            key: space.shape[:-2] for key, space in action_space.spaces.items()
        }
        assert all([len(aps) == 1 for aps in self.action_plane_shapes.values()])

        # MLP for non-linear transformations
        if self.enable_per_unit_resnet:
            self.sap_mlp = nn.Sequential(
                nn.Linear((flags.hidden_dim // 16), 1),
            )
            self.move_mlp = nn.Sequential(
                nn.Linear((flags.hidden_dim // 16) * (6*6), (flags.hidden_dim // 16) * (6*6) // 2),
                nn.LeakyReLU(),
                nn.Linear((flags.hidden_dim // 16) * (6*6) // 2, 6),
            )
            #self.move_mlp = nn.Sequential(
            #    nn.Linear((flags.hidden_dim // 16) * (6 * 6), 6),
            #)
        else:
            self.sap_mlp = nn.Sequential(
                nn.Linear(in_channels, 1),
            )
            self.move_mlp = nn.Sequential(
                nn.Linear(in_channels * 225, 6),
            )

        # Action heads
        self.actors = ['worker', 'sapper']

        self.aam_name = '_with_mask' if flags.enable_sap_masks else '_without_mask'

        if self.enable_per_unit_resnet:
            self.small_resnet = nn.Sequential(
                ResidualBlockNoMask(
                    in_channels=flags.hidden_dim,
                    out_channels=flags.hidden_dim // 2,
                    kernel_size=flags.kernel_size,
                    normalize=flags.normalize,
                    activation=nn.LeakyReLU,
                    squeeze_excitation=flags.enable_se_in_per_unit_resnet,
                ),
                ResidualBlockNoMask(
                    in_channels=flags.hidden_dim // 2,
                    out_channels=flags.hidden_dim // 4,
                    kernel_size=flags.kernel_size,
                    normalize=flags.normalize,
                    activation=nn.LeakyReLU,
                    squeeze_excitation=flags.enable_se_in_per_unit_resnet,
                )
                ,
                ResidualBlockNoMask(
                    in_channels=flags.hidden_dim // 4,
                    out_channels=flags.hidden_dim // 8,
                    kernel_size=flags.kernel_size,
                    normalize=flags.normalize,
                    activation=nn.LeakyReLU,
                    squeeze_excitation=flags.enable_se_in_per_unit_resnet,
                ),
                ResidualBlockNoMask(
                    in_channels=flags.hidden_dim // 8,
                    out_channels=flags.hidden_dim // 16,
                    kernel_size=flags.kernel_size,
                    normalize=flags.normalize,
                    activation=nn.LeakyReLU,
                    squeeze_excitation=flags.enable_se_in_per_unit_resnet,
                ),
            )

            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=flags.hidden_dim // 16,
                    out_channels=flags.hidden_dim // 16,
                    kernel_size=(5, 5),
                    stride=2,
                    padding='valid'
                ),
                nn.LeakyReLU()
            )


        self.sliser = Sliser()

        self.feature_merger = nn.Sequential(
            nn.Linear(
                103 + 5 + in_channels + 3 + 1,  # One-hot encoding replaces embeddings
                in_channels * 2,
                ),
            nn.LeakyReLU(),
            nn.Linear(
                in_channels * 2,  # One-hot encoding replaces embeddings
                in_channels,
                ),
            nn.LeakyReLU()
        )

    def forward(
            self,
            x: torch.Tensor,
            prediction: torch.Tensor,
            available_actions_mask: Dict[str, torch.Tensor],
            sample: bool,
            units_masks: Dict[str, torch.Tensor],
            actions_per_square: Optional[int] = 1,
            profiler: ScopedProfiler = ScopedProfiler(enabled=False),
            one_player=None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        with profiler.block("prepare"):

            policy_logits_out_robot = {}
            actions_out_robot = {}

            batch_size, num_channels, map_height, map_width = x.shape
            x = torch.cat([x, prediction], dim=1)

            patches = self.sliser(x, units_masks, one_player)
            combined_features = patches.view(batch_size, -1, num_channels + 3 + 1, 15 * 15)
            combined_features = combined_features.permute(0, 1, 3, 2).contiguous()

            # One-hot encoding for discrete features
            #discrete_flat = units_masks['embedding_features'].to(torch.int64)  # [B * P]
            #discrete_one_hot = F.one_hot(discrete_flat, num_classes=self.num_classes).float()  # Shape: [B * P, 103]

            #discrete_one_hot = discrete_one_hot.any(dim=2).float().view(-1, self.num_classes)  # Convert to binary presence (OR operation over C_discrete dimension)
            #discrete_one_hot = discrete_one_hot.permute(0, 3, 1, 2).contiguous()  # Shape: [B * P, 875, H, W]

            #discrete_one_hot = discrete_one_hot.view(-1, 103)
            # Continuous features
            umoh = units_masks['one_hot_encoded_embedding_features']
            umcf = units_masks['continues_features']

            if one_player is not None:
                umoh = umoh[:, 16 * one_player:16 *(one_player + 1), ...]
                umcf = umcf[:, 16 * one_player:16 *(one_player + 1), ...]

            discrete_one_hot = umoh.view(-1, 103)
            continuous_flat = umcf.view(-1, 5)  # [B * P, 5]

            combined_per_unit_features = torch.cat([discrete_one_hot, continuous_flat], dim=-1)  # [B * P, 103 + 5]

            # Expand unit features across all action positions
            combined_per_unit_features_expanded = combined_per_unit_features.unsqueeze(0).unsqueeze(0).view(batch_size, 16, 1, 108)
            combined_per_unit_features_expanded = combined_per_unit_features_expanded.expand(batch_size, 16, 225, 108)

            # Merge features
            combined_features = torch.cat([combined_features, combined_per_unit_features_expanded], dim=-1)
            combined_features = self.feature_merger(combined_features)

        np = 1 if (one_player is not None) else 2

        if self.enable_per_unit_resnet:
            combined_features = combined_features.view(batch_size * 16, 15, 15, num_channels)
            combined_features = self.small_resnet(combined_features.permute(0, 3, 1, 2).contiguous())

            for key in self.actors:
                profiler.begin_block(key)
                #n_actions = self.n_actions[key]
                #action_plane_shape = self.action_plane_shapes[key]

                if key == 'sapper':
                    n_actions = 225
                    action_plane_shape = (1,)
                    d = combined_features.permute(0, 2, 3, 1).contiguous()
                    robot_logits = self.sap_mlp(d).squeeze(-1)
                else:
                    n_actions = 6
                    action_plane_shape = (1,)
                    d = self.conv(combined_features).permute(0, 2, 3, 1).contiguous()
                    d = d.view(batch_size, 16, 6*6, -1)
                    combined_flatten = d.flatten(start_dim=-2, end_dim=-1)
                    robot_logits = self.move_mlp(combined_flatten)

                robot_logits = robot_logits.view(batch_size // np, np, 16, n_actions, *action_plane_shape)
                robot_logits = robot_logits.permute(0, 4, 1, 2, 3).contiguous()

                aam = available_actions_mask[key+self.aam_name]
                if one_player is not None:
                    aam = aam[:, :, one_player, ...].unsqueeze(2)
                orig_dtype = aam.dtype
                aam_new_type = aam.to(dtype=torch.int64)
                aam_filled = torch.where(
                    (~aam).all(dim=-1, keepdim=True),
                    torch.ones_like(aam_new_type),
                    aam_new_type.to(dtype=torch.int64)
                ).to(orig_dtype)

                assert robot_logits.shape == aam_filled.shape
                robot_logits = robot_logits + torch.where(
                    aam_filled,
                    torch.zeros_like(robot_logits),
                    torch.zeros_like(robot_logits) + float("-inf")
                )

                actions_robot = DictSeparateActorResn.logits_to_actions(robot_logits.view(-1, n_actions), sample, actions_per_square)
                policy_logits_out_robot[key] = robot_logits
                actions_out_robot[key] = actions_robot.view(*robot_logits.shape[:-1], -1)

                profiler.end_block(key)

        else:
            for key in self.actors:
                profiler.begin_block(key)
                #n_actions = self.n_actions[key]
                #action_plane_shape = self.action_plane_shapes[key]

                if key == 'sapper':
                    n_actions = 225
                    action_plane_shape = (1,)
                    robot_logits = self.sap_mlp(combined_features).squeeze(-1)
                else:
                    n_actions = 6
                    action_plane_shape = (1,)
                    combined_flatten = combined_features.flatten(start_dim=-2, end_dim=-1)
                    robot_logits = self.move_mlp(combined_flatten)

                robot_logits = robot_logits.view(batch_size // np, np, 16, n_actions, *action_plane_shape)
                robot_logits = robot_logits.permute(0, 4, 1, 2, 3).contiguous()

                aam = available_actions_mask[key+self.aam_name]
                if one_player is not None:
                    aam = aam[:, :, one_player, ...].unsqueeze(2)
                orig_dtype = aam.dtype
                aam_new_type = aam.to(dtype=torch.int64)
                aam_filled = torch.where(
                    (~aam).all(dim=-1, keepdim=True),
                    torch.ones_like(aam_new_type),
                    aam_new_type.to(dtype=torch.int64)
                ).to(orig_dtype)

                assert robot_logits.shape == aam_filled.shape
                robot_logits = robot_logits + torch.where(
                    aam_filled,
                    torch.zeros_like(robot_logits),
                    torch.zeros_like(robot_logits) + float("-inf")
                )

                actions_robot = DictSeparateActorResn.logits_to_actions(robot_logits.view(-1, n_actions), sample, actions_per_square)
                policy_logits_out_robot[key] = robot_logits
                actions_out_robot[key] = actions_robot.view(*robot_logits.shape[:-1], -1)

                profiler.end_block(key)

        return policy_logits_out_robot, actions_out_robot





class DictSeparateActorResn(nn.Module):
    def __init__(
            self,
            in_channels: int,
            action_space: gym.spaces.Dict,
            flags,
    ):
        super(DictSeparateActorResn, self).__init__()
        if not all([isinstance(space, gym.spaces.MultiDiscrete) for space in action_space.spaces.values()]):
            act_space_types = {key: type(space) for key, space in action_space.spaces.items()}
            raise ValueError(f"All action spaces must be MultiDiscrete. Found: {act_space_types}")
        if not all([len(space.shape) == 3 for space in action_space.spaces.values()]):
            act_space_ndims = {key: space.shape for key, space in action_space.spaces.items()}
            raise ValueError(f"All action spaces must have 3 dimensions. Found: {act_space_ndims}")
        if not all([space.nvec.min() == space.nvec.max() for space in action_space.spaces.values()]):
            act_space_n_acts = {key: np.unique(space.nvec) for key, space in action_space.spaces.items()}
            raise ValueError(f"Each action space must have the same number of actions throughout the space. "
                             f"Found: {act_space_n_acts}")
        self.n_actions = {
            key: space.nvec.max() for key, space in action_space.spaces.items()
        }
        self.action_plane_shapes = {
            key: space.shape[:-2] for key, space in action_space.spaces.items()
        }
        assert all([len(aps) == 1 for aps in self.action_plane_shapes.values()])


        self.enable_per_unit_resnet = flags.enable_per_unit_resnet

        # MLP for non-linear transformations
        # Simplified MLP
        self.sap_mlp = nn.Sequential(
            nn.Linear(in_channels, 1),
        )
        self.move_mlp = nn.Sequential(
            nn.Linear(in_channels * 225, 6),
        )

        # Optional: Remove attention if unnecessary
        #self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)

        # Action heads
        self.actors = ['worker', 'sapper']

        self.sliser = Sliser()

        self.embedding = nn.Embedding(
            num_embeddings=103,  # Total unique discrete values
            embedding_dim=32
        )

        self.feature_merger = nn.Sequential(
            *[
                nn.Linear(
                    32 * 4 + 5 + in_channels + 3 + 1,
                    in_channels,
                    ),
                nn.LeakyReLU()
            ]
        )

    #def transform(self, sap_action):




    def forward(
            self,
            x: torch.Tensor,
            prediction: torch.Tensor,
            available_actions_mask: Dict[str, torch.Tensor],
            sample: bool,
            units_masks: Dict[str, torch.Tensor],
            actions_per_square: Optional[int] = 1,  # Example default for MAX_OVERLAPPING_ACTIONS
            profiler: ScopedProfiler = ScopedProfiler(enabled=False)
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

        with profiler.block("prepare"):

            policy_logits_out_robot = {}
            actions_out_robot = {}


            batch_size, num_channels, map_height, map_width = x.shape

            x = torch.cat([x, prediction], dim=1)

            patches = self.sliser(x, units_masks)

            #print(patches.shape)
            #torch.cuda.synchronize()
            #patches = self.small_resnet_compiled(patches)
            #print(patches.shape)
            #patches = self.small_resnet(patches)
            #print(patches.shape)
            #assert False
            #torch.cuda.synchronize()
            #print(patches.shape)


            #torch.cuda.synchronize()
            combined_features = patches.view(batch_size, -1, num_channels + 3 + 1, 15 * 15)
            #torch.cuda.synchronize()

            #torch.cuda.synchronize()
            #avg_for_move = combined_features.
            #torch.cuda.synchronize()
            #print(combined_features.shape)


            #torch.cuda.synchronize()
            combined_features = combined_features.permute(0, 1, 3, 2).contiguous()
            #torch.cuda.synchronize()


            #print(combined_features.shape)

            #torch.cuda.synchronize()

            discrete_flat = units_masks['embedding_features'].view(-1).contiguous().to(torch.int32)
            discrete_embedded = self.embedding(discrete_flat)
            discrete_embedded = discrete_embedded.view(-1, 32 * 4)
            continues_flat = units_masks['continues_features'].view(-1, 5)

            combined_per_unit_features = torch.cat([discrete_embedded, continues_flat], dim=-1)
            # 32 x 199

            combined_per_unit_features_expanded = combined_per_unit_features.unsqueeze(0).unsqueeze(0).view(batch_size, 16, 1, 133)
            combined_per_unit_features_expanded = combined_per_unit_features_expanded.expand(batch_size, 16, 225, 133)

            #merged = self.feature_merger(combined_per_unit_features)
            # 32 x 64

            #merged_expanded = merged.unsqueeze(0).unsqueeze(0).view(batch_size, 16, 1, 64)
            #merged_expanded = merged_expanded.expand(batch_size, 16, 225, 64)

            combined_features = torch.cat([combined_features, combined_per_unit_features_expanded], dim=-1)

            combined_features = self.feature_merger(combined_features)

            #combined_features_for_sap = torch.cat([combined_features, merged_expanded], dim=-1)
            #combined_flatten = combined_features.flatten(start_dim=-2, end_dim=-1)
            #combined_features_for_move = torch.cat([combined_flatten, merged])

        for key in self.actors:
            profiler.begin_block(key)
            n_actions = self.n_actions[key]
            action_plane_shape = self.action_plane_shapes[key]

            if key == 'sapper':
                robot_logits = self.sap_mlp(combined_features).squeeze(-1)
            else:
                #combined_flatten = combined_features.flatten(start_dim=-2, end_dim=-1)
                #combined_flatten = torch.cat([combined_flatten, merged.view(batch_size, 16, -1)], dim=-1)
                combined_flatten = combined_features.flatten(start_dim=-2, end_dim=-1)
                robot_logits = self.move_mlp(combined_flatten)

            robot_logits = robot_logits.view(batch_size // 2, 2, 16, n_actions, *action_plane_shape)
            robot_logits = robot_logits.permute(0, 4, 1, 2, 3).contiguous()

            aam = available_actions_mask[key]
            orig_dtype = aam.dtype
            aam_new_type = aam.to(dtype=torch.int64)
            aam_filled = torch.where(
                (~aam).all(dim=-1, keepdim=True),
                torch.ones_like(aam_new_type),
                aam_new_type.to(dtype=torch.int64)
            ).to(orig_dtype)

            assert robot_logits.shape == aam_filled.shape
            robot_logits = robot_logits + torch.where(
                aam_filled,
                torch.zeros_like(robot_logits),
                torch.zeros_like(robot_logits) + float("-inf")
            )

            actions_robot = DictSeparateActorResn.logits_to_actions(robot_logits.view(-1, n_actions), sample, actions_per_square)
            policy_logits_out_robot[key] = robot_logits
            actions_out_robot[key] = actions_robot.view(*robot_logits.shape[:-1], -1)
            profiler.end_block(key)


        return policy_logits_out_robot, actions_out_robot


    @staticmethod
    @torch.no_grad()
    def logits_to_actions(logits: torch.Tensor, sample: bool, actions_per_square: Optional[int]) -> torch.Tensor:
        if actions_per_square is None:
            actions_per_square = logits.shape[-1]
        if sample:
            probs = F.softmax(logits, dim=-1)
            # In case there are fewer than MAX_OVERLAPPING_ACTIONS available actions, we add a small eps value
            probs = torch.where(
                (probs > 0.).sum(dim=-1, keepdim=True) >= actions_per_square,
                probs,
                probs + 1e-10
            )
            return torch.multinomial(
                probs,
                num_samples=min(actions_per_square, probs.shape[-1]),
                replacement=False
            )
        else:
            return logits.argsort(dim=-1, descending=True)[..., :actions_per_square]


class BaselineLayer(nn.Module):
    def __init__(self, in_channels: int, reward_space: RewardSpec, n_value_heads: int):
        super(BaselineLayer, self).__init__()
        assert n_value_heads >= 1
        self.reward_min = reward_space.reward_min
        self.reward_max = reward_space.reward_max
        self.multi_headed = False

        self.linear = nn.Sequential(
            nn.Linear((in_channels + 7 + 1) * 2, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1),
        )
        if reward_space.zero_sum:
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.Sigmoid()
        #if not reward_space.only_once:
            # Expand reward space to n_steps for rewards that occur more than once
        reward_space_expanded = reward_space.scaler
        self.reward_min *= reward_space_expanded
        self.reward_max *= reward_space_expanded

        self.interaction_encoder = nn.MultiheadAttention(embed_dim=in_channels + 7 + 1, num_heads=4)



    def forward(self, x: torch.Tensor,  units_masks, value_head_idxs: Optional[torch.Tensor]):
        """
        Expects an input of shape b * 2, n_channels, x, y
        Returns an output of shape b, 2
        """
        # Average feature planes

        x = torch.flatten(x, start_dim=-2, end_dim=-1).mean(dim=-1)
        x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-8)


        af = units_masks['additional_features'].view(-1, 7)
        #print(af)
        #print(units_masks['additional_features'].shape)
        #print(x.shape)
        x = torch.cat([x, af], dim=1)
        x = torch.nn.functional.pad(x, (0, 1))

        batch_size, feature_dim = x.shape

        #arr = []
        #for i in range(1, len(x), 2):
        #    arr.append(x[i - 1])
        #    arr.append(x[i])
        #    arr.append(x[i])
        #    arr.append(x[i - 1])

        #x1 = torch.stack(arr, dim=0)

        # Create base indices for each pair
        pairs = torch.arange(0, batch_size).reshape(-1, 2)
        #print(pairs)

        # For each row (e.g., [0, 1]), we want the pattern: [0, 1, 1, 0]
        pattern = torch.tensor([0, 1, 1, 0], dtype=torch.long, device=pairs.device)
        reordered_indices = pairs[:, pattern].reshape(-1)
        #print(reordered_indices)
        # Apply indexing to reorder
        result = x[reordered_indices]

        x = result.view(batch_size, -1)
        #print(x.shape)
        #assert False

        #print(x1.shape, result.shape, (x1 - result).sum())
        #assert False

        #print(x.shape)
        #player0 = x[0::2]  # Even indices for player0
        #print(player0.shape)
        #player1 = x[1::2]  # Odd indices for player1
        #print(player1.shape)

        # Interleave and duplicate based on the desired pattern
        #result = torch.cat([player0, player1, player1, player0], dim=0)
        #print(result.shape)

        # Project and reshape input

        #print(x.shape)
        # Interaction encoding
        x = x.reshape(batch_size, 2, -1)
        #print(x.shape)
        x = x.permute(1, 0, 2)  # (batch_size, n_agents, features) -> (n_agents, batch_size, features)
        x, _ = self.interaction_encoder(x, x, x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, n_agents, features)

        x = x.reshape(batch_size, -1)
        #print(x.shape)

        #assert False

        if self.multi_headed:
            x = self.linear(x, value_head_idxs.squeeze()).view(-1, 2)
        else:
            x = self.linear(x).view(-1, 2)
        # Rescale to [0, 1], and then to the desired reward space
        x = self.activation(x)

        return x * (self.reward_max - self.reward_min) + self.reward_min




class BaselineLayerV2(nn.Module):
    def __init__(self, flags, in_channels: int, reward_space: RewardSpec, n_value_heads: int):
        super(BaselineLayerV2, self).__init__()
        assert n_value_heads >= 1
        self.reward_min = reward_space.reward_min
        self.reward_max = reward_space.reward_max

        self.linear = nn.Sequential(
            nn.Linear(128, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1),
        )
        if reward_space.zero_sum:
            self.activation = nn.Softmax(dim=-1)
        else:
            self.activation = nn.Sigmoid()

        reward_space_expanded = reward_space.scaler
        self.reward_min *= reward_space_expanded
        self.reward_max *= reward_space_expanded

        self.attention_to_own = CrossAttentionTransformer(
            dim=128,
            depth=1,
            heads=8,
            dim_head=flags.transformer_dim_head,
            mlp_dim=256,
            kv_dim=in_channels,
        )
        self.attention_to_enemy = CrossAttentionTransformer(
            dim=128,
            depth=1,
            heads=8,
            dim_head=flags.transformer_dim_head,
            mlp_dim=256,
            kv_dim=in_channels,
        )
        self.query_preproc = nn.Linear(7, 128)



    def forward(self, x: torch.Tensor,  units_masks, value_head_idxs: Optional[torch.Tensor]):
        """
        Expects an input of shape b * 2, n_channels, x, y
        Returns an output of shape b, 2
        """
        # Average feature planes

        bs2, n_channels, map_height, map_width = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        assert x.shape == (bs2, map_height, map_width, n_channels)

        queries = self.query_preproc(units_masks['additional_features'].view(-1, 7)).unsqueeze(1)

        own_data = x.view(bs2, map_height * map_width, n_channels)
        enemy_data = torch.flip(
            x.view(bs2//2, 2, map_height, map_width, n_channels),
            dims=[1]).view(bs2, map_height * map_width, n_channels)

        queries = self.attention_to_own(queries, own_data)
        queries = self.attention_to_enemy(queries, enemy_data)

        x = self.linear(queries).view(-1, 2)
        x = self.activation(x)

        return x * (self.reward_max - self.reward_min) + self.reward_min


class BasicActorCriticNetwork(nn.Module):
    def __init__(
            self,
            input_model: nn.Module,
            resnet: nn.Module,
            base_out_channels: int,
            action_space: gym.spaces.Dict,
            reward_space: RewardSpec,
            actor_critic_activation: Callable = nn.ReLU,
            n_action_value_layers: int = 2,
            n_value_heads: int = 1,
            flags=None
    ):
        super(BasicActorCriticNetwork, self).__init__()
        self.dict_input_layer = DictInputLayer()
        self.input_model = input_model

        self.enable_resnet = flags.enable_resnet

        self.learner_batch_size = flags.batch_size * (flags.unroll_length + 1)
        self.flags_batch_size = flags.batch_size

        if self.enable_resnet:
            self.resnet = resnet


        self.base_out_channels = base_out_channels

        if n_action_value_layers < 2:
            raise ValueError("n_action_value_layers must be >= 2 in order to use spectral_norm")

        self.enable_transformer = flags.enable_transformer

        if self.enable_transformer:
            assert flags.n_transformer_blocks > 0
            if flags.n_transformer_blocks == 1:
                self.transformer = SpatialTransformer(
                    hidden_dim=self.base_out_channels,
                    height=24,
                    width=24,
                    num_heads=4
                )
            else:
                self.transformer = nn.Sequential(*[SpatialTransformer(
                    hidden_dim=self.base_out_channels,
                    height=24,
                    width=24,
                    num_heads=4
                ) for _ in range(flags.n_transformer_blocks)])

        self.enable_transformer_v2 = flags.enable_transformer_v2

        if self.enable_transformer_v2:
            self.transformer_v2 = TransformerWithPosEmbed(
                dim=self.base_out_channels,
                depth=flags.n_transformer_v2_blocks,
                heads=4,
                dim_head=flags.transformer_dim_head,
                mlp_dim=256
            )

        self.actor_base = self.make_spectral_norm_head_base(
            n_layers=n_action_value_layers,
            n_channels=self.base_out_channels,
            activation=actor_critic_activation
        )

        self.enable_per_unit_resnet = flags.enable_per_unit_resnet


        if flags.use_embedding_input:
            self.actor = DictSeparateActorResn(self.base_out_channels, action_space, flags)
        else:
            self.actor = DictSeparateActorResnBinary(self.base_out_channels, action_space, flags)


        self.baseline_base = self.make_spectral_norm_head_base(
            n_layers=n_action_value_layers,
            n_channels=self.base_out_channels,
            activation=actor_critic_activation
        )


        if not flags.use_baseline_v2:
            self.baseline = BaselineLayer(
                in_channels=self.base_out_channels,
                reward_space=reward_space,
                n_value_heads=n_value_heads,
            )
        else:
            self.baseline = BaselineLayerV2(
                flags,
                in_channels=self.base_out_channels,
                reward_space=reward_space,
                n_value_heads=n_value_heads,
            )

        self.predictor = Predictor(in_channels=self.base_out_channels, flags=flags)

        try:
            ksz = flags.lstm_k_size
        except AttributeError:
            ksz = 3

        self.enable_lstm = flags.enable_lstm
        if self.enable_lstm:
            self.convlstm = ConvLSTM(
                input_dim=self.base_out_channels,
                hidden_dim=128,
                kernel_size=(ksz, ksz),
                batch_first=False,
                bias=True,
                return_sequence=True
            )
            self.lstm_output_merger = nn.Sequential(
                nn.Conv2d(self.base_out_channels + 128 * 2, self.base_out_channels * 2, (1, 1)),
                nn.LeakyReLU(),
                nn.Conv2d(self.base_out_channels * 2, self.base_out_channels, (1, 1)),
                nn.LeakyReLU()
            )


    def forward(
            self,
            x: Dict[str, Union[dict, torch.Tensor]],
            sample: bool = True,
            one_player = None,
            profiler : ScopedProfiler = ScopedProfiler(enabled=False),
            **actor_kwargs
    ) -> Dict[str, Any]:
        with profiler.block("dict_input_layer"):
            x, available_actions_mask, units_masks, rnn_hidden_state_h, rnn_hidden_state_c, prev_prediction, done = self.dict_input_layer(x)

        with profiler.block("input_model"):
            base_out = self.input_model(x, prev_prediction, one_player)

        batch_size_p = base_out.shape[0]
        n_p = 1 if (one_player is not None) else 2

        if self.enable_resnet:
            base_out = self.resnet(base_out)

        rnn_hidden_state_h = rnn_hidden_state_h.view(batch_size_p, *rnn_hidden_state_h.shape[2:])
        rnn_hidden_state_c = rnn_hidden_state_c.view(batch_size_p, *rnn_hidden_state_c.shape[2:])
        origian_shape = rnn_hidden_state_h.shape

        if self.enable_lstm:
            if batch_size_p == self.learner_batch_size * n_p:
                base_out = base_out.view(-1, self.flags_batch_size * n_p, *base_out.shape[1:])
                rnn_hidden_state_h = rnn_hidden_state_h.view(-1, self.flags_batch_size * n_p, *rnn_hidden_state_h.shape[1:])[0]
                rnn_hidden_state_c = rnn_hidden_state_c.view(-1, self.flags_batch_size * n_p, *rnn_hidden_state_c.shape[1:])[0]
                done = done.view(-1, self.flags_batch_size, *done.shape[1:])
            else:
                done = None
                base_out = base_out.unsqueeze(0)

            rnn_hidden_state_h, rnn_hidden_state_c = self.convlstm(base_out, hidden_states_h=rnn_hidden_state_h, hidden_states_c=rnn_hidden_state_c, done=done, one_player=one_player)
            rnn_hidden_state_h = rnn_hidden_state_h.permute(1, 0, 2, 3, 4).contiguous().view(origian_shape)
            rnn_hidden_state_c = rnn_hidden_state_c.permute(1, 0, 2, 3, 4).contiguous().view(origian_shape)
            base_out = self.lstm_output_merger(torch.cat([base_out.view(batch_size_p, *base_out.shape[2:]), rnn_hidden_state_h, rnn_hidden_state_c], dim=1))

        if self.enable_transformer:
            base_out = self.transformer(base_out)

        if self.enable_transformer_v2:
            bs, channels, h, w = base_out.shape
            base_out = base_out.permute(0, 2, 3, 1).contiguous()
            assert base_out.shape == (bs, h, w, channels)
            base_out = self.transformer_v2(base_out)
            base_out = base_out.permute(0, 3, 1, 2).contiguous()
            assert base_out.shape == (bs, channels, h, w)


        prediction = self.predictor(base_out)


        with profiler.block("actor"):
            policy_logits, actions = self.actor(
                self.actor_base(base_out),
                prediction,
                available_actions_mask=available_actions_mask,
                sample=sample,
                units_masks=units_masks,
                profiler=profiler,
                one_player=one_player,
                **actor_kwargs
            )

        with profiler.block("baseline"):
            if one_player is not None:
                baseline = None
            else:
                baseline = self.baseline(self.baseline_base(base_out), units_masks, None)

        return dict(
            actions_GPU=actions,
            policy_logits_GPU_CPU=policy_logits,
            baseline_GPU=baseline,
            prediction_GPU_CPU=prediction.view(batch_size_p // n_p, n_p, 3, 24, 24),
            rnn_hidden_state_h_GPU=rnn_hidden_state_h.view(batch_size_p // n_p, n_p, *rnn_hidden_state_h.shape[1:]),
            rnn_hidden_state_c_GPU=rnn_hidden_state_c.view(batch_size_p // n_p, n_p, *rnn_hidden_state_c.shape[1:]),
        )

    def sample_actions(self, *args, **kwargs):
        return self.forward(*args, sample=True, **kwargs)

    def select_best_actions(self, *args, **kwargs):
        return self.forward(*args, sample=False, **kwargs)

    @staticmethod
    def make_spectral_norm_head_base(n_layers: int, n_channels: int, activation: Callable) -> nn.Module:
        """
        Returns the base of an action or value head, with the final layer of the base/the semifinal layer of the
        head spectral normalized.
        NB: this function actually returns a base with n_layer - 1 layers, leaving the final layer to be filled in
        with the proper action or value output layer.
        """
        assert n_layers >= 2
        layers = []
        for i in range(n_layers - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, (1, 1)))
            layers.append(activation())
        layers.append(
            nn.utils.spectral_norm(nn.Conv2d(n_channels, n_channels, (1, 1)))
        )
        layers.append(activation())

        return nn.Sequential(*layers)
