import os

from typing import Callable, Dict, Optional, Tuple, Union

import gym.spaces
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _index_select(embedding_layer: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
    out = embedding_layer.weight.index_select(0, x.view(-1))
    return out.view(*x.shape, -1)


def _forward_select(embedding_layer: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
    return embedding_layer(x)



def _player_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=1)


def _player_cat(x: torch.Tensor) -> torch.Tensor:
    return torch.flatten(x, start_dim=1, end_dim=2)


class DictInputLayer(nn.Module):
    @staticmethod
    def forward(
            x: Dict[str, Union[Dict, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        return (x["obs"],
                x["info"]["GPU1_available_actions_mask"],
                x["info"]["GPU1_units_masks"],
                x["info"]["rnn_hidden_state_h_GPU"],
                x["info"]["rnn_hidden_state_c_GPU"],
                x["info"]["prediction_GPU_CPU"],
                x["done_GPU_CPU"])


class ConvEmbeddingInputLayer(nn.Module):
    def __init__(
            self,
            obs_space: gym.spaces.Dict,
            embedding_dim: int,
            out_dim: int,
            n_merge_layers: int = 1,
            activation: Callable = nn.LeakyReLU,
            obs_space_prefix: str = ""
    ):
        super(ConvEmbeddingInputLayer, self).__init__()


        self.obs_space_prefix = obs_space_prefix

        self.embedding_dim = embedding_dim
        self.out_dim = out_dim

        # Embedding for discrete features
        self.embedding = nn.Embedding(
            num_embeddings=875,  # Total unique discrete values
            embedding_dim=embedding_dim
        )

        n_discrete_features = 72
        n_continuous_features = 74

        # Merge layers: Conv2D to merge continuous + embedded discrete features
        self.merge_layers = nn.Sequential(
            *[
                 nn.Conv2d(
                     embedding_dim * n_discrete_features + n_continuous_features + 3,
                     out_dim,
                     (1, 1)
                 ),
                 activation()
             ] * n_merge_layers
        )


        # Final Conv2D to 64 output channels
        #self.final_conv = nn.Conv2d(
        #    in_channels=out_dim,
        #    out_channels=out_dim,
        #    kernel_size=1,
        #    stride=1,
        #    padding=1
        #)


    def forward(self, x, prev_prediction) -> torch.Tensor:
        # x is a dictionary with keys 'continues_features' and 'discrete_features'
        continuous = x[self.obs_space_prefix + "GPU1_continues_features"]  # Shape: [B, 1, P, C_continuous, H, W]
        discrete = x[self.obs_space_prefix + "GPU1_discrete_features"]    # Shape: [B, 1, P, C_discrete, H, W]

        B, _, P, C_discrete, H, W = discrete.shape

        # Move P to batch dimension for discrete features
        discrete = discrete.view(B * P, C_discrete, H, W)  # Shape: [B * P, C_discrete, H, W]

        # Apply embedding
        discrete_flat = discrete.view(B * P, -1).contiguous().to(dtype=torch.int32)  # Shape: [B * P, C_discrete * H * W]
        embedded = self.embedding(discrete_flat)  # Shape: [B * P, C_discrete * H * W, embedding_dim]

        # Reshape back to spatial dimensions
        embedded = embedded.view(B * P, C_discrete, H, W, self.embedding_dim)  # Shape: [B * P, C_discrete, H, W, embedding_dim]
        embedded = embedded.permute(0, 4, 1, 2, 3).contiguous()  # Shape: [B * P, embedding_dim, C_discrete, H, W]
        embedded = embedded.view(B * P, -1, H, W)  # Flatten embedding dimension: [B * P, C_discrete * embedding_dim, H, W]

        # Move P to batch dimension for continuous features
        continuous_flat = continuous.view(B * P, -1, H, W)  # Shape: [B * P, C_continuous, H, W]

        prev_prediction_flat = prev_prediction.view(B * P, -1, H, W)

        prev_prediction_flat = torch.sigmoid(prev_prediction_flat)

        # Merge continuous and embedded discrete features
        combined = torch.cat([continuous_flat, embedded, prev_prediction_flat], dim=1)  # Shape: [B * P, C_continuous + C_discrete * embedding_dim, H, W]

        # Apply merge layers
        output = self.merge_layers(combined)  # Shape: [B * P, out_dim, H, W]

        # Final Conv2D to produce 64 output channels
        #output = self.final_conv(combined)  # Shape: [B * P, 64, H, W]

        # Reshape back to include P in batch
        #output = output.view(B, P, 64, H, W)  # Shape: [B, P, 64, H, W]

        return output


class BinaryInputLayer(nn.Module):
    def __init__(
            self,
            num_classes: int = 875,  # The total unique discrete values
            out_dim: int = 64,
            n_merge_layers: int = 1,
            activation: Callable = nn.LeakyReLU,
            obs_space_prefix = ""
    ):
        super(BinaryInputLayer, self).__init__()

        self.obs_space_prefix = obs_space_prefix

        self.num_classes = num_classes  # 875 unique discrete values
        self.out_dim = out_dim

        n_continuous_features = 74  # Continuous feature count

        # Merge layers: Conv2D to merge continuous + one-hot discrete features
        self.merge_layers = nn.Sequential(
            nn.Conv2d(
                num_classes + n_continuous_features + 3,  # 3 = prev_prediction channels
                num_classes // 2,
                (1, 1)
            ),
            activation(),
            nn.Conv2d(
                num_classes // 2,  # 3 = prev_prediction channels
                out_dim,
                (1, 1)
            ),
            activation()
        )

    def forward(self, x, prev_prediction, one_player=None) -> torch.Tensor:
        """
        x: dictionary with keys 'GPU1_continues_features' and 'GPU1_discrete_features'
        - Continuous shape: [B, 1, P, C_continuous, H, W]
        - Discrete shape: [B, 1, P, C_discrete, H, W]  (Each pixel has C_discrete values, but they don't overlap)
        """

        continuous = x[self.obs_space_prefix + "GPU1_continues_features"]  # Shape: [B, 1, P, C_continuous, H, W]
        output = x[self.obs_space_prefix + "GPU1_one_hot_encoded_discrete_features"]

        if one_player is not None:
            continuous = continuous[:, :, one_player, ...].unsqueeze(2)
            output = output[:, :, one_player, ...].unsqueeze(2)
            #prev_prediction = prev_prediction[:, one_player, ...].unsqueeze(1)

        B, _, P, _, H, W = output.shape

        # Move P to batch dimension for continuous features
        continuous_flat = continuous.view(B * P, -1, H, W)  # Shape: [B * P, C_continuous, H, W]
        output = output.view(B * P, -1, H, W)

        # Process previous predictions
        prev_prediction_flat = prev_prediction.view(B * P, -1, H, W)
        prev_prediction_flat = torch.sigmoid(prev_prediction_flat)

        # Merge continuous, one-hot discrete, and prev_prediction features
        output = torch.cat([output, continuous_flat, prev_prediction_flat], dim=1)  # Shape: [B * P, total_channels, H, W]

        # Apply merge layers
        output = self.merge_layers(output)  # Shape: [B * P, out_dim, H, W]

        return output
