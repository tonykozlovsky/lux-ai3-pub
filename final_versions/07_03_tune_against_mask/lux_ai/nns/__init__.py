import logging
import torch
from torch import nn
from typing import Optional

from .models import BasicActorCriticNetwork
from .in_blocks import ConvEmbeddingInputLayer, BinaryInputLayer
from .conv_blocks import ResidualBlockNoMask, ParallelDilationResidualBlock
from ..lux_gym import create_flexible_obs_space, obs_spaces
from ..utility_constants import BOARD_SIZE
import os


def create_model(
        flags,
        device: torch.device,
        teacher_model_flags: Optional = None,
        is_teacher_model: bool = False
) -> nn.Module:
    obs_space = create_flexible_obs_space(flags, teacher_model_flags)
    if isinstance(obs_space, obs_spaces.MultiObs):
        if is_teacher_model:
            obs_space_prefix = "teacher_"
        else:
            obs_space_prefix = "student_"
        assert obs_space_prefix in obs_space.named_obs_spaces, f"{obs_space_prefix} not in {obs_space.named_obs_spaces}"
    else:
        obs_space_prefix = ""

    return _create_model(
        teacher_model_flags if is_teacher_model else flags,
        device,
        obs_space,
        obs_space_prefix
    )


def _create_model(
        flags,
        device: torch.device,
        obs_space: obs_spaces.BaseObsSpace,
        obs_space_prefix: str
):
    act_space = flags.act_space()
    if flags.use_embedding_input:
        conv_embedding_input_layer = ConvEmbeddingInputLayer(
            obs_space=obs_space.get_obs_spec(),
            embedding_dim=flags.embedding_dim,
            out_dim=flags.hidden_dim,
            n_merge_layers=flags.n_merge_layers,
            obs_space_prefix=obs_space_prefix
        )
    else:
        conv_embedding_input_layer = BinaryInputLayer(out_dim=flags.hidden_dim, obs_space_prefix=obs_space_prefix)
    if flags.model_arch == "conv_model":
        resnet = nn.Sequential(*[ResidualBlockNoMask(
            in_channels=flags.hidden_dim,
            out_channels=flags.hidden_dim,
            kernel_size=flags.kernel_size,
            normalize=flags.normalize,
            activation=nn.LeakyReLU,
        ) for _ in range(flags.n_blocks)])
    elif flags.model_arch == "conv_model_pd":
        resnet = nn.Sequential(*[ParallelDilationResidualBlock(
            in_channels=flags.hidden_dim,
            out_channels=flags.hidden_dim,
            kernel_size=flags.kernel_size,
            normalize=flags.normalize,
            activation=nn.LeakyReLU,
        ) for _ in range(flags.n_blocks)])
    else:
        raise NotImplementedError(f"Model_arch: {flags.model_arch}")

    model = BasicActorCriticNetwork(
        input_model=conv_embedding_input_layer,
        resnet=resnet,
        base_out_channels=flags.hidden_dim,
        action_space=act_space.get_action_space(),
        reward_space=flags.reward_space.get_reward_spec(),
        n_value_heads=1,
        flags=flags
    )
    return model.to(device=device)
