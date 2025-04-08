import torch
from typing import Optional

from ..torchbeast.profiler import ScopedProfiler

from . import act_spaces, obs_spaces, reward_spaces
from .sb3 import SB3Wrapper
from .wrappers import RewardSpaceWrapper, LoggingEnv, VecEnv, PytorchEnv, DictEnv

ACT_SPACES_DICT = {
    key: val for key, val in act_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, act_spaces.BaseActSpace)
}
OBS_SPACES_DICT = {
    key: val for key, val in obs_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, obs_spaces.BaseObsSpace)
}
REWARD_SPACES_DICT = {
    key: val for key, val in reward_spaces.__dict__.items()
    if isinstance(val, type) and issubclass(val, reward_spaces.BaseRewardSpace)
}
#
#REWARD_SPACES_DICT.update({
#    key: val for key, val in multi_subtask.__dict__.items()
#    if isinstance(val, type) and issubclass(val, reward_spaces.BaseRewardSpace)
#})


def create_flexible_obs_space(flags, teacher_flags: Optional) -> obs_spaces.BaseObsSpace:
    if teacher_flags is not None and teacher_flags.obs_space != flags.obs_space:
        # Train a student using a different observation space than the teacher
        return obs_spaces.MultiObs({
            "teacher_": teacher_flags.obs_space(),
            "student_": flags.obs_space()
        })
    else:
        return flags.obs_space()


def create_env(actor_index, flags, device: torch.device, teacher_flags: Optional = None,
               profiler: ScopedProfiler = ScopedProfiler(enabled=False), example=False) -> DictEnv:
    envs = []
    for i in range(flags.n_actor_envs):
        env = SB3Wrapper(
            actor_index * flags.n_actor_envs + i,
            act_space=flags.act_space(),
            obs_space=create_flexible_obs_space(flags, teacher_flags),
            flags=flags,
            example=example,
            profiler=profiler,
        )
        reward_space = create_reward_space(flags)
        env = RewardSpaceWrapper(env, reward_space, profiler=profiler)
        env = env.obs_space.wrap_env(env, flags, profiler=profiler)
        env = LoggingEnv(env, reward_space, profiler=profiler)
        envs.append(env)
    env = VecEnv(envs, flags=flags)
    env = PytorchEnv(env, device, profiler=profiler)
    env = DictEnv(env)
    return env


def create_reward_space(flags) -> reward_spaces.BaseRewardSpace:
    return flags.reward_space(**flags.reward_space_kwargs)
