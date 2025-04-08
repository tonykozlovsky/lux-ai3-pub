import copy
import logging
import math
import gym
import numpy as np
import torch

from ..utility_constants import BOARD_SIZE
from ..torchbeast.profiler import ScopedProfiler


class RewardSpaceWrapper(gym.Wrapper):
    def __init__(self, env, reward_space, profiler):
        super(RewardSpaceWrapper, self).__init__(env)
        self.reward_space = reward_space
        self.profiler = profiler

    def _get_rewards_and_done_3(self, obs, env_reward, env_done, info):
        if obs is None or env_reward is None or env_done is None or info is None: # kaggle
            return None, None
        rewards, done = self.reward_space.compute_rewards_and_done(obs, env_reward, env_done, info)
        return rewards, done

    def reset(self, **kwargs):
        reset = super(RewardSpaceWrapper, self).reset(**kwargs)
        obs3, env_reward, env_done, info3 = reset

        result = obs3, *self._get_rewards_and_done_3(obs3, env_reward, env_done, info3), info3
        return result

    def step(self, action):
        with self.profiler("RewardSpaceWrapper.step"):
            with self.profiler("super.step"):
                step = super(RewardSpaceWrapper, self).step(action)

            with self.profiler("get_rewards_and_done"):
                obs3, env_reward, env_done, info3 = step
                result = obs3, *self._get_rewards_and_done_3(obs3, env_reward, env_done, info3), info3
        return result


class LoggingEnv(gym.Wrapper):
    def __init__(self, env, reward_space, profiler):
        super(LoggingEnv, self).__init__(env)
        self.reward_space = reward_space
        self.vals_peak = {}
        self.reward_sums = {}
        self.profiler = profiler

    def aggregate_reward(self, reward):
        agg_reward = np.array([0., 0.])

        for key, val in reward.items():
            x, mult, r = val
            agg_reward += x * mult

        return agg_reward

    def scale_reward(self, reward):
        scaler = self.reward_space.get_reward_spec().scaler
        return reward * scaler


    def reset(self, **kwargs):
        reset = super(LoggingEnv, self).reset(**kwargs)

        self.reward_sums = {}

        obs3, reward3, done3, info3 = reset

        if obs3 is None or reward3 is None or done3 is None or info3 is None: # kaggle
            return obs3, reward3, done3, info3
        agg_reward = self.aggregate_reward(reward3)

        return obs3, self.scale_reward(agg_reward).astype(np.float32), done3, self.info3(info3, reward3, agg_reward)

    def info3(self, info, rewards, agg_reward):
        info = copy.copy(info)

        logs = {}

        agg_reward_sum = self.reward_sums.get("agg_rewards", np.zeros(2, dtype=float))
        agg_reward_sum += agg_reward
        self.reward_sums["agg_rewards"] = agg_reward_sum

        #logs["agg_rewards"] = [agg_reward_sum[0]]
        #logs["max_agg_rewards"] = [np.max(agg_reward_sum)]

        for key, val in rewards.items():
            x, mult, r = val
            reward_sum = self.reward_sums.get(f"{key}_{r + 1}", np.zeros(2, dtype=float))
            if math.isnan(reward_sum[0]):
                reward_sum = np.zeros(2, dtype=float)
            reward_sum += x * mult
            self.reward_sums[f"{key}_{r + 1}"] = reward_sum
            for i in range(5):
                if i != r:
                    full_key = f"{key}_{i + 1}"
                    if full_key not in self.reward_sums:
                        self.reward_sums[full_key] = [math.nan, math.nan]


            reward_sum = self.reward_sums.get(f"{key}_{r + 1}_metric", np.zeros(2, dtype=float))
            if math.isnan(reward_sum[0]):
                reward_sum = np.zeros(2, dtype=float)
            reward_sum += x
            self.reward_sums[f"{key}_{r + 1}_metric"] = reward_sum
            for i in range(5):
                if i != r:
                    full_key = f"{key}_{i + 1}_metric"
                    if full_key not in self.reward_sums:
                        self.reward_sums[full_key] = [math.nan, math.nan]

        for key, val in self.reward_sums.items():
            logs[key] = [val[0]]



        info.update({f"LOGGING_CPU_{key}": np.array(val, dtype=np.float32) for key, val in logs.items()})

        for key in ["params", "full_params", "state", "discount", "final_observation", "final_state", "player_0", "player_1"]:
            info.pop(key, None)
        return info


    def step(self, action):
        with self.profiler("LoggingEnv.step"):
            with self.profiler("super.step"):
                step = super(LoggingEnv, self).step(action)

            obs3, reward3, done3, info3 = step

            if obs3 is None or reward3 is None or done3 is None or info3 is None:
                return obs3, reward3, done3, info3

            agg_reward = self.aggregate_reward(reward3)
            result = obs3, self.scale_reward(agg_reward), done3, self.info3(info3, reward3, agg_reward)

        return result


class VecEnv(gym.Env):
    def __init__(self, envs, flags):
        self.envs = envs
        self.last_outs = [() for _ in range(len(self.envs))]
        self.flags = flags

    @staticmethod
    def _stack_dict(x):
        if isinstance(x[0], dict):
            return {key: VecEnv._stack_dict([i[key] for i in x]) for key in x[0].keys()}
        else:
            return np.stack([arr for arr in x], axis=0)

    @staticmethod
    def _vectorize_env_outs(env_outs):
        obs_list, reward_list, done_list, info_list = zip(*env_outs)
        obs_stacked = VecEnv._stack_dict(obs_list)
        reward_stacked = np.array(reward_list, dtype=np.float32)
        done_stacked = np.array(done_list)
        info_stacked = VecEnv._stack_dict(info_list)
        return obs_stacked, reward_stacked, done_stacked, info_stacked

    def reset(self, force: bool = False, custom_seeds = None, kaggle_observations=None, **kwargs):
        if force or custom_seeds is not None:
            # noinspection PyArgumentList
            self.last_outs = []
            for i, env in enumerate(self.envs):
                if kaggle_observations is not None:
                    if isinstance(kaggle_observations, tuple):
                        self.last_outs.append(
                            env.reset(kaggle_observation=(kaggle_observations[0][i], kaggle_observations[1]), force=force, **kwargs)
                        )
                    else:
                        self.last_outs.append(
                            env.reset(kaggle_observation=kaggle_observations[i], force=force, **kwargs)
                        )
                else:
                    if custom_seeds is not None:
                        self.last_outs.append(env.reset(force=force, custom_seed=custom_seeds[i], **kwargs))
                    else:
                        self.last_outs.append(env.reset(force=force, **kwargs))

            return VecEnv._vectorize_env_outs(self.last_outs)

        for i, env in enumerate(self.envs):
            # Check if env finished
            if self.last_outs[i][2]:
                # noinspection PyArgumentList
                self.last_outs[i] = env.reset(**kwargs)
        return VecEnv._vectorize_env_outs(self.last_outs)

    def step(self, action):
        if self.flags.kaggle:
            if action is not None:
                actions = [
                    {key: val[i] for key, val in action.items()} for i in range(1)
                ]
            else:
                actions = [None]
        else:
            actions = [
                {key: val[i] for key, val in action.items()} for i in range(len(self.envs))
            ]
        if self.flags.kaggle:
            self.last_outs = [env.step(a) for env, a in zip(self.envs[:1], actions[:1])]
        else:
            self.last_outs = [env.step(a) for env, a in zip(self.envs, actions)]
        result = VecEnv._vectorize_env_outs(self.last_outs)
        return result

    def close(self):
        return [env.close() for env in self.envs]

    def seed(self, seed) -> list:
        if seed is not None:
            return [env.seed(seed + i) for i, env in enumerate(self.envs)]
        else:
            return [env.seed(seed) for i, env in enumerate(self.envs)]

    @property
    def unwrapped(self):
        return [env.unwrapped for env in self.envs]

    @property
    def action_space(self):
        return [env.action_space for env in self.envs]

    @property
    def observation_space(self):
        return [env.observation_space for env in self.envs]

    @property
    def metadata(self):
        return [env.metadata for env in self.envs]


class PytorchEnv(gym.Wrapper):
    def __init__(self, env, device: torch.device = torch.device("cpu"), profiler: ScopedProfiler = ScopedProfiler(enabled=False)):
        super(PytorchEnv, self).__init__(env)
        self.device = device
        self.profiler = profiler

    def reset(self, **kwargs):
        result = tuple([self._to_tensor(out, idx) for idx, out in enumerate(super(PytorchEnv, self).reset(**kwargs))])
        return result

    def step(self, action):
        with self.profiler("PytorchEnv.step"):
            with self.profiler("precalc"):
                if action is not None:
                    action = {key: val.cpu().numpy() for key, val in action.items()}

            with self.profiler("super.step"):
                step_result = super(PytorchEnv, self).step(action)

            with self.profiler("result"):
                result = tuple([self._to_tensor(out, idx) for idx, out in enumerate(step_result)])
        return result

    def _to_tensor(self, x, idx, is_cpu=True, k=''):
        if isinstance(x, dict):
            return {key: self._to_tensor(val, idx, is_cpu and ('GPU1_' not in key), k + '_' + key) for key, val in x.items()}
        else:
            target_device = torch.device("cpu") if is_cpu else self.device
            try:
                if x.dtype == np.object_:
                    assert x[0] is None
                    return None
                if x.dtype != np.float32 and x.dtype != bool and x.dtype != np.int16:
                    print("Unexpected type:", k, idx, x.dtype, x.shape, x)
                    assert False
                    #pass
                x = torch.from_numpy(x)
                #if x.dtype == np.float32:
                #    x = x.to(torch.bfloat16)
            except TypeError:
                return None
            if target_device == torch.device("cpu"):
                return x
            else:
                x = x.pin_memory() # ?
                return x.to(target_device, non_blocking=True)



class DictEnv(gym.Wrapper):
    def __init__(self, env):
        super(DictEnv, self).__init__(env)

    @staticmethod
    def _dict_env_out(env_out: tuple) -> dict:
        obs, reward, done, info = env_out
        assert "obs" not in info.keys()
        assert "reward_GPU_CPU" not in info.keys()
        assert "done_GPU_CPU" not in info.keys()
        return dict(
            obs=obs,
            reward_GPU_CPU=reward,
            done_GPU_CPU=done,
            info=info
        )

    def reset(self, **kwargs):
        return DictEnv._dict_env_out(super(DictEnv, self).reset(**kwargs))

    def step(self, action):
        return DictEnv._dict_env_out(super(DictEnv, self).step(action))
