import copy
from dataclasses import dataclass
from enum import Enum
import logging
import sys
from abc import ABC, abstractmethod

import gym
import matplotlib.pyplot as plt
import numpy as np

from ..utility_constants import BOARD_SIZE

P = 2 # Player count


class BaseObsSpace(ABC):
    # NB: Avoid using Discrete() space, as it returns a shape of ()
    # NB: "_COUNT" keys indicate that the value is used to scale the embedding of another value
    @abstractmethod
    def get_obs_spec(
            self
    ):
        pass

    @abstractmethod
    def wrap_env(self, env) -> gym.Wrapper:
        pass


class MultiObs(BaseObsSpace):
    def __init__(self, named_obs_spaces, *args, **kwargs):
        super(MultiObs, self).__init__(*args, **kwargs)
        self.named_obs_spaces = named_obs_spaces

    def get_obs_spec(
            self
    ) -> gym.spaces.Dict:
        return gym.spaces.Dict({
            name + key: val
            for name, obs_space in self.named_obs_spaces.items()
            for key, val in obs_space.get_obs_spec().spaces.items()
        })

    def wrap_env(self, env, flags, profiler) -> gym.Wrapper:
        return _MultiObsWrapper(env, self.named_obs_spaces, flags=flags, profiler=profiler)


class _MultiObsWrapper(gym.Wrapper):
    def __init__(self, env, named_obs_spaces, flags, profiler):
        super(_MultiObsWrapper, self).__init__(env)
        self.named_obs_space_wrappers = {key: val.wrap_env(env, flags=flags, profiler=profiler) for key, val in named_obs_spaces.items()}

    def reset(self, **kwargs):
        observation, reward, done, info = self.env.reset(**kwargs)
        return self.observation(observation), reward, done, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        return {
            name + key: val
            for name, obs_space in self.named_obs_space_wrappers.items()
            for key, val in obs_space.observation(observation).items()
        }

class FixedShapeObs(BaseObsSpace, ABC):
    pass


class Obs3(FixedShapeObs):
    def get_obs_spec(self):
        board_dims = BOARD_SIZE
        x = board_dims[0]
        y = board_dims[1]
        return gym.spaces.Dict({
            "my_units_presense_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "my_units_presense_mb": gym.spaces.MultiBinary((1, P, x, y)), # ok

            "my_units_n_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "my_units_n_mb": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 16 + 1), #

            "my_unit_max_energy": gym.spaces.Box(0., 1., (1, P, x, y)), 
            "my_unit_max_energy_move_mb": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 20 + 1), #
            "my_unit_max_energy_sap_mb": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 13 + 1), #

            "my_unit_sum_energy": gym.spaces.Box(0., 1., (1, P, x, y)), # ok

            "enemy_units_presense_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "enemy_units_presense_mb": gym.spaces.MultiBinary((1, P, x, y)), # ok

            "enemy_units_n_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "enemy_units_n_mb": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 16 + 1), #

            "enemy_unit_max_energy": gym.spaces.Box(0., 1., (1, P, x, y)), 
            "enemy_unit_max_energy_move_mb": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 20 + 1), #
            "enemy_unit_max_energy_sap_mb": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 13 + 1), #

            "enemy_unit_sum_energy": gym.spaces.Box(0., 1., (1, P, x, y)), # ok

            "asteroid_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "asteroid_mb": gym.spaces.MultiBinary((1, P, x, y)), # tile type can be united

            "nebula_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "nebula_mb": gym.spaces.MultiBinary((1, P, x, y)), # tile type can be united


            "anomaly_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "anomaly_mb": gym.spaces.MultiBinary((1, P, x, y)), # tile type can be united


            "tile_energy_box": gym.spaces.Box(-1., 1., (1, P, x, y)), # ok
            "tile_energy_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 40 + 1), # can be reduced ?

            "energy_known_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "energy_known_mb": gym.spaces.MultiBinary((1, P, x, y)), # ok

            "tile_energy_sign_box" : gym.spaces.Box(-1., 1., (1, P, x, y)), # ok
            "tile_energy_sign_d" : gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 2 + 1), # ok

            "sensor_mask": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "sensor_mask_mb": gym.spaces.MultiBinary((1, P, x, y)), # ok

            "potential_relic_reduced_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "potential_relic_reduced_mb": gym.spaces.MultiBinary((1, P, x, y)), # ok

            "potential_relic_0_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "potential_relic_0_mb": gym.spaces.MultiBinary((1, P, x, y)), # ok

            "potential_relic_1_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "potential_relic_1_mb": gym.spaces.MultiBinary((1, P, x, y)), # ok

            "potential_relic_2_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "potential_relic_2_mb": gym.spaces.MultiBinary((1, P, x, y)), # ok

            "not_relic_reduced_box": gym.spaces.Box(0., 1., (1, P, x, y)), #
            "not_relic_reduced_mb": gym.spaces.MultiBinary((1, P, x, y)), #

            "not_relic_0_box": gym.spaces.Box(0., 1., (1, P, x, y)), #
            "not_relic_0_mb": gym.spaces.MultiBinary((1, P, x, y)), #


            "not_relic_1_box": gym.spaces.Box(0., 1., (1, P, x, y)), #
            "not_relic_1_mb": gym.spaces.MultiBinary((1, P, x, y)), #


            "not_relic_2_box": gym.spaces.Box(0., 1., (1, P, x, y)), #
            "not_relic_2_mb": gym.spaces.MultiBinary((1, P, x, y)), #


            "guarranted_relic_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "guarranted_relic_mb": gym.spaces.MultiBinary((1, P, x, y)), # ok

            "relic_score_box": gym.spaces.Box(0., 1., (1, P, x, y)), # need properly fill

            "relic_node_box": gym.spaces.Box(0., 1., (1, P, x, y)), # ok
            "relic_node_mb": gym.spaces.MultiBinary((1, P, x, y)), # ok


            "enemy_on_relic_probability": gym.spaces.Box(0., 1., (1, P, x, y)), # ok


            "unit_move_cost_box": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "unit_sap_cost_box": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "unit_sap_range_box": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "unit_sensor_range_box": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "nebula_energy_reduction_box": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "sap_dropoff_factor_box": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "unit_energy_void_factor_box": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "nebula_vision_reduction_min_box": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "nebula_vision_reduction_max_box": gym.spaces.Box(0., 1., shape=(1, P)), # ok


            "unit_move_cost_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 5), # ok
            "unit_sap_cost_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 21), # ok
            "unit_sap_range_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 5), # ok
            "unit_sensor_range_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 4), # ok
            "nebula_energy_reduction_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 6 + 1), # ok
            "sap_dropoff_factor_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 3 + 1), # ok [-1, 0, 1, 2]
            "unit_energy_void_factor_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 4 + 1), # [-1, 0, 1, 2, 3].  [0.0625, 0.125, 0.25, 0.375],
            "nebula_vision_reduction_min_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 8), # [0, 1, 2, 3, 4, 5, 6, 7]
            "nebula_vision_reduction_max_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 8), # [0, 1, 2, 3, 4, 5, 6, 7]


            "my_points_plus": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "my_points_plus_d" : gym.spaces.MultiDiscrete(np.zeros((1, P)) + 16 + 1), # ok

            "enemy_points_plus": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "enemy_points_plus_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 16 + 1), # ok

            "advantage_points": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "advantage_points_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 32 + 1), # ok

            "relic_nodes_found": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "relic_nodes_found_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 3 + 1), # ok

            #[-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15]
            "shift_param": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "shift_param_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 8 + 1), # ok

            "steps_to_shift": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "steps_to_shift_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 10 + 1 + 1), # ok [-1, ..., 10]

            "shift_side": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "shift_side_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 2 + 1), # ok [-1, 0, 1]

            "possible_positions_for_relic_node_spawn_reduced": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "possible_positions_for_relic_node_spawn_reduced_mb": gym.spaces.MultiBinary((1, P, x, y)),

            "possible_positions_for_relic_node_spawn_0": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "possible_positions_for_relic_node_spawn_0_mb": gym.spaces.MultiBinary((1, P, x, y)),

            "possible_positions_for_relic_node_spawn_1": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "possible_positions_for_relic_node_spawn_1_mb": gym.spaces.MultiBinary((1, P, x, y)),

            "possible_positions_for_relic_node_spawn_2": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "possible_positions_for_relic_node_spawn_2_mb": gym.spaces.MultiBinary((1, P, x, y)),
            

            "possible_positions_for_relic_node_timer_reduced": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "possible_positions_for_relic_node_timer_reduced_mb": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 10 + 1),

            "possible_positions_for_relic_node_timer_0": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "possible_positions_for_relic_node_timer_0_mb": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 10 + 1),

            "possible_positions_for_relic_node_timer_1": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "possible_positions_for_relic_node_timer_1_mb": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 10 + 1),

            "possible_positions_for_relic_node_timer_2": gym.spaces.Box(0., 1., shape=(1, P, x, y)),
            "possible_positions_for_relic_node_timer_2_mb": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 10 + 1),

            "my_points": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "enemy_points": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "team_points_advantage": gym.spaces.Box(-1., 1., shape=(1, P)), # ok
            "team_points_advantage_ratio": gym.spaces.Box(0., 1., shape=(1, P)), # ok

            "turn": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "turn_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 11), # ok

            "round": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "round_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 5), # ok

            "winning": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "winning_d": gym.spaces.MultiDiscrete(np.zeros((1, P)) + 3), # ok


            "border_box": gym.spaces.Box(0., 1., (1, P, x, y)),
            "border_mb": gym.spaces.MultiBinary((1, P, x, y)),

            "distance_to_my_spawn_box": gym.spaces.Box(0., 1., (1, P, x, y)),
            "distance_to_my_spawn_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 46 + 1), # ok

            "distance_to_enemy_spawn_box": gym.spaces.Box(0., 1., (1, P, x, y)),
            "distance_to_enemy_spawn_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 46 + 1), # ok

            "side_of_map_box": gym.spaces.Box(0., 1., (1, P, x, y)),
            "side_of_map_mb": gym.spaces.MultiBinary((1, P, x, y)),

            "tile_unseen_ticks": gym.spaces.Box(0., 1., (1, P, x, y)),
            "tile_unseen_ticks_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 20 + 1),

            "enemy_seen_ticks_ago": gym.spaces.Box(0., 1., (1, P, x, y)),
            "enemy_seen_ticks_ago_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 20 + 1),

            "enemy_seen_energy": gym.spaces.Box(0., 1., (1, P, x, y)),
            "enemy_seen_energy_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 50 + 2),

            "my_seen_ticks_ago": gym.spaces.Box(0., 1., (1, P, x, y)),
            "my_seen_ticks_ago_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 20 + 1),

            "my_seen_energy": gym.spaces.Box(0., 1., (1, P, x, y)),
            "my_seen_energy_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 50 + 2),

            "my_unexpected_energy_change": gym.spaces.Box(0., 1., (1, P, x, y)),
            "my_unexpected_energy_change_in_saps": gym.spaces.Box(0., 1., (1, P, x, y)),

            "enemy_unexpected_energy_change": gym.spaces.Box(0., 1., (1, P, x, y)),
            "enemy_unexpected_energy_change_in_saps": gym.spaces.Box(0., 1., (1, P, x, y)),

            "my_unexpected_energy_change_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 41),
            "my_unexpected_energy_change_in_saps_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 41),

            "enemy_unexpected_energy_change_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 41),
            "enemy_unexpected_energy_change_in_saps_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 41),

            "no_more_spawn": gym.spaces.MultiBinary((1, P)), # ok

            "need_exploration_0": gym.spaces.MultiBinary((1, P)), # ok
            "need_exploration_1": gym.spaces.MultiBinary((1, P)), # ok
            "need_exploration_2": gym.spaces.MultiBinary((1, P)), # ok

            "my_wins_box": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "my_wins_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 6),

            "enemy_wins_box": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "enemy_wins_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 6),

            "my_wins_advantage": gym.spaces.Box(0., 1., shape=(1, P)), # ok
            "my_wins_advantage_d": gym.spaces.MultiDiscrete(np.zeros((1, P, x, y)) + 11),
        })

    def wrap_env(self, env, flags, profiler) -> gym.Wrapper:
        return _Obs3(env, flags=flags, profiler=profiler)

class SpaceType(Enum):
    DISCRETE = "Discrete"
    CONTINUOUS = "Continuous"


@dataclass
class SpaceInfo:
    type: SpaceType
    size: int


def _get_simplified_spec():
    heavy_spec = Obs3().get_obs_spec().spaces

    spec = {}
    for key, val in heavy_spec.items():
        if isinstance(val, gym.spaces.MultiBinary):
            spec[key] = SpaceInfo(SpaceType.DISCRETE, 2)
        elif isinstance(val, gym.spaces.MultiDiscrete):
            spec[key] = SpaceInfo(SpaceType.DISCRETE, int(val.nvec.min()))
        else:
            spec[key] = SpaceInfo(SpaceType.CONTINUOUS, -1)

    return spec


simplified_spec = _get_simplified_spec()

class _Obs3(gym.Wrapper):

    def __init__(self, env: gym.Env, flags, profiler):
        super(_Obs3, self).__init__(env)
        self._empty_obs = {}
        for key, spec in Obs3().get_obs_spec().spaces.items():
            if isinstance(spec, gym.spaces.MultiBinary) or isinstance(spec, gym.spaces.MultiDiscrete):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.int64)
            elif isinstance(spec, gym.spaces.Box):
                self._empty_obs[key] = np.zeros(spec.shape, dtype=np.float32) + spec.low
            else:
                raise NotImplementedError(f"{type(spec)} is not an accepted observation space.")

        self.steps = 0
        self.flags = flags
        self.profiler = profiler

    def reset(self, **kwargs):
        reset = self.env.reset(**kwargs)

        obs3, reward3, done3, info3 = reset

        result = self.observation(obs3), reward3, done3, info3
        return result

    def step(self, action):
        with self.profiler("_Obs3.step"):
            with self.profiler("super.step"):
                step = self.env.step(action)

            obs3, reward3, done3, info3 = step

            if obs3 is None or reward3 is None or done3 is None or info3 is None: # kaggle
                return obs3, reward3, done3, info3

            with self.profiler("generate observation"):
                result = self.observation(obs3), reward3, done3, info3

        return result


    def observation(self, env_obs):
        with self.profiler("copy"):
            obs = {
                key: val.copy() if val.ndim == 2 else val[:, :, :24, :24].copy()
                for key, val in self._empty_obs.items()
            }

        self.profiler.begin_block("fill data")

        for player_idx, (key, player_obs) in enumerate(env_obs.items()):
            positions = player_obs['units']['position']
            energies = player_obs['units']['energy']

            for pos, energy in zip(positions[player_idx], energies[player_idx]):
                if pos[0] == -1 or energy < 0:
                    continue
                x, y = pos
                obs['my_units_presense_box'][0, player_idx, x, y] = 1
                obs['my_units_presense_mb'][0, player_idx, x, y] = 1

                obs['my_units_n_box'][0, player_idx, x, y] += 1
                obs['my_units_n_mb'][0, player_idx, x, y] += 1

                obs['my_unit_max_energy'][0, player_idx, x, y] = max(obs['my_unit_max_energy'][0, player_idx, x, y], energy)

                n_moves = energy // player_obs['known_game_params']['unit_move_cost']
                n_moves = min(n_moves, 20)
                obs['my_unit_max_energy_move_mb'][0, player_idx, x, y] = max(obs['my_unit_max_energy_move_mb'][0, player_idx, x, y], n_moves)

                n_saps = energy // player_obs['known_game_params']['unit_sap_cost']
                n_saps = min(n_saps, 13)
                obs['my_unit_max_energy_sap_mb'][0, player_idx, x, y] = max(obs['my_unit_max_energy_sap_mb'][0, player_idx, x, y], n_saps)

                obs['my_unit_sum_energy'][0, player_idx, x, y] += energy

            obs['my_units_n_box'][0, player_idx] /= 16.
            obs['my_unit_max_energy'][0, player_idx]  /= 400.
            obs['my_unit_sum_energy'][0, player_idx]  /= 400. * 16.

            for pos, energy in zip(positions[1 - player_idx], energies[1 - player_idx]):
                if pos[0] == -1 or energy < 0:
                    continue
                x, y = pos
                obs['enemy_units_presense_box'][0, player_idx, x, y] = 1
                obs['enemy_units_presense_mb'][0, player_idx, x, y] = 1

                obs['enemy_units_n_box'][0, player_idx, x, y] += 1
                obs['enemy_units_n_mb'][0, player_idx, x, y] += 1

                obs['enemy_unit_max_energy'][0, player_idx, x, y] = max(obs['enemy_unit_max_energy'][0, player_idx, x, y], energy)

                n_moves = energy // player_obs['known_game_params']['unit_move_cost']
                n_moves = min(n_moves, 20)
                obs['enemy_unit_max_energy_move_mb'][0, player_idx, x, y] = max(obs['enemy_unit_max_energy_move_mb'][0, player_idx, x, y], n_moves)

                n_saps = energy // player_obs['known_game_params']['unit_sap_cost']
                n_saps = min(n_saps, 13)
                obs['enemy_unit_max_energy_sap_mb'][0, player_idx, x, y] = max(obs['enemy_unit_max_energy_sap_mb'][0, player_idx, x, y], n_saps)

                obs['enemy_unit_sum_energy'][0, player_idx, x, y] += energy

            obs['enemy_units_n_box'][0, player_idx]  /= 16.
            obs['enemy_unit_max_energy'][0, player_idx]  /= 400.
            obs['enemy_unit_sum_energy'][0, player_idx]  /= 400. * 16.


            #map_features = player_obs['map_features']
            tile_type = player_obs['remember_type']

            obs['asteroid_box'][0, player_idx, :, :] = (tile_type == 2)
            obs['asteroid_mb'][0, player_idx, :, :] = (tile_type == 2)

            obs['nebula_box'][0, player_idx, :, :] = (tile_type == 1)
            obs['nebula_mb'][0, player_idx, :, :] = (tile_type == 1)

            obs['anomaly_box'][0, player_idx, :, :] = (tile_type == 3)
            obs['anomaly_mb'][0, player_idx, :, :] = (tile_type == 3)

            obs['tile_energy_box'][0, player_idx, :, :] = player_obs['remember_map_energy'] / 20.
            obs['tile_energy_d'][0, player_idx, :, :] = player_obs['remember_map_energy'] + 20

            #assert np.min(obs['tile_energy_d'][0, player_idx, :, :]) >= 0 and np.max(obs['tile_energy_d'][0, player_idx, :, :]) <= 40

            obs['energy_known_box'][0, player_idx, :, :] = player_obs['remember_map_energy_known']
            obs['energy_known_mb'][0, player_idx, :, :] = player_obs['remember_map_energy_known']

            obs['tile_energy_sign_box'][0, player_idx, :, :] = np.sign(player_obs['remember_map_energy'])
            obs['tile_energy_sign_d'][0, player_idx, :, :] = np.sign(player_obs['remember_map_energy']) + 1

            #assert np.min(obs['tile_energy_sign_d'][0, player_idx, :, :]) >= 0 and np.max(obs['tile_energy_sign_d'][0, player_idx, :, :]) <= 2

            obs['sensor_mask'][0, player_idx] = player_obs['sensor_mask']
            obs['sensor_mask_mb'][0, player_idx] = player_obs['sensor_mask']


            obs["potential_relic_reduced_box"][0, player_idx] = player_obs['potential_relic_reduced']
            obs['potential_relic_reduced_mb'][0, player_idx] = player_obs['potential_relic_reduced']

            obs["potential_relic_0_box"][0, player_idx] = player_obs['potential_relic_0']
            obs['potential_relic_0_mb'][0, player_idx] = player_obs['potential_relic_0']

            obs["potential_relic_1_box"][0, player_idx] = player_obs['potential_relic_1']
            obs['potential_relic_1_mb'][0, player_idx] = player_obs['potential_relic_1']

            obs["potential_relic_2_box"][0, player_idx] = player_obs['potential_relic_2']
            obs['potential_relic_2_mb'][0, player_idx] = player_obs['potential_relic_2']

            obs["not_relic_reduced_box"][0, player_idx] = player_obs['not_relic_reduced']
            obs['not_relic_reduced_mb'][0, player_idx] = player_obs['not_relic_reduced']

            obs["not_relic_0_box"][0, player_idx] = player_obs['not_relic_0']
            obs['not_relic_0_mb'][0, player_idx] = player_obs['not_relic_0']

            obs["not_relic_1_box"][0, player_idx] = player_obs['not_relic_1']
            obs['not_relic_1_mb'][0, player_idx] = player_obs['not_relic_1']

            obs["not_relic_2_box"][0, player_idx] = player_obs['not_relic_2']
            obs['not_relic_2_mb'][0, player_idx] = player_obs['not_relic_2']









            obs["guarranted_relic_box"][0, player_idx] = player_obs['guarranted_relic']
            obs['guarranted_relic_mb'][0, player_idx] = player_obs['guarranted_relic']


            obs["relic_score_box"][0, player_idx] = player_obs['relic_score'] / max(1, np.max(player_obs['relic_score']))

            for relic in player_obs['remember_relic_nodes']:
                x, y = relic
                if x == -1:
                    continue
                obs['relic_node_box'][0, player_idx, x, y] = 1
                obs['relic_node_mb'][0, player_idx, x, y] = 1


            obs['unit_move_cost_box'][0, player_idx] = (player_obs['known_game_params']['unit_move_cost'] - 1) / 4.
            obs['unit_sap_cost_box'][0, player_idx] = (player_obs['known_game_params']['unit_sap_cost'] - 30) / 20.
            obs['unit_sap_range_box'][0, player_idx] = (player_obs['known_game_params']['unit_sap_range'] - 3) / 4.
            obs['unit_sensor_range_box'][0, player_idx] = (player_obs['known_game_params']['unit_sensor_range'] - 1) / 3.

            obs['unit_energy_void_factor_box'][0, player_idx] = (player_obs['unit_energy_void_factor'] + 1) / 5
            #assert obs['unit_energy_void_factor_box'][0, player_idx] >= 0 and obs['unit_energy_void_factor_box'][0, player_idx] <= 1

            obs['unit_energy_void_factor_d'][0, player_idx] = player_obs['unit_energy_void_factor'] + 1
            #assert obs['unit_energy_void_factor_d'][0, player_idx] >= 0 and obs['unit_energy_void_factor_d'][0, player_idx] <= 4

            obs['nebula_vision_reduction_min_box'][0, player_idx] = (player_obs['min_nebula_vision_reduction']) / 7
            #assert obs['nebula_vision_reduction_min_box'][0, player_idx] >= 0 and obs['nebula_vision_reduction_min_box'][0, player_idx] <= 1

            obs['nebula_vision_reduction_min_d'][0, player_idx] = player_obs['min_nebula_vision_reduction']
            #assert obs['nebula_vision_reduction_min_d'][0, player_idx] >= 0 and obs['nebula_vision_reduction_min_d'][0, player_idx] <= 7

            obs['nebula_vision_reduction_max_box'][0, player_idx] = (player_obs['max_nebula_vision_reduction']) / 7
            #assert obs['nebula_vision_reduction_max_box'][0, player_idx] >= 0 and obs['nebula_vision_reduction_max_box'][0, player_idx] <= 1

            obs['nebula_vision_reduction_max_d'][0, player_idx] = player_obs['max_nebula_vision_reduction']
            #assert obs['nebula_vision_reduction_max_d'][0, player_idx] >= 0 and obs['nebula_vision_reduction_max_d'][0, player_idx] <= 7

            n_red_value = player_obs['nebula_tile_energy_reduction']
            if n_red_value == -1:
                n_red_value = 0
            elif n_red_value == 0:
                n_red_value = 1
            elif n_red_value == 1:
                n_red_value = 2
            elif n_red_value == 2:
                n_red_value = 3
            elif n_red_value == 3:
                n_red_value = 4
            elif n_red_value == 5:
                n_red_value = 5
            elif n_red_value == 25:
                n_red_value = 6
            else:
                raise NotImplementedError(f"nebula_tile_energy_reduction={n_red_value}")

            obs['nebula_energy_reduction_box'][0, player_idx] = n_red_value / 6.
            obs['nebula_energy_reduction_d'][0, player_idx] = n_red_value
            #assert obs['nebula_energy_reduction_d'][0, player_idx] >= 0 and obs['nebula_energy_reduction_d'][0, player_idx] <= 6


            values = [0.25, 0.5, 1]
            n_sap_value = -1

            for idx, value in enumerate(values):
                if abs(value - player_obs['unit_sap_dropoff_factor']) < 1e-3:
                    n_sap_value = idx
                    break

            assert n_sap_value >= -1 and n_sap_value <= 2

            obs['sap_dropoff_factor_box'][0, player_idx] = (n_sap_value + 1) / 3
            obs['sap_dropoff_factor_d'][0, player_idx] = n_sap_value + 1
            #assert obs['sap_dropoff_factor_d'][0, player_idx] >= 0 and obs['sap_dropoff_factor_d'][0, player_idx] <= 3

            obs['unit_move_cost_d'][0, player_idx] = player_obs['known_game_params']['unit_move_cost'] - 1
            #assert obs['unit_move_cost_d'][0, player_idx] >= 0 and obs['unit_move_cost_d'][0, player_idx] <= 4

            obs['unit_sap_cost_d'][0, player_idx] = player_obs['known_game_params']['unit_sap_cost'] - 30
            #assert obs['unit_sap_cost_d'][0, player_idx] >= 0 and obs['unit_sap_cost_d'][0, player_idx] <= 20

            obs['unit_sap_range_d'][0, player_idx] = player_obs['known_game_params']['unit_sap_range'] - 3
            #assert obs['unit_sap_range_d'][0, player_idx] >= 0 and obs['unit_sap_range_d'][0, player_idx] <= 4

            obs['unit_sensor_range_d'][0, player_idx] = player_obs['known_game_params']['unit_sensor_range'] - 1
            #assert obs['unit_sensor_range_d'][0, player_idx] >= 0 and obs['unit_sensor_range_d'][0, player_idx] <= 3


            my_points = player_obs['team_points'][player_idx]
            enemy_points = player_obs['team_points'][1 - player_idx]
            my_prev_points = player_obs['prev_team_points'][player_idx]
            enemy_prev_points = player_obs['prev_team_points'][1 - player_idx]

            my_delta = my_points - my_prev_points
            #assert my_delta >= 0 and my_delta <= 16
            enemy_delta = enemy_points - enemy_prev_points
            #assert enemy_delta >= 0 and enemy_delta <= 16
            advantage = (my_delta - enemy_delta)
            #assert advantage >= -16 and advantage <= 16

            obs['my_points_plus'][0, player_idx] = my_delta / 16.
            obs['enemy_points_plus'][0, player_idx] = enemy_delta / 16.
            obs['advantage_points'][0, player_idx] = advantage / 16.

            obs['my_points_plus_d'][0, player_idx] = my_delta
            #assert obs['my_points_plus_d'][0, player_idx] >= 0 and obs['my_points_plus_d'][0, player_idx] <= 16
            obs['enemy_points_plus_d'][0, player_idx] = enemy_delta
            #assert obs['enemy_points_plus_d'][0, player_idx] >= 0 and obs['enemy_points_plus_d'][0, player_idx] <= 16
            obs['advantage_points_d'][0, player_idx] = advantage + 16
            #assert obs['advantage_points_d'][0, player_idx] >= 0 and obs['advantage_points_d'][0, player_idx] <= 32


            obs['relic_nodes_found'][0, player_idx] = np.sum(player_obs['remember_relic_nodes'][:, 0] != -1) / 6.
            #assert obs['relic_nodes_found'][0, player_idx] >= 0 and obs['relic_nodes_found'][0, player_idx] <= 1

            obs['relic_nodes_found_d'][0, player_idx] = np.sum(player_obs['remember_relic_nodes'][:, 0] != -1) // 2
            #assert obs['relic_nodes_found_d'][0, player_idx] >= 0 and obs['relic_nodes_found_d'][0, player_idx] <= 3


            n_unseen_relics = 0

            indices = np.argwhere(player_obs['guarranted_relic'] == 1)  # Get indexes where the condition is met
            for index in indices:
                x, y = index
                if not player_obs['sensor_mask'][x, y]:
                    n_unseen_relics += 1

            probability = min(1, enemy_delta / max(1, n_unseen_relics))

            if probability > 0.:
                for index in indices:
                    x, y = index
                    if not player_obs['sensor_mask'][x, y]:
                        obs['enemy_on_relic_probability'][0, player_idx, x, y] = probability

            shift_param_idx = player_obs['shift_param'][1] + 1
            #assert shift_param_idx >= 0 and shift_param_idx <= 8
            obs['shift_param'][0, player_idx] = shift_param_idx / 8.
            obs['shift_param_d'][0, player_idx] = shift_param_idx

            steps_to_shift = min(player_obs['steps_to_shift'], 10)
            #assert steps_to_shift >= -1 and steps_to_shift <= 10

            obs['steps_to_shift'][0, player_idx] = 1 if steps_to_shift == -1 else steps_to_shift / 10.
            obs['steps_to_shift_d'][0, player_idx] = steps_to_shift + 1

            shift_side = player_obs['shift_side']
            #assert shift_side >= -1 and shift_side <= 1
            obs['shift_side'][0, player_idx] = (shift_side + 1) / 2.
            obs['shift_side_d'][0, player_idx] = shift_side + 1

            obs['possible_positions_for_relic_node_spawn_reduced'][0, player_idx] = player_obs['possible_positions_for_relic_node_spawn_reduced']
            obs['possible_positions_for_relic_node_spawn_reduced_mb'][0, player_idx] = player_obs['possible_positions_for_relic_node_spawn_reduced']

            obs['possible_positions_for_relic_node_spawn_0'][0, player_idx] = player_obs['possible_positions_for_relic_node_spawn_0']
            obs['possible_positions_for_relic_node_spawn_0_mb'][0, player_idx] = player_obs['possible_positions_for_relic_node_spawn_0']

            obs['possible_positions_for_relic_node_spawn_1'][0, player_idx] = player_obs['possible_positions_for_relic_node_spawn_1']
            obs['possible_positions_for_relic_node_spawn_1_mb'][0, player_idx] = player_obs['possible_positions_for_relic_node_spawn_1']

            obs['possible_positions_for_relic_node_spawn_2'][0, player_idx] = player_obs['possible_positions_for_relic_node_spawn_2']
            obs['possible_positions_for_relic_node_spawn_2_mb'][0, player_idx] = player_obs['possible_positions_for_relic_node_spawn_2']

            timer = player_obs['possible_positions_for_relic_node_timer_reduced']
            obs['possible_positions_for_relic_node_timer_reduced'][0, player_idx] = np.minimum(timer, 100) / 100.
            #assert np.min(obs['possible_positions_for_relic_node_timer_reduced'][0, player_idx]) >= 0 and np.max(obs['possible_positions_for_relic_node_timer_reduced'][0, player_idx]) <= 1

            obs['possible_positions_for_relic_node_timer_reduced_mb'][0, player_idx] = np.minimum(timer, 10)
            #assert np.min(obs['possible_positions_for_relic_node_timer_reduced_mb'][0, player_idx]) >= 0 and np.max(obs['possible_positions_for_relic_node_timer_reduced_mb'][0, player_idx]) <= 10



            timer = player_obs['possible_positions_for_relic_node_timer_0']
            obs['possible_positions_for_relic_node_timer_0'][0, player_idx] = np.minimum(timer, 100) / 100.
            #assert np.min(obs['possible_positions_for_relic_node_timer_0'][0, player_idx]) >= 0 and np.max(obs['possible_positions_for_relic_node_timer_0'][0, player_idx]) <= 1

            obs['possible_positions_for_relic_node_timer_0_mb'][0, player_idx] = np.minimum(timer, 10)
            #assert np.min(obs['possible_positions_for_relic_node_timer_0_mb'][0, player_idx]) >= 0 and np.max(obs['possible_positions_for_relic_node_timer_0_mb'][0, player_idx]) <= 10


            timer = player_obs['possible_positions_for_relic_node_timer_1']
            obs['possible_positions_for_relic_node_timer_1'][0, player_idx] = np.minimum(timer, 100) / 100.
            #assert np.min(obs['possible_positions_for_relic_node_timer_1'][0, player_idx]) >= 0 and np.max(obs['possible_positions_for_relic_node_timer_1'][0, player_idx]) <= 1

            obs['possible_positions_for_relic_node_timer_1_mb'][0, player_idx] = np.minimum(timer, 10)
            #assert np.min(obs['possible_positions_for_relic_node_timer_1_mb'][0, player_idx]) >= 0 and np.max(obs['possible_positions_for_relic_node_timer_1_mb'][0, player_idx]) <= 10


            timer = player_obs['possible_positions_for_relic_node_timer_2']
            obs['possible_positions_for_relic_node_timer_2'][0, player_idx] = np.minimum(timer, 100) / 100.
            #assert np.min(obs['possible_positions_for_relic_node_timer_2'][0, player_idx]) >= 0 and np.max(obs['possible_positions_for_relic_node_timer_2'][0, player_idx]) <= 1

            obs['possible_positions_for_relic_node_timer_2_mb'][0, player_idx] = np.minimum(timer, 10)
            #assert np.min(obs['possible_positions_for_relic_node_timer_2_mb'][0, player_idx]) >= 0 and np.max(obs['possible_positions_for_relic_node_timer_2_mb'][0, player_idx]) <= 10




            my_points = player_obs['team_points'][player_idx]
            enemy_points = player_obs['team_points'][1 - player_idx]

            is_w = -1
            if my_points == enemy_points:
                is_w = 0
            elif my_points > enemy_points:
                is_w = 1


            obs['my_points'][0, player_idx] = np.clip(my_points / 1000.0, 0.0, 1.0)
            obs['enemy_points'][0, player_idx] = np.clip(enemy_points / 1000.0, 0.0, 1.0)

            obs['team_points_advantage'][0, player_idx] = np.clip((my_points - enemy_points) / 100.0, -1.0, 1.0)
            obs['team_points_advantage_ratio'][0, player_idx] = np.clip((my_points - enemy_points) / max(1, enemy_points), -1.0, 1.0)

            obs['winning'][0, player_idx] = is_w
            obs['winning_d'][0, player_idx] = is_w + 1

            obs['turn'][0, player_idx] = player_obs['match_steps'] / 100
            #assert obs['turn'][0, player_idx] >= 0 and obs['turn'][0, player_idx] <= 1

            obs['turn_d'][0, player_idx] = player_obs['match_steps'] // 10
            #assert obs['turn_d'][0, player_idx] >= 0 and obs['turn_d'][0, player_idx] <= 10

            obs['round'][0, player_idx] = (max(0, (player_obs['steps'] - 1)) // 101) / 4
            #assert obs['round'][0, player_idx] >= 0 and obs['round'][0, player_idx] <= 1

            obs['round_d'][0, player_idx] = (max(0, (player_obs['steps'] - 1)) // 101)
            #assert obs['round_d'][0, player_idx] >= 0 and obs['round_d'][0, player_idx] <= 4

            bb = obs['border_box'][0, player_idx]
            bb_d = obs['border_mb'][0, player_idx]
            for x in range(24):
                bb[x, 0] = 1
                bb[x, 23] = 1
                bb[0, x] = 1
                bb[23, x] = 1
                bb_d[x, 0] = 1
                bb_d[x, 23] = 1
                bb_d[0, x] = 1
                bb_d[23, x] = 1

            som = obs['side_of_map_box']
            som_d = obs['side_of_map_mb']

            d2s = obs['distance_to_my_spawn_box']
            d2s_d = obs['distance_to_my_spawn_d']
            for x in range(24):
                for y in range(24):
                    d2s[0, player_idx, x, y] = x + y
                    d2s_d[0, player_idx, x, y] = x + y
                    if x + y < 23:
                        som[0, player_idx, x, y] = 1
                        som_d[0, player_idx, x, y] = 1

            obs['distance_to_enemy_spawn_box'][0, player_idx] = 23 + 23 - d2s[0, player_idx]
            obs['distance_to_enemy_spawn_d'][0, player_idx] = 23 + 23 - d2s_d[0, player_idx]

            #assert (obs['distance_to_my_spawn_d'][0, player_idx] >= 0).all() and (obs['distance_to_my_spawn_d'][0, player_idx] <= 46).all()
            #assert (obs['distance_to_enemy_spawn_d'][0, player_idx] >= 0).all() and (obs['distance_to_enemy_spawn_d'][0, player_idx] <= 46).all()

            obs['distance_to_my_spawn_box'][0, player_idx] /= 46
            #assert (obs['distance_to_my_spawn_box'][0, player_idx] >= 0).all() and (obs['distance_to_my_spawn_box'][0, player_idx] <= 1).all()

            obs['distance_to_enemy_spawn_box'][0, player_idx] /= 46
            #assert (obs['distance_to_enemy_spawn_box'][0, player_idx] >= 0).all() and (obs['distance_to_enemy_spawn_box'][0, player_idx] <= 1).all()


            tut = np.minimum(player_obs['tile_unseen_ticks'], 20)
            obs["tile_unseen_ticks"][0, player_idx] = tut / 20.
            #assert (obs["tile_unseen_ticks"][0, player_idx] >= 0).all() and (obs["tile_unseen_ticks"][0, player_idx] <= 1).all()
            obs["tile_unseen_ticks_d"][0, player_idx] = tut
            #assert (obs["tile_unseen_ticks_d"][0, player_idx] >= 0).all() and (obs["tile_unseen_ticks_d"][0, player_idx] <= 20).all()

            esta = np.minimum(player_obs['enemy_seen_ticks_ago'], 20)
            obs["enemy_seen_ticks_ago"][0, player_idx] = esta / 20.
            #assert (obs["enemy_seen_ticks_ago"][0, player_idx] >= 0).all() and (obs["enemy_seen_ticks_ago"][0, player_idx] <= 1).all()
            obs["enemy_seen_ticks_ago_d"][0, player_idx] = esta
            #assert (obs["enemy_seen_ticks_ago_d"][0, player_idx] >= 0).all() and (obs["enemy_seen_ticks_ago_d"][0, player_idx] <= 20).all()

            ese = np.minimum(player_obs['enemy_seen_energy'], 400)
            obs["enemy_seen_energy"][0, player_idx] = (ese + 2) / 402.
            #assert (obs["enemy_seen_energy"][0, player_idx] >= 0).all() and (obs["enemy_seen_energy"][0, player_idx] <= 1).all()
            obs["enemy_seen_energy_d"][0, player_idx] = np.minimum(ese, 50) + 2
            #assert (obs["enemy_seen_energy_d"][0, player_idx] >= 0).all() and (obs["enemy_seen_energy_d"][0, player_idx] <= 52).all()


            msta = np.minimum(player_obs['my_seen_ticks_ago'], 20)
            obs["my_seen_ticks_ago"][0, player_idx] = msta / 20.
            #assert (obs["my_seen_ticks_ago"][0, player_idx] >= 0).all() and (obs["my_seen_ticks_ago"][0, player_idx] <= 1).all()
            obs["my_seen_ticks_ago_d"][0, player_idx] = msta
            #assert (obs["my_seen_ticks_ago_d"][0, player_idx] >= 0).all() and (obs["my_seen_ticks_ago_d"][0, player_idx] <= 20).all()

            mse = np.minimum(player_obs['my_seen_energy'], 400)
            obs["my_seen_energy"][0, player_idx] = (mse + 2) / 402.
            #assert (obs["my_seen_energy"][0, player_idx] >= 0).all() and (obs["my_seen_energy"][0, player_idx] <= 1).all()
            obs["my_seen_energy_d"][0, player_idx] = np.minimum(mse, 50) + 2
            #assert (obs["my_seen_energy_d"][0, player_idx] >= 0).all() and (obs["my_seen_energy_d"][0, player_idx] <= 52).all()



            x = np.clip(player_obs['my_unexpected_energy_change'], -100, 100)
            obs["my_unexpected_energy_change"][0, player_idx] = x / 100.
            #assert (obs["my_unexpected_energy_change"][0, player_idx] >= -1).all() and (obs["my_unexpected_energy_change"][0, player_idx] <= 1).all()
            
            x = np.clip(player_obs['my_unexpected_energy_change'], -20, 20)
            obs["my_unexpected_energy_change_d"][0, player_idx] = x + 20
            #assert (obs["my_unexpected_energy_change_d"][0, player_idx] >= 0).all() and (obs["my_unexpected_energy_change_d"][0, player_idx] <= 40).all()


            x = np.clip(player_obs['my_unexpected_energy_change_in_saps'], -10, 10)
            obs["my_unexpected_energy_change_in_saps"][0, player_idx] = x / 10.
            #assert (obs["my_unexpected_energy_change_in_saps"][0, player_idx] >= -1).all() and (obs["my_unexpected_energy_change_in_saps"][0, player_idx] <= 1).all()


            x = np.clip(player_obs['my_unexpected_energy_change_in_saps'], -20, 20)
            obs["my_unexpected_energy_change_in_saps_d"][0, player_idx] = x + 20
            #assert (obs["my_unexpected_energy_change_in_saps_d"][0, player_idx] >= 0).all() and (obs["my_unexpected_energy_change_in_saps_d"][0, player_idx] <= 40).all()


            x = np.clip(player_obs['enemy_unexpected_energy_change'], -100, 100)
            obs["enemy_unexpected_energy_change"][0, player_idx] = x / 100.
            #assert (obs["enemy_unexpected_energy_change"][0, player_idx] >= -1).all() and (obs["enemy_unexpected_energy_change"][0, player_idx] <= 1).all()

            x = np.clip(player_obs['enemy_unexpected_energy_change'], -20, 20)
            obs["enemy_unexpected_energy_change_d"][0, player_idx] = x + 20
            #assert (obs["enemy_unexpected_energy_change_d"][0, player_idx] >= 0).all() and (obs["enemy_unexpected_energy_change_d"][0, player_idx] <= 40).all()


            x = np.clip(player_obs['enemy_unexpected_energy_change_in_saps'], -10, 10)
            obs["enemy_unexpected_energy_change_in_saps"][0, player_idx] = x / 10.
            #assert (obs["enemy_unexpected_energy_change_in_saps"][0, player_idx] >= -1).all() and (obs["enemy_unexpected_energy_change_in_saps"][0, player_idx] <= 1).all()


            x = np.clip(player_obs['enemy_unexpected_energy_change_in_saps'], -20, 20)
            obs["enemy_unexpected_energy_change_in_saps_d"][0, player_idx] = x + 20
            #assert (obs["enemy_unexpected_energy_change_in_saps_d"][0, player_idx] >= 0).all() and (obs["enemy_unexpected_energy_change_in_saps_d"][0, player_idx] <= 40).all()

            obs['no_more_spawn'][0, player_idx] = player_obs['no_more_spawn']
            obs['need_exploration_0'][0, player_idx] = player_obs['need_exploration_0']
            obs['need_exploration_1'][0, player_idx] = player_obs['need_exploration_1']
            obs['need_exploration_2'][0, player_idx] = player_obs['need_exploration_2']

            # can affect how old models play they can play worse, and its very hard to support obs changes
            if False: # it adds assymetric play and feature is useful when not training on game win reward
                my_wins = player_obs['team_wins'][player_idx]
                obs['my_wins_box'][0, player_idx] = my_wins / 5.
                assert (obs['my_wins_box'][0, player_idx] >= 0).all() and (obs['my_wins_box'][0, player_idx] <= 1).all()
                obs['my_wins_d'][0, player_idx] = my_wins
                assert (obs['my_wins_d'][0, player_idx] >= 0).all() and (obs['my_wins_d'][0, player_idx] <= 5).all()


                enemy_wins = player_obs['team_wins'][1 - player_idx]
                obs['enemy_wins_box'][0, player_idx] = enemy_wins / 5.
                assert (obs['enemy_wins_box'][0, player_idx] >= 0).all() and (obs['enemy_wins_box'][0, player_idx] <= 1).all()
                obs['enemy_wins_d'][0, player_idx] = enemy_wins
                assert (obs['enemy_wins_d'][0, player_idx] >= 0).all() and (obs['enemy_wins_d'][0, player_idx] <= 5).all()

                my_wins_advantage = my_wins - enemy_wins
                obs['my_wins_advantage'][0, player_idx] = my_wins_advantage / 5.
                assert (obs['my_wins_advantage'][0, player_idx] >= -1).all() and (obs['my_wins_advantage'][0, player_idx] <= 1).all()
                obs['my_wins_advantage_d'][0, player_idx] = my_wins_advantage + 5
                assert (obs['my_wins_advantage_d'][0, player_idx] >= 0).all() and (obs['my_wins_advantage_d'][0, player_idx] <= 10).all()

            pass

        self.profiler.end_block("fill data")

        # with self.profiler("concatenate old"):
        #     spec = Obs3().get_obs_spec().spaces

        #     continuous_features_old = []
        #     discrete_features_old = []
        #     shift = 0
        #     for key, val in spec.items():
        #         if isinstance(val, gym.spaces.MultiBinary) or isinstance(val, gym.spaces.MultiDiscrete):
        #             if isinstance(val, gym.spaces.MultiBinary):
        #                 space_size = 2
        #             else:
        #                 space_size = int(val.nvec.min())
        #             arr = obs[key] + shift

        #             if arr.shape == (1, P, 24, 24):
        #                 pass
        #             elif arr.shape == (1, P):
        #                 reshaped_arr = np.zeros((1, P, 24, 24), dtype=arr.dtype)
        #                 for player_idx in range(P):
        #                     reshaped_arr[0, player_idx, :, :] = arr[0, player_idx]  # Broadcast value
        #                 arr = reshaped_arr

        #                 # arr = np.broadcast_to(arr.reshape(1, P, 1, 1), (1, P, 24, 24))
        #             else:
        #                 print("What", key, arr.shape, file=sys.stderr)

        #             discrete_features_old.append(np.expand_dims(arr, axis=2))
        #             shift += space_size
        #         else:
        #             arr = obs[key]
        #             if arr.shape == (1, P, 24, 24):
        #                 pass
        #             elif arr.shape == (1, P):
        #                 reshaped_arr = np.zeros((1, P, 24, 24), dtype=arr.dtype)
        #                 for player_idx in range(P):
        #                     reshaped_arr[0, player_idx, :, :] = arr[0, player_idx]  # Broadcast value
        #                 arr = reshaped_arr

        #                 # arr = np.broadcast_to(arr.reshape(1, P, 1, 1), (1, P, 24, 24))
        #             else:
        #                 print("What", key, arr.shape, file=sys.stderr)
        #             continuous_features_old.append(np.expand_dims(arr, axis=2))

        #     with self.profiler("actually concatenate"):
        #         continuous_features_old = np.concatenate(continuous_features_old, axis=2)
        #         discrete_features_old = np.concatenate(discrete_features_old, axis=2)

        with self.profiler("concatenate"):
            spec = simplified_spec

            discrete_features_cnt = 0
            continuous_features_cnt = 0
            for key, val in spec.items():
                if val.type == SpaceType.DISCRETE:
                    discrete_features_cnt += 1
                else:
                    continuous_features_cnt += 1

            shifts_list = []

            with self.profiler("discrete"):
                discrete_features = np.zeros((1, P, discrete_features_cnt, 24, 24), dtype=np.int16)
                index = 0
                shift = 0
                for key, val in spec.items():
                    if val.type == SpaceType.DISCRETE:
                        arr = obs[key]

                        if arr.shape == (1, P, 24, 24):
                            discrete_features[0, :, index, :, :] = arr
                        else:
                            assert arr.shape == (1, P)
                            discrete_features[0, :, index, :, :] = np.broadcast_to(arr.reshape(1, P, 1, 1), (1, P, 24, 24))

                        shifts_list.append(shift)

                        index += 1
                        shift += val.size

                shift_array = np.array(shifts_list, dtype=discrete_features.dtype).reshape(1, 1, -1, 1, 1)
                discrete_features += shift_array

            with self.profiler("continuous"):
                continuous_features = np.zeros((1, P, continuous_features_cnt, 24, 24), dtype=np.float32)
                index = 0
                for key, val in spec.items():
                    if val.type == SpaceType.CONTINUOUS:
                        arr = obs[key]
                        if arr.shape == (1, P, 24, 24):
                            continuous_features[0, :, index, :, :] = arr
                        else:
                            assert arr.shape == (1, P)
                            continuous_features[0, :, index, :, :] = np.broadcast_to(arr.reshape(1, P, 1, 1), (1, P, 24, 24))
                        index += 1

        # diff = np.sum(np.abs(discrete_features - discrete_features_old))
        # print(f"discrete diff={diff}")

        # diff = np.sum(np.abs(continuous_features - continuous_features_old))
        # print(f"continuous diff={diff}")

        # comment need to debug

        #one_player_2d = {}

        #for key, val in env_obs['player_1'].items():
        #    if isinstance(val, np.ndarray) and val.shape == (24, 24):
        #        val = val.T
        #    one_player_2d[key] = val

        #one_player_2d_fin = {}

        #for key, val in obs.items():
        #    val = val[0, 0].T
        #    one_player_2d_fin[key] = val

        #stepp = env_obs['player_1']['steps']

        #if stepp == 128:
        #    pass

        assert np.min(discrete_features) >= 0 and np.max(discrete_features) < 875

        if self.flags.use_embedding_input:

            new_obs = {
                'GPU1_continues_features': continuous_features,
                'GPU1_discrete_features': discrete_features,
            }


            return new_obs

        with self.profiler("one_hot"):
            # Initialize one-hot encoded array

            with self.profiler("zeros"):
                one_hot_encoded_discrete_features = np.zeros((1, P, 24, 24, 875), dtype=bool)

            # Reshape discrete_features to (1, P, 24, 24, discrete_features_cnt)
            with self.profiler("discrete_features_trans"):
                discrete_features = np.transpose(discrete_features, (0, 1, 3, 4, 2))  # Shape: [1, P, 24, 24, discrete_features_cnt]

            # Vectorized one-hot encoding using NumPy's advanced indexing
            with self.profiler("put_along_axis"):
                np.put_along_axis(one_hot_encoded_discrete_features, discrete_features, 1, axis=-1)

            #discrete_features = np.transpose(discrete_features, (0, 1, 4, 2, 3))  # Shape: [1, P, discrete_features_cnt, 24, 24]
            with self.profiler("oh_trans"):
                one_hot_encoded_discrete_features = np.transpose(one_hot_encoded_discrete_features, (0, 1, 4, 2, 3))

            new_obs = {
                'GPU1_continues_features': continuous_features,
                'GPU1_one_hot_encoded_discrete_features': one_hot_encoded_discrete_features,
            }


        return new_obs
