import copy
from dataclasses import dataclass
import logging
import random
import gym
import numpy as np

from ..torchbeast.profiler import ScopedProfiler

from ..utility_constants import BOARD_SIZE
from ..utility_constants import MAX_UNITS

from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from luxai_s3.params import EnvParams

import math
import sys
import secrets, time, os

import gym
import numpy as np
import torch
from collections import defaultdict
from enum import Enum
from luxai_s3.params import EnvParams
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode

from ..utility_constants import BOARD_SIZE, MAX_UNITS
from functools import lru_cache

class TileType(int, Enum):
    EMPTY = 0
    NEBULA = 1
    ASTEROID = 2
    ANOMALY = 3


@dataclass
class Unit:
    position: tuple[int, int]
    energy: int
    prev_position: tuple[int, int]
    previous_energy: int
    idx: int

    def __repr__(self):
        return f"Unit(position={self.position}, energy={int(self.energy)}, prev_position={self.prev_position}, previous_energy={int(self.previous_energy)})"

def apply_kernel_at_cell(arr, x, y, kernel):
    """
    Applies a convolution-like kernel to a single cell (x, y) of a 2D array.

    This function updates the input array in-place by adding the kernel's values to the sub-region
    centered at (x, y). The kernel is assumed to be centered at
      (kernel.shape[0]//2, kernel.shape[1]//2)
    and only the overlapping portion is applied if (x, y) is near an edge.

    Parameters
    ----------
    arr : np.ndarray
        2D array to be updated.
    x : int
        Row index of the cell where the kernel is applied.
    y : int
        Column index of the cell where the kernel is applied.
    kernel : np.ndarray
        2D kernel (e.g., a 5x5 array) containing the values to add.

    Returns
    -------
    None
        The array is updated in-place.
    """
    # Ensure arr is a NumPy array.
    arr = np.asarray(arr)

    # Get kernel dimensions and compute its "radius" along each axis.
    k_rows, k_cols = kernel.shape
    r_x = k_rows // 2  # vertical radius
    r_y = k_cols // 2  # horizontal radius

    # Determine the indices in the array where the kernel will be applied.
    # (These indices are clamped to the array's boundaries.)
    arr_x_start = max(x - r_x, 0)
    arr_x_end   = min(x + r_x + 1, arr.shape[0])
    arr_y_start = max(y - r_y, 0)
    arr_y_end   = min(y + r_y + 1, arr.shape[1])

    # Determine the corresponding indices in the kernel.
    # When the cell is near an edge, only a sub-region of the kernel is used.
    ker_x_start = r_x - (x - arr_x_start)
    ker_x_end   = ker_x_start + (arr_x_end - arr_x_start)
    ker_y_start = r_y - (y - arr_y_start)
    ker_y_end   = ker_y_start + (arr_y_end - arr_y_start)

    # Update the sub-region of the array by adding the corresponding part of the kernel.
    arr[arr_x_start:arr_x_end, arr_y_start:arr_y_end] += kernel[ker_x_start:ker_x_end, ker_y_start:ker_y_end]


@lru_cache(maxsize=None)
def generate_kernel(size: int) -> np.ndarray:
    """
    Generate a square kernel of the specified odd size where the outer perimeter is 1,
    and the values increase by 1 toward the center.

    For a kernel of size N (where N is odd), the number of rings is (N + 1) // 2.
    Each cell's value is computed as 1 plus the minimum distance from that cell to any edge.

    Parameters
    ----------
    size : int
        The size of the kernel (must be an odd number).

    Returns
    -------
    kernel : np.ndarray
        A 2D NumPy array of shape (size, size) with the described ring pattern.

    Example
    -------
    >>> generate_kernel(5)
    array([[1, 1, 1, 1, 1],
           [1, 2, 2, 2, 1],
           [1, 2, 3, 2, 1],
           [1, 2, 2, 2, 1],
           [1, 1, 1, 1, 1]])
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number")

    # Create a grid of indices for the rows and columns.
    i = np.arange(size, dtype=np.int16)[:, None]  # Column vector of row indices.
    j = np.arange(size, dtype=np.int16)[None, :]  # Row vector of column indices.

    # Each cell's value is 1 plus its minimum distance from any edge.
    kernel = 1 + np.minimum(np.minimum(i, j), np.minimum(size - 1 - i, size - 1 - j))

    # Mark the array as read-only so it isn't modified accidentally.
    kernel.flags.writeable = False
    return kernel


class SB3Wrapper(gym.Env):
    def __init__(
        self,
        env_id,
        obs_space,
        act_space,
        flags,
        example,
        profiler: ScopedProfiler = ScopedProfiler(enabled=False),
    ) -> None:
        super(SB3Wrapper, self).__init__()
        self.flags = flags
        self.profiler = profiler

        self.n_games_played = 0
        self.example = example
        self.env_id = env_id
        self.initial_reset = True
        self.state = None
        self.known_game_params = None

        self.kaggle_player_id = None

        #shifts
        self.ok = 0
        self.not_ok = 0

        self.board_dims = BOARD_SIZE
        self.obs_space = obs_space
        self.action_space = act_space
        self.observation_space = self.obs_space.get_obs_spec()

        if self.flags.kaggle:
            self.env3 = None
        else:
            if self.env_id == 0 and not example:
                self.replay = True
                self.env3 = LuxAIS3GymEnv(numpy_output=True)
                self.env3 = RecordEpisode(self.env3, save_dir="episodes")
                print("Recording", self.env_id)
            else:
                self.replay = False
                self.env3 = LuxAIS3GymEnv(numpy_output=True)

        self.real_done = True

        self.precomputed_shift_ticks = self.precompute_shift_ticks()
        #print(self.precumputed_shift_ticks)
        #assert False
        self.prev_cells_received_sap = None
        self.prev_previous_sap_units = None


    def precompute_shift_ticks(self):
        result = dict()
        self.shift_steps_by_param_idx = dict()
        self.shift_steps_by_param_idx[-1] = set()

        for i in range(1, 550):
            for param_idx, param in enumerate([-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15]):
                if (i - 1) * abs(param) % 1 > i * abs(param) % 1:
                    if i + 1 not in result:
                        result[i + 1] = []
                    result[i + 1].append((param_idx, param))

                    if param_idx not in self.shift_steps_by_param_idx:
                        self.shift_steps_by_param_idx[param_idx] = set()
                    self.shift_steps_by_param_idx[param_idx].add(i + 1)

        #print(result)
        #print(self.shift_steps_by_param_idx)
        return result



    def dummy_actions_taken(self):
        return {
            key: np.zeros(space.shape + (len(self.action_space.ACTION_MEANINGS[key]),), dtype=bool)
            for key, space in self.action_space.get_action_space().spaces.items()
        }
    def dummy_actions(self):
        return {'player_0': np.zeros((16, 3), dtype=np.int16),
                'player_1': np.zeros((16, 3), dtype=np.int16)}

    def calculate_nebula_tile_energy_reduction(self, player_idx, obs, state):
        map_features = obs['map_features']
        tiles = obs['remember_type']
        energy = map_features['energy']

        positions = obs['units']['position'][player_idx]
        energies = obs['units']['energy'][player_idx]

        while len(positions) > 0 and list(positions[-1]) == [-1, -1]:
            positions = positions[:-1]
            energies = energies[:-1]

        # for p, e in zip(positions, energies):
        #     print(f"{p}: {e}")

        def find_previous_unit_energy(x, y):
            candidate_for_previous_unit_energy = -1
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    prvx, prvy = x + i, y + j
                    for unit_position, unit_energy in zip(state['prev_obs']['units']['position'][player_idx], state['prev_obs']['units']['energy'][player_idx]):
                        if tuple(unit_position) == (prvx, prvy):
                            if candidate_for_previous_unit_energy != -1:
                                return -1 # multiple candidates
                            candidate_for_previous_unit_energy = unit_energy

            return candidate_for_previous_unit_energy

        potential_reductions = [0, 1, 2, 3, 5, 25]
        for unit_position, unit_energy in zip(positions, energies):
            unit_position = tuple(unit_position)
            if unit_position == (-1, -1):
                continue
            x, y = unit_position
            # we want to only check those NEBULA tiles that were nebulas in the previous step, because otherwise the energy reduction is not happening
            if tiles[x][y] == TileType.NEBULA and state['prev_obs']['remember_type'][x][y] == TileType.NEBULA:
                if unit_energy > 0:
                    previous_unit_energy = find_previous_unit_energy(x, y)
                    if previous_unit_energy != -1:
                        expected_unit_energy = previous_unit_energy + energy[x][y] - self.known_game_params['unit_move_cost']
                        diff = expected_unit_energy - unit_energy
                        closest_reduction = min(potential_reductions, key=lambda x: abs(x - diff))
                        if "nebula_tile_energy_reduction_freq" not in state:
                            state["nebula_tile_energy_reduction_freq"] = defaultdict(int)
                        state["nebula_tile_energy_reduction_freq"][closest_reduction] += 1

        obs['nebula_tile_energy_reduction'] = -1
        if freq := state.get("nebula_tile_energy_reduction_freq"):
            # print(f"freq: {freq}")
            obs['nebula_tile_energy_reduction'] = max(freq, key=freq.get)

    def match_enemy_units(self, player_idx, obs, prev_obs, ignore_invalid=False):
        units: list[Unit] = []
        for i, (unit_position, unit_energy) in enumerate(zip(obs['units']['position'][1 - player_idx], obs['units']['energy'][1 - player_idx])):
            if unit_position[0] == -1:
                continue

            previous_candidate = prev_obs['units']['position'][1 - player_idx][i], prev_obs['units']['energy'][1 - player_idx][i]
            if ignore_invalid and np.all(previous_candidate[0] == (-1, -1)):
                continue
            if np.all(previous_candidate[0] == (-1, -1)) or abs(previous_candidate[0][0] - unit_position[0]) + abs(previous_candidate[0][1] - unit_position[1]) <= 1:
                # this should always be true, but for some reason fails super rarely
                prev_unit_position, prev_unit_energy = previous_candidate
                units.append(Unit(unit_position, unit_energy, prev_unit_position, prev_unit_energy, i))
            #else:
            #    print(f"enemy unit {i} did not match!")
            #    print(f"enemy_units:\n{obs['units']['position'][1 - player_idx]}")
            #    print(f"prev enemy units:\n{prev_obs['units']['position'][1 - player_idx]}")

        return units

    def match_my_units(self, player_idx, obs, prev_obs, ignore_invalid):
        units: list[Unit] = []
        for i, (unit_position, unit_energy) in enumerate(zip(obs['units']['position'][player_idx], obs['units']['energy'][player_idx])):
            if unit_position[0] == -1:
                continue

            previous_candidate = prev_obs['units']['position'][player_idx][i], prev_obs['units']['energy'][player_idx][i]
            if ignore_invalid and np.all(previous_candidate[0] == (-1, -1)):
                continue
            if np.all(previous_candidate[0] == (-1, -1)) or abs(previous_candidate[0][0] - unit_position[0]) + abs(previous_candidate[0][1] - unit_position[1]) <= 1:
                # this should always be true, but for some reason fails super rarely
                prev_unit_position, prev_unit_energy = previous_candidate
                units.append(Unit(unit_position, unit_energy, prev_unit_position, prev_unit_energy, i))
            #else:
            #    print(f"enemy unit {i} did not match!")
            #    print(f"enemy_units:\n{obs['units']['position'][1 - player_idx]}")
            #    print(f"prev enemy units:\n{prev_obs['units']['position'][1 - player_idx]}")

        return units


    def calculate_unit_sap_dropoff_factor(self, player_idx, obs, state, cells_received_sap):
        positions = obs['units']['position'][player_idx]
        energies = obs['units']['energy'][player_idx]

        sap_range = self.known_game_params['unit_sap_range']

        if 'prev_obs' not in state or not cells_received_sap:
            obs['unit_sap_dropoff_factor'] = -1
            return

        enemy_units = self.match_enemy_units(player_idx, obs, state['prev_obs'])

        potential_sap_dropoff_factors = [0.25, 0.5, 1]

        enemies_received_direct_sap = defaultdict(lambda: [0, 0])

        enemies_received_dropoff_sap = defaultdict(lambda: [0, 0])
        for position_received_sap, (total_sap, total_shots) in cells_received_sap.items():
            for enemy_idx, enemy in enumerate(enemy_units):
                if abs(enemy.position[0] - position_received_sap[0]) <= 1 and abs(enemy.position[1] - position_received_sap[1]) <= 1:
                    if np.any(enemy.position != position_received_sap):
                        enemies_received_dropoff_sap[enemy_idx][0] += total_sap
                        enemies_received_dropoff_sap[enemy_idx][1] += total_shots
                    else:
                        enemies_received_direct_sap[enemy_idx][0] += total_sap
                        enemies_received_direct_sap[enemy_idx][1] += total_shots

        for enemy_idx, (total_sap, total_shots) in enemies_received_dropoff_sap.items():
            enemy = enemy_units[enemy_idx]
            if enemy.previous_energy != -1:
                expected_energy = enemy.previous_energy
                if np.any(enemy.position != enemy.prev_position):
                    expected_energy -= self.known_game_params['unit_move_cost']
                if enemy.energy >= 0:
                    # This is a hack, it seems that if unit is killed with sap, it's energy is not modified by the energy field
                    expected_energy += obs['map_features']['energy'][enemy.position[0]][enemy.position[1]]
                if enemy_idx in enemies_received_direct_sap:
                    expected_energy -= enemies_received_direct_sap[enemy_idx][0]

                diff = expected_energy - enemy.energy

                normalized_diff = diff / self.known_game_params['unit_sap_cost'] / total_shots

                closest_factor = min(potential_sap_dropoff_factors, key=lambda x: abs(x - normalized_diff))

                if abs(closest_factor - normalized_diff) > 0.01:
                    expected_energy -= self.known_game_params['unit_sap_cost']
                    diff = expected_energy - enemy.energy
                    normalized_diff = diff / self.known_game_params['unit_sap_cost'] / total_shots
                    closest_factor2 = min(potential_sap_dropoff_factors, key=lambda x: abs(x - normalized_diff))

                    if abs(closest_factor2 - normalized_diff) < 0.001: # it's highly likely that this unit did fire too
                        closest_factor = closest_factor2

                if "unit_sap_dropoff_factor_freq" not in state:
                    state["unit_sap_dropoff_factor_freq"] = defaultdict(int)
                state["unit_sap_dropoff_factor_freq"][closest_factor] += 1

        obs['unit_sap_dropoff_factor'] = -1
        if freq := state.get("unit_sap_dropoff_factor_freq"):
            #print(f"unit_sap_dropoff_factor_freq: {freq}")
            obs['unit_sap_dropoff_factor'] = max(freq, key=freq.get)

    def calculate_nebula_tile_vision_reduction(self, player_idx, obs, state):
        min_vision_reduction = state.get('min_nebula_vision_reduction', 0)
        max_vision_reduction = state.get('max_nebula_vision_reduction', 7)

        if 'remember_tile_type' not in state:
            obs['min_nebula_vision_reduction'] = min_vision_reduction
            obs['max_nebula_vision_reduction'] = max_vision_reduction
            return

        if min_vision_reduction == max_vision_reduction:
            obs['min_nebula_vision_reduction'] = min_vision_reduction
            obs['max_nebula_vision_reduction'] = max_vision_reduction
            return

        previous_remember_tiles = state['remember_tile_type']

        unit_sensor_range = self.known_game_params['unit_sensor_range'] + 1
        kernel = generate_kernel(unit_sensor_range * 2 - 1)

        vision_map = np.zeros((24, 24), dtype=np.int16)
        for position in obs['units']['position'][player_idx]:
            if position[0] == -1:
                continue

            x, y = position
            vision_map[x, y] += 10
            apply_kernel_at_cell(vision_map, x, y, kernel)

        sensor_mask = obs['sensor_mask']
        cur_visible_tiles = obs['map_features']['tile_type']

        for i in range(24):
            for j in range(24):
                v = vision_map[i, j]
                if sensor_mask[i, j] and cur_visible_tiles[i][j] == TileType.NEBULA and previous_remember_tiles[i][j] == TileType.NEBULA:
                    assert v > 0 # otherwise it should not be in sensor_mask
                    max_vision_reduction = min(max_vision_reduction, v - 1)
                elif not sensor_mask[i, j] and v:
                    min_vision_reduction = max(min_vision_reduction, v)

        # print("Vision map:")
        # for i, row in enumerate(vision_map):
        #     row_str = []
        #     for j, val in enumerate(row):
        #         if sensor_mask[i, j]:
        #             if cur_visible_tiles[i][j] == TileType.NEBULA:
        #                 row_str.append(f"\033[31;42m{val:2}\033[0m")  # Red foreground with green background
        #             else:
        #                 row_str.append(f"\033[31m{val:2}\033[0m")  # Red foreground only
        #         else:
        #             row_str.append(f"{val:2}")
        #     print(" ".join(row_str))

        # print(f"min_vision_reduction: {min_vision_reduction}, max_vision_reduction: {max_vision_reduction}")

        #assert min_vision_reduction <= max_vision_reduction
        #if min_vision_reduction > max_vision_reduction:
        #    print("IXANEZIS ASSERT FAILED: ", min_vision_reduction, max_vision_reduction)

        max_vision_reduction = max(min_vision_reduction, max_vision_reduction)

        state['min_nebula_vision_reduction'] = min_vision_reduction
        state['max_nebula_vision_reduction'] = max_vision_reduction

        obs['min_nebula_vision_reduction'] = min_vision_reduction
        obs['max_nebula_vision_reduction'] = max_vision_reduction


    def calculate_unit_energy_void_factor(self, player_idx, obs, state):
        if 'prev_obs' not in state:
            obs['unit_energy_void_factor'] = -1
            return


        if "unit_energy_void_factor_freq" not in state:
            state["unit_energy_void_factor_freq"] = defaultdict(int)

        potential_energy_void_factors = [0.0625, 0.125, 0.25, 0.375]

        my_units = self.match_my_units(player_idx, obs, state['prev_obs'], True)
        enemy_units = self.match_enemy_units(player_idx, obs, state['prev_obs'], True)

        for my_unit in my_units:
            my_unit_moved = np.any(my_unit.position != my_unit.prev_position)

            for enemy_unit in enemy_units:
                enemy_unit_moved = np.any(enemy_unit.position != enemy_unit.prev_position)

                if abs(my_unit.position[0] - enemy_unit.position[0]) + abs(my_unit.position[1] - enemy_unit.position[1]) != 1:
                    continue

                my_unit_final_energy = my_unit.energy
                if obs['remember_map_energy_known'][my_unit.position[0]][my_unit.position[1]]:
                    my_unit_final_energy -= obs['remember_map_energy'][my_unit.position[0]][my_unit.position[1]]
                if obs['nebula_tile_energy_reduction'] != -1 and obs['remember_type'][my_unit.position[0]][my_unit.position[1]] == TileType.NEBULA:
                    my_unit_final_energy -= obs['nebula_tile_energy_reduction']

                enemy_unit_final_energy = enemy_unit.energy
                if obs['remember_map_energy_known'][enemy_unit.position[0]][enemy_unit.position[1]]:
                    enemy_unit_final_energy -= obs['remember_map_energy'][enemy_unit.position[0]][enemy_unit.position[1]]
                if obs['nebula_tile_energy_reduction'] != -1 and obs['remember_type'][enemy_unit.position[0]][enemy_unit.position[1]] == TileType.NEBULA:
                    enemy_unit_final_energy -= obs['nebula_tile_energy_reduction']

                my_unit_prev_energy = my_unit.previous_energy
                if my_unit_moved:
                    my_unit_prev_energy -= self.known_game_params['unit_move_cost']

                enemy_unit_prev_energy = enemy_unit.previous_energy
                if enemy_unit_moved:
                    enemy_unit_prev_energy -= self.known_game_params['unit_move_cost']

                for i in range(len(potential_energy_void_factors)):
                    estimated_energies_my_unit = int(my_unit_prev_energy) - int(enemy_unit_prev_energy * potential_energy_void_factors[i])
                    if my_unit_final_energy == estimated_energies_my_unit:
                        state["unit_energy_void_factor_freq"][i] += 1

                for i in range(len(potential_energy_void_factors)):
                    estimated_energies_enemy_unit = int(enemy_unit_prev_energy) - int(my_unit_prev_energy * potential_energy_void_factors[i])
                    if enemy_unit_final_energy == estimated_energies_enemy_unit:
                        state["unit_energy_void_factor_freq"][i] += 1

        obs['unit_energy_void_factor'] = -1
        if freq := state.get("unit_energy_void_factor_freq"):
            #print(f"unit_sap_dropoff_factor_freq: {freq}")
            obs['unit_energy_void_factor'] = max(freq, key=freq.get)





    def update_unexpected_energy_change(self, player, player_idx, obs, info, state, previous_sap_units):
        my_unexpected_energy_change = np.zeros((24, 24), dtype=np.int16)
        enemy_unexpected_energy_change = np.zeros((24, 24), dtype=np.int16)

        my_unexpected_energy_change_in_saps = np.zeros((24, 24), dtype=np.int16)
        enemy_unexpected_energy_change_in_saps = np.zeros((24, 24), dtype=np.int16)

        if 'prev_obs' not in state:
            obs['my_unexpected_energy_change'] = my_unexpected_energy_change
            obs['enemy_unexpected_energy_change'] = enemy_unexpected_energy_change
            obs['my_unexpected_energy_change_in_saps'] = my_unexpected_energy_change_in_saps
            obs['enemy_unexpected_energy_change_in_saps'] = enemy_unexpected_energy_change_in_saps
            return

        my_units = self.match_my_units(player_idx, obs, state['prev_obs'], True)
        enemy_units = self.match_enemy_units(player_idx, obs, state['prev_obs'], True)

        sap_dropoff_factor = obs['unit_sap_dropoff_factor']
        sap_dropoff_factor = max(sap_dropoff_factor, 0.25)
        sap_damage = int(self.known_game_params['unit_sap_cost'] * sap_dropoff_factor)


        for my_unit in my_units:
            my_unit_moved = np.any(my_unit.position != my_unit.prev_position)

            expected_energy = my_unit.previous_energy
            if my_unit_moved:
                expected_energy -= self.known_game_params['unit_move_cost']

            if obs['remember_map_energy_known'][my_unit.position[0]][my_unit.position[1]]:
                expected_energy += obs['remember_map_energy'][my_unit.position[0]][my_unit.position[1]]

            if obs['nebula_tile_energy_reduction'] != -1 and obs['remember_type'][my_unit.position[0]][my_unit.position[1]] == TileType.NEBULA:
                expected_energy -= obs['nebula_tile_energy_reduction']

            if my_unit.idx in previous_sap_units:
                expected_energy -= self.known_game_params['unit_sap_cost']

            unexpected_energy_change = my_unit.energy - expected_energy
            my_unexpected_energy_change[my_unit.position[0], my_unit.position[1]] = unexpected_energy_change


            my_unexpected_energy_change_in_saps[my_unit.position[0], my_unit.position[1]] = unexpected_energy_change // sap_damage


        for enemy_unit in enemy_units:
            enemy_unit_moved = np.any(enemy_unit.position != enemy_unit.prev_position)

            expected_energy = enemy_unit.previous_energy
            if enemy_unit_moved:
                expected_energy -= self.known_game_params['unit_move_cost']

            if obs['remember_map_energy_known'][enemy_unit.position[0]][enemy_unit.position[1]]:
                expected_energy += obs['remember_map_energy'][enemy_unit.position[0]][enemy_unit.position[1]]

            if obs['nebula_tile_energy_reduction'] != -1 and obs['remember_type'][enemy_unit.position[0]][enemy_unit.position[1]] == TileType.NEBULA:
                expected_energy -= obs['nebula_tile_energy_reduction']

            unexpected_energy_change = enemy_unit.energy - expected_energy
            enemy_unexpected_energy_change[enemy_unit.position[0], enemy_unit.position[1]] = unexpected_energy_change

            enemy_unexpected_energy_change_in_saps[enemy_unit.position[0], enemy_unit.position[1]] = unexpected_energy_change // sap_damage


        obs['my_unexpected_energy_change'] = my_unexpected_energy_change
        obs['enemy_unexpected_energy_change'] = enemy_unexpected_energy_change
        obs['my_unexpected_energy_change_in_saps'] = my_unexpected_energy_change_in_saps
        obs['enemy_unexpected_energy_change_in_saps'] = enemy_unexpected_energy_change_in_saps





    def update_shifts(self, player, player_idx, obs, info, state):
        sensor_mask = obs['sensor_mask']
        steps = int(obs['steps'])
        map_features = obs['map_features']
        tile_type = map_features['tile_type']
        prev_obs = state.get('prev_obs', None)

        if prev_obs is None:
            obs['shift_param'] = (-1, -1)
            obs['shift_side'] = -1
            obs['steps_to_shift'] = -1
            return

        shifts_voting_data_positive = state.get('shifts_voting_data_positive', dict())

        #prev_open_state_map_features = prev_state.get('open_state_map_features', None)

        #open_state = info['state'][player]

        #if prev_open_state_map_features is not None:
        #    if (prev_open_state_map_features != open_state['map_features']['tile_type']).any():
        #        print('real shift:', steps)

        #prev_state['open_state_map_features'] = open_state['map_features']['tile_type']

        prev_tile_type = prev_obs['map_features']['tile_type']

        if steps in self.precomputed_shift_ticks:
            stationary_array = []

            for i in range(24):
                for j in range(24):
                    if sensor_mask[i, j]:
                        if tile_type[i, j] == TileType.NEBULA or tile_type[i, j] == TileType.ASTEROID:
                            stationary_array.append((i, j, tile_type[i, j]))


            best_shift = (0, 0)
            best_shift_score = 0
            scores = []
            for cur_shift in [(0, 0), (-1, 1), (1, -1)]:
                shift_score = 0
                for pos in stationary_array:
                    i, j, type = pos
                    si = i + cur_shift[0]
                    if si < 0:
                        si += 24
                    if si >= 24:
                        si -= 24
                    sj = j + cur_shift[1]
                    if sj < 0:
                        sj += 24
                    if sj >= 24:
                        sj -= 24
                    if prev_tile_type[si, sj] == type:
                        shift_score += 1
                scores.append(shift_score)
                if shift_score > best_shift_score:
                    best_shift_score = shift_score
                    best_shift = cur_shift
            #print('scores:', scores)

            if best_shift[0] != 0:
                #print('found shift', steps, best_shift)
                for param_idx, param in self.precomputed_shift_ticks[steps]:
                    if best_shift[0] * param > 0:
                        continue
                    if param_idx not in shifts_voting_data_positive:
                        shifts_voting_data_positive[param_idx] = 0
                    shifts_voting_data_positive[param_idx] += 1

        best_variant = (-1, -1)

        #print("voting data:")
        max_votes = 0
        for param_idx, param in enumerate([-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15]):
            max_votes = max(max_votes, shifts_voting_data_positive.get(param_idx, 0))
            #print(param_idx, param, shifts_voting_data_positive.get(param_idx, 0))
        #print()

        if max_votes != 0:
            best_variants = []
            for param_idx, param in enumerate([-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15]):
                if shifts_voting_data_positive.get(param_idx, 0) == max_votes:
                    best_variants.append((param, param_idx))
            if len(best_variants) > 0:
                if best_variants[0][0] < 0:
                    best_variant = max(best_variants)
                else:
                    best_variant = min(best_variants)

        #print("BEST VARIANT:", steps, best_variant, self.shift_steps_by_param_idx[best_variant[1]])

        obs['shift_param'] = best_variant

        shift_steps = self.shift_steps_by_param_idx[best_variant[1]]

        if len(shift_steps) > 0:
            min_step = max(shift_steps)
            for step in shift_steps:
                if step > steps:
                    min_step = min(min_step, step)
            obs['steps_to_shift'] = min_step - steps
            obs['shift_side'] = best_variant[0] > 0
        else:
            obs['steps_to_shift'] = -1
            obs['shift_side'] = -1

        state['shifts_voting_data_positive'] = shifts_voting_data_positive


    def apply_shift(self, player, player_idx, obs, info, state):
        remember_tile_type = state.get('remember_tile_type', np.zeros((24, 24), dtype=int) - 1)
        steps = int(obs['steps'])

        shift_param, shift_param_idx = obs['shift_param']
        if steps in self.shift_steps_by_param_idx[shift_param_idx]:
            new_remember_tile_type = np.zeros((24, 24), dtype=int) - 1
            shift = (-1, 1) if shift_param < 0 else (1, -1)
            for i in range(24):
                for j in range(24):
                    si = i + shift[0]
                    if si >= 24:
                        si -= 24
                    if si < 0:
                        si += 24

                    sj = j + shift[1]
                    if sj >= 24:
                        sj -= 24
                    if sj < 0:
                        sj += 24
                    new_remember_tile_type[si, sj] = remember_tile_type[i, j]
            state['remember_tile_type'] = new_remember_tile_type

    def update_map_tiles(self, player, player_idx, obs, info, state):
        remember_tile_type = state.get('remember_tile_type', np.full((24, 24), -1, dtype=int))

        units = obs['units']
        map_features = obs['map_features']
        tile_type = map_features['tile_type']
        sensor_mask = obs['sensor_mask']
        current_nebula = (tile_type == TileType.NEBULA)
        current_anomaly = np.zeros((24, 24), dtype=bool)

        sensor_range_param = self.known_game_params['unit_sensor_range']
        nebula_tile_vision_reduction = state.get('nebula_tile_vision_reduction', None)

        # ---- ✅ Vectorized Sensor Range Calculation ----
        unit_positions = units['position'][player_idx]
        unit_positions = unit_positions[unit_positions[:, 0] != -1]  # Remove invalid units

        for x, y in unit_positions:
            if current_nebula[x, y] and nebula_tile_vision_reduction is None:
                continue

            x_range = np.clip(np.arange(x - sensor_range_param, x + sensor_range_param + 1), 0, 23)
            y_range = np.clip(np.arange(y - sensor_range_param, y + sensor_range_param + 1), 0, 23)

            grid_x, grid_y = np.meshgrid(x_range, y_range, indexing='ij')
            current_anomaly[grid_x, grid_y] |= ~sensor_mask[grid_x, grid_y]

        # ---- ✅ Vectorized Memory Update ----
        mask = sensor_mask.astype(bool)  # Convert sensor_mask to boolean
        remember_tile_type[mask] = tile_type[mask]
        remember_tile_type[23 - np.where(mask)[1], 23 - np.where(mask)[0]] = tile_type[mask]  # Symmetry update

        # ---- ✅ Efficient Anomaly Update ----
        anomaly_mask = np.logical_and(current_anomaly, remember_tile_type == -1)
        remember_tile_type[anomaly_mask] = TileType.ANOMALY
        remember_tile_type[23 - np.where(anomaly_mask)[1], 23 - np.where(anomaly_mask)[0]] = TileType.ANOMALY  # Symmetry update

        # ✅ Assign updated memory
        obs['remember_type'] = np.copy(remember_tile_type)
        state['remember_tile_type'] = remember_tile_type



    def update_energy(self, player, player_idx, obs, info, state):
        remember_map_energy = state.get('remember_map_energy', np.zeros((24, 24), dtype=int))
        remember_map_energy_known = state.get('remember_map_energy_known', np.zeros((24, 24), dtype=bool))
        map_features = obs['map_features']
        map_energy = map_features['energy']
        sensor_mask = obs['sensor_mask']

        energy_invalidated = False
        for i in range(24):
            for j in range(24):
                if sensor_mask[i, j]:
                    if remember_map_energy_known[i, j] and remember_map_energy[i, j] != map_energy[i, j]:
                        energy_invalidated = True

        if energy_invalidated:
            if self.flags.zero_energy_on_invalidation:
                remember_map_energy *= 0
            # clear remember_map_energy_known without creating new array (its bool ant its cannot be multiplied)
            remember_map_energy_known[:] = 0

        for i in range(24):
            for j in range(24):
                if sensor_mask[i, j]:
                    remember_map_energy[i, j] = map_energy[i, j]
                    remember_map_energy_known[i, j] = 1

                    si = 23 - j
                    sj = 23 - i
                    remember_map_energy[si, sj] = map_energy[i, j]
                    remember_map_energy_known[si, sj] = 1

        if False and info != None: # not works when all units dead
            open_state = info['state'][player]
            for i in range(24):
                for j in range(24):
                    if remember_map_energy_known[i, j]:
                        assert remember_map_energy[i, j] == open_state['map_features']['energy'][i, j]

        obs['remember_map_energy'] = np.copy(remember_map_energy)
        obs['remember_map_energy_known'] = np.copy(remember_map_energy_known)
        state['remember_map_energy'] = remember_map_energy
        state['remember_map_energy_known'] = remember_map_energy_known

    def process_event_round_0_spawn_ends(self, player, player_idx, obs, info, state):
        state['event_round_0_spawn_ends_fired'] = True
        if not state['relic_0_found']:
            state['enable_possible_positions_removing_0'] = True
        state['enable_not_relic_marking_0'] = True


    def process_event_round_1_spawn_ends(self, player, player_idx, obs, info, state):
        state['event_round_1_spawn_ends_fired'] = True
        if not state['relic_1_found']:
            state['enable_possible_positions_removing_1'] = True
        state['enable_not_relic_marking_1'] = True


    def process_event_round_2_spawn_ends(self, player, player_idx, obs, info, state):
        state['event_round_2_spawn_ends_fired'] = True

        if state['no_more_spawn']:
            return

        state['no_more_spawn'] = True

        if not state['relic_2_found']:
            state['enable_possible_positions_removing_2'] = True
        state['enable_not_relic_marking_2'] = True

    def process_event_round_0_starts(self, player, player_idx, obs, info, state):
        #print("process_event_round_0_starts")
        state['possible_positions_for_relic_node_spawn_0'] = np.ones((24, 24), dtype=bool)
        state['possible_positions_for_relic_node_spawn_1'] = np.zeros((24, 24), dtype=bool)
        state['possible_positions_for_relic_node_spawn_2'] = np.zeros((24, 24), dtype=bool)

        state['possible_positions_for_relic_node_timer_0'] = np.zeros((24, 24), dtype=int)
        state['possible_positions_for_relic_node_timer_1'] = np.zeros((24, 24), dtype=int)
        state['possible_positions_for_relic_node_timer_2'] = np.zeros((24, 24), dtype=int)

        state['remember_potential_relic_0'] = np.zeros((24, 24), dtype=bool)
        state['remember_potential_relic_1'] = np.zeros((24, 24), dtype=bool)
        state['remember_potential_relic_2'] = np.zeros((24, 24), dtype=bool)

        state['remember_guarranted_relic'] = np.zeros((24, 24), dtype=bool)

        state['remember_not_relic_0'] = np.zeros((24, 24), dtype=bool)
        state['remember_not_relic_1'] = np.ones((24, 24), dtype=bool)
        state['remember_not_relic_2'] = np.ones((24, 24), dtype=bool)

        state['enable_possible_positions_removing_0'] = False
        state['enable_possible_positions_removing_1'] = False
        state['enable_possible_positions_removing_2'] = False

        state['no_more_spawn'] = False
        state['relic_found_tick'] = np.zeros((6, 1), dtype=int)

        state['enable_possible_positions_timer_0'] = True
        state['enable_possible_positions_timer_1'] = False
        state['enable_possible_positions_timer_2'] = False

        state['enable_not_relic_marking_0'] = False
        state['enable_not_relic_marking_1'] = False
        state['enable_not_relic_marking_2'] = False

        state['relic_score'] = np.zeros((24, 24), dtype=np.float32)


        state['event_round_0_spawn_ends_fired'] = False
        state['event_round_1_spawn_ends_fired'] = False
        state['event_round_2_spawn_ends_fired'] = False

        state['relic_0_found'] = False
        state['relic_1_found'] = False
        state['relic_2_found'] = False

        pass


    def process_event_round_1_starts(self, player, player_idx, obs, info, state):
        #print("process_event_round_1_starts")
        state['possible_positions_for_relic_node_spawn_1'] = np.ones((24, 24), dtype=bool)
        state['possible_positions_for_relic_node_timer_1'] = np.zeros((24, 24), dtype=int)
        state['remember_not_relic_1'][:] = 0
        state['enable_possible_positions_timer_1'] = True

        pass


    def process_event_round_2_starts(self, player, player_idx, obs, info, state):
        if state['no_more_spawn']:
            return

        state['possible_positions_for_relic_node_spawn_2'] = np.ones((24, 24), dtype=bool)
        state['possible_positions_for_relic_node_timer_2'] = np.zeros((24, 24), dtype=int)
        state['remember_not_relic_2'][:] = 0
        state['enable_possible_positions_timer_2'] = True



    def fill_potential(self, state, arr_p, round_number):
        remember_not_relic = state[f'remember_not_relic_{round_number}']
        remember_potential_relic = state[f'remember_potential_relic_{round_number}']
        remember_guarranted_relic = state['remember_guarranted_relic']
        for x, y in arr_p:
            for i in [-2, -1, 0, 1, 2]:
                for j in [-2, -1, 0, 1, 2]:
                    si = x + i
                    sj = y + j
                    if si < 0 or si >= 24 or sj < 0 or sj >= 24 or remember_guarranted_relic[si, sj] or remember_not_relic[si, sj]:
                        continue
                    #assert not remember_potential_relic[si, sj]
                    remember_potential_relic[si, sj] = True

        inversed_remember_potential_relic = np.logical_not(remember_potential_relic)
        inversed_remember_potential_relic = np.logical_and(inversed_remember_potential_relic, remember_guarranted_relic == False)

        remember_not_relic = np.logical_or(remember_not_relic, inversed_remember_potential_relic)

        state[f'remember_potential_relic_{round_number}'] = remember_potential_relic
        state[f'remember_not_relic_{round_number}'] = remember_not_relic


    def process_event_round_0_relic_found(self, player, player_idx, obs, info, state):

        self.fill_potential(state, [state['remember_relic_nodes'][0], state['remember_relic_nodes'][0 + 3]], 0)

        state['enable_not_relic_marking_0'] = True

        state['relic_found_tick'][0] = int(obs['steps'])

        state['possible_positions_for_relic_node_spawn_0'][:] = False
        state['possible_positions_for_relic_node_timer_0'][:] = 0

        state['enable_possible_positions_removing_0'] = False
        state['enable_possible_positions_timer_0'] = False

        state['relic_0_found'] = True



        pass

    def process_event_round_1_relic_found(self, player, player_idx, obs, info, state):

        self.fill_potential(state,  [state['remember_relic_nodes'][1], state['remember_relic_nodes'][1 + 3]], 1)

        state['enable_not_relic_marking_1'] = True


        state['relic_found_tick'][1] = int(obs['steps'])


        state['possible_positions_for_relic_node_spawn_1'][:] = 0
        state['possible_positions_for_relic_node_timer_1'] *= 0

        state['enable_possible_positions_removing_1'] = False
        state['enable_possible_positions_timer_1'] = False

        state['relic_1_found'] = True

        pass

    def process_event_round_2_relic_found(self, player, player_idx, obs, info, state):

        self.fill_potential(state, [state['remember_relic_nodes'][2], state['remember_relic_nodes'][2 + 3]], 2)

        state['enable_not_relic_marking_2'] = True

        state['relic_found_tick'][2] = int(obs['steps'])

        state['possible_positions_for_relic_node_spawn_2'][:] = 0
        state['possible_positions_for_relic_node_timer_2'] *= 0

        state['enable_possible_positions_removing_2'] = False
        state['enable_possible_positions_timer_2'] = False

        state['relic_2_found'] = True

        pass

    def process_event_event_round_0_all_explored(self, player, player_idx, obs, info, state):
        x, y = state['remember_relic_nodes'][0]
        assert x != -1
        pass

    def process_event_event_round_1_all_explored(self, player, player_idx, obs, info, state):
        #print("process_event_event_round_1_all_explored")
        x, y = state['remember_relic_nodes'][0]
        assert x != -1
        x, y = state['remember_relic_nodes'][1]
        if x == -1:
            state['no_more_spawn'] = True
        pass

    def process_event_event_round_2_all_explored(self, player, player_idx, obs, info, state):
        #print("process_event_event_round_2_all_explored")
        x, y = state['remember_relic_nodes'][0]
        assert x != -1
        x1, y1 = state['remember_relic_nodes'][1]
        x2, y2 = state['remember_relic_nodes'][2]

        if x1 == -1:
            assert x2 == -1

        state['no_more_spawn'] = True
        pass

    def update_guarranted_relic(self, player, player_idx, obs, info, state, open_state):
        if 'prev_obs' not in state:
            prev_team_points = np.zeros(2, dtype=int)
        else:
            prev_team_points = state['prev_obs']['team_points']
        cur_team_points = obs['team_points']
        match_changed = False

        if obs['match_steps'] == 0:
            match_changed = True
            cur_team_points = np.zeros(2, dtype=int)
            prev_team_points = np.zeros(2, dtype=int)

        if match_changed:
            return


        delta_points = cur_team_points - prev_team_points

        points = delta_points[player_idx]



        positions = obs['units']['position'][player_idx]
        energies = obs['units']['energy'][player_idx]


        unique_units_positions = []
        for position, energy in zip(positions, energies):
            if position[0] == -1 or energy < 0:
                continue
            position = (position[0], position[1])
            if position not in unique_units_positions:
                unique_units_positions.append(position)


        remember_guarranted_relic = state['remember_guarranted_relic']

        remember_potential_relic_0 = state['remember_potential_relic_0']
        remember_potential_relic_1 = state['remember_potential_relic_1']
        remember_potential_relic_2 = state['remember_potential_relic_2']

        remember_not_relic_0 = state['remember_not_relic_0']
        remember_not_relic_1 = state['remember_not_relic_1']
        remember_not_relic_2 = state['remember_not_relic_2']

        relic_score = state['relic_score']

        combined_not_relic = np.logical_and.reduce([remember_not_relic_0, remember_not_relic_1, remember_not_relic_2])


        n_units_on_guarranted_relic = 0
        n_units_not_on_guarranted_relic = 0
        n_units_on_not_relic = 0
        for x, y in unique_units_positions:
            if remember_guarranted_relic[x, y]:
                n_units_on_guarranted_relic += 1
            else:
                n_units_not_on_guarranted_relic += 1
            if combined_not_relic[x, y]:
                n_units_on_not_relic += 1

        if points < n_units_on_guarranted_relic:
            print("points:", obs['match_steps'], int(obs['steps']), points, n_units_on_guarranted_relic, n_units_not_on_guarranted_relic, n_units_on_not_relic)
        assert points >= n_units_on_guarranted_relic
        if points - n_units_on_guarranted_relic == n_units_not_on_guarranted_relic - n_units_on_not_relic:
            for x, y in unique_units_positions:
                if not remember_guarranted_relic[x, y] and not combined_not_relic[x, y]:
                    remember_guarranted_relic[x, y] = 1
                    remember_guarranted_relic[23 - y, 23 - x] = 1
                    remember_potential_relic_0[x, y] = 0
                    remember_potential_relic_0[23 - y, 23 - x] = 0
                    remember_potential_relic_1[x, y] = 0
                    remember_potential_relic_1[23 - y, 23 - x] = 0
                    remember_potential_relic_2[x, y] = 0
                    remember_potential_relic_2[23 - y, 23 - x] = 0

        n_check_positions = 0
        for x, y in unique_units_positions:
            if not remember_guarranted_relic[x, y] and not combined_not_relic[x, y]:
                n_check_positions += 1

        n_more_points = points - n_units_on_guarranted_relic
        if n_check_positions > 0 and n_more_points > 0:
            for x, y in unique_units_positions:
                if not remember_guarranted_relic[x, y] and not combined_not_relic[x, y]:
                    relic_score[x, y] += n_more_points / n_check_positions

        relic_score[remember_guarranted_relic] = 0
        relic_score[combined_not_relic] = 0

        if info != None:
            actual_weights = np.logical_and(open_state['relic_nodes_map_weights'] <= np.sum(open_state['relic_nodes_mask']) // 2, open_state['relic_nodes_map_weights'] > 0)
            rnmw = np.array(actual_weights) > 0
            eq = remember_guarranted_relic == 1
            assert(np.sum(rnmw[eq]) == np.sum(eq))

            real_not_relic_0 = np.logical_not(open_state['relic_nodes_map_weights'] == 1)
            assert np.sum(np.logical_and(real_not_relic_0 == False, remember_not_relic_0 == True)) == 0

            if np.sum(open_state['relic_nodes_mask']) // 2 >= 2:
                real_not_relic_1 = np.logical_not(open_state['relic_nodes_map_weights'] == 2)
                assert np.sum(np.logical_and(real_not_relic_1 == False, remember_not_relic_1 == True)) == 0

            if np.sum(open_state['relic_nodes_mask']) // 2 >= 3:
                real_not_relic_2 = np.logical_not(open_state['relic_nodes_map_weights'] == 3)
                assert np.sum(np.logical_and(real_not_relic_2 == False, remember_not_relic_2 == True)) == 0

        state['relic_score'] = relic_score

        state['remember_guarranted_relic'] = remember_guarranted_relic
        state['remember_potential_relic_0'] = remember_potential_relic_0
        state['remember_potential_relic_1'] = remember_potential_relic_1
        state['remember_potential_relic_2'] = remember_potential_relic_2
        state['remember_not_relic_0'] = remember_not_relic_0
        state['remember_not_relic_1'] = remember_not_relic_1
        state['remember_not_relic_2'] = remember_not_relic_2

        pass

    def update_not_relic(self, player, player_idx, obs, info, state, open_state):
        if 'prev_obs' not in state:
            prev_team_points = np.zeros(2, dtype=int)
        else:
            prev_team_points = state['prev_obs']['team_points']
        cur_team_points = obs['team_points']
        match_changed = False

        if obs['match_steps'] == 0:
            match_changed = True
            cur_team_points = np.zeros(2, dtype=int)
            prev_team_points = np.zeros(2, dtype=int)

        if match_changed:
            return

        if player == 'player_1' and obs['steps'] == 50:
            pass

        delta_points = cur_team_points - prev_team_points

        points = delta_points[player_idx]


        positions = obs['units']['position'][player_idx]
        energies = obs['units']['energy'][player_idx]


        unique_units_positions = []
        for position, energy in zip(positions, energies):
            if position[0] == -1 or energy < 0:
                continue
            position = (position[0], position[1])
            if position not in unique_units_positions:
                unique_units_positions.append(position)

        remember_guarranted_relic = state['remember_guarranted_relic']
        remember_potential_relic_0 = state['remember_potential_relic_0']
        remember_potential_relic_1 = state['remember_potential_relic_1']
        remember_potential_relic_2 = state['remember_potential_relic_2']
        remember_not_relic_0 = state['remember_not_relic_0']
        remember_not_relic_1 = state['remember_not_relic_1']
        remember_not_relic_2 = state['remember_not_relic_2']

        n_units_on_guarranted_relic = 0
        for x, y in unique_units_positions:
            if remember_guarranted_relic[x, y]:
                n_units_on_guarranted_relic += 1


        if points < n_units_on_guarranted_relic:
            print("points:", obs['match_steps'], obs['steps'], points, n_units_on_guarranted_relic)
        assert points >= n_units_on_guarranted_relic

        if points == n_units_on_guarranted_relic:
            for x, y in unique_units_positions:
                if not remember_guarranted_relic[x, y]:
                    if state['enable_not_relic_marking_0']:
                        remember_not_relic_0[x, y] = True
                        remember_not_relic_0[23 - y, 23 - x] = True
                        remember_potential_relic_0[x, y] = False
                        remember_potential_relic_0[23 - y, 23 - x] = False

                    if state['enable_not_relic_marking_1']:
                        remember_not_relic_1[x, y] = True
                        remember_not_relic_1[23 - y, 23 - x] = True
                        remember_potential_relic_1[x, y] = False
                        remember_potential_relic_1[23 - y, 23 - x] = False

                    if state['enable_not_relic_marking_2']:
                        remember_not_relic_2[x, y] = True
                        remember_not_relic_2[23 - y, 23 - x] = True
                        remember_potential_relic_2[x, y] = False
                        remember_potential_relic_2[23 - y, 23 - x] = False

        if info != None:
            actual_weights = np.logical_and(open_state['relic_nodes_map_weights'] <= np.sum(open_state['relic_nodes_mask']) // 2, open_state['relic_nodes_map_weights'] > 0)
            rnmw = np.array(actual_weights) > 0
            eq = remember_guarranted_relic == 1
            assert(np.sum(rnmw[eq]) == np.sum(eq))

            real_not_relic_0 = np.logical_not(open_state['relic_nodes_map_weights'] == 1)
            assert np.sum(np.logical_and(real_not_relic_0 == False, remember_not_relic_0 == True)) == 0

            if np.sum(open_state['relic_nodes_mask']) // 2 >= 2:
                real_not_relic_1 = np.logical_not(open_state['relic_nodes_map_weights'] == 2)
                assert np.sum(np.logical_and(real_not_relic_1 == False, remember_not_relic_1 == True)) == 0

            if np.sum(open_state['relic_nodes_mask']) // 2 >= 3:
                real_not_relic_2 = np.logical_not(open_state['relic_nodes_map_weights'] == 3)
                assert np.sum(np.logical_and(real_not_relic_2 == False, remember_not_relic_2 == True)) == 0


        state['remember_guarranted_relic'] = remember_guarranted_relic
        state['remember_potential_relic_0'] = remember_potential_relic_0
        state['remember_potential_relic_1'] = remember_potential_relic_1
        state['remember_potential_relic_2'] = remember_potential_relic_2
        state['remember_not_relic_0'] = remember_not_relic_0
        state['remember_not_relic_1'] = remember_not_relic_1
        state['remember_not_relic_2'] = remember_not_relic_2

    def round_1_spawn_detected(self, player, player_idx, obs, info, state, open_state):
        if 'prev_obs' not in state:
            prev_team_points = np.zeros(2, dtype=int)
        else:
            prev_team_points = state['prev_obs']['team_points']
        cur_team_points = obs['team_points']
        match_changed = False

        if obs['match_steps'] == 0:
            match_changed = True
            cur_team_points = np.zeros(2, dtype=int)
            prev_team_points = np.zeros(2, dtype=int)

        if match_changed:
            return

        delta_points = cur_team_points - prev_team_points

        points = delta_points[player_idx]


        positions = obs['units']['position'][player_idx]
        energies = obs['units']['energy'][player_idx]


        unique_units_positions = []
        for position, energy in zip(positions, energies):
            if position[0] == -1 or energy < 0:
                continue
            position = (position[0], position[1])
            if position not in unique_units_positions:
                unique_units_positions.append(position)

        remember_guarranted_relic = state['remember_guarranted_relic']
        remember_not_relic_0 = state['remember_not_relic_0']

        for x, y in unique_units_positions:
            if remember_guarranted_relic[x, y]:
                points -= 1

        assert points >= 0

        if points > 0:
            n_candidates = 0
            for x, y in unique_units_positions:
                if remember_guarranted_relic[x, y]:
                    continue
                if remember_not_relic_0[x, y]:
                    continue
                n_candidates += 1
            if points > n_candidates:
                return True

        return False

    def round_2_spawn_detected(self, player, player_idx, obs, info, state, open_state):
        if 'prev_obs' not in state:
            prev_team_points = np.zeros(2, dtype=int)
        else:
            prev_team_points = state['prev_obs']['team_points']
        cur_team_points = obs['team_points']
        match_changed = False

        if obs['match_steps'] == 0:
            match_changed = True
            cur_team_points = np.zeros(2, dtype=int)
            prev_team_points = np.zeros(2, dtype=int)

        if match_changed:
            return

        delta_points = cur_team_points - prev_team_points


        points = delta_points[player_idx]


        positions = obs['units']['position'][player_idx]
        energies = obs['units']['energy'][player_idx]


        unique_units_positions = []
        for position, energy in zip(positions, energies):
            if position[0] == -1 or energy < 0:
                continue
            position = (position[0], position[1])
            if position not in unique_units_positions:
                unique_units_positions.append(position)

        remember_guarranted_relic = state['remember_guarranted_relic']
        remember_not_relic_0 = state['remember_not_relic_0']
        remember_not_relic_1 = state['remember_not_relic_1']

        for x, y in unique_units_positions:
            if remember_guarranted_relic[x, y]:
                points -= 1

        assert points >= 0

        if points > 0:
            n_candidates = 0
            for x, y in unique_units_positions:
                if remember_guarranted_relic[x, y]:
                    continue
                if remember_not_relic_0[x, y] and remember_not_relic_1[x, y]:
                    continue
                n_candidates += 1
            if points > n_candidates:
                return True

        return False

    def update_relic(self, player, player_idx, obs, info, state, open_state, prev_state):
        steps = obs['steps']
        round_number = max(0, (steps - 1)) // 101

        #print('round_number:', round_number, steps, match_steps)

        relic_nodes = obs['relic_nodes']
        relic_nodes_found_on_this_tick = []

        remember_relic_nodes = state.get('remember_relic_nodes', np.zeros((6, 2), dtype=int) - 1)

        for idx, relic_node in enumerate(relic_nodes):
            if relic_node[0] == -1:
                continue
            x, y = relic_node
            if remember_relic_nodes[idx][0] == -1:
                relic_nodes_found_on_this_tick.append((x, y))
            remember_relic_nodes[idx][0] = x
            remember_relic_nodes[idx][1] = y

            sx = 23 - y
            sy = 23 - x
            if remember_relic_nodes[(idx + 3) % 6][0] == -1:
                relic_nodes_found_on_this_tick.append((sx, sy))
            if remember_relic_nodes[(idx + 3) % 6][0] != -1:
                assert remember_relic_nodes[(idx + 3) % 6][0] == sx
                assert remember_relic_nodes[(idx + 3) % 6][1] == sy
            remember_relic_nodes[(idx + 3) % 6][0] = sx
            remember_relic_nodes[(idx + 3) % 6][1] = sy
        state['remember_relic_nodes'] = remember_relic_nodes

        obs['remember_relic_nodes'] = np.copy(remember_relic_nodes)

        #if player == 'player_1' and steps == 254:
        #    pass
        # events
        # round_0_starts
        # round_1_starts
        # round_2_starts
        # round_0_relic_found
        # round_1_relic_found
        # round_2_relic_found
        # round_0_spawn_ends
        # round_1_spawn_ends
        # round_2_spawn_ends
        # round_0_all_explored
        # round_1_all_explored
        # round_2_all_explored


        event_round_0_starts = round_number == 0 and steps == 0
        if event_round_0_starts:
            self.process_event_round_0_starts(player, player_idx, obs, info, state)

        #print(round_number, steps, match_steps)
        event_round_1_starts = round_number == 1 and steps == 102
        if event_round_1_starts:
            self.process_event_round_1_starts(player, player_idx, obs, info, state)

        event_round_2_starts = round_number == 2 and steps == 203
        if event_round_2_starts:
            self.process_event_round_2_starts(player, player_idx, obs, info, state)

        prev_remember_relic_nodes = prev_state['remember_relic_nodes']
        event_round_0_relic_found = prev_remember_relic_nodes[0][0] != remember_relic_nodes[0][0]
        if event_round_0_relic_found:
            self.process_event_round_0_relic_found(player, player_idx, obs, info, state)

        event_round_1_relic_found = prev_remember_relic_nodes[1][0] != remember_relic_nodes[1][0]
        if event_round_1_relic_found:
            self.process_event_round_1_relic_found(player, player_idx, obs, info, state)


        event_round_2_relic_found = prev_remember_relic_nodes[2][0] != remember_relic_nodes[2][0]
        if event_round_2_relic_found:
            self.process_event_round_2_relic_found(player, player_idx, obs, info, state)

        event_round_0_spawn_ends = state['event_round_0_spawn_ends_fired'] == False and round_number == 0 and (steps == 50 or np.sum(obs['team_points']) > 0)
        if event_round_0_spawn_ends:
            self.process_event_round_0_spawn_ends(player, player_idx, obs, info, state)


        event_round_1_spawn_ends = state['event_round_1_spawn_ends_fired'] == False and round_number == 1 and (steps == 151 or self.round_1_spawn_detected(player, player_idx, obs, info, state, open_state))
        if event_round_1_spawn_ends:
            self.process_event_round_1_spawn_ends(player, player_idx, obs, info, state)

        event_round_2_spawn_ends = state['event_round_2_spawn_ends_fired'] == False and round_number == 2 and (steps == 252 or self.round_2_spawn_detected(player, player_idx, obs, info, state, open_state))
        if event_round_2_spawn_ends:
            self.process_event_round_2_spawn_ends(player, player_idx, obs, info, state)

        possible_positions_for_relic_node_spawn_0 = state.get('possible_positions_for_relic_node_spawn_0', None)
        possible_positions_for_relic_node_spawn_1 = state.get('possible_positions_for_relic_node_spawn_1', None)
        possible_positions_for_relic_node_spawn_2 = state.get('possible_positions_for_relic_node_spawn_2', None)

        prev_possible_positions_for_relic_node_spawn_0 = np.copy(possible_positions_for_relic_node_spawn_0) if possible_positions_for_relic_node_spawn_0 is not None else None
        prev_possible_positions_for_relic_node_spawn_1 = np.copy(possible_positions_for_relic_node_spawn_1) if possible_positions_for_relic_node_spawn_1 is not None else None
        prev_possible_positions_for_relic_node_spawn_2 = np.copy(possible_positions_for_relic_node_spawn_2) if possible_positions_for_relic_node_spawn_2 is not None else None

        flipped_sensor_mask = obs['sensor_mask'][::-1, ::-1].T

        combined_sensor_mask = np.logical_or(obs['sensor_mask'], flipped_sensor_mask)
        combined_not_sensor_mask = np.logical_not(combined_sensor_mask)

        if state['enable_possible_positions_removing_0']:
            possible_positions_for_relic_node_spawn_0[combined_sensor_mask] = 0
            pass

        if state['enable_possible_positions_removing_1']:
            possible_positions_for_relic_node_spawn_1[combined_sensor_mask] = 0
            pass

        if state['enable_possible_positions_removing_2']:
            possible_positions_for_relic_node_spawn_2[combined_sensor_mask] = 0
            pass


        possible_positions_for_relic_node_timer_0 = state.get('possible_positions_for_relic_node_timer_0', None)
        possible_positions_for_relic_node_timer_1 = state.get('possible_positions_for_relic_node_timer_1', None)
        possible_positions_for_relic_node_timer_2 = state.get('possible_positions_for_relic_node_timer_2', None)

        if state['enable_possible_positions_timer_0']:
            possible_positions_for_relic_node_timer_0[combined_sensor_mask] = 0
            possible_positions_for_relic_node_timer_0[combined_not_sensor_mask] += 1

        if state['enable_possible_positions_timer_1']:
            possible_positions_for_relic_node_timer_1[combined_sensor_mask] = 0
            possible_positions_for_relic_node_timer_1[combined_not_sensor_mask] += 1

        if state['enable_possible_positions_timer_2']:
            possible_positions_for_relic_node_timer_2[combined_sensor_mask] = 0
            possible_positions_for_relic_node_timer_2[combined_not_sensor_mask] += 1



        event_round_0_all_explored = not event_round_0_relic_found and prev_possible_positions_for_relic_node_spawn_0 is not None and np.sum(prev_possible_positions_for_relic_node_spawn_0) != 0 and np.sum(possible_positions_for_relic_node_spawn_0) == 0
        if event_round_0_all_explored:
            self.process_event_event_round_0_all_explored(player, player_idx, obs, info, state)

        event_round_1_all_explored = not event_round_1_relic_found and prev_possible_positions_for_relic_node_spawn_1 is not None and np.sum(prev_possible_positions_for_relic_node_spawn_1) != 0 and np.sum(possible_positions_for_relic_node_spawn_1) == 0
        if event_round_1_all_explored:
            self.process_event_event_round_1_all_explored(player, player_idx, obs, info, state)

        event_round_2_all_explored = not event_round_2_relic_found and prev_possible_positions_for_relic_node_spawn_2 is not None and np.sum(prev_possible_positions_for_relic_node_spawn_2) != 0 and np.sum(possible_positions_for_relic_node_spawn_2) == 0
        if event_round_2_all_explored:
            self.process_event_event_round_2_all_explored(player, player_idx, obs, info, state)

        prev_remember_guarranted_relic = prev_state['remember_guarranted_relic']
        prev_remember_potential_relic_reduced = prev_state['remember_potential_relic_reduced']
        prev_remember_not_relic_reduced = prev_state['remember_not_relic_reduced']

        self.update_not_relic(player, player_idx, obs, info, state, open_state)
        self.update_guarranted_relic(player, player_idx, obs, info, state, open_state)


        obs['guarranted_relic'] = np.copy(state['remember_guarranted_relic'])
        obs['prev_guarranted_relic'] = np.copy(prev_remember_guarranted_relic)


        state['remember_potential_relic_reduced'] = np.logical_or.reduce([state['remember_potential_relic_0'], state['remember_potential_relic_1'], state['remember_potential_relic_2']])

        obs['potential_relic_reduced'] = np.copy(state['remember_potential_relic_reduced'])
        obs['potential_relic_0'] = np.copy(state['remember_potential_relic_0'])
        obs['potential_relic_1'] = np.copy(state['remember_potential_relic_1'])
        obs['potential_relic_2'] = np.copy(state['remember_potential_relic_2'])

        obs['prev_potential_relic_reduced'] = np.copy(prev_remember_potential_relic_reduced)

        state['remember_not_relic_reduced'] = np.logical_and.reduce([state['remember_not_relic_0'], state['remember_not_relic_1'], state['remember_not_relic_2']])

        obs['not_relic_reduced'] = np.copy(state['remember_not_relic_reduced'])
        obs['not_relic_0'] = np.copy(state['remember_not_relic_0'])
        obs['not_relic_1'] = np.copy(state['remember_not_relic_1'])
        obs['not_relic_2'] = np.copy(state['remember_not_relic_2'])


        obs['prev_not_relic_reduced'] = np.copy(prev_remember_not_relic_reduced)



        obs['relic_score'] = np.copy(state['relic_score'])

        obs['possible_positions_for_relic_node_spawn_reduced'] = np.logical_or.reduce([possible_positions_for_relic_node_spawn_0, possible_positions_for_relic_node_spawn_1, possible_positions_for_relic_node_spawn_2])

        obs['possible_positions_for_relic_node_spawn_0'] = np.copy(possible_positions_for_relic_node_spawn_0)
        obs['possible_positions_for_relic_node_spawn_1'] = np.copy(possible_positions_for_relic_node_spawn_1)
        obs['possible_positions_for_relic_node_spawn_2'] = np.copy(possible_positions_for_relic_node_spawn_2)


        obs['possible_positions_for_relic_node_timer_reduced'] = np.maximum.reduce([possible_positions_for_relic_node_timer_0, possible_positions_for_relic_node_timer_1, possible_positions_for_relic_node_timer_2])


        obs['possible_positions_for_relic_node_timer_0'] = np.copy(possible_positions_for_relic_node_timer_0)
        obs['possible_positions_for_relic_node_timer_1'] = np.copy(possible_positions_for_relic_node_timer_1)
        obs['possible_positions_for_relic_node_timer_2'] = np.copy(possible_positions_for_relic_node_timer_2)




        obs['no_more_spawn'] = np.copy(state['no_more_spawn'])


        assert np.sum(np.logical_and(obs['guarranted_relic'], obs['potential_relic_reduced'])) == 0
        assert np.sum(np.logical_and(obs['guarranted_relic'], obs['not_relic_reduced'])) == 0
        assert np.sum(np.logical_and(obs['potential_relic_reduced'], obs['not_relic_reduced'])) == 0


        return

    def update_team_points(self, player, player_idx, obs, info, state):
        if 'prev_obs' in state:
            prev_team_points = state['prev_obs']['team_points']
        else:
            prev_team_points = np.zeros(2, dtype=int)
        if obs['match_steps'] == 0:
            prev_team_points = np.zeros(2, dtype=int)
        obs['prev_team_points'] = np.copy(prev_team_points)

    def update_team_wins(self, player, player_idx, obs, info, state):
        if 'prev_obs' in state:
            prev_team_wins = state['prev_obs']['team_wins']
        else:
            prev_team_wins = np.zeros(2, dtype=int)
        obs['prev_team_wins'] = np.copy(prev_team_wins)

    def update_tile_unseen_ticks(self, player, player_idx, obs, info, state):
        tile_unseen_ticks = state.get('tile_unseen_ticks', np.zeros((24, 24), dtype=int) + 500)

        tile_unseen_ticks[obs['sensor_mask']] = 0
        tile_unseen_ticks[obs['sensor_mask'] == False] += 1

        state['tile_unseen_ticks'] = tile_unseen_ticks
        obs['tile_unseen_ticks'] = np.copy(tile_unseen_ticks)


    def update_my_seen(self, player, player_idx, obs, info, state):
        my_seen_ticks_ago = state.get('my_seen_ticks_ago', np.zeros((24, 24), dtype=int) + 500)

        my_seen_ticks_ago += 1

        positions = obs['units']['position'][player_idx]
        #energies = obs['units']['energy'][1 - player_idx]
        for idx, position in enumerate(positions):
            x, y = position[0], position[1]
            if x == -1:
                continue
            my_seen_ticks_ago[x, y] = 0

        state['my_seen_ticks_ago'] = my_seen_ticks_ago
        obs['my_seen_ticks_ago'] = np.copy(my_seen_ticks_ago)

        my_seen_energy = state.get('my_seen_energy', np.zeros((24, 24), dtype=int) - 2)

        positions = obs['units']['position'][player_idx]
        energies = obs['units']['energy'][player_idx]
        for idx, position in enumerate(positions):
            x, y = position[0], position[1]
            if x == -1:
                continue
            if energies[idx] >= 0:
                my_seen_energy[x, y] = energies[idx]
            else:
                my_seen_energy[x, y] = -1

        state['my_seen_energy'] = my_seen_energy
        obs['my_seen_energy'] = np.copy(my_seen_energy)

    def update_enemy_seen(self, player, player_idx, obs, info, state):
        enemy_seen_ticks_ago = state.get('enemy_seen_ticks_ago', np.zeros((24, 24), dtype=int) + 500)

        enemy_seen_ticks_ago += 1

        positions = obs['units']['position'][1 - player_idx]
        #energies = obs['units']['energy'][1 - player_idx]
        for idx, position in enumerate(positions):
            x, y = position[0], position[1]
            if x == -1:
                continue
            enemy_seen_ticks_ago[x, y] = 0

        state['enemy_seen_ticks_ago'] = enemy_seen_ticks_ago
        obs['enemy_seen_ticks_ago'] = np.copy(enemy_seen_ticks_ago)

        enemy_seen_energy = state.get('enemy_seen_energy', np.zeros((24, 24), dtype=int) - 2)

        positions = obs['units']['position'][1 - player_idx]
        energies = obs['units']['energy'][1 - player_idx]
        for idx, position in enumerate(positions):
            x, y = position[0], position[1]
            if x == -1:
                continue
            if energies[idx] >= 0:
                enemy_seen_energy[x, y] = energies[idx]
            else:
                enemy_seen_energy[x, y] = -1

        state['enemy_seen_energy'] = enemy_seen_energy
        obs['enemy_seen_energy'] = np.copy(enemy_seen_energy)


    def update_state_for_player(self, player, player_idx, obs, info, cells_received_sap, previous_sap_units):
        state = self.state.get(player, {})

        prev_state = {
            'remember_relic_nodes': np.copy(state.get('remember_relic_nodes', np.zeros((6, 2), dtype=int) - 1)),
            'remember_guarranted_relic': np.copy(state.get('remember_guarranted_relic', np.zeros((24, 24), dtype=bool))),
            'remember_potential_relic_reduced': np.copy(state.get('remember_potential_relic_reduced', np.zeros((24, 24), dtype=bool))),
            'remember_not_relic_reduced': np.copy(state.get('remember_not_relic_reduced', np.zeros((24, 24), dtype=bool))),
        }

        with self.profiler('open_state'):
            open_state = None
        open_params = None
        if info != None:
            open_state = info['state'][player]
            if 'full_params' in info:
                state['open_params'] = info['full_params']

                open_params = state.get('open_params', None)

            assert (open_state['relic_nodes_map_weights'] == open_state['relic_nodes_map_weights'][::-1, ::-1].T).all()


        obs['known_game_params'] = self.known_game_params


        # This function has to go before 'update_map_tiles' !!!, because it uses previous 'remember_tile_type'
        # nebula_tile_vision_reduction=list(range(0, 8))
        with self.profiler('calculate_nebula_tile_vision_reduction'):
            self.calculate_nebula_tile_vision_reduction(player_idx, obs, state)

        with self.profiler('points_wins'):
            self.update_team_points(player, player_idx, obs, info, state)
            self.update_team_wins(player, player_idx, obs, info, state)

        with self.profiler('update_shifts'):
            self.update_shifts(player, player_idx, obs, info, state)

        with self.profiler('apply_shift'):
            self.apply_shift(player, player_idx, obs, info, state)

        with self.profiler('update_map_tiles'):
            self.update_map_tiles(player, player_idx, obs, info, state)

        with self.profiler('update_energy'):
            self.update_energy(player, player_idx, obs, info, state)

        with self.profiler('update_relic'):
            self.update_relic(player, player_idx, obs, info, state, open_state, prev_state)

        with self.profiler('update_tile_unseen_ticks'):
            self.update_tile_unseen_ticks(player, player_idx, obs, info, state)

        with self.profiler('update_seen'):
            self.update_enemy_seen(player, player_idx, obs, info, state)
            self.update_my_seen(player, player_idx, obs, info, state)


        #unit_move_cost=list(range(1, 6)),
        # known

        #unit_sensor_range=[1, 2, 3, 4],
        # known

        with self.profiler('calculate_nebula_tile_energy_reduction'):
            self.calculate_nebula_tile_energy_reduction(player_idx, obs, state) # ok may be can be improved?

        with self.profiler('calculate_unit_sap_dropoff_factor'):
            self.calculate_unit_sap_dropoff_factor(player_idx, obs, state, cells_received_sap)

        with self.profiler('calculate_unit_energy_void_factor'):
            self.calculate_unit_energy_void_factor(player_idx, obs, state)


        with self.profiler('update_unexpected_energy_change'):
            self.update_unexpected_energy_change(player, player_idx, obs, info, state, previous_sap_units)

        obs['need_exploration_0'] = np.copy(state['enable_possible_positions_removing_0'])
        obs['need_exploration_1'] = np.copy(state['enable_possible_positions_removing_1'])
        obs['need_exploration_2'] = np.copy(state['enable_possible_positions_removing_2'])

        #unit_sap_cost=list(range(30, 51)),
        # known

        #unit_sap_range=list(range(3, 8)),
        # known

        #unit_sap_dropoff_factor=[0.25, 0.5, 1],
        # ixanezis

        #unit_energy_void_factor=[0.0625, 0.125, 0.25, 0.375],
        # need implement !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        #nebula_tile_drift_speed=[-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15],
        # implemented in self.update_shifts

        #energy_node_drift_speed=[0.01, 0.02, 0.03, 0.04, 0.05],
        # not needed ? Too hard and useless ?

        #energy_node_drift_magnitude=list(range(3, 6)),
        # not needed ? Too hard and useless ?


        #obs['seen_local'] = state.get('seen_local', np.zeros((24, 24), dtype=bool))
        #obs['prev_seen_local'] = state.get('prev_seen_local', np.zeros((24, 24), dtype=bool))

        obs['prev_units'] = copy.deepcopy(state['prev_obs']['units']) if 'prev_obs' in state else None

        if info != None:
            # enemy
            open_units_mask = np.array(open_state['units_mask'][1 - player_idx])
            open_energy = np.array(open_state['units']['energy'][1 - player_idx])

            prev_energy_sum = state.get('energy_sum', 0)
            basic_energy = np.sum(open_units_mask) * 100
            energy_sum = np.sum(open_energy[open_units_mask]) - basic_energy

            obs['energy_sum'] = energy_sum - prev_energy_sum
            state['energy_sum'] = energy_sum


            prev_energy_dead = state.get('energy_dead', 0)
            energy_dead = len(open_energy[open_units_mask][open_energy[open_units_mask] < 0])

            obs['energy_dead'] = energy_dead - prev_energy_dead
            state['energy_dead'] = energy_dead

            # my
            open_units_mask = np.array(open_state['units_mask'][player_idx])
            open_energy = np.array(open_state['units']['energy'][player_idx])

            prev_energy_sum = state.get('my_energy_sum', 0)
            basic_energy = np.sum(open_units_mask) * 100
            energy_sum = np.sum(open_energy[open_units_mask]) - basic_energy

            obs['my_energy_sum'] = energy_sum - prev_energy_sum
            state['my_energy_sum'] = energy_sum


            prev_energy_dead = state.get('my_energy_dead', 0)
            energy_dead = len(open_energy[open_units_mask][open_energy[open_units_mask] < 0])

            obs['my_energy_dead'] = energy_dead - prev_energy_dead
            state['my_energy_dead'] = energy_dead

            #obs['sap_energy'] = open_state['delta_energy']

            obs['n_points_positions_on_map'] = np.sum(open_state['relic_nodes_map_weights'] > 0)
            obs['n_relic_nodes_on_map'] = np.sum(open_state['relic_nodes_mask'] > 0)


        state['prev_obs'] = obs

        return state


    def update_state(self, obs, info, cells_received_sap, previous_sap_units):
        if cells_received_sap == None:
            cells_received_sap = [None, None]

        if previous_sap_units == None:
            previous_sap_units = [[], []]

        if self.state == None:
            self.state = {'player_0':{}, 'player_1':{}}

        for player_idx, player in enumerate(['player_0', 'player_1']):
            with self.profiler('update_state_for_player'):
                # print(f"update_state_for_player {player_idx}, step {obs[player]['steps']}")
                self.state[player] = self.update_state_for_player(player, player_idx, obs[player], info, cells_received_sap[player_idx], previous_sap_units[player_idx])


    def step(self, action):
        if self.flags.kaggle:
            #print(action, file=sys.stderr)
            if action is None:
                actions_processed3 = self.dummy_actions()
                actions_taken3 = self.dummy_actions_taken()
                cells_received_sap = None
                previous_sap_units = [[], []]
            else:
                actions_processed3, actions_taken3, cells_received_sap, previous_sap_units = self.process_actions(action, self.known_game_params['unit_sap_range'], self.known_game_params['unit_sap_cost'], self.known_game_params['unit_move_cost'], self.flags)

            self.prev_cells_received_sap = cells_received_sap
            self.prev_previous_sap_units = previous_sap_units

            info3 = {}
            info3["LOGGING_CPU_kaggle_actions"] = actions_processed3
            info3["actions_taken_GPU_CPU"] = actions_taken3

            return None, None, None, info3
        else:
            self.profiler.begin_block("SB3Wrapper.step")

            with self.profiler("precalc"):
                if action is None:
                    actions_processed3 = self.dummy_actions()
                    actions_taken3 = self.dummy_actions_taken()
                    cells_received_sap = None
                    previous_sap_units = [[], []]
                else:
                    actions_processed3, actions_taken3, cells_received_sap, previous_sap_units = self.process_actions(action, self.known_game_params['unit_sap_range'], self.known_game_params['unit_sap_cost'], self.known_game_params['unit_move_cost'], self.flags)

            with self.profiler('lux step'):
                obs, reward, terminated, truncated, info3 = self.env3.step(actions_processed3)

            with self.profiler('flip'):
                obs = self.flip_obs(obs)
                info3 = self.flip_info_step(info3)

            with self.profiler('update_state'):
                self.update_state(obs, info3, cells_received_sap, previous_sap_units)

            info3["actions_taken_GPU_CPU"] = actions_taken3

            with self.profiler('get_available_actions_mask'):
                info3["GPU1_available_actions_mask"] = self.action_space.get_available_actions_mask(
                    obs,
                    self.known_game_params['unit_sap_range'],
                    self.known_game_params['unit_sap_cost'],
                    self.known_game_params['unit_move_cost'],
                    self.flags.enable_sap_masks
                )

            self.prev_obs3 = obs

            done3 = False
            for key, value in terminated.items():
                if value:
                    done3 = True
            for key, value in truncated.items():
                if value:
                    done3 = True

            if self.flags.enable_game_early_stop:
                minrew = min(reward['player_0'], reward['player_1'])
                maxrew = max(reward['player_0'], reward['player_1'])
                remrew = 5 - minrew - maxrew
                if minrew + remrew < maxrew:
                    done3 = True

            self.real_done = done3

            if not self.flags.five_rounds or self.example:
                steps = obs['player_0']['steps']
                match_steps = obs['player_0']['match_steps']

                if steps != match_steps and match_steps == 0:
                    done3 = True

            with self.profiler('calc_units_mask'):
                unit_masks = self.calc_units_mask(obs, info3)

            self.profiler.end_block("SB3Wrapper.step")
            return obs, reward, done3, unit_masks

    def reset(self, kaggle_observation=None, force=False, custom_seed=None, **kwargs):
        if force:
            self.n_games_played = 0
            self.real_done = True

        if self.flags.kaggle:
            assert kaggle_observation != None

            if isinstance(kaggle_observation, tuple):
                obs3, kaggle_player_id = kaggle_observation
                self.kaggle_player_id = kaggle_player_id
            else:
                obs3 = kaggle_observation

            self.known_game_params = obs3['player_0']['env_cfg']

            obs3 = self.flip_obs(obs3)

            self.update_state(obs3, None, self.prev_cells_received_sap, self.prev_previous_sap_units)

            info3 = {}
            info3["actions_taken_GPU_CPU"] = self.dummy_actions_taken()
            info3["GPU1_available_actions_mask"] = self.action_space.get_available_actions_mask(
                obs3,

                self.known_game_params['unit_sap_range'],
                self.known_game_params['unit_sap_cost'],
                self.known_game_params['unit_move_cost'],
                self.flags.enable_sap_masks
            )
            self.prev_obs3 = obs3

            return obs3, None, None, self.calc_units_mask(self.prev_obs3, info3)
        else:
            if self.real_done:
                self.n_games_played += 1


                self.state = None
                self.known_game_params = None
                #env_params = EnvParams(map_type=1, max_steps_in_match=100)

                if custom_seed is not None:
                    seed = custom_seed
                else:
                    seed = (secrets.randbits(64) ^ int(time.time_ns()) ^ int.from_bytes(os.urandom(8), "big")) % 4000000000
                    if self.flags.replay:
                        seed = 1853128757
                    if self.example:
                        seed = self.n_games_played  * 1000 + self.env_id

                #print('REAL DONE', self.env_id, seed)

                #obs3, info3 = self.env3.reset(seed=seed, options=dict(params=env_params))
                obs3, info3 = self.env3.reset(seed=seed)
                #if self.example:
                    #print("Reset", self.env_id, seed)

                self.known_game_params = info3['params']

                # print(f"known game params: {self.known_game_params}")

                obs3 = self.flip_obs(obs3)
                info3 = self.flip_info_reset(info3)

                self.update_state(obs3, info3, None, None)

                self.prev_obs3 = obs3

                result = self.step(None)

                if self.initial_reset:
                    if self.env_id == 0 or self.example or (custom_seed is not None):
                        n_iters = 0
                    else:
                        n_iters = random.randint(0, 503)
                    for i in range(n_iters):
                        result = self.step(None)
                    self.initial_reset = False

                return result
            else:
                #print('FAKE DONE', self.env_id)
                result = self.step(None)
                return result

    def process_actions(self, action, sap_range, sap_cost, move_cost, flip_learning):
        return self.action_space.process_actions(
            action,
            self.prev_obs3,
            sap_range,
            sap_cost,
            move_cost,
            flip_learning,
            self.kaggle_player_id
        )


    def calc_units_mask(self, input_obs, info3):
        indexes = []
        x_cord = []
        y_cord = []

        #x_cord_c = []
        #x_cord_d = []

        #y_cord_c = []
        #y_cord_d = []

        energy_arr_c = []

        n_energy_move_c = []
        n_energy_move_d = []

        n_energy_sap_c = []
        n_energy_sap_d = []

        number_c = []
        number_d = []

        is_fake_c = []
        is_fake_d = []

        for player_idx, obs in enumerate(input_obs.values()):
            n_units = 0
            numbers = np.zeros((24, 24), dtype=int)
            for pos_idx, position in enumerate(obs['units']['position'][player_idx]):
                energy = obs['units']['energy'][player_idx][pos_idx]
                if position[0] == -1 or energy < 0:
                    indexes.append([0])
                    x_cord.append(0)
                    y_cord.append(0)


                    #x_cord_c.append(0)
                    #x_cord_d.append(-1)

                    #y_cord_c.append(0)
                    #y_cord_d.append(-1)

                    energy_arr_c.append(0)

                    n_energy_move_c.append(0)
                    n_energy_move_d.append(-1)

                    n_energy_sap_c.append(0)
                    n_energy_sap_d.append(-1)

                    number_c.append(0)
                    number_d.append(-1)

                    is_fake_c.append(1)
                    is_fake_d.append(1)

                    n_units += 1
                    continue
                x = position[0]
                y = position[1]
                numbers[x, y] += 1

                indexes.append([x * 24 + y])
                x_cord.append(x)
                y_cord.append(y)

                #x_cord_c.append(x)
                #x_cord_d.append(x)

                #y_cord_c.append(y)
                #y_cord_d.append(y)


                energy_arr_c.append(energy)


                n_energy_move_c.append(energy / self.known_game_params['unit_move_cost'])

                n_moves = int(energy / self.known_game_params['unit_move_cost'] + 1e-3)
                n_moves = min(n_moves, 66)
                n_energy_move_d.append(n_moves)

                n_energy_sap_c.append(energy / self.known_game_params['unit_sap_cost'])

                n_saps = int(energy / self.known_game_params['unit_sap_cost'] + 1e-3)
                n_saps = min(n_saps, 13)
                n_energy_sap_d.append(n_saps)

                number_c.append(numbers[x, y])
                number_d.append(numbers[x, y])

                is_fake_c.append(0)
                is_fake_d.append(0)

                n_units += 1

            assert n_units == 16

        assert len(x_cord) == 32

        result = {}
        result['indexes'] = np.array(indexes).astype(np.int16)
        result['x_cord'] = np.array(x_cord).astype(np.int16)
        result['y_cord'] = np.array(y_cord).astype(np.int16)


        # continues features
        #x_cord_c = np.array(x_cord_c).astype(np.float32) / 23.
        #assert np.max(x_cord_c) <= 1.0 and np.min(x_cord_c) >= 0.0

        #y_cord_c = np.array(y_cord_c).astype(np.float32) / 23.
        #assert np.max(y_cord_c) <= 1.0 and np.min(y_cord_c) >= 0.0

        energy_arr_c = np.array(energy_arr_c).astype(np.float32) / 400.
        assert np.max(energy_arr_c) <= 1.0 and np.min(energy_arr_c) >= 0.0

        n_energy_move_c = np.array(n_energy_move_c).astype(np.float32) / 400.
        assert np.max(n_energy_move_c) <= 1.0 and np.min(n_energy_move_c) >= 0.0

        n_energy_sap_c = np.array(n_energy_sap_c).astype(np.float32) / 14.
        assert np.max(n_energy_sap_c) <= 1.0 and np.min(n_energy_sap_c) >= 0.0

        number_c = np.array(number_c).astype(np.float32) / 16.
        assert np.max(number_c) <= 1.0 and np.min(number_c) >= 0.0

        is_fake_c = np.array(is_fake_c).astype(np.float32)
        assert np.max(is_fake_c) <= 1.0 and np.min(is_fake_c) >= 0.0

        # embedding features
        embedding_features = {}
        #embedding_features['x_cord_d'] = np.array(x_cord_d).astype(np.int32)
        #assert np.max(embedding_features['x_cord_d']) <= 23 and np.min(embedding_features['x_cord_d']) >= -1
        #embedding_features['y_cord_d'] = np.array(y_cord_d).astype(np.int32)
        #assert np.max(embedding_features['y_cord_d']) <= 23 and np.min(embedding_features['y_cord_d']) >= -1

        embedding_features['n_energy_move_d'] = np.array(n_energy_move_d).astype(np.int32)
        assert np.max(embedding_features['n_energy_move_d']) <= 66 and np.min(embedding_features['n_energy_move_d']) >= -1
        embedding_features['n_energy_sap_d'] = np.array(n_energy_sap_d).astype(np.int32)
        assert np.max(embedding_features['n_energy_sap_d']) <= 13 and np.min(embedding_features['n_energy_sap_d']) >= -1
        embedding_features['number_d'] = np.array(number_d).astype(np.int32)
        assert np.max(embedding_features['number_d']) <= 16 and np.min(embedding_features['number_d']) >= -1
        embedding_features['is_fake_d'] = np.array(is_fake_d).astype(np.int32)
        assert np.max(embedding_features['is_fake_d']) <= 1 and np.min(embedding_features['is_fake_d']) >= 0

        discrete_size = {
            #'x_cord_d': 23 + 1 + 1, # possible values [-1, 0, ..., 23]
            #'y_cord_d': 23 + 1 + 1, # possible values [-1, 0, ..., 23]
            'n_energy_move_d': 66 + 1 + 1, # possible values [-1, 0, ..., 66]
            'n_energy_sap_d': 13 + 1 + 1, # possible values [-1, 0, ..., 13]
            'number_d': 16 + 1 + 1, # possible values [-1, 0, ..., 16]
            'is_fake_d': 1 + 1, # possible values [0, 1]
        }

        #embedding_features['x_cord_d'] += 1 # to start from 0
        #embedding_features['y_cord_d'] += 1 # to start from 0
        embedding_features['n_energy_move_d'] += 1 # to start from 0
        embedding_features['n_energy_sap_d'] += 1 # to start from 0
        embedding_features['number_d'] += 1 # to start from 0


        shift = 0
        for key, size in discrete_size.items():
            embedding_features[key] += shift
            shift += size
        # 103

        #result['continues_features'] = np.stack([x_cord_c, y_cord_c, energy_arr_c, n_energy_move_c, n_energy_sap_c, number_c, is_fake_c], axis=1).astype(np.float32)
        result['continues_features'] = np.stack([energy_arr_c, n_energy_move_c, n_energy_sap_c, number_c, is_fake_c], axis=1).astype(np.float32)
        result['embedding_features'] = np.stack([embedding_features[key] for key in discrete_size.keys()], axis=1).astype(np.int16)

        if not self.flags.use_embedding_input:
            one_hot_encoded_embedding_features = np.zeros((32, 103), dtype=bool)
            np.put_along_axis(
                one_hot_encoded_embedding_features,  # Target array
                result['embedding_features'],  # Indices (reshaped for broadcasting)
                1,  # Value to assign (set one-hot to 1)
                axis=-1  # Apply along last axis (103 classes)
            )

            result['one_hot_encoded_embedding_features'] = one_hot_encoded_embedding_features

        assert np.max(result['embedding_features']) < 103 and np.min(result['embedding_features']) >= 0

        my_points_arr = []
        enemy_points_arr = []
        winning = []
        team_points_advantage = []
        team_points_advantage_ratio = []
        turn = []
        round_ = []

        for player_idx, obs in enumerate(input_obs.values()):
            #print(obs.keys())
            my_points = obs['team_points'][player_idx]
            enemy_points = obs['team_points'][1 - player_idx]

            is_w = 0
            if my_points == enemy_points:
                is_w = 0.5
            elif my_points > enemy_points:
                is_w = 1.0
            winning.append(is_w)
            my_points_arr.append(np.clip(my_points / 1000.0, 0.0, 1.0))
            enemy_points_arr.append(np.clip(enemy_points / 1000.0, 0.0, 1.0))

            advantage = np.clip((my_points - enemy_points) / 100.0, -1.0, 1.0)
            team_points_advantage.append(advantage)

            advantage_ratio = np.clip((my_points - enemy_points) / max(1, enemy_points), -1.0, 1.0)
            team_points_advantage_ratio.append(advantage_ratio)

            turn.append(obs['match_steps'] / 100)
            round_.append((max(0, (obs['steps'] - 1)) // 101) / 4)

        my_points_arr = np.array(my_points_arr).astype(np.float32)
        enemy_points_arr = np.array(enemy_points_arr).astype(np.float32)
        winning = np.array(winning).astype(np.float32)
        team_points_advantage = np.array(team_points_advantage).astype(np.float32)
        team_points_advantage_ratio = np.array(team_points_advantage_ratio).astype(np.float32)
        turn = np.array(turn).astype(np.float32)
        round_ = np.array(round_).astype(np.float32)

        additional_features = np.stack([my_points_arr, enemy_points_arr, winning, team_points_advantage, team_points_advantage_ratio, turn, round_], axis=1).astype(np.float32)

        #print(additional_features.shape)
        #assert False
        result['additional_features'] = additional_features

        info3['GPU1_units_masks'] = result
        return info3

    def flip_arr(self, arr):
        for pos_idx, position in enumerate(arr):
            x, y = position
            if x == -1:
                continue
            sx = 23 - y
            sy = 23 - x
            if self.flags.flip_axes:
                arr[pos_idx][0] = sy
                arr[pos_idx][1] = sx
            else:
                arr[pos_idx][0] = sx
                arr[pos_idx][1] = sy

        return arr

    def flip_arr_2d(self, arr):
        if self.flags.flip_axes:
            arr[:] = arr[::-1, ::-1][:]
        else:
            arr[:] = arr[::-1, ::-1].T[:]
        return arr

    def flip_obs(self, obs):
        # obs has keys player_0, player_1
        # player_1 has keys dict_keys(['units', 'units_mask', 'sensor_mask', 'map_features', 'relic_nodes', 'relic_nodes_mask', 'team_points', 'team_wins', 'steps', 'match_steps'])
        # obs['player_1']['units'] has position and energy

        if self.flags.flip_learning:
            self.flip_arr(obs['player_1']['units']['position'][0])
            self.flip_arr(obs['player_1']['units']['position'][1])

            # energy unchanged
            # units_mask unchanged

            self.flip_arr_2d(obs['player_1']['sensor_mask'])

            # obs['player_1']['map_features'] has energy and tile_type

            self.flip_arr_2d(obs['player_1']['map_features']['energy'])
            self.flip_arr_2d(obs['player_1']['map_features']['tile_type'])

            self.flip_arr(obs['player_1']['relic_nodes']) # (1, 6)

        # relic_nodes_mask unchanged # (1, 6)

        # 'team_points', 'team_wins', 'steps', 'match_steps' unchanged

        #print(obs['player_1']['relic_nodes_mask'])
        #assert False
        return obs

    def flip_info_step(self, info):
        #print(info.keys())
        # info keys are dict_keys(['discount', 'final_observation', 'final_state', 'player_0', 'player_1'])


        istate = info['final_state']

        state = {'player_0' : {}, 'player_1' : {}}

        state['player_0']['units'] = {'position' : np.array(istate.units.position),
                                      'energy' : np.array(istate.units.energy)}
        state['player_0']['energy_nodes'] = np.array(istate.energy_nodes)
        state['player_0']['relic_nodes'] = np.array(istate.relic_nodes)
        state['player_0']['relic_nodes_mask'] = np.array(istate.relic_nodes_mask)
        state['player_0']['relic_node_configs'] = np.array(istate.relic_node_configs)
        state['player_0']['relic_nodes_map_weights'] = np.array(istate.relic_nodes_map_weights)
        state['player_0']['map_features'] = {'energy' : np.array(istate.map_features.energy), 'tile_type' : np.array(istate.map_features.tile_type)}
        #state['player_0']['sensor_mask'] = np.array(istate.sensor_mask)
        #state['player_0']['vision_power_map'] = np.array(istate.vision_power_map)

        state['player_0']['steps'] = istate.steps
        state['player_0']['match_steps'] = istate.match_steps

        state['player_0']['team_points'] = np.array(istate.team_points)
        state['player_0']['team_wins'] = np.array(istate.team_wins)
        state['player_0']['units_mask'] = np.array(istate.units_mask)

        state['player_0']['delta_energy_recieve'] = np.array(istate.delta_energy_recieve)
        state['player_0']['delta_energy_shoot'] = np.array(istate.delta_energy_shoot)

        #if np.sum(state['player_0']['delta_energy']) != 0:
        #    print(state['player_0']['delta_energy'])
        #    assert False

        state['player_1'] = copy.deepcopy(state['player_0'])

        if self.flags.flip_learning:
            #state_copy.units.position = np.array(state_copy.units.position)
            self.flip_arr(state['player_1']['units']['position'][0])
            self.flip_arr(state['player_1']['units']['position'][1])

            self.flip_arr(state['player_1']['energy_nodes'])
            self.flip_arr(state['player_1']['relic_nodes'])

            for rnc in state['player_1']['relic_node_configs']:
                self.flip_arr_2d(rnc)

            self.flip_arr_2d(state['player_1']['relic_nodes_map_weights'])
            self.flip_arr_2d(state['player_1']['map_features']['energy'])
            self.flip_arr_2d(state['player_1']['map_features']['tile_type'])
            #self.flip_arr_2d(state['player_1']['sensor_mask'])
            #self.flip_arr_2d(state['player_1']['vision_power_map'])

        info['state'] = state
        return info


    def flip_info_reset(self, info):
        #print(info.keys())
        # info has keys dict_keys(['params', 'full_params', 'state'])

        # state has | units (position, energy) |  units_mask | energy_nodes (array of poses) | energy_node_fns ??? |  energy_nodes_mask  |  relic_nodes (array of positions) |
        # relic_node_configs (array of 2d arrays, should be flipped) | relic_nodes_mask (1d arr) | relic_nodes_map_weights (2d arr flip) | map_features (energy, tile_type) flip |
        # sensor_mask (2d flip) | vision_power_map (2d flip)
        # team_points=Array([0, 0], dtype=int32), team_wins=Array([0, 0], dtype=int32), steps=Array(0, dtype=int32, weak_type=True), match_steps=Array(0, dtype=int32, weak_type=True))

        istate = info['state']

        state = {'player_0' : {}, 'player_1' : {}}

        state['player_0']['units'] = {'position' : np.array(istate.units.position),
                                      'energy' : np.array(istate.units.energy)}
        state['player_0']['energy_nodes'] = np.array(istate.energy_nodes)
        state['player_0']['relic_nodes'] = np.array(istate.relic_nodes)
        state['player_0']['relic_nodes_mask'] = np.array(istate.relic_nodes_mask)
        state['player_0']['relic_node_configs'] = np.array(istate.relic_node_configs)
        state['player_0']['relic_nodes_map_weights'] = np.array(istate.relic_nodes_map_weights)
        state['player_0']['map_features'] = {'energy' : np.array(istate.map_features.energy), 'tile_type' : np.array(istate.map_features.tile_type)}
        #state['player_0']['sensor_mask'] = np.array(istate.sensor_mask)
        #state['player_0']['vision_power_map'] = np.array(istate.vision_power_map)

        state['player_0']['steps'] = istate.steps
        state['player_0']['match_steps'] = istate.match_steps

        state['player_0']['team_points'] = np.array(istate.team_points)
        state['player_0']['team_wins'] = np.array(istate.team_wins)
        state['player_0']['units_mask'] = np.array(istate.units_mask)

        state['player_0']['delta_energy_recieve'] = np.array(istate.delta_energy_recieve)
        state['player_0']['delta_energy_shoot'] = np.array(istate.delta_energy_shoot)



        state['player_1'] = copy.deepcopy(state['player_0'])

        if self.flags.flip_learning:
            #state_copy.units.position = np.array(state_copy.units.position)
            self.flip_arr(state['player_1']['units']['position'][0])
            self.flip_arr(state['player_1']['units']['position'][1])

            self.flip_arr(state['player_1']['energy_nodes'])
            self.flip_arr(state['player_1']['relic_nodes'])

            for rnc in state['player_1']['relic_node_configs']:
                self.flip_arr_2d(rnc)

            self.flip_arr_2d(state['player_1']['relic_nodes_map_weights'])
            self.flip_arr_2d(state['player_1']['map_features']['energy'])
            self.flip_arr_2d(state['player_1']['map_features']['tile_type'])

            #self.flip_arr_2d(state['player_1']['sensor_mask'])
            #self.flip_arr_2d(state['player_1']['vision_power_map'])

        info['state'] = state
        return info
