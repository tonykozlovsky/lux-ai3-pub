from collections import defaultdict
import ctypes
import os
import sys
from abc import ABC, abstractmethod
from functools import lru_cache

import gym
import numpy as np

MAX_OVERLAPPING_ACTIONS = 1

from ..utility_constants import MAX_UNITS, MODEL_PATH

DIRECTIONS = ['0', '1', '2', '3', '4']


class BaseActSpace(ABC):
    @abstractmethod
    def get_action_space(self):
        pass

    @abstractmethod
    def process_actions(self, actions, obs, sap_range, sap_cost, move_cost):
        pass

    @abstractmethod
    def get_available_actions_mask(self, obs, sap_range, sap_cost, move_cost, enable_sap_masks):
        pass

    @staticmethod
    @abstractmethod
    def actions_taken_to_distributions(self, actions_taken):
        pass


class BasicActionSpace(BaseActSpace):
    def get_action_meanings(self):
        action_meanings = {
            "worker": [],
            "sapper": []
        }

        for d in DIRECTIONS:
            action_meanings["worker"].append(f"MOVE_{d}")

        action_meanings["worker"].append(f"SAP")

        for i in range(-7, 8):
            for j in range(-7, 8):
                action_meanings["sapper"].append(f"SAP_{i}_{j}")

        return action_meanings


    def __init__(self):

        self.ACTION_MEANINGS = self.get_action_meanings()

        self.ACTION_MEANINGS_TO_IDX = {
            actor: {
                action: idx for idx, action in enumerate(actions)
            } for actor, actions in self.ACTION_MEANINGS.items()
        }

        if not MODEL_PATH:
            mylib_dir = os.getcwd()
        else:
            mylib_dir = os.path.join(os.path.dirname(__file__), "..", "..", MODEL_PATH)

        self.mylib = ctypes.cdll.LoadLibrary(os.path.join(mylib_dir, 'mylib.so'))

        self.mylib.CreateMyClass.restype = ctypes.c_void_p

        self.mylib.DeleteMyClass.argtypes = [ctypes.c_void_p]

        self.mylib.Reset.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]

        self.mylib.AddUnit_0.argtypes = [ctypes.c_void_p] + [ctypes.c_int32] * 5
        self.mylib.AddUnit_1.argtypes = [ctypes.c_void_p] + [ctypes.c_int32] * 5
        self.mylib.AddUnit_2.argtypes = [ctypes.c_void_p] + [ctypes.c_int32] * 5
        self.mylib.AddUnit_3.argtypes = [ctypes.c_void_p] + [ctypes.c_int32] * 5

        self.mylib.AddRelic.argtypes = [ctypes.c_void_p] + [ctypes.c_int32] * 3

        self.mylib.CalcAAM.argtypes = [ctypes.c_void_p]
        self.mylib.FillAAMForLight.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        self.my_class = self.mylib.CreateMyClass()



    def __del__(self):
        self.mylib.DeleteMyClass(self.my_class)

    @lru_cache(maxsize=None)
    def get_action_space(self):

        # Player count
        p = 2
        return gym.spaces.Dict({
            "worker": gym.spaces.MultiDiscrete(np.zeros((1, p, MAX_UNITS), dtype=int) + len(self.ACTION_MEANINGS["worker"])),
            "sapper": gym.spaces.MultiDiscrete(np.zeros((1, p, MAX_UNITS), dtype=int) + len(self.ACTION_MEANINGS["sapper"]))
        })

    @lru_cache(maxsize=None)
    def get_action_space_expanded_shape(self, *args, **kwargs):
        action_space = self.get_action_space()
        action_space_expanded = {}
        for key, val in action_space.spaces.items():
            action_space_expanded[key] = val.shape + (len(self.ACTION_MEANINGS[key]),)
        return action_space_expanded

    def flip_action(self, action, player_idx, flip_axes):
        directions = np.array(
            [
                [0, 0],  # Do nothing
                [0, -1],  # Move up
                [1, 0],  # Move right
                [0, 1],  # Move down
                [-1, 0],  # Move left
            ],
            dtype=int,
        )

        if player_idx == 0:
            return action

        if action == 0 or action == 5:
            return action

        if action <= 4:
            prev_x = 0
            prev_y = 0
            next_x = directions[action][0]
            next_y = directions[action][1]

            prev_x_f = 23 - prev_y
            prev_y_f = 23 - prev_x

            next_x_f = 23 - next_y
            next_y_f = 23 - next_x

            delta = np.array([next_x_f - prev_x_f, next_y_f - prev_y_f])

            if flip_axes:
                delta = np.array([delta[1], delta[0]])

            for i in range(5):
                if np.all(delta == directions[i]):
                    return i

        assert False

    def flip_sap(self, sap_x, sap_y, player_idx, flip_axes):
        if player_idx == 0:
            return sap_x, sap_y
        prev_x = 0
        prev_y = 0
        next_x = sap_x
        next_y = sap_y

        prev_x_f = 23 - prev_y
        prev_y_f = 23 - prev_x
        next_x_f = 23 - next_y
        next_y_f = 23 - next_x

        delta = np.array([next_x_f - prev_x_f, next_y_f - prev_y_f])

        if flip_axes:
            return delta[1], delta[0]
        else:
            return delta[0], delta[1]


    def process_actions(self, actions, obs, sap_range, sap_cost, move_cost, flags, kaggle_player_id):
        board_dims = (24, 24)

        actions_taken = {
            key: np.zeros(space, dtype=bool) for key, space in self.get_action_space_expanded_shape(board_dims).items()
        }

        actions_worker = actions['worker']
        actions_sapper = actions['sapper']

        lux_action = dict()

        cells_received_sap = [defaultdict(lambda: [0, 0]) for _ in range(2)]
        previous_sap_units = [[] for _ in range(2)]

        one_player = actions_worker.shape[1] != 2

        if one_player:
            assert kaggle_player_id is not None

        for player_idx, (key, val) in enumerate(obs.items()):
            assert(key == f'player_{player_idx}')
            lux_action[key] = []

            if one_player and player_idx != kaggle_player_id:
                lux_action[key] = np.zeros((16, 3), dtype=np.int16)
                continue

            actions_idx = 0 if one_player else player_idx

            positions = val['units']['position'][player_idx]
            energies = val['units']['energy'][player_idx]
            for pos_idx, (position, energy) in enumerate(zip(positions, energies)):
                if position[0] == -1:
                    lux_action[key].append([0, 0, 0])
                else:
                    action_worker = actions_worker[0, actions_idx, pos_idx, 0]

                    is_sap = action_worker == 5
                    if is_sap:
                        action_sapper = actions_sapper[0, actions_idx, pos_idx, 0]

                        sap_x = action_sapper // 15 - 7
                        sap_y = action_sapper % 15 - 7

                        try:
                            assert sap_x >= -sap_range and sap_x <= sap_range
                            assert sap_y >= -sap_range and sap_y <= sap_range
                            assert position[0] + sap_x >= 0 and position[0] + sap_x < 24
                            assert position[1] + sap_y >= 0 and position[1] + sap_y < 24
                            assert energy >= sap_cost
                        except Exception as e:
                            print("SAPS: ", sap_x, sap_y, sap_range, "player_idx: ", player_idx, obs['player_0']['steps'], energy, sap_cost, "pos:", position, "pos_idx:", pos_idx)
                            raise e


                        flipped_sap_x, flipped_sap_y = self.flip_sap(sap_x, sap_y, player_idx, flags.flip_axes) if flags.flip_learning else (sap_x, sap_y)
                        lux_action[key].append([5, flipped_sap_x, flipped_sap_y])

                        actions_taken['sapper'][0, player_idx, pos_idx, action_sapper] = True
                        cells_received_sap[player_idx][(position[0] + sap_x, position[1] + sap_y)][0] += sap_cost
                        cells_received_sap[player_idx][(position[0] + sap_x, position[1] + sap_y)][1] += 1
                        previous_sap_units[player_idx].append(pos_idx)
                    else:
                        flipped_action_worker = self.flip_action(action_worker, player_idx, flags.flip_axes) if flags.flip_learning else action_worker
                        if flipped_action_worker != 0 and flipped_action_worker != 5:
                            assert energy >= move_cost
                        lux_action[key].append([flipped_action_worker, 0, 0])

                    actions_taken['worker'][0, player_idx, pos_idx, action_worker] = True

            lux_action[key] = np.array(lux_action[key]).astype(np.int16)

        return lux_action, actions_taken, cells_received_sap, previous_sap_units

    def fill_gt(self, available_actions_mask, obs, sap_range, sap_cost, move_cost):
        available_actions_mask['ground_truth_CPU'] = np.zeros((2, 3, 24, 24), dtype=bool)

        for player_idx, (key, player_obs) in enumerate(obs.items()):
            positions = player_obs['units']['position']
            energies = player_obs['units']['energy']

            for pos_idx, position in enumerate(positions[player_idx]):
                energy = energies[player_idx][pos_idx]
                if position[0] == -1 or energy < 0:
                    continue
                available_actions_mask['ground_truth_CPU'][1 - player_idx][0][position[0]][position[1]] = True

                for i in range(-1, 2):
                    for j in range(-1, 2):
                        x = position[0] + i
                        y = position[1] + j
                        if x < 0 or x >= 24 or y < 0 or y >= 24:
                            continue
                        available_actions_mask['ground_truth_CPU'][1 - player_idx][1][x][y] = True

            available_actions_mask['ground_truth_CPU'][1 - player_idx][2] = player_obs['sensor_mask']




    def get_available_actions_mask(self, obs, sap_range, sap_cost, move_cost, enable_sap_masks):
        assert sap_range != None
        assert sap_cost != None
        assert move_cost != None

        available_actions_mask = {
            key: np.zeros(space.shape + (len(self.ACTION_MEANINGS[key]),), dtype=bool)
            for key, space in self.get_action_space().spaces.items()
        }

        available_actions_mask['sapper_with_mask'] = np.zeros_like(available_actions_mask['sapper'])
        available_actions_mask['worker_with_mask'] = np.zeros_like(available_actions_mask['worker'])
        available_actions_mask['sapper_without_mask'] = np.zeros_like(available_actions_mask['sapper'])
        available_actions_mask['worker_without_mask'] = np.zeros_like(available_actions_mask['worker'])

        available_actions_mask.pop('sapper')
        available_actions_mask.pop('worker')

        self.fill_gt(available_actions_mask, obs, sap_range, sap_cost, move_cost)



        directions = np.array(
            [
                [0, 0],  # Do nothing
                [0, -1],  # Move up
                [1, 0],  # Move right
                [0, 1],  # Move down
                [-1, 0],  # Move left
            ],
            dtype=int,
        )


        mylib = self.mylib
        my_class = self.my_class

        for name, esm in [("_without_mask", 0), ("_with_mask", 1)]:
            mylib.Reset(my_class, ctypes.c_int32(sap_range), ctypes.c_int32(sap_cost), ctypes.c_int32(esm))

            for player_idx, (key, player_obs) in enumerate(obs.items()):
                assert(key == f'player_{player_idx}')
                positions = player_obs['units']['position']
                energies = player_obs['units']['energy']

                for pos_idx, position in enumerate(positions[player_idx]):
                    if position[0] == -1:
                        if player_idx == 0:
                            mylib.AddUnit_0(my_class, ctypes.c_int32(pos_idx), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(1))
                        else:
                            mylib.AddUnit_1(my_class, ctypes.c_int32(pos_idx), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(1))
                        continue
                    else:
                        energy = energies[player_idx][pos_idx]
                        if player_idx == 0:
                            mylib.AddUnit_0(my_class, ctypes.c_int32(pos_idx), ctypes.c_int32(position[0]), ctypes.c_int32(position[1]), ctypes.c_int32(energy), ctypes.c_int32(0))
                        else:
                            mylib.AddUnit_1(my_class, ctypes.c_int32(pos_idx), ctypes.c_int32(position[0]), ctypes.c_int32(position[1]), ctypes.c_int32(energy), ctypes.c_int32(0))

                for pos_idx, position in enumerate(positions[1 - player_idx]):
                    if position[0] == -1:
                        if player_idx == 0:
                            mylib.AddUnit_2(my_class, ctypes.c_int32(pos_idx), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(1))
                        else:
                            mylib.AddUnit_3(my_class, ctypes.c_int32(pos_idx), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(1))
                        continue
                    else:
                        energy = energies[1 - player_idx][pos_idx]
                        if player_idx == 0:
                            mylib.AddUnit_2(my_class, ctypes.c_int32(pos_idx), ctypes.c_int32(position[0]), ctypes.c_int32(position[1]), ctypes.c_int32(energy), ctypes.c_int32(0))
                        else:
                            mylib.AddUnit_3(my_class, ctypes.c_int32(pos_idx), ctypes.c_int32(position[0]), ctypes.c_int32(position[1]), ctypes.c_int32(energy), ctypes.c_int32(0))

                indices = np.argwhere(player_obs['guarranted_relic'] == 1)  # Get indexes where the condition is met
                for index in indices:
                    x, y = index
                    if not player_obs['sensor_mask'][x, y]:
                        mylib.AddRelic(my_class, ctypes.c_int32(player_idx), ctypes.c_int32(x), ctypes.c_int32(y))


            mylib.CalcAAM(my_class)
            mylib.FillAAMForLight(my_class, available_actions_mask[f"sapper{name}"].ctypes.data_as(ctypes.POINTER(ctypes.c_bool)))

            for player_idx, (key, player_obs) in enumerate(obs.items()):
                assert(key == f'player_{player_idx}')
                positions = player_obs['units']['position']
                energies = player_obs['units']['energy']
                tile_type = player_obs['remember_type']

                for pos_idx, position in enumerate(positions[player_idx]):
                    if position[0] == -1:
                        continue
                    else:
                        energy = energies[player_idx][pos_idx]
                        for action_worker in range(6):
                            if action_worker == 0:
                                is_blocked = False
                            elif action_worker == 5:
                                is_blocked = energy < sap_cost or np.sum(available_actions_mask[f"sapper{name}"][:, player_idx, pos_idx, :]) == 0
                            else:
                                new_pos = position + directions[action_worker]
                                is_blocked = new_pos[0] < 0 or new_pos[0] >= 24 or new_pos[1] < 0 or new_pos[1] >= 24 or tile_type[new_pos[0], new_pos[1]] == 2 or energy < move_cost

                            available_actions_mask[f"worker{name}"][
                            :,
                            player_idx,
                            pos_idx,
                            action_worker
                            ] = not is_blocked

        return available_actions_mask


    def actions_taken_to_distributions(self, actions_taken):
        out = {}
        for space, actions in actions_taken.items():
            out[space] = {
                self.ACTION_MEANINGS[space][i]: actions[..., i].sum()
                for i in range(actions.shape[-1])
            }
        return out
