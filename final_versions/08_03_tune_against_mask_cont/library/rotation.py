import copy

import numpy as np

def rotate_secondary_diagonal(array, size = 24):
    # coordinate rotate
    if len(array) == 2:
        array[0], array[1] = size - array[1] - 1, size - array[0] - 1
    # matrix rotate
    else:
        array = np.flipud(np.fliplr(np.transpose(array)))
    return array



def no_rotate(array):
    return array
    pass


def rotate_main_diagonal(array):
    # coordinate rotate
    if len(array) == 2:
        array[0], array[1] = array[1], array[0]
    # matrix rotate
    else:
        array = np.transpose(array)
    return array


def rotate_two_diagonal(array, size = 24):
    array = rotate_main_diagonal(array)
    array = rotate_secondary_diagonal(array, size)
    return array


def rotate_obs(obs, rotate_by_diagonal):
    obs_rotated = copy.deepcopy(obs)

    # rotate units
    for unit_idx, units in enumerate(obs_rotated['units']['position']):
        for idx, unit_position in enumerate(units):
            if unit_position[0] != -1 and unit_position[1] != -1:
                obs_rotated['units']['position'][unit_idx, idx] = rotate_by_diagonal(unit_position)

    # rotate sensor mask
    obs_rotated['sensor_mask'] = rotate_by_diagonal(obs_rotated['sensor_mask'])

    # rotate matrix
    obs_rotated['map_features']['energy'] = rotate_by_diagonal(obs_rotated['map_features']['energy'])
    obs_rotated['map_features']['tile_type'] = rotate_by_diagonal(obs_rotated['map_features']['tile_type'])

    # rotate relic_nodes
    for node_idx, node in enumerate(obs_rotated['relic_nodes']):
        if node[0] != -1 and node[1] != -1:
            obs_rotated['relic_nodes'][node_idx] = rotate_by_diagonal(node)
    return obs_rotated


def compare_dict(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if isinstance(dict1[key], np.ndarray):
            if not np.array_equal(dict1[key], dict2[key]):
                return False
        elif isinstance(dict1[key], dict):
            compare_dict(dict1[key], dict2[key])
        else:
            if dict1[key] != dict2[key]:
                return False
    return True

def rotate_action_main_diagonal(action):
    actions = {
        0: 0, # 0 - do nothing
        1: 4, # 1 - up
        2: 3, # 2 - right
        3: 2, # 3 - down
        4: 1, # 4 - left
        5: 5 # 5 - sap
    }
    return actions[action]

def rotate_action_secondary_diagonal(action):
    actions = {
        0: 0, # 0 - do nothing
        1: 2, # 1 - up
        2: 1, # 2 - right
        3: 4, # 3 - down
        4: 3, # 4 - left
        5: 5 # 5 - sap
    }
    return actions[action]

def rotate_action_two_diagonal(action):
    return rotate_action_main_diagonal(rotate_action_secondary_diagonal(action))

def rotate_action(action, rotate_by_diagonal):
    return rotate_by_diagonal(action)
