import copy
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

class RewardSpec(NamedTuple):
    reward_min: float
    reward_max: float
    zero_sum: bool
    #only_once: bool
    scaler: float


class BaseRewardSpace(ABC):
    """
    A class used for defining a reward space and/or done state for either the full game or a sub-task
    """
    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def get_reward_spec() -> RewardSpec:
        pass

    @abstractmethod
    def compute_rewards_and_done(self, obs, env_reward, env_done, info):
        pass


class FullGameRewardSpace(BaseRewardSpace):
    """
    A class used for defining a reward space for the full game.
    """
    @abstractmethod
    def compute_rewards_and_done(self, obs, env_reward, env_done, info):
        pass


def all_rewards_and_metrics(obs, env_reward, env_done, info, mult, points_mult, game_win_mult):
    round = max(0, (obs['player_0']['steps'] - 1)) // 101
    assert(round >= 0 and round < 5)
    round = 0


    rewards_dict = {}


    points_reward_scaler = 16. / min(16., max(1., obs['player_0']['n_points_positions_on_map'] * 0.5))
    #print(obs['player_0']['n_points_positions_on_map'], points_reward_scaler)
    #assert False

    plus_points_reward = np.array([0., 0.])
    prev_team_points = obs['player_0']['prev_team_points']
    cur_team_points = obs['player_0']['team_points']
    plus_points_reward = cur_team_points - prev_team_points

    assert plus_points_reward[0] >= 0 and plus_points_reward[1] >= 0



    rewards_dict['plus_points_reward'] = (plus_points_reward, 0.0005 * points_reward_scaler * points_mult, round)



    minus_points_reward = np.array([0., 0.])

    minus_points_reward[0] = -plus_points_reward[1]
    minus_points_reward[1] = -plus_points_reward[0]

    rewards_dict['minus_points_reward'] = (minus_points_reward, 0.0005 * points_reward_scaler * points_mult, round)




    match_win_reward = np.array([0., 0.])
    prev_team_wins = obs['player_0']['prev_team_wins']
    cur_team_wins = obs['player_0']['team_wins']
    match_win_reward = cur_team_wins - prev_team_wins


    mwrr = -np.array([match_win_reward[1], match_win_reward[0]])

    # pos neg
    rewards_dict['match_win_reward'] = ((mwrr + match_win_reward), 1., round)


    game_win_reward = np.array([0., 0.])
    if env_done:
        cur_team_wins = obs['player_0']['team_wins']
        if cur_team_wins[0] > cur_team_wins[1]:
            game_win_reward[0] = 1
            game_win_reward[1] = -1
        else:
            game_win_reward[0] = -1
            game_win_reward[1] = 1
    rewards_dict['game_win_reward'] = (game_win_reward, 10. * game_win_mult, round)


    relic_found_reward = np.array([0., 0.])
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        prev_guarranted_relic = player_obs['prev_guarranted_relic']
        cur_guarranted_relic = player_obs['guarranted_relic']
        relic_found_reward[player_idx] = np.sum(np.logical_and(cur_guarranted_relic == True, prev_guarranted_relic == False))

    assert relic_found_reward[0] >= 0 and relic_found_reward[1] >= 0

    rewards_dict['relic_found_reward'] = (relic_found_reward, 0.08 * 2 * mult, round)


    potential_relic_found_reward = np.array([0., 0.])
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        prev_potential_relic = player_obs['prev_potential_relic_reduced'] == False
        cur_potential_relic = player_obs['potential_relic_reduced'] == True
        potential_relic_found_reward[player_idx] = np.sum(np.logical_and(cur_potential_relic, prev_potential_relic))
        if potential_relic_found_reward[player_idx] > 0:
            pass

    assert potential_relic_found_reward[0] >= 0 and potential_relic_found_reward[1] >= 0

    rewards_dict['potential_relic_found_reward'] = (potential_relic_found_reward, 0.008*2 * mult, round)


    not_relic_found_reward = np.array([0., 0.])
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        mask = player_obs['prev_potential_relic_reduced'] > 0
        prev_not_relic = player_obs['prev_not_relic_reduced'] == 0
        cur_not_relic = player_obs['not_relic_reduced'] == 1
        not_relic_found_reward[player_idx] = np.sum(np.logical_and.reduce([cur_not_relic, prev_not_relic, mask]))
        if not_relic_found_reward[player_idx] > 0:
            pass

    assert not_relic_found_reward[0] >= 0 and not_relic_found_reward[1] >= 0

    rewards_dict['not_relic_found_reward'] = (not_relic_found_reward, 0.004 * 10 * mult, round)



    unit_stacked_reward = np.array([0., 0.])

    for player_idx, (key, player_obs) in enumerate(obs.items()):
        positions = player_obs['units']['position']
        used = np.zeros((24, 24), dtype=int)
        for position in positions[player_idx]:
            if position[0] == -1:
                continue
            else:
                x = position[0]
                y = position[1]
                if used[x, y] == 1:
                    unit_stacked_reward[player_idx] += 1
                used[x, y] = 1

    rewards_dict['unit_stacked_reward'] = (unit_stacked_reward, 0 * mult* 0, round)


    # how much enemy got damage
    enemy_sap_recieve_energy_reward = np.array([0., 0.])
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        delta_energy = player_obs['delta_energy_recieve']
        enemy_sap_recieve_energy_reward[player_idx] = -np.sum(delta_energy[1 - player_idx])

    rewards_dict['enemy_sap_recieve_energy_reward'] = (enemy_sap_recieve_energy_reward, 0.001 * 0.5 * mult, round)


    # how much me got damage
    my_sap_recieve_energy_reward = np.array([0., 0.])
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        delta_energy = player_obs['delta_energy_recieve']
        my_sap_recieve_energy_reward[player_idx] = -np.sum(delta_energy[player_idx])

    rewards_dict['my_sap_recieve_energy_reward'] = (my_sap_recieve_energy_reward, -0.001 * 0.5 * mult * 0, round)


    # how much enemy wasted energy on shots
    enemy_sap_shoot_energy_reward = np.array([0., 0.])
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        delta_energy = player_obs['delta_energy_shoot']
        enemy_sap_shoot_energy_reward[player_idx] = -np.sum(delta_energy[1 - player_idx])

    rewards_dict['enemy_sap_shoot_energy_reward'] = (enemy_sap_shoot_energy_reward, 0. * mult* 0, round)


    # how much me wasted energy on shots
    my_sap_shoot_energy_reward = np.array([0., 0.])
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        delta_energy = player_obs['delta_energy_shoot']
        my_sap_shoot_energy_reward[player_idx] = -np.sum(delta_energy[player_idx])

    rewards_dict['my_sap_shoot_energy_reward'] = (my_sap_shoot_energy_reward, -0.001 * 0.25 * mult * 0, round)


    sap_efficiency_reward = np.array([0., 0.])

    sap_efficiency_reward[0] = enemy_sap_recieve_energy_reward[0] - my_sap_shoot_energy_reward[0]
    sap_efficiency_reward[1] = enemy_sap_recieve_energy_reward[1] - my_sap_shoot_energy_reward[1]

    rewards_dict['sap_efficiency_reward'] = (sap_efficiency_reward, 0 * mult* 0, round)




    enemy_dead_reward = np.array([0., 0.])
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        enemy_energy_dead = max(0, player_obs['energy_dead'])
        enemy_dead_reward[player_idx] = enemy_energy_dead

    rewards_dict['enemy_dead_reward'] = (enemy_dead_reward, 0.02 * mult, round)




    enemy_seen_reward = np.array([0., 0.])

    for player_idx, (key, player_obs) in enumerate(obs.items()):
        positions = player_obs['units']['position']
        k = 0
        for position in positions[1 - player_idx]:
            if position[0] == -1:
                continue
            else:
                k += 1
        enemy_seen_reward[player_idx] = k

    rewards_dict['enemy_seen_reward'] = (enemy_seen_reward, 0.001 * mult* 0, round)


    my_seen_reward = np.array([0., 0.]) # negative

    for player_idx, (key, player_obs) in enumerate(obs.items()):
        other_player_obs = obs['player_0'] if player_idx == 1 else obs['player_1']
        positions = other_player_obs['units']['position']
        k = 0
        for position in positions[player_idx]:
            if position[0] == -1:
                continue
            else:
                k += 1
        my_seen_reward[player_idx] = k

    rewards_dict['my_seen_reward'] = (my_seen_reward, -0.001 * mult* 0, round)



    my_dead_reward = np.array([0., 0.])
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        my_energy_dead = max(0, player_obs['my_energy_dead'])
        my_dead_reward[player_idx] = my_energy_dead

    rewards_dict['my_dead_reward'] = (my_dead_reward, -0.02 * mult, round)


    explore_reward = np.array([0., 0.]) # broken
    #for player_idx, (key, player_obs) in enumerate(obs.items()):
    #    prev_seen_local = player_obs['prev_seen_local']
    #    seen_local = player_obs['seen_local']
    #    mask = player_obs['seen_local'] > 0
    #    if np.sum(player_obs['remember_relic_nodes']) < 6:
    #        explore_reward[player_idx] = np.sum(seen_local[mask] - prev_seen_local[mask])

    #assert explore_reward[0] >= 0 and explore_reward[1] >= 0

    rewards_dict['explore_reward'] = (explore_reward, 1./576. * mult* 0, round)


    my_stuck_reward = np.array([0., 0.]) # not realy easy need to run dijxtra and check closest path to +energy position
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        positions = player_obs['units']['position']
        energies = player_obs['units']['energy']
        for idx, position in enumerate(positions[player_idx]):
            if position[0] == -1:
                continue
            energy = energies[player_idx][idx]
            if energy == 0:
                my_stuck_reward[player_idx] += 1

    rewards_dict['my_stuck_reward'] = (my_stuck_reward, -0.0002 * mult, round)


    sensor_mask_reward = np.array([0., 0.])
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        sensor_mask_reward[player_idx] = np.sum(player_obs['sensor_mask'])

    rewards_dict['sensor_mask_reward'] = (sensor_mask_reward, 0.000025 * mult * 0, round)

    near_units_reward = np.array([0., 0.])
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        positions = player_obs['units']['position']
        energies = player_obs['units']['energy']
        for idx, position_1 in enumerate(positions[player_idx]):
            if position_1[0] == -1:
                continue
            energy_1 = energies[player_idx][idx]
            if energy_1 <= 0:
                continue
            if player_obs['guarranted_relic'][position_1[0], position_1[1]] == True:
                continue
            for jdx, position_2 in enumerate(positions[player_idx]):
                if jdx <= idx:
                    continue
                if position_2[0] == -1:
                    continue
                energy_2 = energies[player_idx][jdx]
                if energy_2 <= 0:
                    continue
                if player_obs['guarranted_relic'][position_2[0], position_2[1]] == True:
                    continue
                if np.abs(position_1[0] - position_2[0]) + np.abs(position_1[1] - position_2[1]) <= 2:
                    near_units_reward[player_idx] += 1

    rewards_dict['near_units_reward'] = (near_units_reward, -0.001 * mult * 0, round)

    my_sum_energy_reward = np.array([0., 0.])
    for player_idx, (key, player_obs) in enumerate(obs.items()):
        positions = player_obs['units']['position']
        energies = player_obs['units']['energy']
        prev_positions = player_obs['prev_units']['position']
        prev_energies = player_obs['prev_units']['energy']
        for idx, position in enumerate(positions[player_idx]):
            if position[0] == -1 or prev_positions[player_idx][idx][0] == -1:
                continue
            if (position[0] == 23 and position[1] == 23) or (position[0] == 0 and position[1] == 0):
                continue
            energy = energies[player_idx][idx]
            prev_energy = prev_energies[player_idx][idx]
            my_sum_energy_reward[player_idx] += energy - prev_energy

    rewards_dict['my_sum_energy_reward'] = (my_sum_energy_reward, 0.0002 * mult* 0, round)


    enemy_sum_energy_reward = np.array([0., 0.])
    enemy_sum_energy_reward[0] = my_sum_energy_reward[1]
    enemy_sum_energy_reward[1] = my_sum_energy_reward[0]
    rewards_dict['enemy_sum_energy_reward'] = (enemy_sum_energy_reward, -0.0002 * mult* 0, round)

    return rewards_dict, env_done



class GameResultReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-7.5,
            reward_max=7.5,
            zero_sum=False,
            #only_once=False,
            scaler = 10
        )

    def __init__(self):
        super(GameResultReward, self).__init__()

    def compute_rewards_and_done(self, obs, env_reward, env_done, info):
        mult = 0.
        points_mult = 1.
        game_win_mult = 0.
        return all_rewards_and_metrics(obs, env_reward, env_done, info, mult, points_mult, game_win_mult)

class GameWinReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-13,
            reward_max=13,
            zero_sum=False,
            #only_once=False,
            scaler = 2.
        )

    def __init__(self):
        super(GameWinReward, self).__init__()

    def compute_rewards_and_done(self, obs, env_reward, env_done, info):
        mult = 0.
        points_mult = 0.
        game_win_mult = 1.
        return all_rewards_and_metrics(obs, env_reward, env_done, info, mult, points_mult, game_win_mult)




class StatefulMultiReward(FullGameRewardSpace):
    @staticmethod
    def get_reward_spec() -> RewardSpec:
        return RewardSpec(
            reward_min=-16.,
            reward_max=32.,
            zero_sum=False,
            #only_once=False,
            scaler = 1.
        )

    def __init__(self):
        super(StatefulMultiReward, self).__init__()


    def compute_rewards_and_done(self, obs, env_reward, env_done, info):
        mult = 1.
        points_mult = 1.
        game_win_mult = 0.
        return all_rewards_and_metrics(obs, env_reward, env_done, info, mult, points_mult, game_win_mult)

