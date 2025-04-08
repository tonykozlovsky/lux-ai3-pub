import pandas as pd
import os
import requests
import json
import time


def request_episodes_list(submission_id = 42596204):
    url = 'https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes'
    params = {
        'ids': [],
        'submissionId': submission_id
    }

    headers = { 'Content-Type': 'application/json' }
    cur_attemp = 0
    max_attemp = 5
    while cur_attemp <= max_attemp:
        try:
            cur_attemp += 1
            response = requests.post(url, json=params, headers=headers, timeout=60)
            if response.status_code != 200:
                print('Failed request for episode list:', response.status_code, response.text)
                time.sleep(1)
            else:
                return response.json()
        except requests.exceptions.Timeout:
            print('Timeout for request episodes list, attempt {}/{}'.format(cur_attemp, max_attemp))
        except Exception as e:
            print('Request for episodes list failed: {}, attempt {}/{}'.format(e, cur_attemp, max_attemp))
    return {}


def get_teams(player_name = 'my'):
    file_path = 'outputs/teams_{}.csv'.format(player_name)
    df = pd.DataFrame(columns=['team_name', 'submission_id', 'submission_date', 'team_id'])

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    return df


def update_teams(data, player_name = 'my'):
    if 'teams' not in data:
        print('no data about teams, skip step update_teams')
        return

    file_path = 'outputs/teams_{}.csv'.format(player_name)
    df = get_teams(player_name)

    for team in data['teams']:
        if not df[(df['team_id'] == team['id']) & (df['submission_id'] == team['publicLeaderboardSubmissionId'])].empty:
            continue
        new_row = {'team_name': team['teamName'], 'team_id': team['id'], 'submission_id': team['publicLeaderboardSubmissionId'], 'submission_date': team['lastSubmissionDate']}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    for submission in data['submissions']:
        if not df[(df['submission_id'] == submission['id'])].empty:
            continue
        team = df.loc[df['team_id'] == submission['teamId']]
        team_name = team['team_name'].iloc[0] if not team['team_name'].empty else ''
        new_row = {'team_name': team_name, 'team_id': submission['teamId'], 'submission_id': submission['id'], 'submission_date': submission['dateSubmitted']}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values(by='team_name')
    df = df.drop_duplicates()
    df.to_csv(file_path, index=False)


def get_episodes_list(player_name = 'my'):
    file_path = 'outputs/episodes_list_{}.csv'.format(player_name)
    columns = ['episode_id', 'enemy_submission_id', 'enemy_name', 'enemy_reward', 'main_reward', 'main_win', 'main_submission_id', 'main_name', 'main_player_id', 'enemy_player_id']
    df = pd.DataFrame(columns=columns)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    return df


def update_episodes_list(data, main_player_submission_id = 42596204, player_name = 'my'):
    rows_changed = 0
    if 'episodes' not in data:
        print('no data about episodes, skip step update_episodes_list')
        return []

    file_path = 'outputs/episodes_list_{}.csv'.format(player_name)
    teams = get_teams(player_name)
    new_episodes = []


    main_player_data = teams.loc[teams['submission_id'] == main_player_submission_id]
    main_team_name = main_player_data['team_name'].iloc[0] if not main_player_data['team_name'].empty else ''

    df = get_episodes_list(player_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

    ['enemy_submission_id', 'enemy_name', 'enemy_reward', 'enemy_player_id']
    for episode in data['episodes']:
        if not df[(df['episode_id'] == episode['id']) & (df['main_submission_id'] == main_player_submission_id)].empty:
            continue

        main_player_id = 0 if episode['agents'][0]['submissionId'] == main_player_submission_id else 1
        new_episodes.append(episode['id'])

        new_row = {'episode_id': episode['id'], 'main_submission_id': main_player_submission_id, 'main_name': main_team_name, 'main_player_id': main_player_id}
        main_reward = episode['agents'][main_player_id].get('reward', 0)
        main_win = 1 if main_reward >= 3 else 0
        new_row['main_reward'] = main_reward
        new_row['main_win'] = main_win

        enemy_player_id = 0 if main_player_id == 1 else 1
        enemy_reward = episode['agents'][enemy_player_id].get('reward', 0)
        enemy_submission_id = episode['agents'][enemy_player_id]['submissionId']
        enemy_data = teams.loc[teams['submission_id'] == enemy_submission_id]
        enemy_team_name = enemy_data['team_name'].iloc[0] if not enemy_data['team_name'].empty else ''

        new_row['enemy_player_id'] = enemy_player_id
        new_row['enemy_reward'] = enemy_reward
        new_row['enemy_submission_id'] = enemy_submission_id
        new_row['enemy_name'] = enemy_team_name
        rows_changed += 1

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df = df.drop_duplicates()
    df = df.sort_values(by='episode_id')
    df.to_csv(file_path, index=False)
    print('update_episodes_list rows changed {}, submission id {}'.format(rows_changed, main_player_submission_id))
    return new_episodes


def request_replay(episode_id, player_name, save = False):
    url = 'https://www.kaggle.com/api/i/competitions.EpisodeService/GetEpisodeReplay'
    params = {
        'episodeId': episode_id
    }

    headers = { 'Content-Type': 'application/json' }
    cur_attemp = 0
    max_attemp = 5
    while cur_attemp <= max_attemp:
        try:
            cur_attemp += 1
            response = requests.post(url, json=params, headers=headers, timeout=60)

            if response.status_code != 200:
                print("Failed request for replay:", response.status_code, response.text, episode_id)
                time.sleep(1)
            else:
                if save:
                    directory_path = 'outputs/{}'.format(player_name)
                    if not os.path.exists(directory_path):
                        os.makedirs(directory_path)
                        print('Create dir: {}'.format(directory_path))

                    with open('{}/replay_{}.json'.format(directory_path, episode_id), 'w') as f:
                        f.write(response.text)
                return response.json()

        except requests.exceptions.Timeout:
            print('Timeout for request_episodes_list, attempt {}/{}'.format(cur_attemp, max_attemp))
        except Exception as e:
            print('Request for replay failed: {}, attempt {}/{}'.format(e, cur_attemp, max_attemp))
    return {}


def get_episodes_extended(player_name = 'my'):
    file_path = 'outputs/episodes_extended_{}.csv'.format(player_name)
    columns = ['episode_id', 'enemy_submission_id', 'enemy_name', 'enemy_reward',
               'main_reward', 'main_win', 'main_submission_id', 'main_name',
               'main_player_id', 'enemy_player_id',
               'energy_node_drift_magnitude', 'energy_node_drift_speed',
               'nebula_tile_drift_speed', 'nebula_tile_energy_reduction',
               'nebula_tile_vision_reduction', 'unit_energy_void_factor',
               'unit_move_cost', 'unit_sap_cost', 'unit_sap_dropoff_factor', 'unit_sap_range', 'unit_sensor_range']
    df = pd.DataFrame(columns=columns)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    return df


def update_episodes_extended(new_episodes, player_name = 'my', cookie = '', save = False):
    rows_changed = 0
    skipped_episodes = 0
    skipped_episodes_count = []
    file_path = 'outputs/episodes_extended_{}.csv'.format(player_name)
    episodes_list = get_episodes_list(player_name)
    df = get_episodes_extended(player_name)
    extended_data_names = ['energy_node_drift_magnitude', 'energy_node_drift_speed',
               'nebula_tile_drift_speed', 'nebula_tile_energy_reduction',
               'nebula_tile_vision_reduction', 'unit_energy_void_factor',
               'unit_move_cost', 'unit_sap_cost', 'unit_sap_dropoff_factor', 'unit_sap_range', 'unit_sensor_range']
    
    episodes_for_retry = []
    skipped_path = 'outputs/skipped_episodes.txt'
    if os.path.exists(skipped_path):
        with open(skipped_path, 'r') as f:
            episodes_for_retry = json.load(f)
    if isinstance(episodes_for_retry, list) and len(episodes_for_retry) > 0:
        new_episodes = list(set(new_episodes + episodes_for_retry))

    print('episodes for retry: ', episodes_for_retry)
    print('total episodes count: ', len(new_episodes))

    if len(new_episodes) == 0:
        print('no new eprisodes, skip step update_episodes_extended')
        return

    for episode_id in new_episodes:
        if not df[(df['episode_id'] == episode_id)].empty:
            print('eprisode already exists, skip it ', episode_id)
            skipped_episodes_count +=1
            continue

        agent_log = []
        if len(cookie) > 0:
            player_id = episodes_list.loc[episodes_list['episode_id'] == episode_id, 'main_player_id'].iloc[0]
            agent_log = request_agent_log(episode_id, player_id, cookie)

        replay = request_replay(episode_id, player_name, save)
        if 'steps' not in replay:
            print('No replay for episode', episode_id)
            continue
        params = replay['steps'][0][0]['info']['replay']['params']
        is_strong = int(0)
        summary_duration = 0
        overtime = 0
        steps = 0

        if len(cookie) > 0:
            if isinstance(agent_log, list) and len(agent_log) > 0 and len(agent_log[0]) > 0:
                text = agent_log[0][0].get('stderr', '')
                if 'HAS_MODEL' in text:
                    is_strong = 1

                steps = len(agent_log)
                for step in agent_log:
                    if len(step) > 0:
                        duration = step[0].get('duration', 0)
                        summary_duration += duration
                        if duration > 3.0:
                            overtime += duration - 3.0
            else:
                print('Skip episode {}. Update cookie and request skipped_episodes'.format(episode_id))
                skipped_episodes.append(episode_id)
                skipped_episodes_count +=1
                continue

        extended_data = { key:  params.get(key, None) for key in extended_data_names }
        extended_data['is_strong'] = is_strong
        extended_data['duration'] = summary_duration
        extended_data['overtime'] = overtime
        extended_data['steps'] = steps
        base_episodes = episodes_list[(episodes_list['episode_id'] == episode_id)]

        for index, episode in base_episodes.iterrows():
            rows_changed += 1
            new_row = extended_data.copy()
            new_row = new_row | episode.to_dict()
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).dropna(how="all")

    df = df.sort_values(by='episode_id')
    df = df.drop_duplicates()
    df.to_csv(file_path, index=False)
    print('update_episodes_extended, rows changed {}'.format(rows_changed))
    print('update_episodes_extended, skipped episodes', skipped_episodes_count)

    with open(skipped_path, 'w') as f:
        json.dump(skipped_episodes, f)

def request_agent_log(episode_id, player_id, cookie, save = False):
    url = 'https://www.kaggle.com/competitions/episodes/{}/agents/{}/logs.json'.format(episode_id, player_id)

    headers = { 'Content-Type': 'application/json', 'Cookie': cookie}

    cur_attemp = 0
    max_attemp = 5
    while cur_attemp <= max_attemp:
        try:
            cur_attemp += 1
            response = requests.get(url, headers=headers, timeout=60)

            if response.status_code != 200:
                print("Failed request for agent log:", response.status_code, response.text, episode_id)
                time.sleep(1)
            else:
                if save:
                    with open('agent_log_{}.json'.format(episode_id), 'w') as f:
                        f.write(response.text)
                return response.json()

        except requests.exceptions.Timeout:
            print('Timeout for agent log, attempt {}/{}'.format(cur_attemp, max_attemp))
        except Exception as e:
            print('Request for agent log failed: {}, attempt {}/{}'.format(e, cur_attemp, max_attemp))
    return {}
