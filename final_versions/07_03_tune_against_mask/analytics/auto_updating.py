import datetime
import json
import schedule
import time
from src import *

def updating():
    config = []
    with open('config2.json', 'r', encoding='utf-8') as file:
        config = json.load(file)

    for params in config:
        submission_id = params['submission_id']
        player_name = params['player_name']
        cookie = params.get('cookie', '')
        print('Request episodes for submission {}'.format(submission_id))
        episodes = request_episodes_list(submission_id)
        print('Update teams')
        update_teams(episodes, player_name)
        print('Update episodes list')
        new_episodes = update_episodes_list(episodes, submission_id, player_name)
        print('Update extended episodes list, new episode count: {}'.format(len(new_episodes)))
        save = False
        if player_name == 'frog':
            save = True
        update_episodes_extended(new_episodes, player_name, cookie, save)
    print('your data was updated. Update time: ', datetime.datetime.now())

def run_schedule():
    updating()
    schedule.every(2).minutes.do(updating)
    while True:
        schedule.run_pending()
        print('timer waiting')
        time.sleep(30)

if __name__ == "__main__":
    try:
        print('Program started')
        run_schedule()
    except Exception as e:
        print('Update failed: ', e)
