import torch
from types import SimpleNamespace
from typing import Optional
import copy
import multiprocessing as mp
import random
import os
import json
import time
import setproctitle
import secrets
import queue
import logging

from .buffer_utils import buffers_apply, stack_buffers, get_cpu_buffers, get_gpu_buffers, fill_buffers_inplace, fill_buffers_inplace_2
from .common import get_input_buffers_for_inference, get_output_buffers_for_inference
from ..profiler import ScopedProfiler
from ...lux_gym import create_env
from .model_inference import ModelInferenceProcess
from omegaconf import OmegaConf
from pathlib import Path
from .load_model import loadd_model
from ...nns import create_model
from ...utils import flags_to_namespace
from .rotate import rotate_action_two_diagonal, rotate_two_diagonal
from .hash import out_env_hashes
import tempfile



@torch.no_grad()
def compete_func(
        flags: SimpleNamespace,
        baseline_flags: SimpleNamespace,
        competitor_flags: SimpleNamespace,
        teacher_flags: Optional[SimpleNamespace],
        actor_index: int,

        baseline_model: torch.nn.Module,
        competitor_model: torch.nn.Module,

        competition_queue: mp.SimpleQueue,
        finished_queue: mp.SimpleQueue,

        shared_finish,
        shared_n_games,
        shared_first_player_wins,
        device,
):
    setproctitle.setproctitle(f"COMPETE_{actor_index}_PROCESS")

    torch.cuda.set_device(f'cuda:{device}')
    torch.set_default_device(f'cuda:{device}')

    try:
        logging.info(f"Compete {actor_index} started.")
        profiler = ScopedProfiler()

        #random.seed(actor_index)

        env = create_env(actor_index, flags, device=device, teacher_flags=teacher_flags, profiler=profiler, example=True)

        while True:
            competition_queue.get()

            #print("START PLAYING")

            #competitor_model.load_state_dict(learner_model.state_dict())

            env_output = env.reset(force=True)

            #print(f"Compete {actor_index} env reset.")

            is_sample = False

            lstm_size = (128, 24, 24) if (baseline_flags.enable_lstm or competitor_flags.enable_lstm) else (1, 1, 1)

            prev_hidden_state_h_baseline = torch.zeros(flags.n_actor_envs, 2, *lstm_size, device=device, dtype=torch.float32)
            prev_hidden_state_c_baseline = torch.zeros(flags.n_actor_envs, 2, *lstm_size, device=device, dtype=torch.float32)
            prev_prediction_baseline = torch.zeros(flags.n_actor_envs, 2, 3, 24, 24, device=device, dtype=torch.float32)

            prev_hidden_state_h_competitor = torch.zeros(flags.n_actor_envs, 2, *lstm_size, device=device, dtype=torch.float32)
            prev_hidden_state_c_competitor = torch.zeros(flags.n_actor_envs, 2, *lstm_size, device=device, dtype=torch.float32)
            prev_prediction_competitor = torch.zeros(flags.n_actor_envs, 2, 3, 24, 24, device=device, dtype=torch.float32)

            def merge(a, b):
                return torch.cat([a[:, 0, :, :, :].unsqueeze(1), b[:, 1, :, :, :].unsqueeze(1)], dim=1)

            def merge2(a, b):
                return {'worker': torch.cat([a['worker'][:, :, 0, :, :].unsqueeze(2), b['worker'][:, :, 1, :, :].unsqueeze(2)], dim=2),
                        'sapper': torch.cat([a['sapper'][:, :, 0, :, :].unsqueeze(2), b['sapper'][:, :, 1, :, :].unsqueeze(2)], dim=2)}

            env_output['info']['rnn_hidden_state_h_GPU'] = merge(prev_hidden_state_h_baseline, prev_hidden_state_h_competitor)
            env_output['info']['rnn_hidden_state_c_GPU'] = merge(prev_hidden_state_c_baseline, prev_hidden_state_c_competitor)
            env_output['info']['prediction_GPU_CPU'] = merge(prev_prediction_baseline, prev_prediction_competitor)

            baseline_output = baseline_model(env_output, sample=is_sample, profiler=profiler)
            competitor_output = competitor_model(env_output, sample=is_sample, profiler=profiler)

            prev_hidden_state_h_baseline = baseline_output['rnn_hidden_state_h_GPU']
            prev_hidden_state_c_baseline = baseline_output['rnn_hidden_state_c_GPU']
            prev_prediction_baseline = baseline_output['prediction_GPU_CPU']

            prev_hidden_state_h_competitor = competitor_output['rnn_hidden_state_h_GPU']
            prev_hidden_state_c_competitor = competitor_output['rnn_hidden_state_c_GPU']
            prev_prediction_competitor = competitor_output['prediction_GPU_CPU']

            n_games_played = 0
            first_player_wins = 0

            while True: #n_games_played < flags.compete_games:
                if shared_finish.value == 1:
                    finished_queue.put(42)
                    break
                baseline_output = baseline_model(env_output, sample=is_sample, profiler=profiler)
                competitor_output = competitor_model(env_output, sample=is_sample, profiler=profiler)

                prev_hidden_state_h_baseline = baseline_output['rnn_hidden_state_h_GPU']
                prev_hidden_state_c_baseline = baseline_output['rnn_hidden_state_c_GPU']
                prev_prediction_baseline = baseline_output['prediction_GPU_CPU']

                prev_hidden_state_h_competitor = competitor_output['rnn_hidden_state_h_GPU']
                prev_hidden_state_c_competitor = competitor_output['rnn_hidden_state_c_GPU']
                prev_prediction_competitor = competitor_output['prediction_GPU_CPU']

                env_output = env.step(merge2(baseline_output['actions_GPU'], competitor_output['actions_GPU']))

                env_output['info']['rnn_hidden_state_h_GPU'] = merge(prev_hidden_state_h_baseline, prev_hidden_state_h_competitor)
                env_output['info']['rnn_hidden_state_c_GPU'] = merge(prev_hidden_state_c_baseline, prev_hidden_state_c_competitor)
                env_output['info']['prediction_GPU_CPU'] = merge(prev_prediction_baseline, prev_prediction_competitor)

                #print("step")
                if env_output["done_GPU_CPU"].any():
                    # Cache reward, done, and info["actions_taken"] from the terminal step
                    cached_reward = env_output["reward_GPU_CPU"]
                    cached_done = env_output["done_GPU_CPU"]
                    cached_info_actions_taken = env_output["info"]["actions_taken_GPU_CPU"]
                    cached_info_logging = {
                        key: val for key, val in env_output["info"].items() if key.startswith("LOGGING_")
                    }
                    n_games_played += env_output["done_GPU_CPU"].sum().item()
                    with shared_n_games.get_lock():
                        shared_n_games.value += env_output["done_GPU_CPU"].sum().item()
                    with shared_first_player_wins.get_lock():
                        shared_first_player_wins.value += (env_output["info"]['LOGGING_CPU_agg_rewards'][env_output["done_GPU_CPU"]] > 0).sum().item()
                    #print("COMPETE DONE: ", env_output["done_GPU_CPU"].sum(), ' n_games_played: ', n_games_played)
                    #print("COMPETE REWARD: ", env_output["info"]['LOGGING_CPU_agg_rewards'])
                    #print("FIRST: ", first_player_wins, 'SECOND:', n_games_played - first_player_wins, ' n_games_played: ', n_games_played)

                    env_output = env.reset()


                    env_output["reward_GPU_CPU"] = cached_reward
                    env_output["done_GPU_CPU"] = cached_done
                    env_output["info"]["actions_taken_GPU_CPU"] = cached_info_actions_taken
                    env_output["info"].update(cached_info_logging)

                    env_output['info']['rnn_hidden_state_h_GPU'] = merge(prev_hidden_state_h_baseline, prev_hidden_state_h_competitor)
                    env_output['info']['rnn_hidden_state_c_GPU'] = merge(prev_hidden_state_c_baseline, prev_hidden_state_c_competitor)
                    env_output['info']['prediction_GPU_CPU'] = merge(prev_prediction_baseline, prev_prediction_competitor)

                    env_output['info']['rnn_hidden_state_h_GPU'][env_output["done_GPU_CPU"]] *= 0
                    env_output['info']['rnn_hidden_state_c_GPU'][env_output["done_GPU_CPU"]] *= 0
                    env_output['info']['prediction_GPU_CPU'][env_output["done_GPU_CPU"]] *= 0

    except KeyboardInterrupt:
        pass  # Return silently.