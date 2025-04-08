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



@torch.no_grad()
def act_func(
        flags: SimpleNamespace,
        teacher_flags: Optional[SimpleNamespace],
        actor_index: int,
        free_queue: mp.Queue,
        full_queue: mp.Queue,
        buffers,
        stats_free_queue: mp.Queue,
        stats_full_queue: mp.Queue,
        stats_buffers,
        is_frozen_teacher: bool = False,
        is_behavior_cloning = False,
        teacher_model=None,
        teacher_model_name=None,
        frozen_actor_model=None,
        frozen_actor_models_path=None,
        actor_model=None,
        device=None,
        batch_type=None,
        torch_compile_lock=None,
):
    frozen_name = 'default'
    if is_frozen_teacher:
        frozen_name = teacher_model_name

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    torch.cuda.set_device(f'cuda:{device}')
    torch.set_default_device(f'cuda:{device}')
    replays_path = Path(os.path.dirname(__file__) + "/../" + "../" + "../") / Path("analytics/outputs/frog")

    def load_replays_list():
        all_replays_list = os.listdir(replays_path)
        return all_replays_list

    if is_behavior_cloning:
        all_replays_list = load_replays_list()
    else:
        all_replays_list = None

    behevior_cloning_replays = []

    behavior_cloning_step_to_take_actions = 0

    def sample_replays(start_from):
        nonlocal behavior_cloning_step_to_take_actions
        behavior_cloning_step_to_take_actions = start_from

        nonlocal behevior_cloning_replays
        behevior_cloning_replays = []
        while len(behevior_cloning_replays) < flags.n_actor_envs:
            behevior_cloning_replays_file = random.choice(all_replays_list)
            try:
                rep = json.load(open(replays_path / behevior_cloning_replays_file))

                team_names = rep['info']['TeamNames']
                victim_position = None
                for i in range(2):
                    if team_names[i] == 'Frog Parade':
                        victim_position = i
                        break
                if victim_position is None:
                    continue

                if victim_position != 1:
                    continue

                rewards = rep['rewards']

                if rewards[victim_position] < rewards[1 - victim_position]: # Frog Lose game
                    continue

                if rep['statuses'][0] != 'DONE' or rep['statuses'][1] != 'DONE':
                    continue

                sap_range = rep['configuration']['env_cfg']['unit_sap_range']
                sap_cost = rep['configuration']['env_cfg']['unit_sap_cost']
                move_cost = rep['configuration']['env_cfg']['unit_move_cost']

                prev_obs = [None, None]
                for step in rep['steps']:
                    for pid in [0, 1]:
                        action = step[pid]['action']
                        obs = prev_obs[pid]
                        if obs != None:
                            o = json.loads(obs['obs'])
                        for pos_id, a in enumerate(action):
                            if a[0] < 0 or a[0] > 5:
                                #print("FIX INVALID ACTION")
                                a[0] = 0
                                a[1] = 0
                                a[2] = 0
                                continue
                            if a[0] < 5:
                                a[1] = 0
                                a[2] = 0
                                if obs != None:
                                    if o['units']['energy'][pid][pos_id] < move_cost:
                                        #print("FIX MOVE ENERGY")
                                        a[0] = 0
                                continue
                            if a[0] == 5:
                                if obs != None:
                                    if o['units']['energy'][pid][pos_id] < sap_cost:
                                        #print("FIX SAP ENERGY")
                                        a[0] = 0
                                        a[1] = 0
                                        a[2] = 0
                                        continue
                                sap_x = a[1]
                                sap_y = a[2]
                                if sap_x < -sap_range or sap_x > sap_range or sap_y < -sap_range or sap_y > sap_range:
                                    #print("FIX SAP")
                                    a[0] = 0
                                    a[1] = 0
                                    a[2] = 0
                                    continue
                                if obs != None:
                                    pos = o['units']['position'][pid][pos_id]
                                    spx = pos[0] + sap_x
                                    spy = pos[1] + sap_y
                                    if spx < 0 or spx >= 24 or spy < 0 or spy >= 24:
                                        #print('FIX SAP POSITION')
                                        a[0] = 0
                                        a[1] = 0
                                        a[2] = 0
                                        continue
                        prev_obs[pid] = step[pid]['observation']


                behevior_cloning_replays.append({'steps': rep['steps'], 'victim_position': victim_position, 'seed': rep['configuration']['seed']})

            except Exception as e:
                print("CANNOT LOAD REPLAY:", replays_path / behevior_cloning_replays_file)
                time.sleep(1)
                continue

    def get_behavior_cloning_actions():
        nonlocal behavior_cloning_step_to_take_actions
        env_actions = []

        for i in range(flags.n_actor_envs):
            env_actions.append([behevior_cloning_replays[i]['steps'][behavior_cloning_step_to_take_actions][0]['action'], behevior_cloning_replays[i]['steps'][behavior_cloning_step_to_take_actions][1]['action']])

        behavior_cloning_step_to_take_actions += 1

        return env_actions

    def bc_actions_to_my(bc_actions):
        worker = torch.zeros((len(bc_actions), 1, 2, 16, 1), dtype=torch.int64)
        sapper = torch.zeros((len(bc_actions), 1, 2, 16, 1), dtype=torch.int64)

        for env_idx, bca in enumerate(bc_actions):
            for pid in range(2):
                for j in range(16):
                    a = bca[pid][j]
                    if pid == 1:
                        a[0] = rotate_action_two_diagonal(a[0])
                        a[1], a[2] = rotate_two_diagonal([a[1] + 7, a[2] + 7], 15)
                        a[1] -= 7
                        a[2] -= 7

                    if a[0] < 5:
                        worker[env_idx, 0, pid, j, 0] = a[0]
                    else:
                        worker[env_idx, 0, pid, j, 0] = 5
                        sapper[env_idx, 0, pid, j, 0] = (a[1] + 7) * 15 + (a[2] + 7)
            pass

        return {'worker': worker, 'sapper': sapper}


    #frozen_model_id = actor_index % num_frozen_actor_models # random.randint(0, num_frozen_actor_models - 1) if num_frozen_actor_models > 0 else None
    if frozen_actor_model is not None:
        frozen_name = frozen_actor_models_path

    first_run_1 = True
    first_run_2 = True
    first_run_3 = True
    first_run_4 = True
    first_run_5 = True

    def _run_model(profiler, env_output, is_sample, state, update_state):
        nonlocal first_run_1
        nonlocal first_run_2
        nonlocal first_run_3
        nonlocal first_run_4
        nonlocal first_run_5

        def merge2(a, b):
            return {'worker': torch.cat([a['worker'][:, :, 0, :, :].unsqueeze(2), b['worker'][:, :, 1, :, :].unsqueeze(2)], dim=2),
                    'sapper': torch.cat([a['sapper'][:, :, 0, :, :].unsqueeze(2), b['sapper'][:, :, 1, :, :].unsqueeze(2)], dim=2)}

        if frozen_actor_model is None and not is_frozen_teacher:
            if not is_behavior_cloning:
                env_output['info']['rnn_hidden_state_h_GPU'] = state['first_model']['prev_hidden_state_h']
                env_output['info']['rnn_hidden_state_c_GPU'] = state['first_model']['prev_hidden_state_c']
                env_output['info']['prediction_GPU_CPU'] = state['first_model']['prev_prediction']

                with profiler.block("inference1"):
                    if first_run_1:
                        with torch_compile_lock:
                            res = actor_model(env_output, sample=is_sample)
                            first_run_1 = False
                    else:
                        res = actor_model(env_output, sample=is_sample)
                if update_state:
                    state['first_model']['prev_hidden_state_h'] = res['rnn_hidden_state_h_GPU']
                    state['first_model']['prev_hidden_state_c'] = res['rnn_hidden_state_c_GPU']
                    state['first_model']['prev_prediction'] = res['prediction_GPU_CPU']

                res['teacher_output'] = copy.copy(res)


                return res, res['actions_GPU']
            else:
                env_output['info']['rnn_hidden_state_h_GPU'] = state['first_model']['prev_hidden_state_h']
                env_output['info']['rnn_hidden_state_c_GPU'] = state['first_model']['prev_hidden_state_c']
                env_output['info']['prediction_GPU_CPU'] = state['first_model']['prev_prediction']

                with profiler.block("inference2"):
                    if first_run_2:
                        with torch_compile_lock:
                            res = actor_model(env_output, sample=is_sample)
                            first_run_2 = False
                    else:
                        res = actor_model(env_output, sample=is_sample)
                if update_state:
                    state['first_model']['prev_hidden_state_h'] = res['rnn_hidden_state_h_GPU']
                    state['first_model']['prev_hidden_state_c'] = res['rnn_hidden_state_c_GPU']
                    state['first_model']['prev_prediction'] = res['prediction_GPU_CPU']

                res['teacher_output'] = copy.copy(res)
                bc_actions = get_behavior_cloning_actions()
                my_bc_actions = bc_actions_to_my(bc_actions)
                res['teacher_output']['actions_GPU'] = my_bc_actions

                return res, my_bc_actions
        else:

            env_output['info']['rnn_hidden_state_h_GPU'] = state['second_model']['prev_hidden_state_h']
            env_output['info']['rnn_hidden_state_c_GPU'] = state['second_model']['prev_hidden_state_c']
            env_output['info']['prediction_GPU_CPU'] = state['second_model']['prev_prediction']

            with profiler.block("inference3"):
                if is_frozen_teacher:
                    if first_run_3:
                        with torch_compile_lock:
                            output_2 = teacher_model(env_output, sample=flags.frozen_teacher_sample)
                            first_run_3 = False
                    else:
                        output_2 = teacher_model(env_output, sample=flags.frozen_teacher_sample)

                else:
                    #print("frozen_model_id:", frozen_model_id, "num_frozen_actor_models: ", num_frozen_actor_models)
                    if first_run_4:
                        with torch_compile_lock:
                            output_2 = frozen_actor_model(env_output, sample=flags.frozen_teacher_sample)
                            first_run_4 = False
                    else:
                        output_2 = frozen_actor_model(env_output, sample=flags.frozen_teacher_sample)
            if update_state:
                state['second_model']['prev_hidden_state_h'] = output_2['rnn_hidden_state_h_GPU']
                state['second_model']['prev_hidden_state_c'] = output_2['rnn_hidden_state_c_GPU']
                state['second_model']['prev_prediction'] = output_2['prediction_GPU_CPU']

            if is_frozen_teacher and flags.frozen_teacher_both_sides:

                output_2['teacher_output'] = copy.copy(output_2)

                return output_2, output_2['actions_GPU']
            else:

                env_output['info']['rnn_hidden_state_h_GPU'] = state['first_model']['prev_hidden_state_h']
                env_output['info']['rnn_hidden_state_c_GPU'] = state['first_model']['prev_hidden_state_c']
                env_output['info']['prediction_GPU_CPU'] = state['first_model']['prev_prediction']

                with profiler.block("inference4"):
                    if first_run_5:
                        with torch_compile_lock:
                            output_1 = actor_model(env_output, sample=is_sample)
                            first_run_5 = False
                    else:
                        output_1 = actor_model(env_output, sample=is_sample)
                if update_state:
                    state['first_model']['prev_hidden_state_h'] = output_1['rnn_hidden_state_h_GPU']
                    state['first_model']['prev_hidden_state_c'] = output_1['rnn_hidden_state_c_GPU']
                    state['first_model']['prev_prediction'] = output_1['prediction_GPU_CPU']

                if is_frozen_teacher:

                    output_1['teacher_output'] = output_2

                else:
                    output_1['teacher_output'] = copy.copy(output_1)

                return output_1, merge2(output_1['actions_GPU'], output_2['actions_GPU'])


    setproctitle.setproctitle(f"ACTOR_{actor_index}_PROCESS_TYPE_{batch_type}_DEVICE_{device}")
    try:
        logging.info(f"Actor {actor_index} started.")
        profiler = ScopedProfiler(enabled=False)

        seed = secrets.randbits(64) ^ int(time.time_ns()) ^ int.from_bytes(os.urandom(8), "big")
        random.seed(seed)

        dbg = False
        if dbg:
            dbg_steps = 1
            file1 = open(f"../debug_hashes_replay_player_{0}.txt", "w")
            file2 = open(f"../debug_hashes_replay_player_{1}.txt", "w")

        env = create_env(actor_index, flags, device=device, teacher_flags=teacher_flags, profiler=profiler)


        custom_seeds = None
        if is_behavior_cloning:
            sample_replays(1)
            custom_seeds = [x['seed'] for x in behevior_cloning_replays]

        env_output = env.reset(force=True, custom_seeds=custom_seeds)

        logging.info(f"Actor {actor_index} env reset.")

        is_sample = not flags.replay

        lstm_size = (128, 24, 24) if flags.enable_lstm else (1, 1, 1)

        state = {
            'first_model': {
                'prev_hidden_state_h' : torch.zeros(flags.n_actor_envs, 2, *lstm_size, device=device, dtype=torch.float32),
                'prev_hidden_state_c' : torch.zeros(flags.n_actor_envs, 2, *lstm_size, device=device, dtype=torch.float32),
                'prev_prediction' : torch.zeros(flags.n_actor_envs, 2, 3, 24, 24, device=device, dtype=torch.float32),
            },
            'second_model': {
                'prev_hidden_state_h' : torch.zeros(flags.n_actor_envs, 2, *lstm_size, device=device, dtype=torch.float32),
                'prev_hidden_state_c' : torch.zeros(flags.n_actor_envs, 2, *lstm_size, device=device, dtype=torch.float32),
                'prev_prediction' : torch.zeros(flags.n_actor_envs, 2, 3, 24, 24, device=device, dtype=torch.float32),
            }

        }

        agent_output, env_actions = _run_model(profiler, env_output, is_sample, state, False)


        if actor_index == 1:
            profiler.print_timings()
            profiler.reset()




        iteration = 0
        ended = 0
        while ended < (1 if flags.five_rounds else 5):
            iteration += 1
            #current_batch = []
            with profiler.block("buffer_idx_get"):
                buffer_idx = free_queue.get()
            stats_buffer_idx = None
            try:
                stats_buffer_idx = stats_free_queue.get_nowait()
            except queue.Empty:
                pass

            #if num_frozen_actor_models > 0 and iteration % change_frozen_actor_every_n_iters == 0:
            #    frozen_model_id = random.randint(0, num_frozen_actor_models - 1)
            #    frozen_name = frozen_actor_models_paths[frozen_model_id]


            profiler.begin_block("act")

            if True:
                #current_buffer = []


                # Write old rollout end.
                if not flags.replay:
                    with profiler.block("current_buffer.append_1"):
                        fill_buffers_inplace_2(buffers[buffer_idx], dict(**env_output, **agent_output), 0)
                        if stats_buffer_idx is not None:
                            fill_buffers_inplace_2(stats_buffers[stats_buffer_idx], dict(**env_output, **agent_output), 0)
                        #current_buffer.append(buffers_apply(dict(**env_output, **agent_output), lambda x: x.to('cpu', non_blocking=False)))
                # Do new rollout.
                profiler.begin_block("unroll")
                for t in range(flags.unroll_length):
                    with profiler.block("inference"):
                        if dbg:
                            for player in ['player_0', 'player_1']:
                                out_env_hashes(env_output, file1 if player == 'player_0' else file2, player, dbg_steps)

                        agent_output, env_actions = _run_model(profiler, env_output, is_sample, state, True)

                        if dbg:
                            for player in ['player_0', 'player_1']:
                                out_env_hashes(agent_output, file1 if player == 'player_0' else file2, player, dbg_steps)

                    with profiler.block("env.step"):
                        env_output = env.step(env_actions)

                        if dbg:
                            dbg_steps += 1

                        env_output['info']['rnn_hidden_state_h_GPU'] = state['first_model']['prev_hidden_state_h']
                        env_output['info']['rnn_hidden_state_c_GPU'] = state['first_model']['prev_hidden_state_c']
                        env_output['info']['prediction_GPU_CPU'] = state['first_model']['prev_prediction']

                    if env_output["done_GPU_CPU"].any():
                        # Cache reward, done, and info["actions_taken"] from the terminal step
                        cached_reward = env_output["reward_GPU_CPU"]
                        cached_done = env_output["done_GPU_CPU"]
                        cached_info_actions_taken = env_output["info"]["actions_taken_GPU_CPU"]
                        cached_info_logging = {
                            key: val for key, val in env_output["info"].items() if key.startswith("LOGGING_")
                        }

                        with profiler.block("env.reset"):
                            custom_seeds = None
                            if is_behavior_cloning:
                                sample_replays(2)
                                custom_seeds = [x['seed'] for x in behevior_cloning_replays]
                            env_output = env.reset(custom_seeds=custom_seeds)

                        env_output["reward_GPU_CPU"] = cached_reward
                        env_output["done_GPU_CPU"] = cached_done
                        env_output["info"]["actions_taken_GPU_CPU"] = cached_info_actions_taken
                        env_output["info"].update(cached_info_logging)

                        for model in 'first_model', 'second_model':
                            state[model]['prev_hidden_state_h'][env_output["done_GPU_CPU"]] *= 0
                            state[model]['prev_hidden_state_c'][env_output["done_GPU_CPU"]] *= 0
                            state[model]['prev_prediction'][env_output["done_GPU_CPU"]] *= 0

                        env_output['info']['rnn_hidden_state_h_GPU'] = state['first_model']['prev_hidden_state_h']
                        env_output['info']['rnn_hidden_state_c_GPU'] = state['first_model']['prev_hidden_state_c']
                        env_output['info']['prediction_GPU_CPU'] = state['first_model']['prev_prediction']


                        if flags.replay:
                            ended += 1

                    if not flags.replay:
                        with profiler.block("current_buffer.append_2"):
                            fill_buffers_inplace_2(buffers[buffer_idx], dict(**env_output, **agent_output), t + 1)
                        if stats_buffer_idx is not None:
                            fill_buffers_inplace_2(stats_buffers[stats_buffer_idx], dict(**env_output, **agent_output), t + 1)
                            #current_buffer.append(buffers_apply(dict(**env_output, **agent_output), lambda x: x.to('cpu', non_blocking=False)))

                #current_batch.append(current_buffer)
                profiler.end_block("unroll")

            if not flags.replay:
                #with profiler.block("unsqueeze"):
                #    for idx, current_buffer in enumerate(current_batch):
                #        for jdx, cb in enumerate(current_buffer):
                #            current_batch[idx][jdx] = buffers_apply(cb, lambda x: x.unsqueeze(0))

                #with profiler.block("stack_buffers1"):
                #    stacked_buffers = [stack_buffers(x, dim=0) for x in current_batch]
                #with profiler.block("stack_buffers2"):
                #    if flags.n_actor_envs != flags.batch_size:
                #        stacked_buffers = stack_buffers(stacked_buffers, dim=1)
                #    else:
                #        stacked_buffers = stacked_buffers[0]
                #with profiler.block("buffers_apply"):
                #    stacked_buffers = buffers_apply(stacked_buffers, lambda x: x.view(flags.unroll_length + 1, flags.batch_size, *x.shape[2:]))

                #with profiler.block("stats_queue_put"):
                #    try:
                #        buffer_idx = stats_free_queue.get_nowait()
                #        with profiler.block("stats_get_cpu_buffers"):
                #            cpu_buffers = get_cpu_buffers(stacked_buffers)
                #        with profiler.block("stats_fill_buffers_inplace"):
                #            fill_buffers_inplace(stats_buffers[buffer_idx], cpu_buffers)
                #        stats_full_queue.put(buffer_idx)
                #    except queue.Empty:
                #        pass

                #with profiler.block("get_gpu_buffers"):
                #    gpu_buffers = get_gpu_buffers(stacked_buffers, device='cpu')


                #with profiler.block("free_queues_get"):
                #    free_queues[actor_index].get()


                #with profiler.block("fill_buffers_inplace"):
                #    fill_buffers_inplace(buffers[buffer_idx], gpu_buffers)

                with profiler.block("full_queue_put"):
                    full_queue.put((buffer_idx, actor_index))

                if stats_buffer_idx is not None:
                    stats_full_queue.put((stats_buffer_idx, frozen_name))

            profiler.end_block("act")

            if actor_index == 1 and iteration % (10 * (flags.batch_size // flags.n_actor_envs)) == 0:
                profiler.print_timings()
                profiler.reset()

        if flags.replay:
            print("Replay ended")
            if dbg:
                file1.close()
                file2.close()

    except KeyboardInterrupt:
        return
