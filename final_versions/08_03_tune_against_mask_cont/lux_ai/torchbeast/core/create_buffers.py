import os

import torch
from types import SimpleNamespace
from typing import Optional
import copy

from .buffer_utils import buffers_apply, stack_buffers, get_cpu_buffers, get_gpu_buffers
from .common import get_input_buffers_for_inference, get_output_buffers_for_inference
from ..profiler import ScopedProfiler
from ...lux_gym import create_env


@torch.no_grad()
def create_buffers_func(
        flags: SimpleNamespace,
        teacher_flags: Optional[SimpleNamespace],
        actor_index: int,
        actor_model: torch.nn.Module,
        buffers,
        stats_buffers,
        learner_gpu_buffers,
        device,
        batch_types
):
    def _run_model(env_output, **kwargs):
        env_output = get_input_buffers_for_inference(env_output)
        res = actor_model(buffers_apply(env_output, lambda x: x.to(device)), **kwargs)
        res = buffers_apply(res, lambda x: x.to('cpu'))

        res['teacher_output'] = copy.copy(res)
        return res

    try:
        profiler = ScopedProfiler()

        env = create_env(actor_index, flags, device=device, teacher_flags=teacher_flags, profiler=profiler)
        env_output = env.reset(force=True)

        is_sample = not flags.replay

        lstm_size = (128, 24, 24) if flags.enable_lstm else (1, 1, 1)

        prev_hidden_state_h = torch.zeros(flags.n_actor_envs, 2, *lstm_size, device=device, dtype=torch.float32)
        prev_hidden_state_c = torch.zeros(flags.n_actor_envs, 2, *lstm_size, device=device, dtype=torch.float32)
        prev_prediction = torch.zeros(flags.n_actor_envs, 2, 3, 24, 24, device=device, dtype=torch.float32)

        env_output['info']['rnn_hidden_state_h_GPU'] = prev_hidden_state_h
        env_output['info']['rnn_hidden_state_c_GPU'] = prev_hidden_state_c
        env_output['info']['prediction_GPU_CPU'] = prev_prediction

        #buffers_for_inference = get_input_buffers_for_inference(env_output)
        #for i in range(flags.num_inference_workers * 2): # * 2 so that we have enough buffers to copy data into while inference workers are working with their own buffers
        #    inference_input_buffers.append(buffers_apply(buffers_for_inference, lambda x: torch.empty_like(x, device='cpu').share_memory_()))

        agent_output = _run_model(env_output, sample=is_sample, profiler=profiler)

        #output_buffers_for_inference = get_output_buffers_for_inference(agent_output)
        #for i in range(flags.num_inference_workers * 2):
        #    inference_output_buffers.append(buffers_apply(output_buffers_for_inference, lambda x: torch.empty_like(x, device='cpu').share_memory_()))

        current_batch = []

        for bi in range(flags.batch_size // flags.n_actor_envs):
            current_buffer = []

            current_buffer.append(buffers_apply(dict(**env_output, **agent_output), lambda x: x.to('cpu', non_blocking=False)))

            for t in range(flags.unroll_length):
                agent_output = _run_model(env_output, sample=is_sample, profiler=profiler)

                prev_hidden_state_h = agent_output['rnn_hidden_state_h_GPU']
                prev_hidden_state_c = agent_output['rnn_hidden_state_c_GPU']
                prev_prediction = agent_output['prediction_GPU_CPU']

                env_output = env.step(agent_output["actions_GPU"])

                env_output['info']['rnn_hidden_state_h_GPU'] = prev_hidden_state_h
                env_output['info']['rnn_hidden_state_c_GPU'] = prev_hidden_state_c
                env_output['info']['prediction_GPU_CPU'] = prev_prediction

                if env_output["done_GPU_CPU"].any():
                    # Cache reward, done, and info["actions_taken"] from the terminal step
                    cached_reward = env_output["reward_GPU_CPU"]
                    cached_done = env_output["done_GPU_CPU"]
                    cached_info_actions_taken = env_output["info"]["actions_taken_GPU_CPU"]
                    cached_info_logging = {
                        key: val for key, val in env_output["info"].items() if key.startswith("LOGGING_")
                    }

                    env_output = env.reset()

                    env_output["reward_GPU_CPU"] = cached_reward
                    env_output["done_GPU_CPU"] = cached_done
                    env_output["info"]["actions_taken_GPU_CPU"] = cached_info_actions_taken
                    env_output["info"].update(cached_info_logging)

                    env_output['info']['rnn_hidden_state_h_GPU'] = prev_hidden_state_h
                    env_output['info']['rnn_hidden_state_c_GPU'] = prev_hidden_state_c
                    env_output['info']['prediction_GPU_CPU'] = prev_prediction

                    env_output['info']['rnn_hidden_state_h_GPU'][env_output["done_GPU_CPU"]] *= 0
                    env_output['info']['rnn_hidden_state_c_GPU'][env_output["done_GPU_CPU"]] *= 0
                    env_output['info']['prediction_GPU_CPU'][env_output["done_GPU_CPU"]] *= 0


                current_buffer.append(buffers_apply(dict(**env_output, **agent_output), lambda x: x.to('cpu', non_blocking=False)))
            current_batch.append(current_buffer)


        for idx, current_buffer in enumerate(current_batch):
            for jdx, cb in enumerate(current_buffer):
                current_batch[idx][jdx] = buffers_apply(cb, lambda x: x.unsqueeze(0))

        stacked_buffers = [stack_buffers(x, dim=0) for x in current_batch]
        single_buffer = stacked_buffers[0]
        stacked_buffers = stack_buffers(stacked_buffers, dim=1)
        #stacked_buffers = buffers_apply(stacked_buffers, lambda x: x.view(flags.unroll_length + 1, flags.batch_size, *x.shape[2:]))

        single_cpu_buffers = get_cpu_buffers(single_buffer)

        for i in range(flags.num_stats_buffers):
            stats_buffers.append(buffers_apply(single_cpu_buffers, lambda x: torch.empty_like(x).share_memory_()))

        single_gpu_buffers = get_gpu_buffers(single_buffer, device='cpu')

        n_actors_by_bt = {
            'frozen_actor': flags.num_frozen_model_actors,
            'frozen_teacher': flags.frozen_teacher_actors,
            'behavior_cloning': flags.behavior_cloning_actors,
            'selfplay': flags.num_actors - flags.num_frozen_model_actors - flags.frozen_teacher_actors - flags.behavior_cloning_actors,
        }
        n_buffers_by_bt = {
            'frozen_actor': flags.num_frozen_model_actors_buffers,
            'frozen_teacher': flags.frozen_teacher_models_buffers,
            'behavior_cloning': flags.behavior_cloning_actors_buffers,
            'selfplay': flags.num_buffers - flags.num_frozen_model_actors_buffers - flags.frozen_teacher_models_buffers - flags.behavior_cloning_actors_buffers,
        }
        for bt in batch_types:
            buffers[bt] = []
            if n_actors_by_bt[bt] == 0:
                continue
            for i in range(n_buffers_by_bt[bt]):
                buffers[bt].append(buffers_apply(single_gpu_buffers, lambda x: torch.empty_like(x).share_memory_()))

        gpu_buffers = get_gpu_buffers(stacked_buffers, device='cpu')

        for i in range(flags.n_learner_devices):
            learner_gpu_buffers.append({})
            for bt in batch_types:
                if n_actors_by_bt[bt] == 0:
                    continue
                device = 'cpu' if os.getenv('MAC') == '1' else (flags.n_actor_devices + i)
                learner_gpu_buffers[-1][bt] = []
                for j in range(flags.prepare_batches):
                    learner_gpu_buffers[-1][bt].append(buffers_apply(gpu_buffers, lambda x: torch.empty_like(x, device=device)))




    except KeyboardInterrupt:
        pass  # Return silently.

