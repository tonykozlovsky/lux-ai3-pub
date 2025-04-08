import os

import setproctitle

from .buffer_utils import buffers_apply, fill_buffers_inplace, stack_buffers, fill_buffers_inplace_3
import torch
import tempfile


def batch_prepare_process_func(flags, full_queue, batch_queue, device, buffers, learner_free_batch_queue, learner_gpu_buffers,
                               #free_queues,
                               free_queue, batch_type):
    try:
        setproctitle.setproctitle(f"BATCH_PREPARE_PROCESS_TYPE_{batch_type}_DEVICE_{device}")

        #os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
        torch.cuda.set_device(f'cuda:{device}')
        torch.set_default_device(f'cuda:{device}')

        print("BATCH PREPARE", "device: ", device, "batch_type: ", batch_type)

        while True:
            batch_parts = []
            buffers_ids = []
            for i in range(flags.batch_size // flags.n_actor_envs):
                #print("GET BUFFER")
                buffer_idx, actor_idx = full_queue.get()
                #print("GOT BUFFER", buffer_idx, actor_idx)
                buffers_ids.append(buffer_idx)
                pinned = buffers_apply(buffers[buffer_idx], lambda x: x.pin_memory()) # 850
                # 750
                #batch_parts.append(buffers_apply(pinned, lambda x: x.to(device, non_blocking=True)))
                batch_parts.append(pinned)

            #if flags.n_actor_envs != flags.batch_size:
            #    stacked_buffers = stack_buffers(batch_parts, dim=1)
            #else:
            #    stacked_buffers = batch_parts[0]

            if os.getenv("MAC") != '1':
                #pinned_buffers = buffers_apply(buffers[buffer_idx], lambda x: x.pin_memory())
                #learner_buffer_idx = learner_free_batch_queue.get()
                #fill_buffers_inplace(learner_gpu_buffers[learner_buffer_idx], pinned_buffers, non_blocking=True)
                #batch_queue.put((buffer_idx, actor_idx, buffers_apply(pinned_buffers, lambda x: x.to(f'cuda:{device}', non_blocking=True))))
                #torch.cuda.synchronize()

                learner_buffer_idx = learner_free_batch_queue.get()
                for part_idx in range(len(batch_parts)):
                    a = part_idx * flags.n_actor_envs
                    b = (part_idx + 1) * flags.n_actor_envs
                    fill_buffers_inplace_3(learner_gpu_buffers[learner_buffer_idx], batch_parts[part_idx], a, b)
                #fill_buffers_inplace(learner_gpu_buffers[learner_buffer_idx], stacked_buffers, non_blocking=True)
                #torch.cuda.empty_cache()
                #with lock:
                torch.cuda.synchronize()

                for buffer_idx in buffers_ids:
                    free_queue.put(buffer_idx)
            else:
                pass
                #pinned_buffers = buffers[buffer_idx]
                #learner_buffer_idx = learner_free_batch_queue.get()
                #fill_buffers_inplace(learner_gpu_buffers[learner_buffer_idx], stacked_buffers, non_blocking=False)
                #for buffer_idx in buffers_ids:
                #    free_queue.put(buffer_idx)
                #batch_queue.put((buffer_idx, actor_idx, buffers_apply(pinned_buffers, lambda x: x.to(f'cuda:{device}', non_blocking=True))))
                #torch.cuda.synchronize()
            #free_queues[actor_idx].put(0)

            batch_queue.put((batch_type, learner_buffer_idx))
    except KeyboardInterrupt:
        return
