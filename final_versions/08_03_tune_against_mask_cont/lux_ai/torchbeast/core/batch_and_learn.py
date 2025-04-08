import torch.distributed as dist
import  os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .learn import learn
from ...nns import create_model
from ..profiler import ScopedProfiler

from .load_model import loadd_model

from .losses_func_frozen_teacher import losses_func_frozen_teacher
from .losses_func_frozen_actor import losses_func_frozen_actor
from .losses_func_selfplay import losses_func_selfplay
from .losses_func_behavior_cloning import losses_func_behavior_cloning
import random
from pathlib import Path
import tempfile

import setproctitle

def setup(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    #torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed process group"""
    dist.destroy_process_group()


def batch_and_learn(i, flags, shared_steps, batch_queues_, batch_types, borders, disk_params, actor_model, teacher_flags, warmup_end_step, shared_multiplier, shared_entropy_multiplier_worker, shared_entropy_multiplier_sapper, shared_pos_weight,
                    stats_queue_learner, cgn,
                    #free_queues,
                    free_queue, learner_free_batch_queues_, learner_gpu_buffers_):
    try:
        world_size = flags.n_learner_devices
        gpu_id = flags.n_actor_devices + i

        #os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
        torch.cuda.set_device(f'cuda:{gpu_id}')
        torch.set_default_device(f'cuda:{gpu_id}')

        rank = i
        setproctitle.setproctitle(f"LEARN_{i}_PROCESS")
        #os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(tempfile.gettempdir(), f'torch_inductor_cache_LEARN_{i}_wat') # persistent compile cache

        if os.getenv("MAC") != '1':
            setup(rank, world_size)

        print("BATCH AND LEARN", "gpu_id: ", gpu_id, "rank: ", rank, world_size)

        device = 'cpu' if os.getenv("MAC") == '1' else gpu_id

        if flags.use_teacher and flags.use_old_teacher_loss:
            teacher_model = create_model(
                flags,
                device,
                teacher_model_flags=teacher_flags,
                is_teacher_model=True
            )
            if os.getenv("USE_TORCH_COMPILE") == "1":
                teacher_model = torch.compile(teacher_model, fullgraph=True, dynamic=False)
                print("TEACHER COMPILED")

            loadd_model(teacher_model, Path(os.path.dirname(__file__) + "/../" + "../" + "../") / Path(flags.teacher_load_dir) / flags.teacher_checkpoint_file)
            teacher_model.eval()
        else:
            teacher_model = None

        learner_model = create_model(flags, device, teacher_model_flags=None, is_teacher_model=False)
        if os.getenv("USE_TORCH_COMPILE") == "1":
            learner_model = torch.compile(learner_model, fullgraph=True, dynamic=False)
        if os.getenv("MAC") != '1':
            learner_model = DDP(learner_model, device_ids=[device], static_graph=True)

        #if checkpoint_state is not None:
        #    loadd_model(learner_model, checkpoint_path)

        if os.getenv("MAC") != '1':
            learner_model.module.load_state_dict(actor_model.state_dict())
        else:
            learner_model.load_state_dict(actor_model.state_dict())

        learner_model.train()
        #learner_model = learner_model.share_memory()
        print("Learner model loaded")


        if os.getenv("USE_TORCH_COMPILE") == "1":
            flags.optimizer_class = torch.compile(flags.optimizer_class, fullgraph=True, dynamic=False)

        optimizer = flags.optimizer_class(
            learner_model.parameters(),
            **flags.optimizer_kwargs
        )



        def lr_lambda(epoch):
            return disk_params['lr_lambda']

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        #if checkpoint_state is not None and not flags.weights_only:
        #    scheduler.load_state_dict(checkpoint_state["scheduler_state_dict"])



        profiler = ScopedProfiler()

        n_by_type = {}

        iteration = 0
        while shared_steps.value < flags.total_steps:
            #print("LEARN ITER", rank, iteration)

            iteration += 1
            # profiler.begin_block("batch_and_learn")

            if True or ("get_batch"):
                want_batch_type = None
                randval = random.random()

                for i in range(len(borders)):
                    if randval < borders[i][1]:
                        want_batch_type = batch_types[i]
                        assert borders[i][1] != borders[i][0]
                        break

                n_by_type[want_batch_type] = n_by_type.get(want_batch_type, 0) + 1
                #print(n_by_type)

                batch_queue = batch_queues_[want_batch_type][rank]
                learner_free_batch_queue = learner_free_batch_queues_[want_batch_type][rank]
                learner_gpu_buffers = learner_gpu_buffers_[want_batch_type]

                batch_type, learner_buffer_idx = batch_queue.get()

                batch = learner_gpu_buffers[learner_buffer_idx]

            if True or ("learn"):
                with shared_steps.get_lock():
                    shared_steps.value += flags.batch_size * flags.unroll_length

                flags.lmb = disk_params['lmb']
                flags.use_bf16 = disk_params['use_bf16'] != 0
                flags.teacher_kl_cost = disk_params['teacher_kl_cost']
                flags.teacher_kl_cost_change_per_step = disk_params['teacher_kl_cost_change_per_step']
                flags.teacher_baseline_cost = disk_params['teacher_baseline_cost']
                flags.prediction_cost = disk_params['prediction_cost']

                warmup = shared_steps.value < warmup_end_step
                if not warmup:
                    learn(
                        device=device,
                        flags=flags,
                        learner_model=learner_model,
                        teacher_model=teacher_model,
                        batch=batch,
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                        warmup=warmup,
                        shared_multiplier=shared_multiplier,
                        shared_entropy_multiplier_worker=shared_entropy_multiplier_worker,
                        shared_entropy_multiplier_sapper=shared_entropy_multiplier_sapper,
                        profiler=profiler,
                        stats_queue_learner=stats_queue_learner,
                        cgn=cgn,
                        losses_func_selfplay=losses_func_selfplay,
                        losses_func_frozen_actor=losses_func_frozen_actor,
                        losses_func_frozen_teacher=losses_func_frozen_teacher,
                        losses_func_behavior_cloning=losses_func_behavior_cloning,
                        shared_pos_weight=shared_pos_weight,
                        batch_type=batch_type
                    )


            if True or ("free_queue_put"):
                learner_free_batch_queue.put(learner_buffer_idx)


            if rank < flags.n_actor_devices and iteration % flags.update_actor_weigts_every_batches == 0:
                if os.getenv("MAC") != '1':
                    actor_model.load_state_dict(learner_model.module.state_dict())
                else:
                    actor_model.load_state_dict(learner_model.state_dict())

            #if iteration <= 10:
            #    print(f"Process {rank} before empty_cache: {torch.cuda.memory_allocated()} {torch.cuda.memory_reserved()} bytes")
            #    torch.cuda.empty_cache()  # Only affects this process
            #    print(f"Process {rank} after empty_cache: {torch.cuda.memory_allocated()} {torch.cuda.memory_reserved()} bytes")



            if iteration % 100 == 0:
                profiler.print_timings()
                profiler.reset()
                print("n_by_type", n_by_type)
    except KeyboardInterrupt:
        return
    except Exception as e:
        print(f"Error in batch_and_learn: {e}")
        raise
    finally:
        cleanup()

