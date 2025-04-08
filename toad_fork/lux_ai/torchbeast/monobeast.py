import logging
import math
import os
import signal
import threading
import time
import timeit
from pathlib import Path
from typing import Union
import setproctitle
import sys
import torch
import psutil
from omegaconf import OmegaConf
import concurrent.futures

from ..nns import create_model
from ..utils import flags_to_namespace

from .core.load_model import loadd_model
from .core.create_buffers import create_buffers_func
from .core.act import act_func
from .core.stats import stats_func, RollingAverage
from .core.batch_prepare import batch_prepare_process_func
from .core.batch_and_learn import batch_and_learn


import multiprocessing as mp








def train(flags):
    setproctitle.setproctitle(f"MAIN_PROCESS")

    t = flags.unroll_length
    b = flags.batch_size

    if flags.use_teacher:
        teacher_flags = OmegaConf.load(Path(os.path.dirname(__file__)).parent.parent / Path(flags.teacher_load_dir) / "config.yaml")
        teacher_flags = flags_to_namespace(OmegaConf.to_container(teacher_flags))

        # put into yaml, not here !!!

        #teacher_flags.use_embedding_input = True
        #teacher_flags.n_transformer_blocks = 1
        #teacher_flags.enable_per_unit_resnet = False
        #teacher_flags.use_old_input = True
        #teacher_flags.use_baseline_v2 = False
        #teacher_flags.enable_transformer_v2 = False
        #teacher_flags.n_transformer_v2_blocks = 0
        #teacher_flags.prediction_size = 3
    else:
        teacher_flags = None

    if flags.load_dir:
        checkpoint_path = Path(os.path.dirname(__file__) + "/../" + "../") / Path(flags.load_dir) / flags.checkpoint_file
        checkpoint_state = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=flags.weights_only)
    else:
        checkpoint_state = None

    actor_models = []

    for i in range(flags.n_actor_devices):
        device = 'cpu' if os.getenv("MAC") == '1' else i
        actor_model = create_model(flags, device, teacher_model_flags=teacher_flags, is_teacher_model=False)
        if os.getenv("USE_TORCH_COMPILE") == "1":
            actor_model = torch.compile(actor_model, fullgraph=True, dynamic=False)

        if checkpoint_state is None:  # Only initialize if no checkpoint is loaded
            pass
        else:

            print("Loading model from checkpoint...")
            try:
                loadd_model(actor_model, checkpoint_path)
            except KeyError as e:
                print(f"Checkpoint missing keys: {e}")
                raise

        actor_model.eval()  # Set to evaluation mode
        actor_model.share_memory()  # Prepare for multiprocessing
        n_trainable_params = sum(p.numel() for p in actor_model.parameters() if p.requires_grad)
        print(f'Training model with {n_trainable_params:,d} parameters.')
        print("Actor model loaded")

        actor_models.append(actor_model)



    all_processes = []

    manager = mp.Manager()
    disk_params = manager.dict()

    disk_params['target_entropy_worker'] = flags.target_entropy_worker
    disk_params['target_entropy_sapper'] = flags.target_entropy_sapper

    disk_params['target_entropy_worker_change_per_step'] = flags.target_entropy_worker_change_per_step
    disk_params['target_entropy_sapper_change_per_step'] = flags.target_entropy_sapper_change_per_step
    disk_params['entropy_multiplier_worker'] = flags.entropy_initial_multiplier_worker
    disk_params['entropy_multiplier_sapper'] = flags.entropy_initial_multiplier_sapper

    disk_params['entropy_mult_change_speed_per_step'] = flags.entropy_mult_change_speed_per_step

    disk_params['lmb'] = flags.lmb
    disk_params['use_bf16'] = flags.use_bf16
    disk_params['teacher_kl_cost'] = flags.teacher_kl_cost
    disk_params['teacher_kl_cost_change_per_step'] = flags.teacher_kl_cost_change_per_step
    disk_params['teacher_baseline_cost'] = flags.teacher_baseline_cost
    disk_params['prediction_cost'] = flags.prediction_cost
    disk_params['lr_lambda'] = flags.lr_lambda
    disk_params['lr_lambda_change_per_step'] = flags.lr_lambda_change_per_step


    # Initialize dictionary
    shared_multiplier =  mp.Value('d', 0.1)
    shared_entropy_multiplier_worker =  mp.Value('d', flags.entropy_initial_multiplier_worker)
    shared_entropy_multiplier_sapper =  mp.Value('d', flags.entropy_initial_multiplier_sapper)
    shared_pos_weight =  [mp.Value('d', 1.), mp.Value('d', 1.), mp.Value('d', 1.)]

    shared_steps = mp.Value('d', 0)

    shared_exit = mp.Value('i', 0)

    warmup_end_step = 0
    if checkpoint_state is not None:
        if not flags.weights_only:
            if "step" in checkpoint_state.keys():
                shared_steps.value = checkpoint_state["step"]
            if "disk_params" in checkpoint_state.keys():
                print("LOADING DISK PARAMS FROM CHECKPOINT")
                for key in checkpoint_state["disk_params"].keys():
                    disk_params[key] = checkpoint_state["disk_params"][key]

        warmup_end_step = shared_steps.value + flags.n_warmup_steps

    batch_types = ['frozen_actor', 'frozen_teacher', 'behavior_cloning', 'selfplay']

    free_queue = {}
    full_queue = {}
    batch_queues = {}
    learner_free_batch_queues = {}


    n_buffers_by_bt = {
        'frozen_actor': flags.num_frozen_model_actors_buffers,
        'frozen_teacher': flags.frozen_teacher_models_buffers,
        'behavior_cloning': flags.behavior_cloning_actors_buffers,
        'selfplay': flags.num_buffers - flags.num_frozen_model_actors_buffers - flags.frozen_teacher_models_buffers - flags.behavior_cloning_actors_buffers,
    }
    for bt in batch_types:
        free_queue[bt] = torch.multiprocessing.Queue(maxsize=n_buffers_by_bt[bt])
        full_queue[bt] = torch.multiprocessing.Queue(maxsize=n_buffers_by_bt[bt])
        batch_queues[bt] = [torch.multiprocessing.Queue(maxsize=flags.prepare_batches) for _ in range(flags.n_learner_devices)]

        learner_free_batch_queues[bt] = []
        for i in range(flags.n_learner_devices):
            learner_free_batch_queues[bt].append(torch.multiprocessing.Queue(maxsize=flags.prepare_batches))

        for learner_free_batch_queue in learner_free_batch_queues[bt]:
            for i in range(flags.prepare_batches):
                learner_free_batch_queue.put(i)


    stats_free_queue = torch.multiprocessing.Queue(maxsize=flags.num_stats_buffers)
    stats_full_queue = torch.multiprocessing.Queue(maxsize=flags.num_stats_buffers)
    stats_queue_learner = torch.multiprocessing.Queue(maxsize=100)



    buffers = {}
    learner_gpu_buffers = []
    stats_buffers = []

    create_buffers_func(flags,
                        teacher_flags,
                        0,
                        actor_models[0],
                        buffers,
                        stats_buffers,
                        learner_gpu_buffers,
                        device = 'cpu' if os.getenv("MAC") == '1' else 0,
                        batch_types = batch_types)
    #torch.cuda.empty_cache()

    if flags.num_frozen_model_actors > 0:
        frozen_actor_models = []
        frozen_actor_model_paths = []
        for i in range(flags.n_actor_devices):
            assert flags.num_frozen_model_actors > 0
            frozen_actor_checkpoint_paths = [
                Path(os.path.dirname(__file__) + "/../" + "../") / Path(f) for f in flags.frozen_actor_checkpoint_paths
            ]
            change_frozen_actor_every_n_iters = flags.change_frozen_actor_every_n_iters

            def _create_frozen_actor_model(frozen_actor_checkpoint_path):
                logging.info(f"Loading frozen actor model from {frozen_actor_checkpoint_path}")
                frozen_model_flags = OmegaConf.load(Path(frozen_actor_checkpoint_path).parent / 'config.yaml')
                frozen_model_flags = flags_to_namespace(OmegaConf.to_container(frozen_model_flags, resolve=True))

                device = 'cpu' if os.getenv("MAC") == '1' else i
                frozen_actor_model = create_model(
                    frozen_model_flags, device, teacher_model_flags=teacher_flags, is_teacher_model=False)
                if os.getenv("USE_TORCH_COMPILE") == "1":
                    frozen_actor_model = torch.compile(frozen_actor_model, fullgraph=True, dynamic=False)
                loadd_model(frozen_actor_model, frozen_actor_checkpoint_path)
                frozen_actor_model.eval()
                return frozen_actor_model

            frozen_actor_models.append([_create_frozen_actor_model(path) for path in frozen_actor_checkpoint_paths])
            frozen_actor_model_paths.append([path.split('/')[2] for path in flags.frozen_actor_checkpoint_paths])
    else:
        assert flags.num_frozen_model_actors == 0
        frozen_actor_models = []
        frozen_actor_model_paths = []
        for i in range(flags.n_actor_devices):
            frozen_actor_models.append([])
            frozen_actor_model_paths.append([])
        change_frozen_actor_every_n_iters = 0


    teacher_models = []
    teacher_model_names = []

    if not flags.use_old_teacher_loss:
        for i in range(flags.n_actor_devices):
            # Load teacher model for KL loss
            if flags.use_teacher:
                if flags.teacher_kl_cost <= 0. and flags.teacher_baseline_cost <= 0.:
                    raise ValueError("It does not make sense to use teacher when teacher_kl_cost <= 0 "
                                     "and teacher_baseline_cost <= 0")

                device = 'cpu' if os.getenv("MAC") == '1' else i
                teacher_model = create_model(
                    flags,
                    device,
                    teacher_model_flags=teacher_flags,
                    is_teacher_model=True
                )
                if os.getenv("USE_TORCH_COMPILE") == "1":
                    teacher_model = torch.compile(teacher_model, fullgraph=True, dynamic=False)
                    print("TEACHER COMPILED")

                loadd_model(teacher_model, Path(os.path.dirname(__file__) + "/../" + "../") / Path(flags.teacher_load_dir) / flags.teacher_checkpoint_file)
                teacher_model.eval()
            else:
                teacher_model = None
                if flags.teacher_kl_cost > 0.:
                    logging.warning(f"flags.teacher_kl_cost is {flags.teacher_kl_cost}, but use_teacher is False. "
                                    f"Setting flags.teacher_kl_cost to 0.")
                if flags.teacher_baseline_cost > 0.:
                    logging.warning(f"flags.teacher_baseline_cost is {flags.teacher_baseline_cost}, but use_teacher is False. "
                                    f"Setting flags.teacher_baseline_cost to 0.")
                flags.teacher_kl_cost = 0.
                flags.teacher_kl_cost_change_per_step = 0.
                flags.teacher_baseline_cost = 0.

            teacher_models.append(teacher_model)
            teacher_model_names.append(flags.teacher_load_dir.split('/')[2])
    else:
        teacher_models = [None] * flags.n_actor_devices
        teacher_model_names = [''] * flags.n_actor_devices


    batch_type_by_actor_id = dict()

    borders = []
    borders.append((0., flags.frozen_actor_probability))
    borders.append((borders[-1][1], borders[-1][1] + flags.frozen_teacher_probability))
    borders.append((borders[-1][1], borders[-1][1] + flags.behavior_cloning_probability))
    borders.append((borders[-1][1], borders[-1][1] + flags.selfplay_probability + 1.))

    print("BORDERS: ", borders)

    torch_compile_lock = mp.Semaphore(8)
    start_lock = mp.Semaphore(8)

    def actor_thr(i):
        if i < flags.num_frozen_model_actors:
            batch_type_by_actor_id[i] = 'frozen_actor'
        elif i < flags.num_frozen_model_actors + flags.frozen_teacher_actors:
            batch_type_by_actor_id[i] = 'frozen_teacher'
        elif i < flags.num_frozen_model_actors + flags.frozen_teacher_actors + flags.behavior_cloning_actors:
            batch_type_by_actor_id[i] = 'behavior_cloning'
        else:
            batch_type_by_actor_id[i] = 'selfplay'
        cur_batch_type = batch_type_by_actor_id[i]
        actor_start = threading.Thread if flags.debug else mp.Process
        device = 'cpu' if os.getenv("MAC") == '1' else (i % flags.n_actor_devices)
        actor = actor_start(
            target=act_func,
            args=(
                flags,
                teacher_flags,
                i,
                free_queue[cur_batch_type],
                full_queue[cur_batch_type],
                buffers[cur_batch_type],
                stats_free_queue,
                stats_full_queue,
                stats_buffers,
                #len(frozen_actor_models[i % flags.n_actor_devices]) if i < flags.num_frozen_model_actors else 0,
                #change_frozen_actor_every_n_iters,
                True if (i >= flags.num_frozen_model_actors and i < flags.num_frozen_model_actors + flags.frozen_teacher_actors) else False, # is frozen teacher
                i >= flags.num_frozen_model_actors + flags.frozen_teacher_actors and
                i < flags.num_frozen_model_actors + flags.frozen_teacher_actors + flags.behavior_cloning_actors, # is behavior cloning
                teacher_models[i % flags.n_actor_devices],
                teacher_model_names[i % flags.n_actor_devices],
                frozen_actor_models[i % flags.n_actor_devices][i % len(frozen_actor_models[i % flags.n_actor_devices])] if i < flags.num_frozen_model_actors else None,
                frozen_actor_model_paths[i % flags.n_actor_devices][i % len(frozen_actor_models[i % flags.n_actor_devices])] if i < flags.num_frozen_model_actors else None,
                actor_models[i % flags.n_actor_devices],
                device,
                cur_batch_type,
                torch_compile_lock
            ),
            name=f"ACTOR_{i}_PROCESS"
        )
        with start_lock:
            actor.start()
        return actor

    with concurrent.futures.ThreadPoolExecutor(max_workers=flags.num_actors) as executor:
        for i in range(flags.num_actors):
            all_processes.append(executor.submit(actor_thr, i))

    if flags.replay:
        for p in all_processes:
            p.join()
        sys.exit(0)

    stats_process = mp.Process(
        target=stats_func,
        args=(flags,
              stats_free_queue,
              stats_full_queue,
              stats_buffers,
              stats_queue_learner, shared_multiplier, shared_entropy_multiplier_worker, shared_entropy_multiplier_sapper, shared_steps, shared_pos_weight,
              disk_params,
              warmup_end_step),
        name="STATS_PROCESS"
    )
    stats_process.start()
    all_processes.append(stats_process)

    print("START BATCH PREPARE")
    if True:
        n_actors_by_bt = {
            'frozen_actor': flags.num_frozen_model_actors,
            'frozen_teacher': flags.frozen_teacher_actors,
            'behavior_cloning': flags.behavior_cloning_actors,
            'selfplay': flags.num_actors - flags.num_frozen_model_actors - flags.frozen_teacher_actors - flags.behavior_cloning_actors,
        }
        for bt in batch_types:
            if n_actors_by_bt[bt] == 0:
                continue
            for i in range(flags.n_learner_devices * flags.n_batch_prepare_processes):
                device = 'cpu' if os.getenv("MAC") == '1' else (flags.n_actor_devices + i // flags.n_batch_prepare_processes)
                batch_prepare_process = mp.Process(
                    target=batch_prepare_process_func,
                    args=(flags, full_queue[bt], batch_queues[bt][i // flags.n_batch_prepare_processes], device, buffers[bt], learner_free_batch_queues[bt][i // flags.n_batch_prepare_processes], learner_gpu_buffers[i // flags.n_batch_prepare_processes][bt],
                          #free_queues,
                          free_queue[bt],bt),
                    name="BATCH_PREPARE_PROCESS"
                )
                batch_prepare_process.start()
                all_processes.append(batch_prepare_process)

    print("END BATCH PREPARE")

    for bt in batch_types:
        for i in range(n_buffers_by_bt[bt]):
            free_queue[bt].put(i)


    for i in range(flags.num_stats_buffers):
        stats_free_queue.put(i)

    logging.info("START ?")

    #if checkpoint_state is not None and flags.load_optimizer:
    #    try:
    #        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    #    except Exception as e:
    #        print(f"Checkpoint missing keys: {e}")

    cgn = torch.nn.utils.clip_grad_norm_

    if False and os.getenv("USE_TORCH_COMPILE") == "1":
        cgn = torch.compile(cgn, fullgraph=True, dynamic=False)

    assert flags.n_actor_devices <= flags.n_learner_devices

    if True:
        if os.getenv("MAC") == '1':
            learn_process = mp.Process(
                target=batch_and_learn,
                args=(0, flags, shared_steps, batch_queues, batch_types, borders, disk_params, actor_models[min(0, len(actor_models) - 1)], teacher_flags, warmup_end_step, shared_multiplier, shared_entropy_multiplier_worker, shared_entropy_multiplier_sapper, shared_pos_weight,
                      stats_queue_learner, cgn,
                      #free_queues,
                      free_queue, learner_free_batch_queues, learner_gpu_buffers[0]),
                name="LEARN_PROCESS"
            )
            learn_process.start()
            all_processes.append(learn_process)
        else:
            for i in range(flags.n_learner_devices):
                learn_process = mp.Process(
                    target=batch_and_learn,
                    args=(i, flags, shared_steps, batch_queues, batch_types, borders, disk_params, actor_models[min(i, len(actor_models) - 1)], teacher_flags, warmup_end_step, shared_multiplier, shared_entropy_multiplier_worker, shared_entropy_multiplier_sapper, shared_pos_weight,
                          stats_queue_learner, cgn,
                          free_queue, learner_free_batch_queues, learner_gpu_buffers[i]),
                    name="LEARN_PROCESS"
                )
                learn_process.start()
                all_processes.append(learn_process)



    def kill_all_children():
        current_process = psutil.Process(os.getpid())  # Get current process
        for child in current_process.children(recursive=True):  # Get all child processes
            print(f"Killing child process {child.pid}")
            child.terminate()  # Terminate each process

        _, alive = psutil.wait_procs(current_process.children(recursive=True), timeout=3)
        for child in alive:
            print(f"Force killing child process {child.pid}")
            child.kill()  # Force kill if needed

    def checkpoint(checkpoint_path: Union[str, Path]):
        print(f"Saving checkpoint to {checkpoint_path}")
        #torch.save(
        #    {
        #        "model_state_dict": actor_model.state_dict(),
        #        #"optimizer_state_dict": optimizer.state_dict(),
        #        #"scheduler_state_dict": scheduler.state_dict(),
        #        "step": int(shared_steps.value),
        #        "disk_params": dict(disk_params)
        #    },
        #    checkpoint_path + ".pt",
        #)
        torch.save(
            {
                "model_state_dict": actor_model.state_dict(),
            },
            checkpoint_path + "_weights.pt"
        )
        print(f"Saved checkpoint to {checkpoint_path}")

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while int(shared_steps.value) < flags.total_steps:
            start_step = int(shared_steps.value)
            start_time = timer()
            time.sleep(10)

            # Save every checkpoint_freq minutes
            if timer() - last_checkpoint_time > flags.checkpoint_freq * 60:
                cp_path = str(int(shared_steps.value)).zfill(int(math.log10(flags.total_steps)) + 1)
                checkpoint(cp_path)
                last_checkpoint_time = timer()

            sps = (int(shared_steps.value) - start_step) / (timer() - start_time)
            bps = (int(shared_steps.value) - start_step) / (t * b) / (timer() - start_time)
            print(f"{os.getcwd()}\nSteps {int(shared_steps.value):d} @ {sps:.1f} SPS / {bps:.1f} BPS.")

            if os.getenv("MAC") != '1':
                for bt in batch_types:
                    bqszs = ''
                    for i in range(flags.n_learner_devices):
                        bqszs += f' {batch_queues[bt][i].qsize()} '
                    print(f"\n  {bt} :    stats_q_learner: {stats_queue_learner.qsize()}, prepare_q: {bqszs}, full_queue: {full_queue[bt].qsize()}, free_queue: {free_queue[bt].qsize()}, stats_free_queue: {stats_free_queue.qsize()}, stats_full_queue: {stats_full_queue.qsize()}\n")

    except KeyboardInterrupt:
        # Try checkpointing and joining actors then quit.
        return
    else:
        print(f"Learning finished after {int(shared_steps.value):d} steps.")
    finally:
        cp_path = str(int(shared_steps.value)).zfill(int(math.log10(flags.total_steps)) + 1)
        checkpoint(cp_path)

        kill_all_children()

        shared_exit.value = 1

