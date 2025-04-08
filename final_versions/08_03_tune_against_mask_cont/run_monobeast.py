import tempfile
import torch
torch.set_float32_matmul_precision('high')
torch._dynamo.config.cache_size_limit = 64
torch.backends.cudnn.benchmark = True
#torch._inductor.config.max_autotune = False

#import torch._inductor.config
#torch._inductor.config.use_triton = False  # ✅ Some versions require this too

import logging
import os
from pathlib import Path

import hydra
import wandb
from lux_ai.torchbeast.monobeast import train
from lux_ai.torchbeast.core.bench import bench
from lux_ai.utils import flags_to_namespace
from omegaconf import DictConfig, OmegaConf
from torch import multiprocessing as mp

import jax

jax.config.update("jax_compilation_cache_dir", f"{tempfile.gettempdir()}/jax_cache")
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.02)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TORCH_MATMUL_PRECISION"] = "high"
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(tempfile.gettempdir(), 'torch_inductor_cache') # persistent compile cache

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(asctime)s] " "%(message)s"
    ),
    level=logging.DEBUG,
)
# set this to INFO or DEBUG to see more about torch compilation
logging.getLogger("torch").setLevel(logging.WARN)
# this suppresses a lot of lines like "Attempting to release lock 281443567413248 on /tmp/torch_inductor_cache/locks/ck6m7mcqt5wtaafdhxlwdmslw3zlpuejnvegbp5aepxw65jh2axp.lock"
logging.getLogger("filelock").setLevel(logging.WARN)

logging.getLogger("jax").setLevel(logging.WARN)


def get_default_flags(flags: DictConfig) -> DictConfig:
    flags = OmegaConf.to_container(flags)
    # Env params
    flags.setdefault("seed", None)
    flags.setdefault("obs_space_kwargs", {})
    flags.setdefault("reward_space_kwargs", {})

    # Training params
    flags.setdefault("use_bf16", True)
    flags.setdefault("discounting", 0.999)
    flags.setdefault("reduction", "mean")
    flags.setdefault("clip_grads", 10.)
    flags.setdefault("checkpoint_freq", 5.)
    flags.setdefault("num_learner_threads", 1)
    flags.setdefault("use_teacher", False)
    flags.setdefault("teacher_baseline_cost", flags.get("teacher_kl_cost", 0.) / 2.)

    # Reloading previous run params
    flags.setdefault("replay", False)
    flags.setdefault("load_dir", None)
    flags.setdefault("checkpoint_file", None)
    flags.setdefault("weights_only", False)
    flags.setdefault("n_warmup_steps", 0)

    # Miscellaneous params
    flags.setdefault("disable_wandb", False)
    flags.setdefault("debug", False)

    return OmegaConf.create(flags)


def validate_flags(flags):
    assert flags.batch_size % flags.n_actor_envs == 0, "Batch size must be divisible by number of envs per actor"

@hydra.main(config_path="conf", config_name="two_gpu", version_base="1.1")
def main(flags: DictConfig):
    cli_conf = OmegaConf.from_cli()
    if Path("config.yaml").exists():
        new_flags = OmegaConf.load("config.yaml")
        flags = OmegaConf.merge(new_flags, cli_conf)

    if False and flags.get("load_dir", None) and not flags.get("weights_only", False):
        assert False
        # this ignores the local config.yaml and replaces it completely with saved one
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_steps=N before and want to increase it
        print("Loading existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load(Path(flags.load_dir) / "config.yaml")
        # Overwrite some parameters
        new_flags = OmegaConf.merge(new_flags, flags)
        flags = OmegaConf.merge(new_flags, cli_conf)

    flags = get_default_flags(flags)
    if os.getenv("SMALL") == "1": # NOT FULLY SUPPORTED FOR ALL BATCH TYPES
        flags.num_actors = 3
        flags.num_buffers = 3
        flags.num_stats_buffers = 1
        flags.n_actor_envs = 1
        flags.batch_size = 1
        flags.prepare_batches = 1
        flags.disable_wandb = True

        if flags.num_frozen_model_actors > 0:
            flags.num_frozen_model_actors = 1
        if flags.num_frozen_model_actors_buffers > 0:
            flags.num_frozen_model_actors_buffers = 1

        if flags.frozen_teacher_actors > 0:
            flags.frozen_teacher_actors = 1
        if flags.frozen_teacher_models_buffers > 0:
            flags.frozen_teacher_models_buffers = 1

        flags.behavior_cloning_actors = 0 # NOT WORKING YET


        # guarantee batches of different types ratio not depending on num_actors
        #flags.frozen_actor_probability = 0.33
        #flags.frozen_teacher_probability = 1. # NOT TESTED PROPERLY
        #flags.behavior_cloning_probability = 0. # NOT WORKING YET
        #flags.selfplay_probability = 1. # all remaining probability is for selfplay

        flags.num_bench_envs = 1
        flags.num_competitors = 1

        flags.n_actor_devices = 1
        flags.n_learner_devices = 1


    if os.getenv("MAC") == "1":
        flags.n_learner_devices = 1 # 4
        flags.n_actor_devices = 1 # 4
        flags.sharing_strategy = "file_system"

    if os.getenv("BENCHMARK") == "1":
        flags.reward_space = "GameResultReward"
        flags.batch_size = flags.num_bench_envs
        flags.n_actor_envs = flags.num_bench_envs

    validate_flags(flags)

    print(OmegaConf.to_yaml(flags, resolve=True))
    OmegaConf.save(flags, "config.yaml")

    flags = flags_to_namespace(OmegaConf.to_container(flags, resolve=True))

    mp.set_sharing_strategy(flags.sharing_strategy)

    lib_name = "mylib.so"
    exec_file = "../../../cpp/main.cpp"
    os.system("g++ -shared -fPIC -o {} {} -O3 -std=c++17".format(os.path.join(os.getcwd(), lib_name), exec_file))
    #os.system(f"pip install {os.path.join(os.getcwd(), '../../../../Lux-Design-S3/src/')}")

    if os.getenv("BENCHMARK") == "1":
        bench(flags)
    else:
        train(flags)


if __name__ == "__main__":
    os.setpgrp()
    mp.set_start_method("spawn", force=True)
    main()
