import setproctitle
import threading
from .load_model import loadd_model
from ...nns import create_model
from .compete import compete_func

import torch
import os
import torch.multiprocessing as mp

from pathlib import Path
import time
import signal
import logging
import traceback
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor
from ...utils import flags_to_namespace
from typing import Optional


def create_model_from_path(device, path, checkpoint_name: Optional[str] = None):
    base_path = Path(os.path.dirname(__file__)).parent.parent.parent / path
    flags = OmegaConf.load(base_path / 'config.yaml')
    flags = flags_to_namespace(OmegaConf.to_container(flags, resolve=True))
    model = create_model(flags, device, teacher_model_flags=None, is_teacher_model=False)
    if os.getenv("USE_TORCH_COMPILE") == "1":
        model = torch.compile(model)
    if checkpoint_name is not None:
        loadd_model(model, base_path / checkpoint_name)
    return model, flags


def bench(flags):
    setproctitle.setproctitle(f"MAIN_PROCESS")

    device = 0

    baseline_model, baseline_flags = create_model_from_path(device, flags.benchmark_baseline_dir, flags.benchmark_baseline_file)
    baseline_model.eval()
    baseline_model.share_memory()  # Prepare for multiprocessing

    competitor_model, competitor_flags = create_model_from_path(device, flags.competitors_dir)
    competitor_model.eval()
    competitor_model.share_memory()

    shared_finish = mp.Value('i', 0)
    shared_n_games = mp.Value('i', 0)
    shared_first_player_wins = mp.Value('i', 0)
    competition_queue = mp.SimpleQueue()
    finished_queue = mp.SimpleQueue()


    def process_starter(i):
        compete_process = mp.Process(
            target=compete_func,
            args=(
                flags,
                baseline_flags,
                competitor_flags,
                None,
                i,

                baseline_model,
                competitor_model,
                competition_queue,
                finished_queue,

                shared_finish,
                shared_n_games,
                shared_first_player_wins,
                device
            ),
            name="COMPETE_PROCESS"
        )
        compete_process.start()
        return compete_process

    with ThreadPoolExecutor(max_workers=flags.num_competitors) as executor:
        compete_processes = list(executor.map(process_starter, range(flags.num_competitors)))

    current_baseline = Path(flags.benchmark_baseline_dir) / flags.benchmark_baseline_file
    current_competitor = None
    competitors_path = Path(os.path.dirname(__file__)).parent.parent.parent / flags.competitors_dir

    try:
        competitor_index = None
        prev_n_games = -1
        while True:
            time.sleep(1)
            files = os.listdir(competitors_path)
            files = [f for f in files if os.path.isfile(competitors_path / f) and f.endswith('_weights.pt')]

            competitor_checkpoints = [(int(f.split('.')[0].replace('_weights', '')), f) for f in files]

            if not files:
                logging.info(f"No checkpoints in {competitors_path}")
                continue

            competitor_checkpoints = sorted(competitor_checkpoints, key=lambda x: x[0])

            if current_competitor:
                baseline_wr = 100 * shared_first_player_wins.value / max(1, shared_n_games.value)
                competitor_wr = 100 - baseline_wr
                if shared_n_games.value != prev_n_games:
                    logging.info(f"{current_baseline} VS {current_competitor} (competitor_index: {competitor_index}), games: {shared_n_games.value}, winrate: {baseline_wr:.1f} / {competitor_wr:.1f}")
                    prev_n_games = shared_n_games.value

            if current_competitor is None or shared_n_games.value >= flags.games_to_test:
                if current_competitor is not None:
                    shared_finish.value = 1

                    for _ in range(flags.num_competitors):
                        finished_queue.get()

                    baseline_wr = 100 * shared_first_player_wins.value / max(1, shared_n_games.value)
                    competitor_wr = 100 - baseline_wr
                    logging.info(f"FINAL {current_baseline} VS {current_competitor} (competitor_index: {competitor_index}), games: {shared_n_games.value}, winrate: {baseline_wr:.1f} / {competitor_wr:.1f}")

                    with open(competitors_path / 'benchmark_results.txt', 'a') as f:
                        f.write(f'{current_baseline} vs {current_competitor}, games: {shared_n_games.value} : {baseline_wr:.2f} / {competitor_wr:.2f}\n')

                    if not flags.fixed_baseline and competitor_wr > 52:
                        loadd_model(baseline_model, current_competitor)
                        current_baseline = current_competitor

                if flags.rerun:
                    if competitor_index is None:
                        if hasattr(flags, 'start_from_checkpoint'):
                            for i, (_, checkpoint) in enumerate(competitor_checkpoints):
                                if checkpoint == flags.start_from_checkpoint:
                                    competitor_index = i
                                    break

                            assert competitor_index is not None, f"Checkpoint {flags.start_from_checkpoint} not found"
                        else:
                            competitor_index = 0 # starts from first checkpoint if not specified
                    else:
                        competitor_index += 1
                        if competitor_index >= len(competitor_checkpoints):
                            break
                else:
                    if len(competitor_checkpoints) - 1 == competitor_index:
                        logging.info(f"No new checkpoints to test, sleeping")
                        current_competitor = None
                        continue

                    competitor_index = len(competitor_checkpoints) - 1

                loadd_model(competitor_model, competitors_path / competitor_checkpoints[competitor_index][1])
                current_competitor = Path(flags.competitors_dir) / competitor_checkpoints[competitor_index][1]

                logging.info(f"LOADED {current_baseline} VS {current_competitor}")

                shared_finish.value = 0
                shared_n_games.value = 0
                shared_first_player_wins.value = 0
                for i in range(flags.num_competitors):
                    competition_queue.put(i)

    except Exception as e:
        logging.error(traceback.format_exc())
        raise
    finally:
        os.killpg(0, signal.SIGKILL)
        pass


