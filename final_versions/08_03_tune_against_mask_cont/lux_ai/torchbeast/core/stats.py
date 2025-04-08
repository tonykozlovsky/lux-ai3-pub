import torch
import os
import time
import setproctitle
from collections import deque
import threading
import wandb
import logging

from .buffer_utils import buffers_apply
from .losses import combine_policy_entropy, combine_policy_entropy_GPT
from .losses import reduce
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

import numpy as np
import math

import torch.nn as nn

import tempfile

class RollingAverage:
    def __init__(self, window_size):
        self.window = deque(maxlen=window_size)
        self.total = 0.0

    def add(self, value):
        if len(self.window) == self.window.maxlen:
            self.total -= self.window[0]
        self.window.append(value)
        self.total += value

    def average(self):
        return self.total / len(self.window) if self.window else 0.0


losses = None
clip_grad = None
lock = threading.Lock()


def init_wandb(flags):
    if not flags.disable_wandb:
        wandb.init(
            config=flags.wandb,  # Converts flags object to a dictionary
            project=flags.wandb.get("project", "default_project"),
            entity=flags.wandb.get("entity"),
            group=flags.wandb.get("group"),
            name=flags.name,
        )
        logging.info("WANDB INIT")


def fill_losses(stats_queue_learner):
    global losses
    global clip_grad
    try:
        while True:
            learner_batch = stats_queue_learner.get()
            #print("GET BATCH L")
            with lock:
                if 'losses' in learner_batch:
                    losses = learner_batch['losses']
                    clip_grad = learner_batch['clip_grad']
    except KeyboardInterrupt:
        return




def fix_keys(stats):
    result = {}

    def recursive(s, prefix):
        if isinstance(s, dict):
            for key, value in s.items():
                if isinstance(value, dict):
                    recursive(value, prefix + key + '.')
                else:
                    result[prefix + key] = value

    recursive(stats, '')

    return result

def smoothed_update(smoothed, stats):
    for key, value in stats.items():
        if key not in smoothed:
            if 'agg_rewards' not in key and 'game_win' not in key and 'match_win' not in key:
                smoothed[key] = RollingAverage(window_size=100)
            else:
                smoothed[key] = RollingAverage(window_size=20)

        smoothed[key].add(value)

def smoothed_get_dict(smoothed, stats):
    result = {key: smoothed[key].average() for key in smoothed}
    for key, value in stats.items():
        result[key + '_RAW_'] = value

    return result



def disk_params_update(disk_params):
    try:
        while True:
            time.sleep(1)
            try:
                try:
                    with open('params.txt', 'r') as f:
                        data = f.read()
                    data = data.split('\n')
                except Exception:
                    print("Creating new params.txt")
                    with open('params.txt', 'w') as f:
                        for key, value in disk_params.items():
                            f.write(f"{key} {value:.12f} .\n")
                        f.flush()
                        os.fsync(f.fileno())
                    continue

                if len(data) < 0:
                    continue

                with open('params.txt', 'w') as f:
                    for line in data:
                        line = line.split(' ')
                        if len(line) != 3:
                            continue
                        name = line[0]
                        value = float(line[1])
                        update = line[2]
                        if name in disk_params and update == 'u':
                            disk_params[name] = value
                        f.write(f"{name} {disk_params[name]:.10f} _\n")
                    f.flush()
                    os.fsync(f.fileno())




            except Exception:
                continue

    except KeyboardInterrupt:
        return


@torch.no_grad()
def stats_func(
        flags,
        stats_free_queue,
        stats_full_queue,
        stats_buffers,
        stats_queue_learner,
        shared_multiplier,
        shared_entropy_multiplier_worker,
        shared_entropy_multiplier_sapper,
        shared_steps,
        shared_pos_weight,
        disk_params,
        warmup_end_step
):
    all_threads = []
    try:
        setproctitle.setproctitle("STATS_PROCESS")
        global losses
        global clip_grad

        smoothed = dict()

        init_wandb(flags)

        rolling_avg = RollingAverage(window_size=10000)

        print("stats_func started")



        prev_steps = None
        prev_steps_worker = None
        prev_steps_sapper = None

        n_batches = 0

        prediction_averages = []


        for i in range(3):
            prediction_averages.append({'positive': RollingAverage(window_size=1000), 'negative': RollingAverage(window_size=1000)})


        lt = threading.Thread(target=fill_losses, args=(stats_queue_learner,))
        lt.start()
        all_threads.append(lt)


        disk_params_update_thread = threading.Thread(target=disk_params_update, args=(disk_params,))
        disk_params_update_thread.start()
        all_threads.append(disk_params_update_thread)

        while True:
            batch_idx, batch_type = stats_full_queue.get()
            batch = stats_buffers[batch_idx]

            steps = shared_steps.value

            if prev_steps == None:
                prev_steps = steps
                prev_steps_worker = steps
                prev_steps_sapper = steps
            delta_steps = 0 if steps < warmup_end_step else (steps - prev_steps)
            prev_steps = steps

            disk_params['target_entropy_worker'] += disk_params['target_entropy_worker_change_per_step'] * delta_steps
            disk_params['target_entropy_sapper'] += disk_params['target_entropy_sapper_change_per_step'] * delta_steps

            disk_params['teacher_kl_cost'] += disk_params['teacher_kl_cost_change_per_step'] * delta_steps

            disk_params['lr_lambda'] += disk_params['lr_lambda_change_per_step'] * delta_steps

            n_batches += 1

            avg_reward = rolling_avg.average()

            if flags.enable_multiplier_scaling:
                multiplier = avg_reward / 8.
            else:
                multiplier = 1.

            shared_multiplier.value = multiplier

            if multiplier != 0.:
                batch['reward_GPU_CPU'] /= multiplier

            batch = buffers_apply(batch, lambda x: x[1:])

            agg_rewards = batch['info']['LOGGING_CPU_agg_rewards']
            valid_values = agg_rewards[batch["done_GPU_CPU"]][~agg_rewards[batch["done_GPU_CPU"]].isnan()]
            if len(valid_values) > 0:
                for value in valid_values.detach().cpu():
                    if value.item() > 0:
                        rolling_avg.add(value.item())

            entropies = {}
            attss = {}

            if batch_type == 'default':
                for act_space, learner_policy_logits in batch['policy_logits_GPU_CPU'].items():
                    actions_taken_mask = batch["info"]["actions_taken_GPU_CPU"][act_space]

                    any_actions_taken = actions_taken_mask.any(dim=-1)

                    combine_policy_entropy_func = combine_policy_entropy_GPT if flags.use_GPT_losses else combine_policy_entropy
                    learner_policy_entropy = combine_policy_entropy_func(
                        learner_policy_logits,
                        any_actions_taken
                    )

                    aats = any_actions_taken.sum().item()

                    attss[act_space] = aats

                    current_entropy = (reduce(
                        learner_policy_entropy,
                        reduction="sum"
                    ) / max(1, aats)).cpu().item()

                    entropies[act_space] = current_entropy

                    if aats > 0:

                        if act_space == 'worker':

                            delta_steps_worker = 0 if steps < warmup_end_step else (steps - prev_steps_worker)
                            prev_steps_worker = steps
                            multiplier_change_worker = disk_params['entropy_mult_change_speed_per_step'] * delta_steps_worker

                            if current_entropy > disk_params['target_entropy_worker']:
                                disk_params['entropy_multiplier_worker'] *= (1 - multiplier_change_worker)
                                if disk_params['entropy_multiplier_worker'] <= 1e-5:
                                    disk_params['entropy_multiplier_worker'] = 0.
                            else:
                                disk_params['entropy_multiplier_worker'] *= (1 + multiplier_change_worker)
                                if disk_params['entropy_multiplier_worker'] < 1e-5:
                                    disk_params['entropy_multiplier_worker'] = 1e-5
                            shared_entropy_multiplier_worker.value = disk_params['entropy_multiplier_worker']

                        elif act_space == 'sapper':

                            delta_steps_sapper = 0 if steps < warmup_end_step else (steps - prev_steps_sapper)
                            prev_steps_sapper = steps
                            multiplier_change_sapper = disk_params['entropy_mult_change_speed_per_step'] * delta_steps_sapper

                            if current_entropy > disk_params['target_entropy_sapper']:
                                disk_params['entropy_multiplier_sapper'] *= (1 - multiplier_change_sapper)
                                if disk_params['entropy_multiplier_sapper'] <= 1e-5:
                                    disk_params['entropy_multiplier_sapper'] = 0.
                            else:
                                disk_params['entropy_multiplier_sapper'] *= (1 + multiplier_change_sapper)
                                if disk_params['entropy_multiplier_sapper'] < 1e-5:
                                    disk_params['entropy_multiplier_sapper'] = 1e-5
                            shared_entropy_multiplier_sapper.value = disk_params['entropy_multiplier_sapper']
                        else:
                            assert False


            #print("ACTOR:", entropies['worker'], entropies['sapper'], attss['worker'], attss['sapper'])


            if n_batches % flags.stats_freq_batches == 0:
                logging_result = {}
                mask = batch["done_GPU_CPU"]
                plus_games = mask.detach().sum().item()
                if plus_games > 0:
                    def get_logging_result_fast(batch, mask, bt):
                        simplified_keys = []
                        values = []
                        mults = []
                        for key, val in batch["info"].items():
                            # Process only keys that start with "LOGGING_" and do not contain "ACTIONS_"
                            if key.startswith("LOGGING_") and "ACTIONS_" not in key:
                                # Extract the portion of the key after the 8th character
                                simplified_keys.append(key[8+4:] + bt)

                                values.append(val.detach())
                                mults.append(1 if (multiplier == 0. or key != 'LOGGING_CPU_agg_rewards') else multiplier)

                        data = torch.stack(values, dim=0)
                        masked_data = data[:, mask]
                        masked_data = masked_data.squeeze(-1)
                        valid_mask = ~masked_data.isnan()  # shape: [N, number_of_True_in_mask]
                        found_any_valid = valid_mask.any(dim=1)  # shape: [N]

                        # If there are valid entries, pick the index of the *first* valid one.
                        #    (torch.argmax returns the *first* occurrence of the max value; for booleans, max=True=1.)
                        first_valid_idx = torch.argmax(valid_mask.int(), dim=1)  # shape: [N]

                        # 9) Gather those "first valid values" into a single 1D result
                        #    Initialize all as nan; fill where found_any_valid is True.
                        N = len(values)
                        result = torch.full((N,), float('nan'), device=mask.device)#, dtype=torch.bfloat16)
                        idx_range = torch.arange(N, device=mask.device)
                        result[found_any_valid] = masked_data[idx_range[found_any_valid], first_valid_idx[found_any_valid]]#.bfloat16()

                        return {key: value.item() / mult for key, value, mult in zip(simplified_keys, result.cpu(), mults) if not torch.isnan(value)}

                    lr_fast = get_logging_result_fast(batch, mask, '')
                    lr_fast_by_type = get_logging_result_fast(batch, mask, '_' + batch_type)

                    logging_result = lr_fast
                    for key, value in lr_fast_by_type.items():
                        if 'agg_rewards' not in key and 'game_win' not in key and 'match_win' not in key:
                            continue
                        logging_result[key] = value

                stats = {
                    "Env": logging_result,
                    "Loss": {},
                    "Entropy": {},
                    "Misc": {
                        "batch_reward_sum": batch["reward_GPU_CPU"].sum().item(),
                        "batch_reward_max": batch["reward_GPU_CPU"].max().item(),
                        "batch_reward_mean": batch["reward_GPU_CPU"].mean().item(),
                        "multiplier": multiplier,
                        "entropy_mult_worker": disk_params['entropy_multiplier_worker'] ,
                        "target_entropy_worker": disk_params['target_entropy_worker'] ,
                        "entropy_mult_sapper": disk_params['entropy_multiplier_sapper'] ,
                        "target_entropy_sapper": disk_params['target_entropy_sapper'] ,
                        "lmb": disk_params['lmb'],
                        "teacher_kl_cost": disk_params['teacher_kl_cost'],
                        "teacher_kl_cost_change_per_step": disk_params['teacher_kl_cost_change_per_step'],
                        "prediction_cost":disk_params['prediction_cost']
                    },
                }

                if len(entropies.values()) > 0:
                    stats["Entropy"]["overall"] = sum(e for e in entropies.values())

                for key, val in entropies.items():
                    if val > 0:
                        stats["Entropy"][key] = val

                with lock:
                    if losses != None:

                        stats['Misc']['learning_rate'] = losses['learning_rate']

                        stats['Misc']['total_norm'] = clip_grad['total_norm']
                        stats['Misc']['clip_grad_value'] = clip_grad['clip_grad_value']

                        if 'vtrace_pg_loss' in losses:
                            stats["Loss"]["vtrace_pg_loss"] = losses['vtrace_pg_loss']

                        if 'upgo_pg_loss' in losses:
                            stats["Loss"]["upgo_pg_loss"] = losses['upgo_pg_loss']

                        if 'baseline_loss' in losses:
                            stats["Loss"]["baseline_loss"] = losses['baseline_loss']

                        if 'teacher_kl_loss' in losses:
                            stats["Loss"]["teacher_kl_loss"] = losses['teacher_kl_loss']

                        if 'teacher_baseline_loss' in losses:
                            stats["Loss"]["teacher_baseline_loss"] = losses['teacher_baseline_loss']

                        if 'teacher_kl_loss_orig' in losses:
                            stats["Loss"]["teacher_kl_loss_orig"] = losses['teacher_kl_loss_orig']

                        if 'teacher_baseline_loss_orig' in losses:
                            stats["Loss"]["teacher_baseline_loss_orig"] = losses['teacher_baseline_loss_orig']

                        if 'imitation_loss' in losses:
                            stats["Loss"]["imitation_loss"] = losses['imitation_loss']

                        if 'imitation_loss_orig' in losses:
                            stats["Loss"]["imitation_loss_orig"] = losses['imitation_loss_orig']

                        if 'entropy_loss' in losses:
                            stats["Loss"]["entropy_loss"] = -losses['entropy_loss']

                        if 'total_loss' in losses:
                            stats["Loss"]["total_loss"] = losses['total_loss']

                        if 'position_loss' in losses:
                            stats["Loss"]["prediction_position_loss"] = losses['position_loss']

                        if 'near_loss' in losses:
                            stats["Loss"]["prediction_near_loss"] = losses['near_loss']

                        if 'sensor_mask_loss' in losses:
                            stats["Loss"]["prediction_sensor_mask_loss"] = losses['sensor_mask_loss']

                        if 'actor_kl_loss' in losses:
                            stats["Loss"]["actor_kl_loss"] = losses['actor_kl_loss']

                        # Batch before : # obs[0], actions[0] | obs[1], actions[0] | obs[2] | actions[1] ...
                        # Batch now : # obs[1], actions[0] | obs[2], actions[1] | obs[3], actions[2] ...

                        if batch_type == 'default':

                            learner_outputs = batch['prediction_GPU_CPU'].cpu()[1:]
                            ground_truth = batch['info']['GPU1_available_actions_mask']['ground_truth_CPU'].cpu()[:-1]

                            # align learner_outputs with ground_truth

                            names = ['unit_position', 'near_position', 'sensor_mask']

                            for i in range(3):
                                targets = ground_truth[1:, :, :, i, :, :].flatten().cpu().numpy()
                                prediction_averages[i]['positive'].add(np.sum(targets))
                                prediction_averages[i]['negative'].add(np.sum(1 - targets))

                            for i in range(3):
                                name = names[i]

                                # Flatten predictions and ground truth
                                probs = torch.sigmoid(learner_outputs[:-1, :, :, i, :, :].flatten().float()).cpu().numpy()
                                loss_gt = ground_truth[1:, :, :, i, :, :]
                                targets = loss_gt.flatten().cpu().numpy()

                                positive_avg_sum = prediction_averages[i]['positive'].average()
                                negative_avg_sum = prediction_averages[i]['negative'].average()
                                shared_pos_weight[i].value = (negative_avg_sum / positive_avg_sum) if positive_avg_sum > 0 else 1.

                                prediction_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(shared_pos_weight[i].value, device='cpu'))
                                prediction_position_loss = flags.prediction_cost * prediction_loss_func(
                                    learner_outputs[:-1, :, :, i, :, :].flatten().float(),
                                    loss_gt.flatten().float(),
                                )

                                # Convert to binary (0 or 1)
                                predictions = (probs > 0.5).astype(float)

                                correct = (predictions == targets).sum().item()
                                total = targets.size
                                accuracy = correct / total

                                # Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
                                true_positives = (predictions * targets).sum().item()  # TP: 1 in predictions and targets
                                false_positives = (predictions * (1 - targets)).sum().item()  # FP: 1 in predictions but 0 in targets
                                false_negatives = ((1 - predictions) * targets).sum().item()  # FN: 0 in predictions but 1 in targets

                                # Precision and Recall
                                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
                                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

                                # Compute F1-score
                                f1 = f1_score(targets, predictions, zero_division=1)

                                # Compute Precision-Recall AUC (PR-AUC)
                                try:
                                    pr_auc = average_precision_score(targets, probs)  # Uses raw probabilities
                                except ValueError:  # Handle edge cases where only one class exists
                                    pr_auc = None

                                # Compute ROC AUC
                                try:
                                    auc = roc_auc_score(targets, probs)  # Use raw probabilities, not binary
                                except ValueError:  # Handle edge cases where only one class exists
                                    auc = None

                                # Store metrics
                                stats["Misc"][f"stats_loss_{name}"] = prediction_position_loss
                                stats["Misc"][f"pos_weight_{name}"] = shared_pos_weight[i].value
                                stats["Misc"][f"accuracy_{name}"] = accuracy
                                stats["Misc"][f"precision_{name}"] = precision
                                stats["Misc"][f"recall_{name}"] = recall
                                stats["Misc"][f"f1_{name}"] = f1
                                if pr_auc is not None and not math.isnan(pr_auc):
                                    stats["Misc"][f"pr_auc_{name}"] = pr_auc
                                if auc is not None and not math.isnan(auc):
                                    stats["Misc"][f"roc_auc_{name}"] = auc






                        losses = None


                stats = fix_keys(stats)
                smoothed_update(smoothed, stats)
                stats = smoothed_get_dict(smoothed, stats)
                #print(stats)
                if not flags.disable_wandb:
                    wandb.log(stats, step=int(shared_steps.value))

            #storage_id = batch['storage_id_GPU_CPU'][0, 0].item()
            #actor_index = storage_id % 10000 // 10
            #print("STATS STORAGE ID:", storage_id)

            #assert checksum_buffers(batch).item() == batch['checksum_GPU_CPU'][0, 0].item(), f"{checksum_buffers(batch).item()} != {batch['checksum_GPU_CPU'][0, 0].item()}"

            #storage_queues[actor_index].put(storage_id)
            stats_free_queue.put(batch_idx)
    except KeyboardInterrupt:
        return
    finally:
        for t in all_threads:
            t.join()