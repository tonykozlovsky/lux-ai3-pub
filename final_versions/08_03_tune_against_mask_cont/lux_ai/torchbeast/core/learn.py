import os

from .stats import RollingAverage

import torch
import torch.nn as nn
import torch.optim
import torch.amp
import math
from types import SimpleNamespace
from typing import Dict, Optional

from ..profiler import ScopedProfiler
from .buffer_utils import buffers_apply
from .losses import compute_teacher_kl_loss, compute_teacher_kl_loss_GPT
from .losses import combine_policy_logits_to_log_probs, combine_policy_entropy, reduce, compute_policy_gradient_loss, compute_baseline_loss


rolling_grad = RollingAverage(window_size=10000)

def learn(
        device,
        flags: SimpleNamespace,
        learner_model: nn.Module,
        teacher_model: Optional[nn.Module],
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        warmup: bool = False,
        shared_multiplier = None,
        shared_entropy_multiplier_worker = None,
        shared_entropy_multiplier_sapper = None,
        profiler: ScopedProfiler = ScopedProfiler(enabled=False),
        stats_queue_learner=None,
        cgn=None,
        losses_func_selfplay=None,
        losses_func_frozen_actor=None,
        losses_func_frozen_teacher=None,
        losses_func_behavior_cloning=None,
        shared_pos_weight=None,
        batch_type=None,
):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=flags.use_bf16):
        multiplier = shared_multiplier.value
        if multiplier != 0.:
            batch['reward_GPU_CPU'] /= multiplier


        with profiler.block("flattened_batch"):
            flattened_batch = buffers_apply(batch, lambda x: torch.flatten(x, start_dim=0, end_dim=1))

        with profiler.block("learner_model"):
            learner_outputs = learner_model(flattened_batch)


        with profiler.block("learner_outputs"):
            learner_outputs = buffers_apply(learner_outputs, lambda x: x.view(flags.unroll_length + 1,
                                                                              flags.batch_size,
                                                                              *x.shape[1:]))

        teacher_output = None
        if flags.use_teacher and flags.use_old_teacher_loss:
            teacher_output = teacher_model(flattened_batch)
            teacher_output = buffers_apply(teacher_output, lambda x: x.view(flags.unroll_length + 1, flags.batch_size, *x.shape[1:]))


        # Take final value function slice for bootstrapping.
        with profiler.block("bootstrap_value"):
            bootstrap_value = learner_outputs["baseline_GPU"][-1]

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        with profiler.block("batch_apply_x"):
            batch = buffers_apply(batch, lambda x: x[1:])


        with profiler.block("leaner_apply_x"):
            learner_outputs = buffers_apply(learner_outputs, lambda x: x[:-1])

        if flags.use_teacher and flags.use_old_teacher_loss:
            teacher_output = buffers_apply(teacher_output, lambda x: x[:-1])


        optimizer.zero_grad()

        actor_kl_loss = None
        losses_func = None

        if batch_type == 'selfplay':
            losses_func = losses_func_selfplay

            vtrace_pg_loss, upgo_pg_loss, baseline_loss, teacher_kl_loss, teacher_baseline_loss, entropy_loss, total_loss, prediction_position_loss, prediction_near_loss, prediction_sensor_mask_loss, teacher_kl_loss_orig, teacher_baseline_loss_orig, imitation_loss, imitation_loss_orig = losses_func(
                device, profiler, flags, batch, learner_outputs, teacher_output, shared_entropy_multiplier_worker, shared_entropy_multiplier_sapper, bootstrap_value, warmup, shared_pos_weight
            )


            with torch.no_grad():
                combined_actor_kl_loss = torch.zeros(
                    (flags.unroll_length, flags.batch_size, 2),
                    device=device
                )
                for act_space in batch["actions_GPU"].keys():
                    actions_taken_mask = batch["info"]["actions_taken_GPU_CPU"][act_space]

                    learner_policy_logits = learner_outputs["policy_logits_GPU_CPU"][act_space]

                    actor_policy_logits = batch["policy_logits_GPU_CPU"][act_space]

                    any_actions_taken = actions_taken_mask.any(dim=-1)
                    actor_kl_loss_func = compute_teacher_kl_loss_GPT if flags.use_GPT_losses else compute_teacher_kl_loss
                    actor_kl_loss = actor_kl_loss_func(
                        learner_policy_logits,
                        actor_policy_logits,
                        any_actions_taken
                    )


                    combined_actor_kl_loss = combined_actor_kl_loss + actor_kl_loss
                actor_kl_loss = reduce(
                    combined_actor_kl_loss,
                    reduction=flags.reduction
                )
                #print("ACTOR KL: ", actor_kl_loss)

        elif batch_type == 'frozen_actor':
            losses_func = losses_func_frozen_actor

            vtrace_pg_loss, upgo_pg_loss, baseline_loss, teacher_kl_loss, teacher_baseline_loss, entropy_loss, total_loss, prediction_position_loss, prediction_near_loss, prediction_sensor_mask_loss, teacher_kl_loss_orig, teacher_baseline_loss_orig, imitation_loss, imitation_loss_orig = losses_func(
                device, profiler, flags, batch, learner_outputs, teacher_output, shared_entropy_multiplier_worker, shared_entropy_multiplier_sapper, bootstrap_value, warmup, shared_pos_weight
            )
        elif batch_type == 'frozen_teacher':
            losses_func = losses_func_frozen_actor

            zero_losses = flags.only_teacher_loss or flags.frozen_teacher_both_sides
            vtrace_pg_loss, upgo_pg_loss, baseline_loss, _, _, entropy_loss, total_loss1, prediction_position_loss, prediction_near_loss, prediction_sensor_mask_loss, _, _, imitation_loss, imitation_loss_orig = losses_func(
                device, profiler, flags, batch, learner_outputs, teacher_output, shared_entropy_multiplier_worker, shared_entropy_multiplier_sapper, bootstrap_value, warmup, shared_pos_weight, zero_losses
            )

            losses_func = losses_func_frozen_teacher
            _, _, _, teacher_kl_loss, teacher_baseline_loss, _, total_loss2, _, _, _, teacher_kl_loss_orig, teacher_baseline_loss_orig, _, _ = losses_func(
                device, profiler, flags, batch, learner_outputs, teacher_output, shared_entropy_multiplier_worker, shared_entropy_multiplier_sapper, bootstrap_value, warmup, shared_pos_weight
            )

            total_loss = total_loss1 + total_loss2
        elif batch_type == 'behavior_cloning':
            losses_func = losses_func_behavior_cloning

            vtrace_pg_loss, upgo_pg_loss, baseline_loss, teacher_kl_loss, teacher_baseline_loss, entropy_loss, total_loss, prediction_position_loss, prediction_near_loss, prediction_sensor_mask_loss, teacher_kl_loss_orig, teacher_baseline_loss_orig, imitation_loss, imitation_loss_orig = losses_func(
                device, profiler, flags, batch, learner_outputs, shared_entropy_multiplier_worker, shared_entropy_multiplier_sapper, bootstrap_value, warmup, shared_pos_weight
            )
        else:
            assert False


        total_loss.backward()





        clip_grad_value = min(rolling_grad.average() * 1.25, 100.)
        if clip_grad_value == 0.:
            clip_grad_value = 10.


        with profiler.block("clip_grad_norm"):
            total_norm = cgn(learner_model.parameters(), clip_grad_value)
        with profiler.block("clip_grad_norm_item"):
            tni = total_norm.item()
            #print("TNI: ", tni)
        with profiler.block("clip_grad_norm_isnan"):
            if not math.isnan(tni):
                rolling_grad.add(tni)

        with profiler.block("optimizer_step"):
            optimizer.step()


        if True:
            if True or ("stats_queue_two"):

                stats = {}
                if vtrace_pg_loss is not None:
                    stats['vtrace_pg_loss'] = vtrace_pg_loss.detach().cpu().item()
                if upgo_pg_loss is not None:
                    stats['upgo_pg_loss'] = upgo_pg_loss.detach().cpu().item()
                if baseline_loss is not None:
                    stats['baseline_loss'] = baseline_loss.detach().cpu().item()
                if teacher_kl_loss is not None:
                    stats['teacher_kl_loss'] = teacher_kl_loss.detach().cpu().item()
                if teacher_baseline_loss is not None:
                    stats['teacher_baseline_loss'] = teacher_baseline_loss.cpu().detach().item()
                if teacher_kl_loss_orig is not None:
                    stats['teacher_kl_loss_orig'] = teacher_kl_loss_orig.detach().cpu().item()
                if teacher_baseline_loss_orig is not None:
                    stats['teacher_baseline_loss_orig'] = teacher_baseline_loss_orig.cpu().detach().item()
                if imitation_loss is not None:
                    stats['imitation_loss'] = imitation_loss.detach().cpu().item()
                if imitation_loss_orig is not None:
                    stats['imitation_loss_orig'] = imitation_loss_orig.detach().cpu().item()
                if entropy_loss is not None:
                    stats['entropy_loss'] = entropy_loss.detach().cpu().item()
                if total_loss is not None:
                    stats['total_loss'] = total_loss.detach().cpu().item()
                if prediction_position_loss is not None:
                    stats['position_loss'] = prediction_position_loss.detach().cpu().item()
                if prediction_near_loss is not None:
                    stats['near_loss'] = prediction_near_loss.detach().cpu().item()
                if prediction_sensor_mask_loss is not None:
                    stats['sensor_mask_loss'] = prediction_sensor_mask_loss.detach().cpu().item()
                if actor_kl_loss is not None:
                    stats['actor_kl_loss'] = actor_kl_loss.detach().cpu().item()

                stats['learning_rate'] = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else 0.


                stats_queue_learner.put({'losses': stats,
                                         'clip_grad' : {'clip_grad_value': clip_grad_value, 'total_norm': tni},
                                         }

                                        )
        if not warmup and flags.enable_lr_scheduler and lr_scheduler is not None:
            lr_scheduler.step()
