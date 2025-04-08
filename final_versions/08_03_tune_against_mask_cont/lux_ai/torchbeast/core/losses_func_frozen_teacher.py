import torch

from .losses import combine_policy_logits_to_log_probs, combine_policy_entropy, reduce, compute_policy_gradient_loss, compute_baseline_loss
from .losses import combine_policy_logits_to_log_probs_GPT, combine_policy_entropy_GPT, compute_policy_gradient_loss_GPT, compute_baseline_loss_GPT
from . import vtrace
from . import td_lambda
from . import upgo
import torch.nn as nn

from .losses import compute_teacher_kl_loss, compute_teacher_kl_loss_GPT

def losses_func_frozen_teacher(rank, profiler, flags, batch, learner_outputs, teacher_output, shared_entropy_multiplier_worker, shared_entropy_multiplier_sapper, bootstrap_value, warmup, shared_pos_weight):

    if True or ("zeros"):
        combined_teacher_kl_loss = torch.zeros(
            (flags.unroll_length, flags.batch_size, 2 if flags.frozen_teacher_both_sides else 1),
            device=rank
        )


    teacher_output = batch['teacher_output']


    if True or ("combine"):
        for act_space in batch["actions_GPU"].keys():
            actions_taken_mask = batch["info"]["actions_taken_GPU_CPU"][act_space]

            if not flags.frozen_teacher_both_sides:
                actions_taken_mask = actions_taken_mask[:, :, :, 1, ...].unsqueeze(3)

            learner_policy_logits = learner_outputs["policy_logits_GPU_CPU"][act_space]

            if not flags.frozen_teacher_both_sides:
                learner_policy_logits = learner_policy_logits[:, :, :, 1, ...].unsqueeze(3)

            teacher_policy_logits = teacher_output["policy_logits_GPU_CPU"][act_space]

            if not flags.frozen_teacher_both_sides:
                teacher_policy_logits = teacher_policy_logits[:, :, :, 1, ...].unsqueeze(3)

            # Only take entropy and KL loss for tiles where at least one action was taken

            any_actions_taken = actions_taken_mask.any(dim=-1)
            teacher_kl_loss_func = compute_teacher_kl_loss_GPT if flags.use_GPT_losses else compute_teacher_kl_loss
            teacher_kl_loss = teacher_kl_loss_func(
                learner_policy_logits,
                teacher_policy_logits,
                any_actions_taken
            )


            combined_teacher_kl_loss = combined_teacher_kl_loss + teacher_kl_loss




    values = learner_outputs["baseline_GPU"]
    if not flags.frozen_teacher_both_sides:
        values = values[:, :, 1].unsqueeze(2)

    compute_baseline_loss_func = compute_baseline_loss_GPT if flags.use_GPT_losses else compute_baseline_loss

    teacher_kl_loss_orig = reduce(
        combined_teacher_kl_loss,
        reduction=flags.reduction
    ) # one value tensor
    teacher_kl_loss = flags.teacher_kl_cost * teacher_kl_loss_orig


    teacher_baseline = teacher_output["baseline_GPU"]
    if not flags.frozen_teacher_both_sides:
        teacher_baseline = teacher_baseline[:, :, 1].unsqueeze(2)

    teacher_baseline_loss_orig = compute_baseline_loss_func(
        values,
        teacher_baseline,
        reduction=flags.reduction
    )
    teacher_baseline_loss = flags.teacher_baseline_cost * teacher_baseline_loss_orig

    total_loss = teacher_kl_loss + teacher_baseline_loss

    if warmup:
        total_loss *= 0

    return None, None, None, teacher_kl_loss, teacher_baseline_loss, None, total_loss, None, None, None, teacher_kl_loss_orig, teacher_baseline_loss_orig, None, None
