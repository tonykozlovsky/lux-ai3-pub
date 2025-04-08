import torch

from .losses import combine_policy_logits_to_log_probs, combine_policy_entropy, reduce, compute_policy_gradient_loss, compute_baseline_loss
from .losses import combine_policy_logits_to_log_probs_GPT, combine_policy_entropy_GPT, compute_policy_gradient_loss_GPT, compute_baseline_loss_GPT
from . import vtrace
from . import td_lambda
from . import upgo
import torch.nn as nn

def losses_func_behavior_cloning(rank, profiler, flags, batch, learner_outputs, teacher_output, shared_entropy_multiplier_worker, shared_entropy_multiplier_sapper, bootstrap_value, warmup, shared_pos_weight):

    if True or ("zeros"):
        combined_imitation_loss = torch.zeros(
            (flags.unroll_length, flags.batch_size, 1),
            device=rank
        )


    teacher_output = batch['teacher_output']


    if True or ("combine"):
        for act_space in batch["actions_GPU"].keys():
            actions_taken_mask = batch["info"]["actions_taken_GPU_CPU"][act_space][:, :, :, 1, ...].unsqueeze(3) # worker (17 3 1 2 16 6) sapper (17 3 1 2 16 225)

            learner_policy_logits = learner_outputs["policy_logits_GPU_CPU"][act_space][:, :, :, 1, ...].unsqueeze(3) # worker (17 3 1 2 16 6) sapper (17 3 1 2 16 225)

            teacher_actions = teacher_output["actions_GPU"][act_space][:, :, :, 1, ...].unsqueeze(3)  # (17, 3, 1, 2, 16)


            #any_actions_taken = actions_taken_mask.any(dim=-1) # worker (17 3 1 2 16)

            what = combine_policy_logits_to_log_probs(learner_policy_logits, teacher_actions, actions_taken_mask)

            imitation_loss = -what
            combined_imitation_loss = combined_imitation_loss + imitation_loss

    dummy_baseline_loss = learner_outputs["baseline_GPU"].sum() * 0.

    imitation_loss_orig = reduce(
        combined_imitation_loss,
        reduction=flags.reduction
    ) # one value tensor
    imitation_loss = flags.imitation_cost * imitation_loss_orig

    #print("IMITATION LOSS", imitation_loss_orig.detach().item())

    total_loss = imitation_loss + dummy_baseline_loss

    if warmup:
        total_loss *= 0

    return None, None, None, None, None, None, total_loss, None, None, None, None, None, imitation_loss, imitation_loss_orig

