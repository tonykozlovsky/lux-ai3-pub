import torch

from .losses import combine_policy_logits_to_log_probs, combine_policy_entropy, reduce, compute_policy_gradient_loss, compute_baseline_loss, compute_teacher_kl_loss
from .losses import combine_policy_logits_to_log_probs_GPT, combine_policy_entropy_GPT, compute_policy_gradient_loss_GPT, compute_baseline_loss_GPT, compute_teacher_kl_loss_GPT
from . import vtrace
from . import td_lambda
from . import upgo
import torch.nn as nn

def losses_func_frozen_actor(rank, profiler, flags, batch, learner_outputs, teacher_output, shared_entropy_multiplier_worker, shared_entropy_multiplier_sapper, bootstrap_value, warmup, shared_pos_weight, zero_losses=False):

    bootstrap_value = bootstrap_value[:, 0].unsqueeze(1)

    if True or ("zeros"):
        combined_behavior_action_log_probs = torch.zeros(
            (flags.unroll_length, flags.batch_size, 1),
            device=rank
        )
        combined_learner_action_log_probs = torch.zeros_like(combined_behavior_action_log_probs)
        combined_learner_entropy = torch.zeros_like(combined_behavior_action_log_probs)
        combined_teacher_kl_loss = torch.zeros_like(combined_behavior_action_log_probs)

    entropy_multipliers = {
        'worker': shared_entropy_multiplier_worker.value,
        'sapper': shared_entropy_multiplier_sapper.value
    }

    # unroll length = 17      batch size = 3         num players = 2        num units = 16
    # worker action space size = 6
    # sapper action space size = 225



    if True or ("combine"):
        for act_space in batch["actions_GPU"].keys():
            actions = batch["actions_GPU"][act_space][:, :, :, 0, :, :].unsqueeze(3) # worker (17 3 1 2 16 1)
            actions_taken_mask = batch["info"]["actions_taken_GPU_CPU"][act_space][:, :, :, 0, :, :].unsqueeze(3) # worker (17 3 1 2 16 6) sapper (17 3 1 2 16 225)

            behavior_policy_logits = batch["policy_logits_GPU_CPU"][act_space][:, :, :, 0, :, :].unsqueeze(3) # worker (17 3 1 2 16 6) sapper (17 3 1 2 16 225)

            combine_policy_logits_to_log_probs_func = combine_policy_logits_to_log_probs_GPT if flags.use_GPT_losses else combine_policy_logits_to_log_probs
            behavior_action_log_probs = combine_policy_logits_to_log_probs_func(behavior_policy_logits,
                                                                                actions,
                                                                                actions_taken_mask
                                                                                ) # worker (17 3 2))

            combined_behavior_action_log_probs = combined_behavior_action_log_probs + behavior_action_log_probs # worker (17 3 2)

            learner_policy_logits = learner_outputs["policy_logits_GPU_CPU"][act_space][:, :, :, 0, :, :].unsqueeze(3) # worker (17 3 1 2 16 6) sapper (17 3 1 2 16 225)

            learner_action_log_probs_func = combine_policy_logits_to_log_probs_GPT if flags.use_GPT_losses else combine_policy_logits_to_log_probs
            learner_action_log_probs = learner_action_log_probs_func(
                learner_policy_logits,
                actions,
                actions_taken_mask
            ) # worker (17 3 2)

            combined_learner_action_log_probs = combined_learner_action_log_probs + learner_action_log_probs # worker (17 3 2)

            # Only take entropy and KL loss for tiles where at least one action was taken
            any_actions_taken = actions_taken_mask.any(dim=-1) # worker (17 3 1 2 16)

            combine_policy_entropy_func = combine_policy_entropy_GPT if flags.use_GPT_losses else combine_policy_entropy
            learner_policy_entropy = combine_policy_entropy_func(
                learner_policy_logits,
                any_actions_taken
            ) # worker (17 3 2)

            combined_learner_entropy = combined_learner_entropy + learner_policy_entropy * entropy_multipliers[act_space]

            if flags.use_old_teacher_loss:
                teacher_kl_loss_func = compute_teacher_kl_loss_GPT if flags.use_GPT_losses else compute_teacher_kl_loss
                teacher_kl_loss = teacher_kl_loss_func(
                    learner_policy_logits,
                    teacher_output["policy_logits_GPU_CPU"][act_space][:, :, :, 0, ...].unsqueeze(3),
                    any_actions_taken
                )


                combined_teacher_kl_loss = combined_teacher_kl_loss + teacher_kl_loss


    # profiler.begin_block("calc_loss")
    # Original discount calculation
    if True or ("discounts"):
        discounts = (~batch["done_GPU_CPU"]).float() * flags.discounting # (17 3)

    if True or ("discounts_one"):
        # Adjust discounts to reset when a new episode starts
        done_mask = batch["done_GPU_CPU"].float()
        # Identify steps where a new episode starts (i.e., ~batch["done"] after a `done=1`)
        new_episode_mask = (done_mask.cumsum(dim=0) > 0) & (~done_mask.bool())
        discounts[new_episode_mask] = 1.0  # Reset discounts for the new episode


    discounts = discounts.unsqueeze(-1).expand_as(combined_behavior_action_log_probs) # (17 3 2)
    values = learner_outputs["baseline_GPU"][:, :, 0].unsqueeze(2) # (17 3 2)
    reward = batch["reward_GPU_CPU"][:, :, 0].unsqueeze(2)

    vtrace_returns_func = vtrace.from_action_log_probs_gpt if flags.use_GPT_losses else vtrace.from_action_log_probs

    vtrace_returns = vtrace_returns_func(combined_behavior_action_log_probs,
                                         combined_learner_action_log_probs,
                                         discounts,
                                         reward,
                                         values,
                                         bootstrap_value,
                                         clip_rho_threshold=1.0,
                                         clip_pg_rho_threshold=1.0)

    td_lambda_returns_func = td_lambda.td_lambda_GPT if flags.use_GPT_losses else td_lambda.td_lambda
    td_lambda_returns = td_lambda_returns_func(
        reward, values, bootstrap_value, discounts, flags.lmb)

    upgo_returns_func = upgo.upgo_GPT if flags.use_GPT_losses else upgo.upgo
    upgo_returns = upgo_returns_func(
        rewards=reward,
        values=values,
        bootstrap_value=bootstrap_value,
        discounts=discounts,
        lmb=flags.lmb
    ) # dist of tensors of (17 3 2)

    compute_policy_gradient_loss_func = compute_policy_gradient_loss_GPT if flags.use_GPT_losses else compute_policy_gradient_loss

    vtrace_pg_loss = compute_policy_gradient_loss_func(
        combined_learner_action_log_probs,
        vtrace_returns.pg_advantages,
        reduction=flags.reduction
    ) # one value tensor

    with torch.no_grad():
        upgo_clipped_importance = torch.minimum(
            vtrace_returns.log_rhos.exp(),
            torch.ones_like(vtrace_returns.log_rhos)
        ).detach() # 17 3 2

    upgo_pg_loss = compute_policy_gradient_loss_func(
        combined_learner_action_log_probs,
        upgo_clipped_importance * upgo_returns.advantages,
        reduction=flags.reduction
    ) # one value tensor

    compute_baseline_loss_func = compute_baseline_loss_GPT if flags.use_GPT_losses else compute_baseline_loss

    baseline_loss = compute_baseline_loss_func(
        values,
        td_lambda_returns.vs,
        reduction=flags.reduction
    ) # one value tensor


    if True or ("entropy"):
        entropy_loss = reduce(
            combined_learner_entropy,
            reduction=flags.reduction
        ) # one value tensor


    if flags.use_old_teacher_loss:
        teacher_kl_loss_orig = reduce(
            combined_teacher_kl_loss,
            reduction=flags.reduction
        ) # one value tensor
    else:
        teacher_kl_loss_orig = torch.zeros_like(baseline_loss)

    teacher_kl_loss = flags.teacher_kl_cost * teacher_kl_loss_orig

    if flags.use_old_teacher_loss:
        teacher_baseline_loss_orig = compute_baseline_loss_func(
            values,
            teacher_output["baseline_GPU"][:, :, 0].unsqueeze(2),
            reduction=flags.reduction
        )
    else:
        teacher_baseline_loss_orig = torch.zeros_like(baseline_loss)

    teacher_baseline_loss = flags.teacher_baseline_cost * teacher_baseline_loss_orig

    if True or ("prediction"):

        # Batch before : # obs[0], outputs[0] | obs[1], outputs[0] | obs[2] | outputs[1] ...
        # Learner before : #  outputs[0] |         outputs[1] |         outputs[2] ...


        # Batch now : # obs[1], outputs[0] | obs[2], outputs[1] | obs[3], outputs[2] ...
        # Learner now : # outputs[0] |           outputs[1] |            outputs[2] ...


        masked_learner_outputs = learner_outputs['prediction_GPU_CPU'][:, :, 0, ...].unsqueeze(2)
        masked_gt = batch['info']['GPU1_available_actions_mask']['ground_truth_CPU'][:, :, 0, ...].unsqueeze(2)
        prediction_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(shared_pos_weight[0].value, device=rank))

        prediction_position_loss = flags.prediction_cost * prediction_loss_func(
            masked_learner_outputs[:, :, :, 0, :, :].flatten().float(),
            masked_gt[:, :, :, 0, :, :].flatten().float(),
        )

        prediction_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(shared_pos_weight[1].value, device=rank))
        prediction_near_loss = flags.prediction_cost * prediction_loss_func(
            masked_learner_outputs[:, :, :, 1, :, :].flatten().float(),
            masked_gt[:, :, :, 1, :, :].flatten().float(),
        )

        prediction_loss_func = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(shared_pos_weight[2].value, device=rank))
        prediction_sensor_mask_loss = flags.prediction_cost * prediction_loss_func(
            masked_learner_outputs[:, :, :, 2, :, :].flatten().float(),
            masked_gt[:, :, :, 2, :, :].flatten().float(),
        )

        if zero_losses:
            vtrace_pg_loss *= 0
            upgo_pg_loss *= 0
            baseline_loss *= 0
            entropy_loss *= 0

        total_loss = (vtrace_pg_loss +
                      upgo_pg_loss +
                      baseline_loss +
                      prediction_position_loss +
                      prediction_near_loss +
                      prediction_sensor_mask_loss +
                      teacher_kl_loss +
                      teacher_baseline_loss
                      - entropy_loss)

    if warmup:
        #total_loss = baseline_loss + teacher_baseline_loss
        #vtrace_pg_loss, upgo_pg_loss, teacher_kl_loss, entropy_loss = torch.zeros(4) + float("nan")
        total_loss *= 0

    # profiler.end_block("calc_loss")

    if not flags.use_old_teacher_loss:
        teacher_kl_loss = None
        teacher_baseline_loss = None
        teacher_kl_loss_orig = None
        teacher_baseline_loss_orig = None

    return vtrace_pg_loss, upgo_pg_loss, baseline_loss, teacher_kl_loss, teacher_baseline_loss, entropy_loss, total_loss, prediction_position_loss, prediction_near_loss, prediction_sensor_mask_loss, teacher_kl_loss_orig, teacher_baseline_loss_orig, None, None
