import torch.nn as nn
import torch
import torch.nn.functional as F


KL_DIV_LOSS = nn.KLDivLoss(reduction="none")


def combine_policy_logits_to_log_probs_GPT(policy_logits, actions, actions_taken_mask):
    """
    policy_logits:        (T, B, 1, P, U, A)
    actions:              (T, B, 1, P, U, 1)
    actions_taken_mask:   (T, B, 1, P, U, A)
    Return shape:         (T, B, P)
    """
    # Compute log-softmax over the last dimension (the action dimension A)
    #num_available_actions = (policy_logits > -1e9).sum(dim=-1, keepdim=True).float()
    #log_probs = F.log_softmax(policy_logits, dim=-1) / (num_available_actions + 1e-8)

    log_probs = F.log_softmax(policy_logits, dim=-1)  # same shape as policy_logits

    # Gather the log-prob of the chosen action
    # actions has shape (T, B, 1, P, U, 1), so we gather along dim=-1
    chosen_log_probs = torch.gather(log_probs, dim=-1, index=actions)
    # chosen_log_probs is now (T, B, 1, P, U, 1)
    chosen_log_probs = chosen_log_probs.squeeze(-1)   # (T, B, 1, P, U)

    # Also gather the "actions_taken_mask" for the chosen action
    chosen_mask = torch.gather(actions_taken_mask, dim=-1, index=actions)
    # chosen_mask is (T, B, 1, P, U, 1) -> remove last dim
    chosen_mask = chosen_mask.squeeze(-1)  # (T, B, 1, P, U)

    # Multiply chosen log-probs by the chosen mask
    # This zeros out log-probs for units that did not actually act
    masked_chosen_log_probs = chosen_log_probs * chosen_mask.float()  # (T, B, 1, P, U)

    # Sum across the U dimension (the number of units)
    # so that we end up with per-player log-prob per (T,B)
    sum_log_probs = masked_chosen_log_probs.sum(dim=-1)  # (T, B, 1, P)
    # If that extra dimension (the "1") is truly 1, remove it
    sum_log_probs = sum_log_probs.squeeze(2)             # (T, B, P)

    return sum_log_probs


def combine_policy_logits_to_log_probs(
        behavior_policy_logits: torch.Tensor,
        actions: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    """
    Combines all policy_logits at a given step to get a single action_log_probs value for that step

    Initial shape: time, batch, 1, players, n_units, n_actions
    Returned shape: time, batch, players
    """
    # Get the action probabilities
    probs = F.softmax(behavior_policy_logits, dim=-1)
    # Ignore probabilities for actions that were not used
    probs = actions_taken_mask * probs
    # Select the probabilities for actions that were taken by stacked agents and sum these
    selected_probs = torch.gather(probs, -1, actions)
    # Convert the probs to conditional probs, since we sample without replacement
    remaining_probability_density = 1. - torch.cat([
        torch.zeros(
            (*selected_probs.shape[:-1], 1),
            device=selected_probs.device,
            dtype=selected_probs.dtype
        ),
        selected_probs[..., :-1].cumsum(dim=-1)
    ], dim=-1)
    # Avoid division by zero
    remaining_probability_density = remaining_probability_density + torch.where(
        remaining_probability_density == 0,
        torch.ones_like(remaining_probability_density),
        torch.zeros_like(remaining_probability_density)
    )
    conditional_selected_probs = selected_probs / remaining_probability_density
    # Remove 0-valued conditional_selected_probs in order to eliminate neg-inf valued log_probs
    conditional_selected_probs = conditional_selected_probs + torch.where(
        conditional_selected_probs == 0,
        torch.ones_like(conditional_selected_probs),
        torch.zeros_like(conditional_selected_probs)
    )
    log_probs = torch.log(conditional_selected_probs)
    # Sum over actions, y and x dimensions to combine log_probs from different actions
    # Squeeze out action_planes dimension as well
    return torch.flatten(log_probs, start_dim=-2, end_dim=-1).sum(dim=-1).squeeze(dim=-2)


def combine_policy_entropy_GPT(policy_logits, any_actions_taken):
    """
    policy_logits:     (T, B, 1, P, U, A)
    any_actions_taken: (T, B, 1, P, U)  [Boolean]
    Return shape:      (T, B, P)
    """
    # Compute log-softmax
    policy_logits = torch.where(
        torch.isfinite(policy_logits), policy_logits, torch.full_like(policy_logits, -1e9)
    )
    log_probs = F.log_softmax(policy_logits, dim=-1)  # (T, B, 1, P, U, A)
    probs = torch.exp(log_probs)                      # (T, B, 1, P, U, A)

    # Entropy for each unit: -sum_a [p(a)*log p(a)]
    entropy_per_unit = -(probs * log_probs).sum(dim=-1)  # (T, B, 1, P, U)
    if False:
        num_available_actions = (policy_logits > -1e9).sum(dim=-1).float()  # (T, B, 1, P, U)
        entropy_per_unit = entropy_per_unit / (num_available_actions + 1e-8)  # Prevent divide-by-zero

    # Only consider units that actually took an action
    masked_entropy = entropy_per_unit * any_actions_taken.float()  # (T, B, 1, P, U)

    # Sum across units
    sum_entropy = masked_entropy.sum(dim=-1)  # (T, B, 1, P)
    sum_entropy = sum_entropy.squeeze(2)      # (T, B, P)

    return sum_entropy


def combine_policy_entropy(
        policy_logits: torch.Tensor,
        actions_taken_mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes and combines policy entropy for a given step.
    NB: We are just computing the sum of individual entropies, not the joint entropy, because I don't think there is
    an efficient way to compute the joint entropy?

    Initial shape: time, batch, action_planes, players, x, y, n_actions
    Returned shape: time, batch, players
    """
    policy = F.softmax(policy_logits, dim=-1)
    log_policy = F.log_softmax(policy_logits, dim=-1)
    log_policy_masked_zeroed = torch.where(
        log_policy.isneginf(),
        torch.zeros_like(log_policy),
        log_policy
    )
    entropies = (policy * log_policy_masked_zeroed).sum(dim=-1)
    assert actions_taken_mask.shape == entropies.shape
    entropies_masked = entropies * actions_taken_mask.float()
    # Sum over y, x, and action_planes dimensions to combine entropies from different actions
    return -entropies_masked.sum(dim=-1).squeeze(dim=-2)


def compute_teacher_kl_loss(
        learner_policy_logits: torch.Tensor,
        teacher_policy_logits: torch.Tensor,
        actions_taken_mask: torch.Tensor,
) -> torch.Tensor:
    learner_policy_log_probs = F.log_softmax(learner_policy_logits, dim=-1)
    teacher_policy = F.softmax(teacher_policy_logits, dim=-1)

    learner_policy_log_probs = torch.where(
        learner_policy_log_probs.isneginf(),
        torch.zeros_like(learner_policy_log_probs),
        learner_policy_log_probs
    )
    teacher_policy = torch.where(
        teacher_policy.isneginf(),
        torch.zeros_like(teacher_policy),
        teacher_policy
    )

    kl_div = F.kl_div(
        learner_policy_log_probs,
        teacher_policy.detach(),
        reduction="none",
        log_target=False
    ).sum(dim=-1)
    assert actions_taken_mask.shape == kl_div.shape
    kl_div_masked = kl_div * actions_taken_mask.float()
    # Sum over y, x, and action_planes dimensions to combine kl divergences from different actions
    return kl_div_masked.sum(dim=-1).squeeze(dim=-2)


def compute_teacher_kl_loss_GPT(learner_logits, teacher_logits, any_actions_taken):
    """
    learner_logits:    (T, B, 1, P, U, A)
    teacher_logits:    (T, B, 1, P, U, A)
    any_actions_taken: (T, B, 1, P, U) [Boolean]
    Return shape:      (T, B, P)
    """
    # Softmax for teacher
    teacher_probs = F.softmax(teacher_logits, dim=-1)    # (T, B, 1, P, U, A)
    # Log-softmax for learner
    learner_log_probs = F.log_softmax(learner_logits, dim=-1)  # (T, B, 1, P, U, A)

    # KL(teacher || learner) = sum_a [ teacher_probs(a) * (log teacher_probs(a) - log learner_probs(a)) ]
    # We'll add a small epsilon inside log() to avoid numerical issues.
    eps = 1e-8
    teacher_log_probs = torch.log(teacher_probs + eps)
    kl_per_action = teacher_probs * (teacher_log_probs - learner_log_probs)  # (T, B, 1, P, U, A)

    # Sum over the action dimension
    kl_per_unit = kl_per_action.sum(dim=-1)  # (T, B, 1, P, U)

    # Mask out units that did not act
    kl_per_unit = kl_per_unit * any_actions_taken.float()

    # Sum across units
    kl_per_step = kl_per_unit.sum(dim=-1)    # (T, B, 1, P)
    kl_per_step = kl_per_step.squeeze(2)     # (T, B, P)

    return kl_per_step


def reduce(losses: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return losses.mean()
    elif reduction == "sum":
        return losses.sum()
    else:
        raise ValueError(f"Reduction must be one of 'sum' or 'mean', was: {reduction}")


def compute_baseline_loss(values: torch.Tensor, value_targets: torch.Tensor, reduction: str) -> torch.Tensor:
    baseline_loss = F.smooth_l1_loss(values, value_targets.detach(), reduction="none")
    return reduce(baseline_loss, reduction=reduction)


def compute_baseline_loss_GPT(values, targets, reduction='mean'):
    """
    values:   (T, B, P)
    targets:  (T, B, P)
    """
    # e.g., 0.5 * MSE is common in many actor-critic setups
    loss = 0.5 * (targets - values).pow(2)

    if reduction == 'mean':
        return (loss ).mean()
    elif reduction == 'sum':
        return (loss).sum()
    else:
        # e.g. 'none' or custom
        return loss

def compute_policy_gradient_loss(
        action_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        reduction: str
) -> torch.Tensor:
    cross_entropy = -action_log_probs.view_as(advantages)

    return reduce(cross_entropy * advantages.detach(), reduction)


def compute_policy_gradient_loss_GPT(log_probs, advantages, reduction='mean'):
    """
    log_probs:  (T, B, P)  e.g., sum of log-probs for the chosen actions
    advantages: (T, B, P)  e.g., from vtrace_returns.pg_advantages
    """
    loss = -advantages * log_probs
    if reduction == 'mean':
        return (loss).mean()
    elif reduction == 'sum':
        return (loss).sum()
    else:
        return loss
