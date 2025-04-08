# This file taken from
#     https://github.com/deepmind/scalable_agent/blob/
#         cd66d00914d56c8ba2f0615d9cdeefcb169a8d70/vtrace.py
# and modified.

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

import collections

import torch
import torch.nn.functional as F


VTraceFromLogitsReturns = collections.namedtuple(
    "VTraceFromLogitsReturns",
    [
        "vs",
        "pg_advantages",
        "log_rhos",
        "behavior_action_log_probs",
        "target_action_log_probs",
    ],
)

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")


def action_log_probs(policy_logits, actions):
    return -F.nll_loss(
        F.log_softmax(policy_logits.view(-1, policy_logits.shape[-1]), dim=-1),
        torch.flatten(actions),
        reduction="none",
    ).view_as(actions)


def from_logits(
        behavior_policy_logits,
        target_policy_logits,
        actions,
        discounts,
        rewards,
        values,
        bootstrap_value,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
):
    """V-trace for softmax policies."""

    target_action_log_probs = action_log_probs(target_policy_logits, actions)
    behavior_action_log_probs = action_log_probs(behavior_policy_logits, actions)
    return from_action_log_probs(
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold
    )


def from_action_log_probs(
        behavior_action_log_probs,
        target_action_log_probs,
        discounts,
        rewards,
        values,
        bootstrap_value,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
):
    log_rhos = target_action_log_probs - behavior_action_log_probs
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
    )
    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behavior_action_log_probs=behavior_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict(),
    )


@torch.no_grad()
def from_importance_weights(
        log_rhos,
        discounts,
        rewards,
        values,
        bootstrap_value,
        clip_rho_threshold=1.0,
        clip_pg_rho_threshold=1.0,
):
    """V-trace from log importance weights."""
    with torch.no_grad():
        rhos = torch.exp(log_rhos)
        if clip_rho_threshold is not None:
            clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
        else:
            clipped_rhos = rhos

        cs = torch.clamp(rhos, max=1.0)
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = torch.cat(
            [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
        )
        deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)

        acc = torch.zeros_like(bootstrap_value)
        result = []
        for t in range(discounts.shape[0] - 1, -1, -1):
            acc = deltas[t] + discounts[t] * cs[t] * acc
            result.append(acc)
        result.reverse()
        vs_minus_v_xs = torch.stack(result)

        # Add V(x_s) to get v_s.
        vs = torch.add(vs_minus_v_xs, values)

        # Advantage for policy gradient.
        vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
        else:
            clipped_pg_rhos = rhos
        pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)


class VTraceReturns_GPT:
    def __init__(self, vs, pg_advantages, log_rhos):
        self.vs = vs
        self.pg_advantages = pg_advantages
        self.log_rhos = log_rhos

@torch.no_grad()
def from_action_log_probs_gpt(behavior_action_log_probs,
                          target_action_log_probs,
                          discounts,
                          rewards,
                          values,
                          bootstrap_value,
                          clip_rho_threshold=1.0,
                          clip_pg_rho_threshold=1.0):
    """
    behavior_action_log_probs: (T, B, P)
    target_action_log_probs:   (T, B, P)
    discounts:                 (T, B, P)
    rewards:                   (T, B, P)
    values:                    (T, B, P)
    bootstrap_value:           (B, P)  [value estimate for t=T+1]
    clip_rho_threshold:        scalar or None
    clip_pg_rho_threshold:     scalar or None

    Returns an object with:
      vs:            (T, B, P)   (the fixed-step value targets)
      pg_advantages: (T, B, P)
      log_rhos:      (T, B, P)
    """
    # log_rhos = log( pi(a|s) / mu(a|s) ) = target_log_prob - behavior_log_prob
    log_rhos = target_action_log_probs - behavior_action_log_probs
    rhos = torch.exp(log_rhos)

    if clip_rho_threshold is not None:
        clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    else:
        clipped_rhos = rhos

    # For "c_s" in the paper, often clamp to [0, 1.0], though some also use clip_rho_threshold
    cs = torch.clamp(rhos, max=1.0)

    # We will do a reverse scan to compute vs
    T = rewards.shape[0]
    B = rewards.shape[1]
    P = rewards.shape[2]

    # Append bootstrapped value to the end
    values_extended = torch.cat([
        values,
        bootstrap_value.unsqueeze(0)  # shape (1, B, P)
    ], dim=0)  # shape => (T+1, B, P)

    vs = []
    vs_plus_1 = bootstrap_value  # shape (B, P) at time T

    for t in reversed(range(T)):
        # shape: (B, P)
        reward_t = rewards[t]
        discount_t = discounts[t]
        value_t = values[t]
        rho_t = clipped_rhos[t]
        c_t = cs[t]

        # v_s = V(x_s) + rho_s [r_s + gamma * V(x_{s+1}) - V(x_s)]
        #        + gamma * c_s [v_{s+1} - V(x_{s+1})]
        # See IMPALA (Espeholt et al. 2018) or the Haiku reference code for detail.
        delta_t = rho_t * (reward_t + discount_t * values_extended[t+1] - value_t)
        vs_t = value_t + delta_t + discount_t * c_t * (vs_plus_1 - values_extended[t+1])

        vs.append(vs_t)
        vs_plus_1 = vs_t

    vs.reverse()
    vs = torch.stack(vs, dim=0)  # (T, B, P)

    # Policy-gradient advantage is ρ_s [r_s + γ v_{s+1} - V(x_s)]
    # but we also clamp ρ in PG if clip_pg_rho_threshold is not None
    if clip_pg_rho_threshold is not None:
        pg_rhos = torch.clamp(rhos, max=clip_pg_rho_threshold)
    else:
        pg_rhos = rhos

    # For the 'critic' term we use vs[t+1], but we can index with the extended array:
    vs_next = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)  # shift by 1 in time

    pg_advantages = pg_rhos * (rewards + discounts * vs_next - values)

    return VTraceReturns_GPT(vs=vs, pg_advantages=pg_advantages, log_rhos=log_rhos)