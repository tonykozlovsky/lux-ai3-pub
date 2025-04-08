import collections
import torch

UPGOReturns = collections.namedtuple("UPGOReturns", "vs advantages")


@torch.no_grad()
def upgo(
        rewards: torch.Tensor,
        values: torch.Tensor,
        bootstrap_value: torch.Tensor,
        discounts: torch.Tensor,
        lmb: float,
) -> UPGOReturns:
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat(
        [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
    )
    target_values = [bootstrap_value]
    for t in range(discounts.shape[0] - 1, -1, -1):
        # noinspection PyUnresolvedReferences
        target_values.append(
            rewards[t] + discounts[t] * torch.max(values_t_plus_1[t],
                                                  (1 - lmb) * values_t_plus_1[t] + lmb * target_values[-1])
        )
    target_values.reverse()
    # Remove bootstrap value from end of target_values list
    target_values = torch.stack(target_values[:-1], dim=0)

    return UPGOReturns(
        vs=target_values,
        advantages=target_values - values
    )


class UPGOReturns_GPT:
    def __init__(self, advantages):
        self.advantages = advantages

@torch.no_grad()
def upgo_GPT(rewards, values, bootstrap_value, discounts, lmb=0.95):
    """
    rewards:         (T, B, P)
    values:          (T, B, P)
    bootstrap_value: (B, P)
    discounts:       (T, B, P)
    lmb:             possibly unused in a naive upgo, or used in a variant

    Returns UPGOReturns object with:
      advantages:  (T, B, P)
    """
    T = rewards.shape[0]
    max_vs = []
    max_vs_plus_1 = bootstrap_value  # (B, P)

    for t in reversed(range(T)):
        # (B, P)
        candidate = rewards[t] + discounts[t] * max_vs_plus_1
        max_vs_t = torch.maximum(values[t], candidate)
        max_vs.append(max_vs_t)
        max_vs_plus_1 = max_vs_t

    max_vs.reverse()
    max_vs = torch.stack(max_vs, dim=0)  # (T, B, P)

    # advantage = (max future value) - current value
    advantages = max_vs - values
    return UPGOReturns_GPT(advantages=advantages)
