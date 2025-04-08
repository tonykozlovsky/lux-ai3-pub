import collections
import torch

TDLambdaReturns = collections.namedtuple("TDLambdaReturns", "vs advantages")


@torch.no_grad()
def td_lambda(
        rewards: torch.Tensor,
        values: torch.Tensor,
        bootstrap_value: torch.Tensor,
        discounts: torch.Tensor,
        lmb: float,
) -> TDLambdaReturns:
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat(
        [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
    )
    target_values = [bootstrap_value]
    for t in range(discounts.shape[0] - 1, -1, -1):
        # noinspection PyUnresolvedReferences
        target_values.append(
            rewards[t] + discounts[t] * ((1 - lmb) * values_t_plus_1[t] + lmb * target_values[-1])
        )
    target_values.reverse()
    # Remove bootstrap value from end of target_values list
    target_values = torch.stack(target_values[:-1], dim=0)

    return TDLambdaReturns(
        vs=target_values,
        advantages=target_values - values
    )

class TDLambdaReturns_GPT:
    def __init__(self, vs):
        self.vs = vs

@torch.no_grad()
def td_lambda_GPT(rewards, values, bootstrap_value, discounts, lmb=0.95):
    """
    rewards:        (T, B, P)
    values:         (T, B, P)
    bootstrap_value (B, P)
    discounts:      (T, B, P)
    lmb:            scalar in [0,1], the lambda parameter

    Returns TDLambdaReturns object with:
      vs:  (T, B, P)
    """
    T = rewards.shape[0]
    vs = []
    vs_plus_1 = bootstrap_value  # (B, P)

    # We'll do a backward pass:
    for t in reversed(range(T)):
        td_target = rewards[t] + discounts[t] * vs_plus_1
        vs_t = values[t] + lmb * (td_target - values[t])
        vs.append(vs_t)
        vs_plus_1 = vs_t

    vs.reverse()
    vs = torch.stack(vs, dim=0)  # (T, B, P)
    return TDLambdaReturns_GPT(vs=vs)
