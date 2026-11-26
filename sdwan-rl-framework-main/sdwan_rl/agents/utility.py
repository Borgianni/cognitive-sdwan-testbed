"""Single source of immediate utility for the greedy baseline and snapshot audits."""
import math

from .config import RLConfig
from .env import SDWANEnv

TUNNELS = ("lte", "satellite")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def valid_tunnels(metrics):
    result = []
    for tunnel in TUNNELS:
        row = metrics.get(tunnel) if isinstance(metrics, dict) else None
        result.append(
            isinstance(row, dict)
            and all(finite(row.get(k)) for k in ("delay", "packet_loss", "throughput"))
            and row["delay"] > 0
            and 0 <= row["packet_loss"] <= 100
            and row["throughput"] >= 0
        )
    return result


def snapshot_utilities(metrics, cfg):
    """Score both actions on one complete snapshot with the actual RL reward.

    A missing/invalid tunnel makes this paired comparison unavailable.
    Do not manufacture metrics, interpret failed probes as excellent links,
    or duplicate the reward formula in the baseline.
    """
    if not all(valid_tunnels(metrics)):
        raise ValueError("Paired utility requires valid telemetry for both tunnels")
    env = SDWANEnv(cfg)
    env.update_from_metrics(metrics)
    scores = tuple(env.reward(a) for a in range(len(TUNNELS)))
    if not all(finite(score) for score in scores):
        raise ValueError("Non-finite snapshot utility")
    return scores


class GreedyUtilityAgent:
    """Causal non-learning comparator, using the same reward as DQN."""
    name = "Greedy utility"

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else RLConfig()
        self.reset()

    def reset(self):
        self.last_action = 0

    def decide(self, metrics):
        valid = valid_tunnels(metrics)
        scores = None
        if all(valid):
            scores = snapshot_utilities(metrics, self.cfg)
            action = max(range(len(TUNNELS)), key=lambda a: (scores[a], -a))
            reason = "maximum_snapshot_utility"
        elif any(valid):
            action = valid.index(True)
            reason = "only_valid_telemetry"
        else:
            action = self.last_action
            reason = "no_valid_telemetry_hold"
        self.last_action = action
        return action, {"utility_by_action": list(scores) if scores is not None else None,
                        "valid_telemetry": valid, "reason": reason}
