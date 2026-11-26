"""Shared reward CLI options for both runner entry points."""
from .config import RLConfig


def add_reward_arguments(parser):
    parser.add_argument("--reward-mode", choices=["normalized", "legacy"], default=RLConfig.reward_mode)
    for flag, field in (
        ("reward-delay-ref-ms", "threshold_delay_ms"),
        ("reward-bandwidth-ref-mbps", "reward_bandwidth_ref_mbps"),
        ("reward-loss-ref-fraction", "reward_loss_ref_fraction"),
        ("weight-delay", "w_b"), ("weight-loss", "w_a"),
        ("weight-bandwidth", "w_c"), ("weight-cost", "w_d"),
        ("lte-cost-index", "lte_cost_index"), ("sat-cost-index", "sat_cost_index"),
    ):
        parser.add_argument("--" + flag, dest=field, type=float, default=getattr(RLConfig, field))


def reward_kwargs(args):
    return {name: getattr(args, name) for name in (
        "reward_mode", "threshold_delay_ms", "reward_bandwidth_ref_mbps",
        "reward_loss_ref_fraction", "w_a", "w_b", "w_c", "w_d",
        "lte_cost_index", "sat_cost_index",
    )}
