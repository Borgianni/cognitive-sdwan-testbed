import math
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class LinkConfig:
    bw_mbps: int
    delay_ms: int
    loss_pct: float
    jitter_ms: int
    max_queue: int

@dataclass
class TopologyConfig:
    lte: LinkConfig = field(default_factory=lambda: LinkConfig(250, 25, 2, 5, 750))
    sat: LinkConfig = field(default_factory=lambda: LinkConfig(200, 60, 3, 30, 1000))
    host_link: LinkConfig = field(default_factory=lambda: LinkConfig(250, 50, 1, 10, 1000))

@dataclass
class RLConfig:
    gamma: float = 0.95
    replay_size: int = 10_000
    batch_size: int = 32
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay: float = 0.995
    lr: float = 1e-4
    # reward weights 
    w_a: float = 3.0   # packet loss penalty
    w_b: float = 2.0   # delay term
    w_c: float = 2.0   # available bw
    w_d: float = 0.2   # cost penalty
    threshold_delay_ms: float = 140.0
    lte_capacity_mbps: float = 250.0
    sat_capacity_mbps: float = 200.0
    # Observation thresholds in percentage points (10.0 means 10%).
    # Set both to 0.0 to reproduce the original zero-loss representation.
    lte_loss_threshold_pct: float = 10.0
    sat_loss_threshold_pct: float = 10.0

    # Provisional RTT observation thresholds in milliseconds (not SLA limits).
    # Separate from threshold_delay_ms, which is used only by the reward.
    lte_delay_threshold_ms: float = 175.0
    sat_delay_threshold_ms: float = 240.0

    # Bandwidth observation uses residual capacity fraction by default.
    # legacy_throughput reproduces the public original implementation;
    # positive_residual reproduces the mathematical predicate in the paper.
    bandwidth_state_mode: str = "residual_threshold"
    lte_residual_threshold: float = 0.20
    sat_residual_threshold: float = 0.20

    # Continuous observation uses a common RTT scale (not an SLA threshold).
    state_representation: str = "binary"
    continuous_delay_scale_ms: float = 100.0

    # Explicit dimensionless reward ratios, preserving the historical scale.
    reward_mode: str = "normalized"
    reward_bandwidth_ref_mbps: float = 1.0
    reward_loss_ref_fraction: float = 0.01
    # Synthetic dimensionless preference indices, not monetary/energy costs.
    lte_cost_index: float = 7.0
    sat_cost_index: float = 20.0

    @classmethod
    def from_record(cls, fields):
        """Old checkpoints/traces predate explicit reward semantics."""
        fields = dict(fields)
        reward_fields = {"reward_mode", "reward_bandwidth_ref_mbps",
                         "reward_loss_ref_fraction", "lte_cost_index", "sat_cost_index"}
        present = reward_fields.intersection(fields)
        if not present:
            fields["reward_mode"] = "legacy"
        elif present != reward_fields:
            raise ValueError("Incomplete recorded reward configuration")
        return cls(**fields)

    def __post_init__(self):
        if self.reward_mode not in ("normalized", "legacy"):
            raise ValueError("Unknown reward mode")
        for name in ("threshold_delay_ms", "reward_bandwidth_ref_mbps",
                     "lte_cost_index", "sat_cost_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        ref = self.reward_loss_ref_fraction
        if isinstance(ref, bool) or not math.isfinite(ref) or not 0 < ref <= 1:
            raise ValueError("reward_loss_ref_fraction must be in (0, 1]")
        for name in ("w_a", "w_b", "w_c", "w_d"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not any(getattr(self, name) > 0 for name in ("w_a", "w_b", "w_c", "w_d")):
            raise ValueError("At least one reward weight must be positive")
        if self.state_representation not in ("binary", "continuous"):
            raise ValueError("Unknown state_representation")
        if (not math.isfinite(self.continuous_delay_scale_ms)
                or self.continuous_delay_scale_ms <= 0.0):
            raise ValueError("continuous_delay_scale_ms must be finite and positive")
        if self.bandwidth_state_mode not in (
            "residual_threshold", "legacy_throughput", "positive_residual"
        ):
            raise ValueError("Unknown bandwidth_state_mode")
        for name in ("lte_capacity_mbps", "sat_capacity_mbps"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("lte_residual_threshold", "sat_residual_threshold"):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("lte_delay_threshold_ms", "sat_delay_threshold_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("lte_loss_threshold_pct", "sat_loss_threshold_pct"):
            if not 0.0 <= getattr(self, name) <= 100.0:
                raise ValueError(f"{name} must be between 0 and 100 percent")


@dataclass
class ControllerConfig:
    base_url: str = "http://127.0.0.1:8080"
    metrics_endpoint: str = "/metrics"
    decision_endpoint: str = "/link_decision"

@dataclass
class ExperimentConfig:
    episodes: int = 60
    ticks_per_episode: int = 20
    degrade_lte_ranges: Tuple[Tuple[int, int], ...] = ((1, 20), (30, 40))