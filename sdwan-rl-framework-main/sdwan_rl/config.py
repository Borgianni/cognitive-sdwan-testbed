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
    threshold_delay_ms: int = 140
    lte_capacity_mbps: float = 250.0
    sat_capacity_mbps: float = 200.0
    # Observation thresholds in percentage points (10.0 means 10%).
    # Set both to 0.0 to reproduce the original zero-loss representation.
    lte_loss_threshold_pct: float = 10.0
    sat_loss_threshold_pct: float = 10.0

    def __post_init__(self):
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