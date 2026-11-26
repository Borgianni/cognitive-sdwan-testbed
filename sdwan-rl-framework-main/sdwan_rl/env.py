import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from .config import RLConfig

A_LTE, A_SAT = 0, 1

@dataclass
class State:
    # Feature order: LTE/LEO delay, LTE/LEO loss, LTE/LEO bandwidth.
    vec: np.ndarray

class SDWANEnv:
    """Reward and configurable six-feature observation of the network."""
    def __init__(self, cfg: RLConfig = None):
        self.cfg = cfg if cfg is not None else RLConfig()
        self.ticks = 0
        self.reset_metrics()

    def reset_metrics(self):
        self.lte = dict(delay=float('inf'), thr=0.0, loss=100.0)
        self.sat = dict(delay=float('inf'), thr=0.0, loss=100.0)

        # Zero throughput is meaningful only after a measurement is received.
        self._has_metrics = False

    @staticmethod
    def residual_capacity_fraction(throughput_mbps: float, capacity_mbps: float) -> float:
        """Estimated unused fraction of configured interface capacity.

        Invalid measurements return NaN, not a fully available link.
        """
        if (not math.isfinite(throughput_mbps) or throughput_mbps < 0.0
                or not math.isfinite(capacity_mbps) or capacity_mbps <= 0.0):
            return float("nan")
        return max(0.0, min(1.0, (capacity_mbps - throughput_mbps) / capacity_mbps))

    def _bandwidth_ok(self, throughput, capacity, threshold):
        if not self._has_metrics or not math.isfinite(throughput) or throughput < 0.0:
            return 0
        if self.cfg.bandwidth_state_mode == "legacy_throughput":
            return int(throughput > 0.0)
        residual = self.residual_capacity_fraction(throughput, capacity)
        if self.cfg.bandwidth_state_mode == "positive_residual":
            return int(residual > 0.0)
        return int(residual >= threshold)

    def _continuous_state(self) -> State:
        if not self._has_metrics:
            return State(vec=np.zeros(6, dtype=np.float32))

        def delay_quality(delay):
            if not math.isfinite(delay) or delay <= 0.0:
                return 0.0
            return 1.0 / (1.0 + delay / self.cfg.continuous_delay_scale_ms)

        def loss_quality(loss):
            if not math.isfinite(loss) or not 0.0 <= loss <= 100.0:
                return 0.0
            return 1.0 - loss / 100.0

        def bandwidth_quality(throughput, capacity):
            residual = self.residual_capacity_fraction(throughput, capacity)
            return residual if math.isfinite(residual) else 0.0

        return State(vec=np.array([
            delay_quality(self.lte['delay']),
            delay_quality(self.sat['delay']),
            loss_quality(self.lte['loss']),
            loss_quality(self.sat['loss']),
            bandwidth_quality(self.lte['thr'], self.cfg.lte_capacity_mbps),
            bandwidth_quality(self.sat['thr'], self.cfg.sat_capacity_mbps),
        ], dtype=np.float32))

    def make_state(self) -> State:
        if self.cfg.state_representation == "continuous":
            return self._continuous_state()
        # The ping probe reports end-to-end RTT, not one-way NetEm delay.
        lte_delay_ok = int(0.0 < self.lte['delay'] <= self.cfg.lte_delay_threshold_ms)
        sat_delay_ok = int(0.0 < self.sat['delay'] <= self.cfg.sat_delay_threshold_ms)
        # Loss is measured end-to-end and expressed as a percentage.
        lte_loss_ok  = int(0.0 <= self.lte['loss'] <= self.cfg.lte_loss_threshold_pct)
        sat_loss_ok  = int(0.0 <= self.sat['loss'] <= self.cfg.sat_loss_threshold_pct)
        lte_bw_ok = self._bandwidth_ok(self.lte['thr'], self.cfg.lte_capacity_mbps,
                                       self.cfg.lte_residual_threshold)
        sat_bw_ok = self._bandwidth_ok(self.sat['thr'], self.cfg.sat_capacity_mbps,
                                       self.cfg.sat_residual_threshold)
        return State(vec=np.array([lte_delay_ok, sat_delay_ok, lte_loss_ok, sat_loss_ok, lte_bw_ok, sat_bw_ok], dtype=np.float32))

    def cost_index(self, action: int) -> float:
        if action not in (A_LTE, A_SAT):
            raise ValueError("Unknown tunnel action")
        return float(self.cfg.sat_cost_index if action == A_SAT else self.cfg.lte_cost_index)

    def reward_terms(self, action: int) -> Dict:
        """Shared scalar reward and signed components for the chosen tunnel.

        Canonical telemetry: RTT ms, loss percent, throughput Mbps.
        Normalized mode fails closed for detectably invalid selected metrics.
        Legacy mode preserves the original numerical floors and formula at
        the historical parameters; it is only a reproduction comparator.
        """
        cost = self.cost_index(action)
        m = self.sat if action == A_SAT else self.lte
        capacity = self.cfg.sat_capacity_mbps if action == A_SAT else self.cfg.lte_capacity_mbps
        d, loss, throughput = m["delay"], m["loss"], m["thr"]
        cfg = self.cfg
        if cfg.reward_mode == "legacy":
            delay_score = math.log(cfg.threshold_delay_ms / max(d, 1e-4))
            loss_score = 1 + math.log(max(loss, 1e-4) + 1)
            residual = max(capacity - throughput, 1e-4)
            bandwidth_score = math.log(residual + 1)
        else:
            if (not self._has_metrics or not all(math.isfinite(v) for v in (d, loss, throughput))
                    or d <= 0 or not 0 <= loss <= 100 or throughput < 0):
                raise ValueError("Invalid selected-tunnel telemetry for normalized reward")
            residual = max(capacity - throughput, 0.0)
            delay_score = math.log(cfg.threshold_delay_ms) - math.log(d)
            loss_score = 1 + math.log1p((loss / 100.0) / cfg.reward_loss_ref_fraction)
            bandwidth_score = math.log1p(residual / cfg.reward_bandwidth_ref_mbps)
        delay_term = cfg.w_b * delay_score
        loss_term = -cfg.w_a * loss_score
        bandwidth_term = cfg.w_c * bandwidth_score
        cost_term = -cfg.w_d * cost
        qos = delay_term + loss_term + bandwidth_term
        reward = qos + cost_term
        if cfg.reward_mode == "normalized" and not math.isfinite(reward):
            raise ValueError("Non-finite normalized reward")
        return {"delay_term": float(delay_term), "loss_term": float(loss_term),
                "bandwidth_term": float(bandwidth_term), "cost_term": float(cost_term),
                "qos_utility": float(qos), "cost_index": cost,
                "residual_mbps": float(residual), "reward": float(reward)}

    def reward(self, action: int) -> float:
        return self.reward_terms(action)["reward"]

    def update_from_metrics(self, metrics: Dict) -> None:
        # Missing probes are explicit nulls on the wire, not zero measurements.
        for name, target in (("lte", self.lte), ("satellite", self.sat)):
            for wire_key, key in (("delay", "delay"), ("throughput", "thr"),
                                  ("packet_loss", "loss")):
                value = metrics[name][wire_key]
                target[key] = float("nan") if value is None else value
        self._has_metrics = True
