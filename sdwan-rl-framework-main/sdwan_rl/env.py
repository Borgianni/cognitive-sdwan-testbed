import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple
from .config import RLConfig

A_LTE, A_SAT = 0, 1

@dataclass
class State:
    # Binary features (delay_ok, delay_ok_sat, loss_ok_lte, loss_ok_sat, bw_ok_lte, bw_ok_sat)
    vec: np.ndarray

class SDWANEnv:
    """A lightweight env wrapper with your reward Eq. (1) and booleanized state."""
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

    def make_state(self) -> State:
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

    def reward(self, action: int) -> float:
        # Available bandwidth = capacity - throughput (clipped)
        ab_lte = max(self.cfg.lte_capacity_mbps - self.lte['thr'], 1e-4)
        ab_sat = max(self.cfg.sat_capacity_mbps - self.sat['thr'], 1e-4)

        # avoid log(0)
        d_lte = max(self.lte['delay'], 1e-4)
        d_sat = max(self.sat['delay'], 1e-4)
        l_lte = max(self.lte['loss'],  1e-4)
        l_sat = max(self.sat['loss'],  1e-4)

        if action == A_SAT:
            R = ( self.cfg.w_b * math.log(self.cfg.threshold_delay_ms / d_sat)
                - self.cfg.w_a * (1 + math.log(l_sat + 1))
                + self.cfg.w_c * math.log(ab_sat + 1)
                - self.cfg.w_d * 20)  # satellite cost (matches your controller’s cost map)
        else:
            R = ( self.cfg.w_b * math.log(self.cfg.threshold_delay_ms / d_lte)
                - self.cfg.w_a * (1 + math.log(l_lte + 1))
                + self.cfg.w_c * math.log(ab_lte + 1)
                - self.cfg.w_d * 7)   # LTE cost
        return float(R)

    def update_from_metrics(self, metrics: Dict) -> None:
        self.lte['delay'] = metrics['lte']['delay']
        self.lte['thr']   = metrics['lte']['throughput']
        self.lte['loss']  = metrics['lte']['packet_loss']
        self.sat['delay'] = metrics['satellite']['delay']
        self.sat['thr']   = metrics['satellite']['throughput']
        self.sat['loss']  = metrics['satellite']['packet_loss']
        self._has_metrics = True
