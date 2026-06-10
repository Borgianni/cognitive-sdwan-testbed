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
    def __init__(self, cfg: RLConfig = RLConfig()):
        self.cfg = cfg
        self.ticks = 0
        self.reset_metrics()

    def reset_metrics(self):
        self.lte = dict(delay=float('inf'), thr=0.0, loss=100.0)
        self.sat = dict(delay=float('inf'), thr=0.0, loss=100.0)

    def make_state(self) -> State:
        lte_delay_ok = 1 if self.lte['delay'] < 700 else 0
        sat_delay_ok = 1 if self.sat['delay'] < 1200 else 0
        lte_loss_ok  = 1 if self.lte['loss'] == 0 else 0
        sat_loss_ok  = 1 if self.sat['loss'] == 0 else 0
        lte_bw_ok    = 1 if self.lte['thr'] > 0 else 0
        sat_bw_ok    = 1 if self.sat['thr'] > 0 else 0
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
