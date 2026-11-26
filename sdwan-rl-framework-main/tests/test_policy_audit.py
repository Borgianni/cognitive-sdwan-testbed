import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sdwan_rl.config import RLConfig
from sdwan_rl.policy_audit import append_trace, compare_observations, weights_digest


class FakeFrozenAgent:
    frozen = True

    def __init__(self):
        self.model = self
        self.weights = np.array([1.0, 2.0], dtype=np.float32)

    def get_weights(self):
        return [self.weights.copy()]

    def q_values(self, state):
        # A deliberately constructed policy, NOT a trained DQN result.
        return np.array([state[0] + state[2], 0.5])


def snapshot(delay, loss):
    return {"lte": {"delay": delay, "packet_loss": loss, "throughput": 10.0},
            "satellite": {"delay": 120.0, "packet_loss": 3.0, "throughput": 10.0}}


class PolicyAuditTests(unittest.TestCase):
    def test_aliased_observations_have_same_greedy_action(self):
        cfg = RLConfig(lte_delay_threshold_ms=700, sat_delay_threshold_ms=1200,
                       lte_loss_threshold_pct=0, sat_loss_threshold_pct=0,
                       bandwidth_state_mode="legacy_throughput")
        result = compare_observations(FakeFrozenAgent(), cfg, snapshot(50, 2), snapshot(300, 20))
        self.assertTrue(result["aliased"])
        self.assertFalse(result["greedy_action_changed"])
        self.assertEqual(result["nominal"]["q_values"], result["degraded"]["q_values"])

    def test_distinct_observations_can_change_greedy_action(self):
        agent = FakeFrozenAgent()
        before = weights_digest(agent)
        result = compare_observations(agent, RLConfig(), snapshot(50, 2), snapshot(300, 20))
        self.assertFalse(result["aliased"])
        self.assertTrue(result["greedy_action_changed"])
        self.assertEqual(before, weights_digest(agent))

    def test_requires_frozen_agent(self):
        agent = FakeFrozenAgent()
        agent.frozen = False
        with self.assertRaises(ValueError):
            compare_observations(agent, RLConfig(), snapshot(50, 2), snapshot(300, 20))

    def test_rejects_invalid_snapshot(self):
        with self.assertRaises(ValueError):
            compare_observations(FakeFrozenAgent(), RLConfig(), snapshot(0, 2), snapshot(300, 20))

    def test_trace_strict_json_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            append_trace(path, {"rtt": float("inf")}, mode="x")
            append_trace(path, {"action": np.int64(1)})
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows, [{"rtt": None}, {"action": 1}])
            with self.assertRaises(FileExistsError):
                append_trace(path, {}, mode="x")


if __name__ == "__main__":
    unittest.main()
