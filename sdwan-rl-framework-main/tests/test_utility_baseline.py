import unittest
from dataclasses import replace

from sdwan_rl.agents.utility import GreedyUtilityAgent, snapshot_utilities
from sdwan_rl.benchmark_audit import snapshot_gap, summarize
from sdwan_rl.config import RLConfig
from sdwan_rl.env import SDWANEnv


def snapshot(delay=50, loss=2, throughput=10):
    return {"lte": {"delay": delay, "packet_loss": loss, "throughput": throughput},
            "satellite": {"delay": 120, "packet_loss": 3, "throughput": 10}}


class UtilityBaselineTests(unittest.TestCase):
    def test_shared_reward_for_both_actions(self):
        for cfg in (RLConfig(), replace(RLConfig(), w_d=0.0, w_a=5.0)):
            m = snapshot(160, 9, 180)
            env = SDWANEnv(cfg)
            env.update_from_metrics(m)
            self.assertEqual(snapshot_utilities(m, cfg), tuple(env.reward(a) for a in (0, 1)))

    def test_responds_to_loss_and_load_at_constant_delay(self):
        agent = GreedyUtilityAgent()
        self.assertEqual(agent.decide(snapshot())[0], 0)
        self.assertEqual(agent.decide(snapshot(loss=90))[0], 1)
        self.assertEqual(agent.decide(snapshot(throughput=249.99))[0], 1)

    def test_cold_start_invalid_metrics_and_reset(self):
        agent = GreedyUtilityAgent()
        self.assertEqual(agent.decide(None)[0], 0)
        for invalid in (0, None, float("nan"), True):
            action, details = agent.decide(snapshot(delay=invalid))
            self.assertEqual(action, 1)
            self.assertIsNone(details["utility_by_action"])
        self.assertEqual(agent.decide(None)[0], 1)
        agent.reset()
        self.assertEqual(agent.decide(None)[0], 0)
        with self.assertRaises(ValueError):
            snapshot_utilities(snapshot(loss=101), RLConfig())

    def test_tie_break_prefers_lte(self):
        cfg = replace(RLConfig(), w_d=0.0, lte_capacity_mbps=200)
        m = snapshot(120, 3, 10)
        self.assertEqual(GreedyUtilityAgent(cfg).decide(m)[0], 0)

    def test_causal_and_hindsight_are_distinct(self):
        cfg = RLConfig()
        action, _ = GreedyUtilityAgent(cfg).decide(snapshot())
        self.assertEqual(snapshot_gap(snapshot(), action, cfg)["gap"], 0)
        later = snapshot(300, 20, 10)
        self.assertGreater(snapshot_gap(later, action, cfg)["gap"], 0)
        self.assertEqual(snapshot_gap(later, 1, cfg)["gap"], 0)

    def test_missing_snapshot_not_counted_as_zero_gap(self):
        rows = [snapshot_gap(None, 0, RLConfig()), snapshot_gap(snapshot(), 0, RLConfig())]
        result = summarize(rows)
        self.assertEqual(result["valid_paired_snapshots"], 1)
        self.assertEqual(result["unavailable_paired_snapshots"], 1)
        self.assertEqual(result["mean_snapshot_gap"], 0)
        self.assertIsNone(summarize([None])["mean_snapshot_gap"])


if __name__ == "__main__":
    unittest.main()
