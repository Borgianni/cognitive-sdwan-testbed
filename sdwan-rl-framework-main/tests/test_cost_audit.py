import math
import unittest
from dataclasses import replace

from sdwan_rl.config import RLConfig
from sdwan_rl.cost_audit import cost_qos_summary
from sdwan_rl.env import SDWANEnv


def row(cfg, action):
    metrics = {"lte": {"delay": 50, "packet_loss": 2, "throughput": 10,
                       "cost_index": cfg.lte_cost_index},
               "satellite": {"delay": 120, "packet_loss": 3, "throughput": 10,
                             "cost_index": cfg.sat_cost_index}}
    env = SDWANEnv(cfg)
    env.update_from_metrics(metrics)
    terms = env.reward_terms(action)
    return {"action": action, "reward": terms["reward"], "measurement_metrics": metrics,
            "reward_terms": terms}


class CostAuditTests(unittest.TestCase):
    def test_ratio_of_sums_not_mean_of_ratios(self):
        cfg = RLConfig()
        rows = [row(cfg, 0), row(cfg, 1)]
        result = cost_qos_summary(rows, cfg)
        self.assertEqual(result["total_selected_cost_index"], 27)
        self.assertEqual(result["mean_selected_cost_index"], 13.5)
        self.assertEqual(result["lte_selection_fraction"], .5)
        self.assertAlmostEqual(result["reward_to_cost_ratio"], sum(r["reward"] for r in rows) / 27)
        self.assertNotAlmostEqual(result["reward_to_cost_ratio"],
                                  sum(r["reward"] / r["reward_terms"]["cost_index"] for r in rows) / 2)
        self.assertAlmostEqual(result["mean_qos_utility_valid_selected"],
                               sum(r["reward_terms"]["qos_utility"] for r in rows) / 2)

    def test_reward_shift_affects_ratio_even_without_action_change(self):
        cfg = RLConfig()
        other = replace(cfg, threshold_delay_ms=280)
        a = cost_qos_summary([row(cfg, 0), row(cfg, 1)], cfg)
        b = cost_qos_summary([row(other, 0), row(other, 1)], other)
        self.assertEqual(a["mean_selected_cost_index"], b["mean_selected_cost_index"])
        self.assertAlmostEqual(b["reward_to_cost_ratio"] - a["reward_to_cost_ratio"],
                               2 * cfg.w_b * math.log(2) / 27)

    def test_cost_free_reward_still_records_cost(self):
        cfg = replace(RLConfig(), w_d=0)
        result = cost_qos_summary([row(cfg, 1)], cfg)
        self.assertEqual(result["mean_selected_cost_index"], 20)
        self.assertEqual(result["mean_qos_utility_valid_selected"], result["total_logged_reward"])

    def test_mismatched_cost_and_reward_rejected(self):
        cfg = RLConfig()
        data = row(cfg, 0)
        data["measurement_metrics"]["lte"]["cost_index"] = 5
        with self.assertRaisesRegex(ValueError, "cost"):
            cost_qos_summary([data], cfg)
        data = row(cfg, 0)
        data["reward_terms"]["cost_term"] = 0
        with self.assertRaisesRegex(ValueError, "decomposition"):
            cost_qos_summary([data], cfg)

    def test_legacy_missing_selected_telemetry_coverage(self):
        cfg = RLConfig(reward_mode="legacy")
        data = row(cfg, 0)
        data["measurement_metrics"]["lte"]["delay"] = 0
        data.pop("reward_terms")
        result = cost_qos_summary([data], cfg)
        self.assertEqual(result["unavailable_selected_snapshots"], 1)
        self.assertIsNone(result["mean_qos_utility_valid_selected"])
        self.assertEqual(result["total_selected_cost_index"], 7)
        with self.assertRaises(ValueError):
            cost_qos_summary([data], RLConfig())


if __name__ == "__main__":
    unittest.main()
