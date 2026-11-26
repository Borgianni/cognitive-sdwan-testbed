import unittest
from dataclasses import asdict, replace
from unittest.mock import patch

from sdwan_rl.config import RLConfig
from sdwan_rl.reward_sensitivity import candidates, offline, make_plan, policy_report


def snapshot(delay=50, loss=2, throughput=10):
    return {"lte": {"delay": delay, "packet_loss": loss, "throughput": throughput},
            "satellite": {"delay": 120, "packet_loss": 3, "throughput": 10}}


def manifest():
    return {"schema_version": 1, "scenarios": [
        {"id": "train", "split": "train", "episodes": 3, "ticks_per_episode": 2, "degrade_lte_ranges": [[1, 1]]},
        {"id": "cal", "split": "calibration", "episodes": 3, "ticks_per_episode": 2, "degrade_lte_ranges": [[2, 2]]},
        {"id": "test", "split": "held_out_test", "episodes": 3, "ticks_per_episode": 2, "degrade_lte_ranges": [[3, 3]]},
    ]}


class RewardSensitivityTests(unittest.TestCase):
    def test_twelve_unique_candidates_and_one_factor(self):
        base = asdict(RLConfig())
        rows = list(candidates(RLConfig()))
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({tuple(asdict(c).values()) for _, c in rows}), 12)
        for _, cfg in rows[1:]:
            self.assertEqual(sum(v != base[k] for k, v in asdict(cfg).items()), 1)

    def test_offline_common_reference_gap(self):
        result = offline([snapshot(), snapshot(160, 9, 180), snapshot(300, 20)], RLConfig())
        self.assertEqual(result["rows"][0]["mean_common_reference_gap"], 0.0)
        self.assertTrue(all(r["mean_common_reference_gap"] >= 0 for r in result["rows"]))
        with self.assertRaises(ValueError):
            offline([], RLConfig())

    def test_plan_counts_and_guard(self):
        plan = make_plan(RLConfig(), manifest(), "train", ["test"], [1, 2])
        self.assertEqual(plan["training_runs"], 24)
        self.assertEqual(plan["frozen_runs"], 24)
        with self.assertRaises(ValueError):
            make_plan(RLConfig(), manifest(), "train", ["cal"], [1])
        with self.assertRaises(ValueError):
            make_plan(RLConfig(), manifest(), "train", ["test"], [1, 1])

    def test_policy_report_complete_and_incomplete(self):
        plan = make_plan(RLConfig(), manifest(), "train", ["test"], [1, 2])
        rows = []
        for config in plan["configs"]:
            for seed in plan["seeds"]:
                rows.append({"agent": "dqn", "rl_config": config["rl_config"], "seed": seed,
                             "scenario_id": "test", "manifest_sha256": plan["manifest_sha256"],
                             "training_protocol": {"scenario_id": "train"}, "model_weights_sha256": str(seed),
                             "mean_common_reference_reward": float(seed), "mean_common_reference_cost_index": 7,
                             "selected_delay_ms": 50, "selected_loss_pct": 2, "selected_residual_mbps": 240,
                             "lte_fraction": 1, "within_episode_switch_fraction": 0})
        with patch("sdwan_rl.reward_sensitivity.score_trace", side_effect=rows):
            result = policy_report(list(range(len(rows))), plan)
        self.assertEqual(result["summaries"][0]["metrics"]["mean_common_reference_reward"]["mean"], 1.5)
        with patch("sdwan_rl.reward_sensitivity.score_trace", side_effect=rows[:-1]):
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                policy_report(list(range(len(rows) - 1)), plan)


if __name__ == "__main__":
    unittest.main()
