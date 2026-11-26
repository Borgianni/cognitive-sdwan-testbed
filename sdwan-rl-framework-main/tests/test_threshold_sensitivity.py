import unittest
from dataclasses import replace

from sdwan_rl.config import RLConfig
from sdwan_rl.threshold_sensitivity import (
    DEFAULT_GRID, FIELDS, configurations, make_plan, sweep, thresholds, validate_grid,
)


def snapshot(delay=50, loss=2, throughput=10):
    return {"lte": {"delay": delay, "packet_loss": loss, "throughput": throughput},
            "satellite": {"delay": 120, "packet_loss": 3, "throughput": 10}}


class ThresholdSensitivityTests(unittest.TestCase):
    def test_factorial_has_729_distinct_configurations(self):
        rows = list(configurations(RLConfig(), DEFAULT_GRID, "factorial"))
        self.assertEqual(len(rows), 729)
        self.assertEqual(len({tuple(thresholds(c).values()) for _, c in rows}), 729)

    def test_ofat_changes_one_parameter_at_a_time(self):
        base = RLConfig()
        rows = list(configurations(base, DEFAULT_GRID))
        self.assertEqual(len(rows), 13)
        self.assertEqual(rows[0], ("baseline", base))
        for _, cfg in rows[1:]:
            self.assertEqual(sum(getattr(cfg, k) != getattr(base, k) for k in FIELDS), 1)

    def test_sweep_preserves_oracle_reward_and_detects_threshold_effect(self):
        rows = sweep([snapshot(), snapshot(300, 20)], RLConfig(), DEFAULT_GRID, "ofat")
        self.assertEqual(len({r["mean_snapshot_oracle_reward"] for r in rows}), 1)
        self.assertEqual(rows[0]["feature_mean"]["lte_delay"], .5)
        self.assertEqual(rows[0]["feature_mean"]["lte_loss"], .5)
        by_id = {r["id"]: r for r in rows}
        self.assertEqual(by_id["lte_delay_threshold_ms_level2"]["feature_mean"]["lte_delay"], 1.0)
        self.assertEqual(by_id["lte_loss_threshold_pct_level2"]["feature_mean"]["lte_loss"], 1.0)

    def test_continuous_input_config_does_not_change_binary_sweep(self):
        cfg = RLConfig(state_representation="continuous")
        rows = list(configurations(cfg, DEFAULT_GRID))
        self.assertEqual(cfg.state_representation, "continuous")
        self.assertTrue(all(c.state_representation == "binary" for _, c in rows))

    def test_custom_grid_validation(self):
        for grid in ({}, {**DEFAULT_GRID, "lte_loss_threshold_pct": [5, 20]},
                     {**DEFAULT_GRID, "lte_residual_threshold": [0, .2, .4]},
                     {**DEFAULT_GRID, "lte_delay_threshold_ms": [100, 175, 175]}):
            with self.subTest(grid=grid):
                with self.assertRaises(ValueError):
                    validate_grid(grid, RLConfig())
        custom = {**DEFAULT_GRID, "lte_delay_threshold_ms": [100, 200, 400]}
        self.assertEqual(validate_grid(custom, replace(RLConfig(), lte_delay_threshold_ms=200))
                         ["lte_delay_threshold_ms"], [100, 200, 400])

    def test_manifest_counts_and_split_roles(self):
        plan = make_plan(RLConfig(), DEFAULT_GRID, [7, 9])
        self.assertEqual(plan["status"], "planned_not_executable")
        self.assertEqual(plan["configurations"], 16)
        self.assertEqual(plan["training_runs"], 32)
        self.assertEqual(plan["frozen_evaluations"], 32)
        self.assertTrue(all(e["training_split"] != e["evaluation_split"] and
                            e["evaluation_mode"] == "frozen" for e in plan["entries"]))
        self.assertEqual(make_plan(RLConfig(), DEFAULT_GRID, [7], False)["training_runs"], 13)
        with self.assertRaises(ValueError):
            make_plan(RLConfig(), DEFAULT_GRID, [7, 7])


if __name__ == "__main__":
    unittest.main()
