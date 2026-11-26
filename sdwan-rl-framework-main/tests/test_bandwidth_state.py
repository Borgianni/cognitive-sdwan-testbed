import unittest

import numpy as np

from sdwan_rl.config import RLConfig
from sdwan_rl.env import SDWANEnv


class BandwidthStateTests(unittest.TestCase):
    def env(self, cfg, lte_thr, sat_thr):
        env = SDWANEnv(cfg)
        env.update_from_metrics({
            "lte": {"delay": 50.0, "throughput": lte_thr, "packet_loss": 2.0},
            "satellite": {"delay": 120.0, "throughput": sat_thr, "packet_loss": 3.0},
        })
        return env

    def test_load_only_impairment_changes_state(self):
        cfg = RLConfig()
        np.testing.assert_array_equal(self.env(cfg, 50, 40).make_state().vec, [1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(self.env(cfg, 225, 40).make_state().vec, [1, 1, 1, 1, 0, 1])
        np.testing.assert_array_equal(self.env(cfg, 50, 180).make_state().vec, [1, 1, 1, 1, 1, 0])

    def test_idle_and_saturated_links(self):
        cfg = RLConfig()
        np.testing.assert_array_equal(self.env(cfg, 0, 0).make_state().vec[4:], [1, 1])
        np.testing.assert_array_equal(self.env(cfg, 250, 250).make_state().vec[4:], [0, 0])

    def test_original_code_and_paper_predicates(self):
        legacy = RLConfig(bandwidth_state_mode="legacy_throughput")
        paper = RLConfig(bandwidth_state_mode="positive_residual")
        np.testing.assert_array_equal(self.env(legacy, 0, 200).make_state().vec[4:], [0, 1])
        np.testing.assert_array_equal(self.env(paper, 0, 200).make_state().vec[4:], [1, 0])
        np.testing.assert_array_equal(self.env(paper, 225, 180).make_state().vec[4:], [1, 1])

    def test_inclusive_boundary_and_independent_thresholds(self):
        cfg = RLConfig(lte_residual_threshold=0.20, sat_residual_threshold=0.25)
        np.testing.assert_array_equal(self.env(cfg, 200, 150).make_state().vec[4:], [1, 1])
        np.testing.assert_array_equal(self.env(cfg, 200.01, 150.01).make_state().vec[4:], [0, 0])

    def test_reset_and_invalid_measurements(self):
        env = SDWANEnv()
        np.testing.assert_array_equal(env.make_state().vec, [0, 0, 0, 0, 0, 0])
        for val in (-1, float("nan"), float("inf")):
            with self.subTest(val=val):
                np.testing.assert_array_equal(self.env(RLConfig(), val, val).make_state().vec[4:], [0, 0])
        env = self.env(RLConfig(), 0, 0)
        env.reset_metrics()
        np.testing.assert_array_equal(env.make_state().vec[4:], [0, 0])

    def test_fraction_bounds(self):
        f = SDWANEnv.residual_capacity_fraction
        self.assertEqual(f(0, 250), 1.0)
        self.assertEqual(f(225, 250), 0.1)
        self.assertEqual(f(300, 250), 0.0)
        self.assertTrue(np.isnan(f(-1, 250)))

    def test_invalid_configuration(self):
        for name in ("lte_residual_threshold", "sat_residual_threshold"):
            for value in (0, -1, 1.01, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        RLConfig(**{name: value})
        for name in ("lte_capacity_mbps", "sat_capacity_mbps"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        RLConfig(**{name: value})
        with self.assertRaises(ValueError):
            RLConfig(bandwidth_state_mode="unknown")

    def test_reward_unchanged_across_bandwidth_modes(self):
        configs = [RLConfig(bandwidth_state_mode=mode) for mode in
                   ("residual_threshold", "legacy_throughput", "positive_residual")]
        envs = [self.env(cfg, 225, 180) for cfg in configs]
        for action in (0, 1):
            self.assertEqual(len({env.reward(action) for env in envs}), 1)


if __name__ == "__main__":
    unittest.main()
