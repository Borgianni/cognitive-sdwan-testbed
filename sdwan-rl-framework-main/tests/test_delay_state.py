import unittest

import numpy as np

from sdwan_rl.config import RLConfig
from sdwan_rl.env import SDWANEnv


class DelayStateTests(unittest.TestCase):
    def state(self, cfg, lte_rtt, sat_rtt):
        env = SDWANEnv(cfg)
        env.update_from_metrics({
            "lte": {"delay": lte_rtt, "throughput": 10.0, "packet_loss": 2.0},
            "satellite": {"delay": sat_rtt, "throughput": 10.0, "packet_loss": 3.0},
        })
        return env.make_state().vec

    def test_delay_only_impairment_changes_observation(self):
        cfg = RLConfig()
        np.testing.assert_array_equal(self.state(cfg, 50, 120), [1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(self.state(cfg, 300, 120), [0, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(self.state(cfg, 50, 300), [1, 0, 1, 1, 1, 1])

    def test_original_thresholds_do_not_detect_main_impairment(self):
        cfg = RLConfig(lte_delay_threshold_ms=700, sat_delay_threshold_ms=1200,
                       lte_loss_threshold_pct=0, sat_loss_threshold_pct=0)
        np.testing.assert_array_equal(self.state(cfg, 50, 120), [1, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(self.state(cfg, 300, 120), [1, 1, 0, 0, 1, 1])

    def test_independent_thresholds_and_inclusive_boundary(self):
        cfg = RLConfig(lte_delay_threshold_ms=100, sat_delay_threshold_ms=200)
        np.testing.assert_array_equal(self.state(cfg, 100, 200)[:2], [1, 1])
        np.testing.assert_array_equal(self.state(cfg, 100.01, 200)[:2], [0, 1])
        np.testing.assert_array_equal(self.state(cfg, 100, 200.01)[:2], [1, 0])

    def test_invalid_or_missing_rtt_is_not_acceptable(self):
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                np.testing.assert_array_equal(self.state(RLConfig(), value, value)[:2], [0, 0])

    def test_invalid_thresholds(self):
        for name in ("lte_delay_threshold_ms", "sat_delay_threshold_ms"):
            for value in (0.0, -1.0, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        RLConfig(**{name: value})

    def test_reward_is_independent_of_observation_thresholds(self):
        first = SDWANEnv(RLConfig())
        second = SDWANEnv(RLConfig(lte_delay_threshold_ms=700, sat_delay_threshold_ms=1200))
        for env in (first, second):
            env.update_from_metrics({
                "lte": {"delay": 300.0, "throughput": 10.0, "packet_loss": 2.0},
                "satellite": {"delay": 120.0, "throughput": 10.0, "packet_loss": 3.0},
            })
        for action in (0, 1):
            self.assertEqual(first.reward(action), second.reward(action))
        self.assertFalse(np.array_equal(first.make_state().vec, second.make_state().vec))


if __name__ == "__main__":
    unittest.main()
