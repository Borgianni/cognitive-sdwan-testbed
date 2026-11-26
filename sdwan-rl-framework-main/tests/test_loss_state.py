import unittest

import numpy as np

from sdwan_rl.config import RLConfig
from sdwan_rl.env import SDWANEnv


class PacketLossStateTests(unittest.TestCase):
    def state(self, cfg, lte_loss, sat_loss, lte_delay=25.0):
        env = SDWANEnv(cfg)
        env.update_from_metrics({
            "lte": {"delay": lte_delay, "throughput": 10.0, "packet_loss": lte_loss},
            "satellite": {"delay": 60.0, "throughput": 10.0, "packet_loss": sat_loss},
        })
        return env.make_state().vec

    def test_nominal_and_degraded_lte_are_distinguishable(self):
        cfg = RLConfig()
        np.testing.assert_array_equal(self.state(cfg, 2.0, 3.0), [1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(self.state(cfg, 20.0, 3.0, 150.0), [1, 1, 0, 1, 1, 1])
        np.testing.assert_array_equal(self.state(cfg, 2.0, 20.0), [1, 1, 1, 0, 1, 1])

    def test_original_zero_loss_encoding(self):
        cfg = RLConfig(lte_loss_threshold_pct=0.0, sat_loss_threshold_pct=0.0)
        np.testing.assert_array_equal(self.state(cfg, 2.0, 3.0), [1, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(self.state(cfg, 20.0, 3.0, 150.0), [1, 1, 0, 0, 1, 1])
        np.testing.assert_array_equal(self.state(cfg, 0.0, 0.0), [1, 1, 1, 1, 1, 1])

    def test_per_tunnel_thresholds_and_inclusive_boundary(self):
        cfg = RLConfig(lte_loss_threshold_pct=5.0, sat_loss_threshold_pct=8.0)
        np.testing.assert_array_equal(self.state(cfg, 5.0, 8.0)[2:4], [1, 1])
        np.testing.assert_array_equal(self.state(cfg, 5.01, 8.01)[2:4], [0, 0])

    def test_invalid_measurements_fail_closed(self):
        np.testing.assert_array_equal(self.state(RLConfig(), -1.0, float("nan"))[2:4], [0, 0])

    def test_invalid_thresholds(self):
        for name in ("lte_loss_threshold_pct", "sat_loss_threshold_pct"):
            for value in (-1.0, 101.0, float("nan"), float("inf")):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(ValueError):
                        RLConfig(**{name: value})

    def test_independent_default_configs_and_reset(self):
        first, second = SDWANEnv(), SDWANEnv()
        first.cfg.lte_loss_threshold_pct = 5.0
        self.assertEqual(second.cfg.lte_loss_threshold_pct, 10.0)
        np.testing.assert_array_equal(second.make_state().vec, [0, 0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
