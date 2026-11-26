import unittest
from dataclasses import replace

import numpy as np

from sdwan_rl.aliasing_audit import compare_encodings
from sdwan_rl.config import RLConfig
from sdwan_rl.env import SDWANEnv


def snapshot(delay=50, loss=2, throughput=10):
    return {"lte": {"delay": delay, "packet_loss": loss, "throughput": throughput},
            "satellite": {"delay": 120, "packet_loss": 3, "throughput": 10}}


class ContinuousStateTests(unittest.TestCase):
    def env(self, cfg, metrics):
        env = SDWANEnv(cfg)
        env.update_from_metrics(metrics)
        return env

    def test_normalized_feature_order_and_shape(self):
        env = self.env(RLConfig(state_representation="continuous"), snapshot())
        s = env.make_state().vec
        np.testing.assert_allclose(s, [100/150, 100/220, .98, .97, .96, .95])
        self.assertEqual(s.shape, (6,))
        self.assertEqual(s.dtype, np.float32)

    def test_continuous_distinguishes_c05_counterexample(self):
        cfg = RLConfig(state_representation="continuous")
        a, b = snapshot(), snapshot(160, 9, 180)
        s_a = self.env(cfg, a).make_state().vec
        s_b = self.env(cfg, b).make_state().vec
        self.assertTrue(np.all(s_a[[0, 2, 4]] > s_b[[0, 2, 4]]))
        np.testing.assert_array_equal(s_a[[1, 3, 5]], s_b[[1, 3, 5]])
        binary = replace(cfg, state_representation="binary")
        np.testing.assert_array_equal(self.env(binary, a).make_state().vec,
                                      self.env(binary, b).make_state().vec)

    def test_binary_thresholds_do_not_change_continuous_features(self):
        cfg = RLConfig(state_representation="continuous")
        other = replace(cfg, lte_delay_threshold_ms=700,
                        lte_loss_threshold_pct=0, lte_residual_threshold=.9,
                        bandwidth_state_mode="legacy_throughput")
        np.testing.assert_array_equal(self.env(cfg, snapshot()).make_state().vec,
                                      self.env(other, snapshot()).make_state().vec)

    def test_delay_scale_and_monotonicity(self):
        cfg = RLConfig(state_representation="continuous", continuous_delay_scale_ms=200)
        self.assertAlmostEqual(float(self.env(cfg, snapshot(200)).make_state().vec[0]), .5)
        values = [self.env(cfg, snapshot(d)).make_state().vec[0] for d in (25, 100, 300, 1000)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))

    def test_invalid_and_reset_are_finite_and_conservative(self):
        cfg = RLConfig(state_representation="continuous")
        for v in (-1, float("nan"), float("inf")):
            s = self.env(cfg, snapshot(v, v, v)).make_state().vec
            np.testing.assert_array_equal(s[[0, 2, 4]], [0, 0, 0])
            self.assertTrue(np.isfinite(s).all())
        env = self.env(cfg, snapshot(0, 101, 300))
        np.testing.assert_array_equal(env.make_state().vec[[0, 2, 4]], [0, 0, 0])
        env.reset_metrics()
        np.testing.assert_array_equal(env.make_state().vec, np.zeros(6))

    def test_idle_saturated_and_loss_boundaries(self):
        cfg = RLConfig(state_representation="continuous")
        s = self.env(cfg, snapshot(loss=0, throughput=0)).make_state().vec
        np.testing.assert_array_equal(s[[2, 4]], [1, 1])
        s = self.env(cfg, snapshot(loss=100, throughput=250)).make_state().vec
        np.testing.assert_array_equal(s[[2, 4]], [0, 0])

    def test_reward_invariant_to_encoding(self):
        cfg = RLConfig()
        a = self.env(cfg, snapshot(160, 9, 180))
        b = self.env(replace(cfg, state_representation="continuous"), snapshot(160, 9, 180))
        for action in (0, 1):
            self.assertEqual(a.reward(action), b.reward(action))

    def test_audit_compares_all_encodings_and_flags_singletons(self):
        cfg = RLConfig(state_representation="continuous")
        report = compare_encodings([snapshot(), snapshot(160, 9, 180)], cfg)
        self.assertGreater(report["revised_binary"]["mean_aliasing_gap"], 0)
        self.assertEqual(report["continuous"]["mean_aliasing_gap"], 0)
        self.assertEqual(report["continuous"]["singleton_observations"], 2)
        self.assertEqual(report["continuous"]["fraction_samples_in_singleton_observations"], 1)
        self.assertEqual(report["revised_binary"]["distinct_observations"], 1)

    def test_invalid_configuration(self):
        for v in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RLConfig(continuous_delay_scale_ms=v)
        with self.assertRaises(ValueError):
            RLConfig(state_representation="unknown")


if __name__ == "__main__":
    unittest.main()
