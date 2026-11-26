import argparse
import math
import unittest
from dataclasses import asdict, replace

from sdwan_rl.config import RLConfig
from sdwan_rl.env import SDWANEnv
from sdwan_rl.agents.utility import snapshot_utilities
from sdwan_rl.reward_cli import add_reward_arguments, reward_kwargs


def metrics(delay=50, loss=2, throughput=10):
    return {"lte": {"delay": delay, "packet_loss": loss, "throughput": throughput},
            "satellite": {"delay": 120, "packet_loss": 3, "throughput": 10}}


def env(cfg, data=None):
    obj = SDWANEnv(cfg)
    obj.update_from_metrics(metrics() if data is None else data)
    return obj


class RewardTests(unittest.TestCase):
    def test_historical_scale_preserved_on_positive_valid_metrics(self):
        a, b = RLConfig(), RLConfig(reward_mode="legacy")
        for action in (0, 1):
            self.assertAlmostEqual(env(a).reward(action), env(b).reward(action), places=12)

    def test_unit_invariance_with_explicit_references(self):
        cfg = RLConfig()
        base = env(cfg).reward(0)
        # Re-express both measured quantities and their physical references.
        alternate = replace(cfg, threshold_delay_ms=140000,
                            lte_capacity_mbps=250000, sat_capacity_mbps=200000,
                            reward_bandwidth_ref_mbps=1000)
        self.assertAlmostEqual(env(alternate, metrics(50000, 2, 10000)).reward(0), base)
        self.assertAlmostEqual(math.log1p((2 / 100) / .01), math.log1p(2 / 1))

    def test_zero_loss_and_saturation_no_artificial_floor(self):
        terms = env(RLConfig(), metrics(loss=0, throughput=300)).reward_terms(0)
        self.assertEqual(terms["loss_term"], -3.0)
        self.assertEqual(terms["bandwidth_term"], 0.0)
        self.assertEqual(terms["residual_mbps"], 0.0)

    def test_cost_configuration_is_shared(self):
        cfg = replace(RLConfig(), lte_cost_index=12)
        obj = env(cfg)
        terms = obj.reward_terms(0)
        self.assertAlmostEqual(terms["cost_term"], -2.4)
        self.assertEqual(terms["cost_index"], 12)
        self.assertEqual(terms["reward"], terms["qos_utility"] + terms["cost_term"])
        self.assertEqual(snapshot_utilities(metrics(), cfg)[0], obj.reward(0))

    def test_delay_reference_common_additive_shift(self):
        cfg = RLConfig()
        other = replace(cfg, threshold_delay_ms=280)
        shifts = [env(other).reward(a) - env(cfg).reward(a) for a in (0, 1)]
        for value in shifts:
            self.assertAlmostEqual(value, cfg.w_b * math.log(2))

    def test_invalid_selected_metrics_not_rewarded(self):
        for data in (metrics(delay=0), metrics(loss=-1), metrics(throughput=float("nan"))):
            with self.assertRaises(ValueError):
                env(RLConfig(), data).reward(0)
        with self.assertRaises(ValueError):
            SDWANEnv(RLConfig()).reward(0)

    def test_configuration_validation_and_cli(self):
        for kwargs in ({"reward_bandwidth_ref_mbps": 0}, {"reward_loss_ref_fraction": 2},
                       {"threshold_delay_ms": float("nan")}, {"w_a": -1},
                       {"sat_cost_index": 0}, {"reward_mode": "bad"}):
            with self.assertRaises(ValueError):
                RLConfig(**kwargs)
        parser = argparse.ArgumentParser()
        add_reward_arguments(parser)
        cfg = RLConfig(**reward_kwargs(parser.parse_args(["--weight-cost", "0",
                                                           "--reward-delay-ref-ms", "200",
                                                           "--sat-cost-index", "35"])))
        self.assertEqual((cfg.w_d, cfg.threshold_delay_ms, cfg.sat_cost_index), (0, 200, 35))

    def test_old_records_are_legacy_not_reinterpreted(self):
        fields = asdict(RLConfig())
        for key in ("reward_mode", "reward_bandwidth_ref_mbps", "reward_loss_ref_fraction",
                    "lte_cost_index", "sat_cost_index"):
            fields.pop(key)
        self.assertEqual(RLConfig.from_record(fields).reward_mode, "legacy")
        self.assertEqual(RLConfig.from_record(asdict(RLConfig())), RLConfig())
        with self.assertRaises(ValueError):
            RLConfig.from_record({**fields, "reward_mode": "normalized"})


if __name__ == "__main__":
    unittest.main()
