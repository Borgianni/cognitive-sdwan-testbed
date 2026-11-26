import argparse
import unittest
from dataclasses import asdict, replace

import numpy as np

from sdwan_rl.config import RLConfig
from sdwan_rl.dqn_core import td_targets, replay_ready, target_sync_due, decay_epsilon
from sdwan_rl.dqn_settings import DQN_FIELDS, CURRENT_DQN, LEGACY_DQN, add_dqn_arguments, dqn_kwargs
from sdwan_rl.runtime_environment import runtime_environment


class DQNCoreTests(unittest.TestCase):
    def test_terminal_target_and_target_max(self):
        q = np.array([[2, 4], [50, 60], [-2, -3]])
        np.testing.assert_allclose(td_targets([1, 7, -1], q, [False, True, False], .95),
                                   [4.8, 7, -2.9], rtol=1e-6)

    def test_bad_target_inputs_rejected(self):
        for rewards, q, dones, gamma in (([1], [[1, 2], [3, 4]], [False], .95),
                                        ([1], [[float("nan"), 2]], [False], .95),
                                        ([1], [[1, 2]], [False], 1)):
            with self.assertRaises(ValueError):
                td_targets(rewards, q, dones, gamma)

    def test_schedule_boundaries(self):
        self.assertFalse(replay_ready(31, 32, False))
        self.assertTrue(replay_ready(32, 32, False))
        self.assertFalse(replay_ready(100, 32, True))
        self.assertFalse(target_sync_due(0, 100))
        self.assertFalse(target_sync_due(99, 100))
        self.assertTrue(target_sync_due(100, 100))
        self.assertTrue(target_sync_due(200, 100))
        self.assertAlmostEqual(decay_epsilon(1, .01, .995), .995)
        self.assertEqual(decay_epsilon(.01, .01, .995), .01)

    def test_settings_and_cli(self):
        cfg = RLConfig()
        self.assertEqual(cfg.dqn_version, CURRENT_DQN)
        self.assertEqual((cfg.dqn_replay_warmup, cfg.dqn_target_update_steps), (32, 100))
        for change in ({"gamma": 1}, {"dqn_replay_warmup": 31}, {"dqn_target_update_steps": 0},
                       {"dqn_gradient_clipnorm": float("nan")}, {"dqn_deterministic_ops": 1},
                       {"eps_end": 2}, {"replay_size": 31}):
            with self.assertRaises(ValueError):
                replace(cfg, **change)
        parser = argparse.ArgumentParser()
        add_dqn_arguments(parser)
        other = RLConfig(**dqn_kwargs(parser.parse_args(["--dqn-target-update-steps", "50",
                                                        "--dqn-deterministic-ops"])))
        self.assertEqual(other.dqn_target_update_steps, 50)
        self.assertTrue(other.dqn_deterministic_ops)

    def test_old_record_is_legacy_not_new_algorithm(self):
        fields = asdict(RLConfig())
        for key in DQN_FIELDS:
            fields.pop(key)
        self.assertEqual(RLConfig.from_record(fields).dqn_version, LEGACY_DQN)
        self.assertEqual(RLConfig.from_record(asdict(RLConfig())), RLConfig())
        with self.assertRaises(ValueError):
            RLConfig.from_record({**fields, "dqn_version": CURRENT_DQN})

    def test_runtime_versions_are_explicit(self):
        info = runtime_environment()
        self.assertTrue(info["python"])
        self.assertTrue(info["packages"]["numpy"])
        self.assertIn("tensorflow", info["packages"])
        self.assertIn("ryu", info["packages"])


if __name__ == "__main__":
    unittest.main()
