import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from sdwan_rl.config import RLConfig
from sdwan_rl.policy_audit import weights_digest, save_checkpoint, load_frozen


@unittest.skipIf(tf is None, "TensorFlow required for DQN integration tests")
class DQNTrainingTests(unittest.TestCase):
    def agent(self, seed=19):
        from sdwan_rl.agents.dqn import DQNAgent
        cfg = RLConfig(batch_size=2, dqn_replay_warmup=3, dqn_target_update_steps=2)
        return DQNAgent.from_config(cfg, seed=seed), cfg

    def add(self, agent, done=False):
        agent.remember(np.ones(6), 0, 2.0, np.ones(6) * .5, done)

    def test_architecture_and_initial_target(self):
        agent, _ = self.agent()
        self.assertEqual([layer.units for layer in agent.model.layers], [64, 32, 2])
        self.assertEqual(agent.model.count_params(), 2594)
        self.assertFalse(agent.target_model.trainable)
        for a, b in zip(agent.model.get_weights(), agent.target_model.get_weights()):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(agent.optimizer.global_clipnorm, 10.0)

    def test_warmup_batch_and_hard_update(self):
        agent, _ = self.agent()
        initial_target = [w.copy() for w in agent.target_model.get_weights()]
        for _ in range(2):
            self.add(agent)
            self.assertIsNone(agent.train_step())
            self.assertEqual(agent.eps, 1.0)
        self.add(agent, done=True)
        first = agent.train_step()
        self.assertEqual(first["optimizer_updates"], 1)
        self.assertEqual(int(agent.optimizer.iterations.numpy()), 1)
        self.assertFalse(first["target_synced"])
        for a, b in zip(initial_target, agent.target_model.get_weights()):
            np.testing.assert_array_equal(a, b)
        self.add(agent)
        second = agent.train_step()
        self.assertTrue(second["target_synced"])
        self.assertEqual(second["target_syncs"], 1)
        self.assertEqual(second["replay_size"], 4)
        self.assertAlmostEqual(agent.eps, .995 ** 2)
        for a, b in zip(agent.model.get_weights(), agent.target_model.get_weights()):
            np.testing.assert_array_equal(a, b)

    def test_seeded_initialization_and_actions(self):
        a, _ = self.agent()
        b, _ = self.agent()
        self.assertEqual(weights_digest(a), weights_digest(b))
        self.assertEqual([a.act(np.ones(6)) for _ in range(20)],
                         [b.act(np.ones(6)) for _ in range(20)])
        c, _ = self.agent(20)
        self.assertNotEqual(weights_digest(a), weights_digest(c))

    def test_frozen_checkpoint_roundtrip_and_no_updates(self):
        agent, cfg = self.agent()
        for _ in range(4):
            self.add(agent)
            agent.train_step()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.keras"
            save_checkpoint(agent, cfg, path)
            other, _ = self.agent()
            load_frozen(other, cfg, path)
            before = weights_digest(other)
            self.assertEqual(before, weights_digest(agent))
            self.add(other)
            self.assertIsNone(other.train_step())
            self.assertEqual(other.training_state(), {"optimizer_updates": 0,
                             "target_syncs": 0, "replay_size": 0, "epsilon": 0.0})
            self.assertEqual(weights_digest(other), before)


if __name__ == "__main__":
    unittest.main()
