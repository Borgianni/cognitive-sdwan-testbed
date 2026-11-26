import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import tensorflow  # noqa: F401
except ImportError:
    tensorflow = None

from sdwan_rl.config import RLConfig
from sdwan_rl.policy_audit import load_frozen, save_checkpoint, weights_digest


@unittest.skipIf(tensorflow is None, "TensorFlow required for DQN integration tests")
class FrozenDQNTests(unittest.TestCase):
    def test_frozen_agent_does_not_learn_or_explore(self):
        from sdwan_rl.agents.dqn import DQNAgent
        agent = DQNAgent(6, 2, batch_size=1)
        state = np.ones(6, dtype=np.float32)
        agent.remember(state, 0, 1.0, state, False)
        agent.freeze()
        before = weights_digest(agent)
        expected = int(np.argmax(agent.q_values(state)))
        for _ in range(5):
            self.assertEqual(agent.act(state), expected)
            action, _, exploratory = agent.traced_action(state)
            self.assertEqual(action, expected)
            self.assertFalse(exploratory)
            agent.remember(state, 1, -2.0, state, False)
            agent.train_step()
        self.assertEqual(weights_digest(agent), before)
        self.assertEqual(len(agent.memory), 1)
        self.assertEqual(agent.eps, 0.0)

    def test_checkpoint_roundtrip_and_config_mismatch(self):
        from sdwan_rl.agents.dqn import DQNAgent
        cfg = RLConfig()
        agent = DQNAgent(6, 2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.keras"
            save_checkpoint(agent, cfg, path)
            other = DQNAgent(6, 2)
            load_frozen(other, cfg, path)
            self.assertTrue(other.frozen)
            self.assertEqual(weights_digest(agent), weights_digest(other))
            with self.assertRaises(ValueError):
                load_frozen(other, RLConfig(lte_loss_threshold_pct=0), path)


if __name__ == "__main__":
    unittest.main()
