import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from sdwan_rl.config import RLConfig
from sdwan_rl.experiment_protocol import resolve_run
from sdwan_rl.learning_curves import report, read_run
from sdwan_rl.policy_audit import validate_run_seed


def manifest():
    return {"schema_version": 1, "scenarios": [
        {"id": "train", "split": "train", "episodes": 2, "ticks_per_episode": 2,
         "degrade_lte_ranges": [[1, 1]]},
        {"id": "cal", "split": "calibration", "episodes": 2, "ticks_per_episode": 2,
         "degrade_lte_ranges": [[1, 2]]},
        {"id": "test", "split": "held_out_test", "episodes": 2, "ticks_per_episode": 2,
         "degrade_lte_ranges": [[2, 2]]},
    ]}


def rows(seed, evaluation=False):
    data = manifest()
    train = resolve_run(data, "train", "train", "online")
    protocol = resolve_run(data, "test", "held_out_test", "frozen") if evaluation else train
    final = f"{seed + 10:064x}"
    header = {"kind": "run", "mode": "frozen" if evaluation else "online",
              "agent_type": "dqn", "run_seed": seed, "protocol": protocol,
              "rl_config": asdict(RLConfig()), "episode_count": 2,
              "initial_weights_sha256": final if evaluation else f"{seed:064x}",
              "training_protocol": train if evaluation else None,
              "training_seed": seed if evaluation else None}
    decisions = [{"kind": "decision", "episode": ep, "tick": t, "reward": seed + ep + t,
                  "epsilon": 0.0 if evaluation else .5, "exploratory": False,
                  "action": 0, "greedy_action": 0, "q_values": [1, 0]}
                 for ep in (1, 2) for t in (0, 1)]
    return [header, *decisions, {"kind": "complete", "weights_sha256": final,
                                 "epsilon": 0 if evaluation else .5,
                                 "replay_size": 0 if evaluation else 4}]


class LearningCurveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def write(self, name, data):
        path = Path(self.directory.name) / name
        path.write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
        return path

    def test_episode_curves_use_independent_runs(self):
        a, b = self.write("a", rows(1)), self.write("b", rows(3))
        result = report([b, a])
        self.assertEqual(result["training"]["seeds"], [1, 3])
        point = result["training"]["curves"][0]["total_reward"]
        self.assertEqual(point["n"], 2)
        self.assertEqual(point["mean"], 7)
        self.assertAlmostEqual(point["sample_sd"], 8 ** .5)
        self.assertIsNone(result["held_out"])

    def test_reject_duplicate_seeds_and_initial_weights(self):
        a = self.write("a", rows(1))
        with self.assertRaises(ValueError):
            report([a, a])
        b = rows(2)
        b[0]["initial_weights_sha256"] = rows(1)[0]["initial_weights_sha256"]
        with self.assertRaisesRegex(ValueError, "initial weights"):
            report([a, self.write("b", b)])

    def test_reject_incomplete_invalid_or_out_of_order_trace(self):
        for data in (rows(1)[:-1], rows(1)[:2] + rows(1)[3:]):
            with self.assertRaises(ValueError):
                read_run(self.write("bad", data), "train")
        for key, value in (("reward", float("nan")), ("tick", 1), ("action", 1)):
            data = rows(1)
            data[1][key] = value
            with self.assertRaises(ValueError):
                read_run(self.write("bad", data), "train")

    def test_frozen_matrix_and_weight_pairing(self):
        training = [self.write("t1", rows(1)), self.write("t2", rows(2))]
        evaluation = [self.write("e1", rows(1, True)), self.write("e2", rows(2, True))]
        result = report(training, evaluation)
        self.assertEqual(result["held_out"]["scenarios"][0]["mean_tick_reward_across_seeds"]["n"], 2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            report(training, evaluation[:1])
        data = rows(1, True)
        data[0]["initial_weights_sha256"] = data[-1]["weights_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            report(training, [self.write("wrong", data), evaluation[1]])

    def test_frozen_exploration_and_changed_weights_rejected(self):
        for key, value in (("exploratory", True), ("epsilon", .1)):
            data = rows(1, True)
            data[1][key] = value
            with self.assertRaises(ValueError):
                read_run(self.write("bad", data), "held_out_test")
        data = rows(1, True)
        data[-1]["weights_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            read_run(self.write("bad", data), "held_out_test")

    def test_seed_validation(self):
        for value in (None, True, -1, 2**32, 1.5):
            with self.assertRaises(ValueError):
                validate_run_seed(value, required=True)
        validate_run_seed(0, required=True)
        validate_run_seed(None)


if __name__ == "__main__":
    unittest.main()
