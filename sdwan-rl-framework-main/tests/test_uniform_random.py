import json
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from sdwan_rl.agents.uniform_random import UniformRandomAgent
from sdwan_rl.agents.utility import snapshot_utilities
from sdwan_rl.benchmark_audit import audit_trace, paired_report, snapshot_gap
from sdwan_rl.config import RLConfig
from sdwan_rl.experiment_protocol import resolve_run


def snapshot():
    return {"lte": {"delay": 50, "packet_loss": 2, "throughput": 10},
            "satellite": {"delay": 120, "packet_loss": 3, "throughput": 10}}


def manifest():
    return {"schema_version": 1, "scenarios": [
        {"id": "train", "split": "train", "episodes": 2, "ticks_per_episode": 2,
         "degrade_lte_ranges": [[1, 1]]},
        {"id": "cal", "split": "calibration", "episodes": 2, "ticks_per_episode": 2,
         "degrade_lte_ranges": [[1, 2]]},
        {"id": "test", "split": "held_out_test", "episodes": 2, "ticks_per_episode": 2,
         "degrade_lte_ranges": [[2, 2]]},
    ]}


def random_trace(seed):
    cfg = RLConfig()
    agent = UniformRandomAgent(seed)
    protocol = resolve_run(manifest(), "test", "held_out_test", "heuristic")
    rows = [{"kind": "run", "agent_type": "random", "mode": "heuristic",
             "checkpoint": None, "initial_weights_sha256": None, "run_seed": seed,
             "training_protocol": None, "training_seed": None, "protocol": protocol,
             "rl_config": asdict(cfg), "heuristic_config": agent.configuration, "episode_count": 2}]
    for ep in (1, 2):
        agent.reset()
        for tick in (0, 1):
            previous = None if tick == 0 else snapshot()
            action, details = agent.decide(previous)
            rows.append({"kind": "decision", "episode": ep, "tick": tick,
                         "action": action, "greedy_action": None, "q_values": None,
                         "epsilon": 0.0, "exploratory": False, "heuristic_details": details,
                         "decision_metrics": previous, "measurement_metrics": snapshot(),
                         "reward": snapshot_utilities(snapshot(), cfg)[action]})
    rows.append({"kind": "complete", "weights_sha256": None, "epsilon": 0.0, "replay_size": 0})
    return rows


class UniformRandomTests(unittest.TestCase):
    def test_seeded_actions_ignore_metrics_and_global_rng(self):
        a, b = UniformRandomAgent(17), UniformRandomAgent(17)
        before = random.getstate()
        self.assertEqual([a.decide(None)[0] for _ in range(100)],
                         [b.decide(snapshot())[0] for _ in range(100)])
        self.assertEqual(random.getstate(), before)

    def test_uniform_distribution_contract_and_reset(self):
        agent = UniformRandomAgent(5)
        reference = random.Random(5)
        for _ in range(50):
            agent.reset()
            action, details = agent.decide(None)
            self.assertEqual(action, reference.randrange(2))
            self.assertEqual(details["action_probabilities"], [.5, .5])

    def test_random_expected_utility_not_lower_bound(self):
        scores = snapshot_utilities(snapshot(), RLConfig())
        expected = snapshot_gap(snapshot(), 0, RLConfig())["expected_uniform_utility"]
        self.assertLess(min(scores), expected)
        self.assertLess(expected, max(scores))

    def test_invalid_configuration(self):
        for seed in (-1, True, 2**32, 1.5):
            with self.assertRaises(ValueError):
                UniformRandomAgent(seed)
        with self.assertRaises(ValueError):
            UniformRandomAgent(1, 0)

    def test_complete_seeded_trace_replay_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            data = random_trace(1)
            def write():
                path.write_text("\n".join(json.dumps(r) for r in data), encoding="utf-8")
            write()
            report = audit_trace(path)
            self.assertEqual(report["ticks"], 4)
            self.assertEqual(report["decision_time_snapshot"]["unavailable_paired_snapshots"], 2)
            self.assertEqual(report["subsequent_measurement_snapshot"]["valid_paired_snapshots"], 4)
            data[1]["action"] = 1 - data[1]["action"]
            write()
            with self.assertRaisesRegex(ValueError, "replayed policy"):
                audit_trace(path)

    def test_paired_matrix_requires_every_run(self):
        with tempfile.TemporaryDirectory() as folder:
            reports = []
            for seed in (1, 2):
                path = Path(folder) / str(seed)
                path.write_text("\n".join(json.dumps(r) for r in random_trace(seed)), encoding="utf-8")
                reports.append(audit_trace(path))
            result = paired_report(reports, [1, 2], ["random"])
            self.assertEqual(result["summaries"][0]["n_seeds"], 2)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                paired_report(reports[:1], [1, 2], ["random"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                paired_report(reports + reports[:1], [1, 2], ["random"])
            reports[1]["rl_config"]["w_a"] = 99
            with self.assertRaisesRegex(ValueError, "common configuration"):
                paired_report(reports, [1, 2], ["random"])


if __name__ == "__main__":
    unittest.main()
