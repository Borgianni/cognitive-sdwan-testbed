import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from sdwan_rl.config import RLConfig
from sdwan_rl.aliasing_audit import audit, compare_encodings, evaluate_snapshot, snapshots_from_trace


def snapshot(delay=50, loss=2, throughput=10):
    return {"lte": {"delay": delay, "packet_loss": loss, "throughput": throughput},
            "satellite": {"delay": 120, "packet_loss": 3, "throughput": 10}}


class AliasingAuditTests(unittest.TestCase):
    def test_revised_binary_still_aliases_conflicting_actions(self):
        cfg = RLConfig()
        a, b = snapshot(), snapshot(160, 9, 180)
        s_a, r_a = evaluate_snapshot(a, cfg)
        s_b, r_b = evaluate_snapshot(b, cfg)
        self.assertEqual(s_a, s_b)
        self.assertGreater(r_a[0], r_a[1])
        self.assertLess(r_b[0], r_b[1])
        report = audit([a, b], cfg)
        self.assertEqual(report["conflicting_observations"], 1)
        self.assertEqual(report["fraction_samples_in_conflicting_observations"], 1.0)
        self.assertAlmostEqual(report["mean_aliasing_gap"], (r_b[1] - r_b[0]) / 2)
        self.assertAlmostEqual(report["mean_aliasing_gap"], 1.3517117217523284)

    def test_revised_thresholds_distinguish_severe_impairment(self):
        reports = compare_encodings([snapshot(), snapshot(300, 20, 10)], RLConfig())
        self.assertEqual(reports["legacy_code"]["distinct_observations"], 1)
        self.assertEqual(reports["legacy_paper"]["distinct_observations"], 1)
        self.assertEqual(reports["revised_binary"]["distinct_observations"], 2)
        self.assertAlmostEqual(reports["revised_binary"]["mean_aliasing_gap"], 0.0)
        self.assertGreater(reports["legacy_code"]["mean_aliasing_gap"], 0.0)

    def test_identical_preferences_have_zero_gap(self):
        report = audit([snapshot(), snapshot(55, 2, 10)], RLConfig())
        self.assertEqual(report["conflicting_observations"], 0)
        self.assertAlmostEqual(report["mean_aliasing_gap"], 0.0)

    def test_sample_weighted_gap(self):
        a, b, c = snapshot(), snapshot(160, 9, 180), snapshot(300, 20, 10)
        pair = audit([a, b], RLConfig())
        mixed = audit([a, b, c], RLConfig())
        self.assertAlmostEqual(mixed["mean_aliasing_gap"], pair["mean_aliasing_gap"] * 2 / 3)

    def test_empty_and_invalid_data_rejected(self):
        for rows in ([], [snapshot(delay=0)], [snapshot(loss=float("nan"))],
                     [snapshot(throughput=-1)], [{}]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    audit(rows, RLConfig())

    def test_trace_uses_decision_metrics_and_skips_reset(self):
        rows = [
            {"kind": "run", "mode": "frozen", "initial_weights_sha256": "abc",
             "rl_config": asdict(RLConfig())},
            {"kind": "decision", "tick": 0, "decision_metrics": None,
             "measurement_metrics": snapshot()},
            {"kind": "decision", "tick": 1, "decision_metrics": snapshot(),
             "measurement_metrics": snapshot(300, 20)},
            {"kind": "complete", "weights_sha256": "abc", "epsilon": 0, "replay_size": 0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            snapshots, cfg, skipped = snapshots_from_trace(path)
            self.assertEqual(snapshots, [snapshot()])
            self.assertEqual(cfg, RLConfig())
            self.assertEqual(skipped, 1)
            rows[-1]["weights_sha256"] = "changed"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with self.assertRaises(ValueError):
                snapshots_from_trace(path)


if __name__ == "__main__":
    unittest.main()
