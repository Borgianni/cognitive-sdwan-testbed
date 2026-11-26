import random
import unittest
from pathlib import Path

from sdwan_rl.experiment_protocol import digest, normalize_manifest, resolve_run, validate_training_provenance
from sdwan_rl.scenario_generation import (
    DEFAULT_SEEDS, generate, overlaps, temporal_signature,
)
from sdwan_rl.experiment_protocol import read_manifest


def template():
    return read_manifest(Path(__file__).resolve().parents[1] / "docs" / "c10_example_manifest.json")


class ScenarioGenerationTests(unittest.TestCase):
    def test_reproducibility_and_global_rng_isolation(self):
        data = template()
        before = random.getstate()
        first = generate(data)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(first, generate(data))
        self.assertEqual(first[1]["manifest_sha256"], digest(first[0]))

    def test_distinct_timing_and_nominal_guards(self):
        data = template()
        manifest, receipt = generate(data)
        rows = manifest["scenarios"]
        self.assertEqual(len(rows), 17)
        self.assertEqual(len({temporal_signature(r) for r in rows}), len(rows))
        self.assertFalse({temporal_signature(r) for r in rows}
                         & {temporal_signature(r) for r in data["scenarios"]})
        for row in rows:
            low, high = receipt["duration_bounds_episodes"][row["split"]]
            for event in row["events"]:
                self.assertGreaterEqual(event["start_episode"], 3)
                self.assertLessEqual(event["end_episode"], row["episodes"] - 2)
                self.assertTrue(low <= event["end_episode"] - event["start_episode"] + 1 <= high)
        test_events = [e for row in rows if row["split"] == "held_out_test" for e in row["events"]]
        self.assertGreater(len({e["start_episode"] for e in test_events}), 1)
        self.assertGreater(len({e["end_episode"] - e["start_episode"] + 1 for e in test_events}), 1)

    def test_profile_and_overlap_preservation(self):
        data = template()
        bases = {r["id"]: r for r in data["scenarios"]}
        manifest, receipt = generate(data)
        provenance = {r["id"]: r["template_id"] for r in receipt["generated_scenarios"]}
        for row in manifest["scenarios"]:
            base = bases[provenance[row["id"]]]
            self.assertEqual(row["traffic_mbps"], base["traffic_mbps"])
            self.assertEqual([(e["tunnel"], e["delay_ms"], e["loss_pct"]) for e in row["events"]],
                             [(e["tunnel"], e["delay_ms"], e["loss_pct"]) for e in base["events"]])
            for i, a in enumerate(row["events"]):
                for j, b in enumerate(row["events"][i + 1:], i + 1):
                    if a["tunnel"] == b["tunnel"]:
                        self.assertGreaterEqual(b["start_episode"] - a["end_episode"] - 1, 1)
                    else:
                        self.assertEqual(overlaps(a, b), overlaps(base["events"][i], base["events"][j]))

    def test_test_seed_changes_only_test_schedules(self):
        a, _ = generate(template())
        b, _ = generate(template(), seeds={**DEFAULT_SEEDS, "held_out_test": 3304})
        self.assertEqual([r for r in a["scenarios"] if r["split"] != "held_out_test"],
                         [r for r in b["scenarios"] if r["split"] != "held_out_test"])
        self.assertNotEqual(a, b)

    def test_provenance_compatible_with_existing_frozen_runner(self):
        manifest, _ = generate(template())
        self.assertEqual(normalize_manifest(manifest), manifest)
        train = resolve_run(manifest, "train_reference_r001", "train", "online")
        test = resolve_run(manifest, "test_joint_r001", "held_out_test", "frozen")
        validate_training_provenance(train, test)

    def test_invalid_or_impossible_settings(self):
        for kwargs in ({"seeds": {**DEFAULT_SEEDS, "held_out_test": DEFAULT_SEEDS["train"]}},
                       {"nominal": 0},
                       {"durations": {"train": [5, 15], "calibration": [3, 9],
                                      "held_out_test": [18, 19]}}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    generate(template(), **kwargs)
        with self.assertRaises(ValueError):
            generate(template(), replicas={"train": 1, "calibration": 1, "held_out_test": 100},
                     durations={"train": [5, 15], "calibration": [3, 4], "held_out_test": [3, 4]},
                     max_attempts=100)


if __name__ == "__main__":
    unittest.main()
