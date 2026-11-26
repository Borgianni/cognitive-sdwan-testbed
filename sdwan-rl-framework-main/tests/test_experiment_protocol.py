import unittest
from copy import deepcopy

from sdwan_rl.experiment_protocol import (digest, normalize_manifest, resolve_run,
                                          validate_training_provenance)


def manifest():
    return {"schema_version": 1, "scenarios": [
        {"id": "train_a", "split": "train", "episodes": 60,
         "ticks_per_episode": 20, "degrade_lte_ranges": [[30, 40]]},
        {"id": "cal_a", "split": "calibration", "episodes": 20,
         "ticks_per_episode": 20, "degrade_lte_ranges": [[5, 8]]},
        {"id": "test_a", "split": "held_out_test", "episodes": 20,
         "ticks_per_episode": 20, "degrade_lte_ranges": [[11, 15]]},
    ]}


class ExperimentProtocolTests(unittest.TestCase):
    def test_valid_distinct_splits_and_fingerprint(self):
        data = manifest()
        train = resolve_run(data, "train_a", "train", "online")
        test = resolve_run(data, "test_a", "held_out_test", "frozen")
        validate_training_provenance(train, test)
        validate_training_provenance(train, resolve_run(data, "cal_a", "calibration", "frozen"))
        self.assertEqual(train["manifest_sha256"], test["manifest_sha256"])
        self.assertNotEqual(train["scenario_sha256"], test["scenario_sha256"])

    def test_renamed_identical_schedule_rejected_even_if_ranges_reordered(self):
        data = manifest()
        data["scenarios"][2].update(episodes=60, degrade_lte_ranges=[[36, 40], [30, 35]])
        with self.assertRaisesRegex(ValueError, "Duplicate effective"):
            normalize_manifest(data)

    def test_canonical_manifest_order_and_ranges(self):
        original = manifest()
        changed = deepcopy(original)
        changed["scenarios"][0]["degrade_lte_ranges"] = [[35, 40], [30, 35]]
        changed["scenarios"].reverse()
        self.assertEqual(digest(normalize_manifest(original)), digest(normalize_manifest(changed)))

    def test_mode_and_split_enforced(self):
        data = manifest()
        for scenario, split, mode in (("train_a", "train", "frozen"),
                                      ("test_a", "held_out_test", "online"),
                                      ("train_a", "held_out_test", "frozen"),
                                      ("unknown", "held_out_test", "frozen")):
            with self.subTest(scenario=scenario, split=split, mode=mode):
                with self.assertRaises(ValueError):
                    resolve_run(data, scenario, split, mode)
        self.assertEqual(resolve_run(None, None, "diagnostic", "frozen"), {"split": "diagnostic"})
        with self.assertRaises(ValueError):
            resolve_run(data, "test_a", "diagnostic", "frozen")

    def test_reject_old_or_mismatched_checkpoint_provenance(self):
        data = manifest()
        test = resolve_run(data, "test_a", "held_out_test", "frozen")
        train = resolve_run(data, "train_a", "train", "online")
        for provenance in (None, {"split": "diagnostic"}, {**train, "scenario_sha256": "wrong"}):
            with self.assertRaises(ValueError):
                validate_training_provenance(provenance, test)
        data["scenarios"][0]["degrade_lte_ranges"] = [[20, 30]]
        with self.assertRaisesRegex(ValueError, "manifests differ"):
            validate_training_provenance(resolve_run(data, "train_a", "train", "online"), test)

    def test_invalid_and_incomplete_manifests(self):
        data = manifest()
        data["scenarios"].pop()
        with self.assertRaises(ValueError):
            normalize_manifest(data)
        for ranges in ([[0, 1]], [[5, 4]], [[2, 61]], [[True, 4]]):
            data = manifest()
            data["scenarios"][0]["degrade_lte_ranges"] = ranges
            with self.assertRaises(ValueError):
                normalize_manifest(data)


if __name__ == "__main__":
    unittest.main()
