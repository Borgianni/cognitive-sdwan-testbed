import unittest
from pathlib import Path

from sdwan_rl.experiment_protocol import read_manifest, resolve_run


class HeuristicProtocolTests(unittest.TestCase):
    def test_heuristic_is_nonlearning_evaluation_only(self):
        path = Path(__file__).resolve().parents[1] / "docs" / "c10_example_manifest.json"
        manifest = read_manifest(path)
        for scenario in manifest["scenarios"]:
            if scenario["split"] == "train":
                with self.assertRaises(ValueError):
                    resolve_run(manifest, scenario["id"], "train", "heuristic")
            else:
                self.assertEqual(resolve_run(manifest, scenario["id"], scenario["split"], "heuristic"),
                                 resolve_run(manifest, scenario["id"], scenario["split"], "frozen"))


if __name__ == "__main__":
    unittest.main()
