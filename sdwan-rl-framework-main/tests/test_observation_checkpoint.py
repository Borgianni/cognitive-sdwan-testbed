import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from sdwan_rl.config import RLConfig
from sdwan_rl.policy_audit import checkpoint_config, load_frozen


class ObservationCheckpointTests(unittest.TestCase):
    def test_legacy_checkpoint_defaults_to_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.keras"
            path.write_bytes(b"test checkpoint placeholder")
            fields = asdict(RLConfig())
            del fields["state_representation"]
            del fields["continuous_delay_scale_ms"]
            meta = {"rl_config": fields, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            Path(str(path) + ".json").write_text(json.dumps(meta))
            cfg, _ = checkpoint_config(path)
            self.assertEqual(cfg.state_representation, "binary")
            self.assertEqual(cfg.continuous_delay_scale_ms, 100)

    def test_mismatch_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.keras"
            path.write_bytes(b"test checkpoint placeholder")
            fields = asdict(RLConfig(state_representation="continuous"))
            meta = {"observation_schema": 2, "rl_config": fields,
                    "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            sidecar = Path(str(path) + ".json")
            sidecar.write_text(json.dumps(meta))
            cfg, _ = checkpoint_config(path)
            self.assertEqual(cfg.state_representation, "continuous")
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                load_frozen(None, RLConfig(), path)
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                load_frozen(None, RLConfig(state_representation="continuous",
                                           continuous_delay_scale_ms=200), path)
            del fields["state_representation"]
            sidecar.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "missing observation"):
                checkpoint_config(path)


if __name__ == "__main__":
    unittest.main()
