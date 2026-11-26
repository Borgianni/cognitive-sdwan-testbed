import argparse
import json
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict
from sdwan_rl.config import RLConfig
from sdwan_rl.telemetry_core import TelemetryConfig
from sdwan_rl.config_files import add_config_files,resolve_configs,read_config
from sdwan_rl.experiment_protocol import read_manifest


class ConfigFileTests(unittest.TestCase):
    def test_reference_files_are_complete(self):
        root=Path(__file__).resolve().parents[1]/"docs"/"reference_smoke_v1"
        cfg=read_config(root/"rl.json",RLConfig)
        tele=read_config(root/"telemetry.json",TelemetryConfig)
        manifest=read_manifest(root/"manifest.json")
        self.assertEqual(cfg.batch_size,4)
        self.assertEqual(cfg.dqn_target_update_steps,2)
        self.assertEqual(tele.observation_lag_ticks,0)
        self.assertEqual(len(manifest["scenarios"]),3)

    def test_file_override_rejected_and_defaults_compatible(self):
        parser=argparse.ArgumentParser()
        parser.add_argument("--weight-delay",dest="w_b",type=float,default=2)
        add_config_files(parser)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"rl.json"
            p.write_text(json.dumps(asdict(RLConfig(w_b=3))))
            tokens=["--rl-config",str(p)]
            args=parser.parse_args(tokens)
            rl,t=resolve_configs(parser,args,RLConfig(),TelemetryConfig(),tokens)
            self.assertEqual(rl.w_b,3)
            self.assertEqual(t,TelemetryConfig())
            tokens+=["--weight-delay=4"]
            with self.assertRaisesRegex(ValueError,"overrides"):
                resolve_configs(parser,parser.parse_args(tokens),RLConfig(),TelemetryConfig(),tokens)
            p.write_text("{}")
            with self.assertRaises(ValueError):
                read_config(p,RLConfig)


if __name__=="__main__":
    unittest.main()
