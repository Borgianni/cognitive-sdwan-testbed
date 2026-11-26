import tempfile
import json
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
from sdwan_rl.config import RLConfig
from sdwan_rl.telemetry_core import TelemetryConfig
from sdwan_rl.statistical_report import episode_windows, make_plan, validate_plan, report, read_inventory
from sdwan_rl.experiment_protocol import resolve_run


def manifest():
    def row(name,split,ranges):
        return {"id":name,"split":split,"episodes":4,"ticks_per_episode":2,"degrade_lte_ranges":ranges}
    return {"schema_version":1,"scenarios":[row("train","train",[[1,1]]),
        row("cal","calibration",[[2,2]]),row("test","held_out_test",[[2,3]])]}


def design():
    return make_plan(manifest(),"train","test",[1,2],RLConfig(),TelemetryConfig())


class StatisticalReportTests(unittest.TestCase):
    def test_windows_inclusive_union_and_recovery(self):
        self.assertEqual(episode_windows({"episodes":60,"degrade_lte_ranges":[[30,40]]}),
                         {"full":list(range(1,61)),"impairment":list(range(30,41)),
                          "last_impairment_episode":[40],"post_impairment":list(range(41,61))})
        self.assertEqual(episode_windows({"episodes":8,"events":[{"start_episode":2,"end_episode":4},
                          {"start_episode":4,"end_episode":6}]})["impairment"],list(range(2,7)))

    def test_plan_locks_family_and_settings(self):
        p = design()
        self.assertEqual(p,design())
        self.assertEqual(validate_plan(p),p)
        self.assertEqual(len(p["primary_family"]),4)
        self.assertEqual(len(p["evaluation_order"]),8)
        p["windows"]["test"]["impairment"] = [3]
        with self.assertRaises(ValueError):
            validate_plan(p)

    def test_missing_runs_exposed_and_no_tests(self):
        r = report(design(),[],[])
        self.assertEqual(r["status"],"incomplete_matrix_descriptive_only")
        self.assertFalse(r["primary_family_ready"])
        self.assertEqual(len(r["inventory"]["evaluation"]),8)
        self.assertTrue(all(row["status"]=="missing" for row in r["inventory"]["evaluation"]))
        self.assertTrue(all(c["effect"]["test"] is None for c in r["contrasts"]))
        self.assertTrue(all(c["inference"]["reject"] is False for c in r["contrasts"] if c["primary"]))

    def test_failed_trace_is_not_a_complete_replicate(self):
        p = design()
        h = {"kind":"run","agent_type":"dqn","run_seed":1,"mode":"online",
             "protocol":resolve_run(p["manifest"],"train","train","online"),
             "rl_config":p["rl_config"],"telemetry_schema":1,"telemetry_config":p["telemetry_config"],"run_id":"training-1"}
        end = {"kind":"failed","error":"synthetic failure","episode":1,"tick":0}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"failed.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in (h,end)))
            row = read_inventory(path,p,"train")
            self.assertEqual(row["status"],"failed")
            self.assertEqual(row["failure"],"synthetic failure")
            r = report(p,[path],[])
            self.assertEqual(r["inventory"]["training"][0]["status"],"failed")
            with self.assertRaisesRegex(ValueError,"Duplicate attempt"):
                report(p,[path,path],[])

    def test_rejects_wrong_config(self):
        p = design()
        h = {"kind":"run","agent_type":"dqn","run_seed":1,"mode":"online",
             "protocol":resolve_run(p["manifest"],"train","train","online"),
             "rl_config":asdict(RLConfig(w_a=4)),"telemetry_schema":1,"telemetry_config":p["telemetry_config"]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"wrong.jsonl"
            path.write_text(json.dumps(h))
            with self.assertRaisesRegex(ValueError,"locked plan"):
                read_inventory(path,p,"train")


if __name__ == "__main__":
    unittest.main()
