import unittest
from sdwan_rl.telemetry_audit import distribution, primary_summary
from sdwan_rl.probe_impact import plan
from sdwan_rl.experiment_protocol import normalize_manifest


def measurement(sent, received, echoes, rtt, probe_bytes=100):
    return {"window_start_s": 1, "window_end_s": 3,
            "probe_payload_bytes_both_directions": probe_bytes,
            "flow_evidence": {"probes": {}},
            "streams": {"primary": {"sent_packets": sent, "received_packets": received,
                                     "echo_received_packets": echoes, "rtt_ms": rtt,
                                     "received_payload_bytes": received * 100}}}


class TelemetryAuditTests(unittest.TestCase):
    def test_distribution(self):
        result = distribution([1, 2, 3, 4])
        self.assertEqual(result["p50"], 2.5)
        self.assertAlmostEqual(result["p95"], 3.85)
        self.assertEqual(distribution([None])["n"], 0)

    def test_primary_count_weighted_metrics(self):
        result = primary_summary([measurement(10, 8, 5, 50), measurement(20, 10, 10, 80)])
        self.assertEqual(result["primary_sent_packets"], 30)
        self.assertEqual(result["primary_loss_pct"], 40)
        self.assertEqual(result["primary_rtt_ms"], 70)
        self.assertEqual(result["probe_payload_bytes_both_directions"], 200)

    def test_no_echo_is_not_zero_delay(self):
        result = primary_summary([measurement(10, 0, 0, None, 0)])
        self.assertEqual(result["primary_loss_pct"], 100)
        self.assertIsNone(result["primary_rtt_ms"])

    def test_probe_plan_counterbalances_and_rejects_test(self):
        def row(identifier, split, start):
            return {"id": identifier, "split": split, "episodes": 4, "ticks_per_episode": 2,
                    "events": [{"tunnel": "lte", "start_episode": start, "end_episode": start,
                                "delay_ms": 150, "loss_pct": 20}],
                    "traffic_mbps": {"lte": 0, "satellite": 0}}
        manifest = normalize_manifest({"schema_version": 2, "scenarios": [
            row("tr", "train", 1), row("cal", "calibration", 2), row("te", "held_out_test", 3)]})
        value = plan(manifest, ["cal"], [1, 2], [0, 1])
        self.assertEqual(len(value["entries"]), 8)
        self.assertEqual(value, plan(manifest, ["cal"], [1, 2], [0, 1]))
        for i in range(0, 8, 2):
            self.assertEqual({r["probes"] for r in value["entries"][i:i+2]}, {True, False})
        with self.assertRaises(ValueError):
            plan(manifest, ["te"], [1])


if __name__ == "__main__":
    unittest.main()
