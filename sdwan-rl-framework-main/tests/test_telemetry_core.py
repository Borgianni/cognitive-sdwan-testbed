import unittest

from sdwan_rl.telemetry_core import (TelemetryConfig, throughput_mbps, stream_specs,
                                    stream_measurement, observation_age, observation_index)


class TelemetryCoreTests(unittest.TestCase):
    def test_counter_units_and_reset(self):
        self.assertAlmostEqual(throughput_mbps({"time_s": 1, "tx_bytes": 100},
                                              {"time_s": 3, "tx_bytes": 250100}), 1)
        with self.assertRaises(ValueError):
            throughput_mbps({"time_s": 1, "tx_bytes": 100}, {"time_s": 2, "tx_bytes": 0})

    def test_probe_budget_and_config_validation(self):
        cfg = TelemetryConfig()
        self.assertEqual(stream_specs(cfg)["lte"]["packets"], 100)
        self.assertEqual(set(stream_specs(cfg, False)), {"primary"})
        self.assertNotIn("observation_lag_ticks", cfg.measurement_settings())
        for kw in ({"window_s": 0}, {"probe_payload_bytes": 1}, {"probe_packets": True},
                   {"observation_lag_ticks": -1}):
            with self.assertRaises(ValueError):
                TelemetryConfig(**kw)

    def test_one_way_loss_distinct_from_echo_loss(self):
        client = {"sent": 100, "sent_payload_bytes": 12800, "echo_received_unique": 80,
                  "rtt_ms": 75, "first_send_s": 1.0, "last_send_s": 2.98,
                  "first_echo_s": 1.1, "last_echo_s": 3.1, "max_send_lateness_s": .001}
        server = {"received_unique": 90, "received_payload_bytes": 11520, "echo_sent": 90,
                  "echo_payload_bytes": 11520, "first_receive_s": 1.04, "last_receive_s": 3.03}
        out = stream_measurement(client, server, stream_specs(TelemetryConfig())["lte"], 1, 4.5, 2)
        self.assertEqual(out["loss_pct"], 10)
        self.assertEqual(out["roundtrip_missing_pct"], 20)
        self.assertEqual(out["rtt_ms"], 75)
        self.assertAlmostEqual(out["offered_payload_mbps"], .0512)
        client["sent"] = 99
        with self.assertRaises(ValueError):
            stream_measurement(client, server, stream_specs(TelemetryConfig())["lte"], 1, 4.5, 2)

    def test_total_loss_is_not_zero_delay(self):
        c = {"sent": 100, "sent_payload_bytes": 12800, "echo_received_unique": 0, "rtt_ms": None,
             "first_send_s": 1., "last_send_s": 2.98, "first_echo_s": None, "last_echo_s": None,
             "max_send_lateness_s": 0}
        s = {"received_unique": 0, "received_payload_bytes": 0, "echo_sent": 0,
             "echo_payload_bytes": 0, "first_receive_s": None, "last_receive_s": None}
        row = stream_measurement(c, s, stream_specs(TelemetryConfig())["lte"], 1, 4.5, 2)
        self.assertEqual(row["loss_pct"], 100)
        self.assertIsNone(row["rtt_ms"])

    def test_causal_lag_and_age(self):
        self.assertEqual([observation_index(i, 2) for i in range(7)], [None, 0, 0, 0, 1, 2, 3])
        spans = {"lte": {"rtt": {"start_s": 1, "end_s": 4}},
                 "satellite": {"rtt": {"start_s": 2, "end_s": 5}}}
        result = observation_age(spans, 6)
        self.assertEqual(result["oldest_support_age_s"], 5)
        self.assertEqual(result["oldest_completion_age_s"], 2)
        self.assertEqual(result["support_overlap_s"], 2)
        with self.assertRaises(ValueError):
            observation_age(spans, 4)


if __name__ == "__main__":
    unittest.main()
