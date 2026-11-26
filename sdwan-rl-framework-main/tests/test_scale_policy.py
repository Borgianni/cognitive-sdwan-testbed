import unittest
import numpy as np
from sdwan_rl.config import RLConfig
from sdwan_rl.scale_cases import catalog, path_residual, make_scenario
from sdwan_rl.scale_policy import PathEnvironment, policy_signature
from sdwan_rl.telemetry_core import TelemetryConfig
from sdwan_rl.scale_audit import stream_summary


class ScalePolicyTests(unittest.TestCase):
    def test_feature_order_and_reused_reward(self):
        cfg = RLConfig(state_representation="continuous")
        for case in ("parallel2", "parallel3", "parallel4", "triangle3"):
            graph = catalog(case)
            env = PathEnvironment(graph, cfg)
            self.assertEqual(env.state().shape, (3*env.n,))
            np.testing.assert_array_equal(env.state(), np.zeros(3*env.n))
            rates = {e["id"]: 10 for e in graph["edges"]}
            metrics = {p["id"]: {"delay": 100*(i+1), "packet_loss": i+1,
                                   **path_residual(graph, p, rates)}
                       for i,p in enumerate(graph["paths"])}
            env.update(metrics)
            np.testing.assert_allclose(env.state()[:env.n], [1/(2+i) for i in range(env.n)])
            np.testing.assert_allclose(env.state()[env.n:2*env.n], [1-(i+1)/100 for i in range(env.n)])
            for i,p in enumerate(graph["paths"]):
                self.assertEqual(env.terms(i)["cost_index"], p["cost_index"])
                self.assertEqual(env.terms(i)["residual_mbps"], metrics[p["id"]]["residual_mbps"])

    def test_signatures_differ_across_catalogs_not_splits(self):
        cfg = RLConfig(state_representation="continuous")
        t = TelemetryConfig()
        a = policy_signature(make_scenario("parallel3", "train"), cfg, t)
        b = policy_signature(make_scenario("parallel3", "held_out_test"), cfg, t)
        c = policy_signature(make_scenario("triangle3", "train"), cfg, t)
        self.assertEqual(a,b)
        self.assertNotEqual(a,c)
        self.assertEqual(a["architecture"], c["architecture"])

    def test_no_reward_for_unavailable_rtt(self):
        graph = catalog("parallel3")
        env = PathEnvironment(graph, RLConfig(state_representation="continuous"))
        rates = {e["id"]: 10 for e in graph["edges"]}
        metrics = {p["id"]: {"delay": None if i == 0 else 120, "packet_loss": 10,
                              **path_residual(graph,p,rates)} for i,p in enumerate(graph["paths"])}
        env.update(metrics)
        self.assertEqual(env.state()[0],0)
        with self.assertRaises(ValueError):
            env.terms(0)
        action, values = env.greedy()
        self.assertNotEqual(action,0)
        self.assertIsNone(values[0])

    def test_sequence_summary_is_receive_weighted(self):
        def m(sent, received, late):
            return {"window_start_s": 1, "window_end_s": 3,
                    "streams": {"secondary": {"sent_packets": sent,"received_packets": received,
                                  "echo_received_packets": received, "rtt_ms": 50,
                                  "received_payload_bytes": 100*received,
                                  "late_sequence_unique": late,"max_sequence_lag": late}}}
        result = stream_summary([m(10,5,2),m(20,15,3)], "secondary")
        self.assertEqual(result["late_sequence_fraction"], .25)
        self.assertEqual(result["late_sequence_unique"], 5)
        self.assertAlmostEqual(result["loss_pct"], 100/3)


if __name__ == "__main__":
    unittest.main()
