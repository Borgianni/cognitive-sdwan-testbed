import unittest
from sdwan_rl.scale_cases import (CASES, architecture, catalog, make_scenario, validate_scenario,
                                  walk, primary_rules, secondary_rules, static_rules, path_residual,
                                  episode_profiles, digest)


class ScaleCaseTests(unittest.TestCase):
    def test_dimensions(self):
        for n, params in ((2, 2594), (3, 2819), (4, 3044)):
            self.assertEqual(architecture(n)["input"], 3 * n)
            self.assertEqual(architecture(n)["output"], n)
            self.assertEqual(architecture(n)["online_trainable_parameters"], params)

    def test_catalog_routes_and_distinct_ports(self):
        for case in CASES:
            graph = catalog(case)
            for site in graph["sites"]:
                ports = [e[s + "_port"] for e in graph["edges"] for s in ("a", "b") if e[s] == site]
                self.assertEqual(len(set(ports)), len(ports))
                self.assertNotIn(1, ports)
            for i, path in enumerate(graph["paths"]):
                rules = primary_rules(graph, i)
                self.assertEqual(len(rules), 2 * (len(path["edges"]) + 1))
                self.assertEqual(len({r["cookie"] for r in rules}), len(rules))
                self.assertEqual(rules[0]["site"], 1)
                self.assertEqual(rules[len(rules)//2-1]["site"], 2)
            with self.assertRaises(ValueError):
                primary_rules(graph, len(graph["paths"]))

    def test_transit_and_competing_flow_share_cb(self):
        graph = catalog("triangle3")
        path = graph["paths"][2]
        hops = walk(graph, path["edges"], 1, 2)
        self.assertEqual([r[0] for r in hops], [1, 3, 2])
        scenario = make_scenario("triangle3", "held_out_test")
        load = next(r for r in scenario["traffic"] if r["src"] == 3)
        self.assertIn(load["edges"][0], path["edges"])
        self.assertEqual(load["rate_mbps"], 150)
        with self.assertRaises(ValueError):
            walk(graph, ["ac", "ac", "p1"], 1, 2)

    def test_bottleneck_residual(self):
        graph = catalog("triangle3")
        result = path_residual(graph, graph["paths"][2], {"ac": 20, "cb": 170})
        self.assertEqual(result["nominal_bottleneck_mbps"], 180)
        self.assertEqual(result["residual_mbps"], 10)
        self.assertEqual(result["equivalent_load_mbps"], 170)
        with self.assertRaises(ValueError):
            path_residual(graph, graph["paths"][2], {"ac": 20})

    def test_split_and_recovery(self):
        tr = make_scenario("triangle3", "train")
        te = make_scenario("triangle3", "held_out_test")
        self.assertEqual(digest(tr["graph"]), digest(te["graph"]))
        self.assertNotEqual(digest(tr), digest(te))
        ev = te["events"][0]
        self.assertEqual(episode_profiles(te, ev["start_episode"])["cb"]["loss_pct"], 22)
        self.assertEqual(episode_profiles(te, te["episodes"])["cb"]["loss_pct"], .5)
        te["traffic"][0]["rate_mbps"] = 99
        with self.assertRaises(ValueError):
            validate_scenario(te)

    def test_static_rules_and_rr(self):
        s = make_scenario("triangle3", "calibration")
        for mode in ("off", "pinned", "rr"):
            rules = static_rules(s, mode)
            self.assertEqual(len({r["cookie"] for r in rules}), len(rules))
        rr = secondary_rules(s["graph"], "rr")
        self.assertEqual(sum(r["out_port"] == "controller" for r in rr), 1)
        self.assertEqual(len([r for r in rr if r["site"] == 2 and "udp_dst" in r["match"]]), 3)
        self.assertEqual(len([r for r in rr if r["site"] == 3 and "udp_dst" in r["match"]]), 1)


if __name__ == "__main__":
    unittest.main()
