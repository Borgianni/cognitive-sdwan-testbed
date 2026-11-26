import unittest
from copy import deepcopy
from unittest.mock import patch

from sdwan_rl.experiment_protocol import normalize_manifest, scenario_digest, resolve_run, validate_training_provenance
from sdwan_rl.scenario_profiles import normalize_v2, episode_profiles, novelty_report
from sdwan_rl.scenario_runtime import change_netem, flow_counter


def event(tunnel="lte", start=3, end=6, delay=150, loss=20):
    return dict(tunnel=tunnel, start_episode=start, end_episode=end, delay_ms=delay, loss_pct=loss)


def row(identifier="train", split="train", events=None, traffic=None):
    return dict(id=identifier, split=split, episodes=20, ticks_per_episode=20,
                events=[event()] if events is None else events,
                traffic_mbps={"lte": 20, "satellite": 0} if traffic is None else traffic)


def manifest():
    return {"schema_version": 2, "scenarios": [
        row(), row("cal", "calibration", [event(delay=100, loss=10)]),
        row("test", "held_out_test", [event("satellite", delay=250, loss=25)],
            {"lte": 0, "satellite": 80})]}


class ScenarioProfileTests(unittest.TestCase):
    def test_both_tunnels_and_recovery(self):
        scenario = row(events=[event(), event("satellite", 5, 8, 250, 25)])
        self.assertEqual(episode_profiles(scenario, 2)["lte"]["delay_ms"], 25)
        both = episode_profiles(scenario, 5)
        self.assertEqual(both["lte"]["delay_ms"], 150)
        self.assertEqual(both["satellite"]["delay_ms"], 250)
        self.assertEqual(episode_profiles(scenario, 7)["lte"]["loss_pct"], 2)
        self.assertEqual(episode_profiles(scenario, 9)["satellite"]["delay_ms"], 60)
        self.assertEqual(both["satellite"]["capacity_mbps"], 200)

    def test_canonical_merge_and_duplicate_rejected(self):
        a = row()
        b = row("test", "held_out_test", [event(start=5), event(end=4)])
        self.assertEqual(scenario_digest(a), scenario_digest(b))
        data = manifest()
        data["scenarios"][2] = b
        with self.assertRaisesRegex(ValueError, "Duplicate effective"):
            normalize_manifest(data)

    def test_invalid_events_and_traffic(self):
        for events in ([event(start=0)], [event(delay=10)], [event(loss=101)],
                       [event(), event(start=5, delay=200)], [event(delay=25, loss=2)]):
            with self.assertRaises(ValueError):
                normalize_v2(row(events=events))
        for traffic in ({"lte": -1, "satellite": 0}, {"lte": 200, "satellite": 50},
                        {"lte": 0, "satellite": float("nan")}):
            with self.assertRaises(ValueError):
                normalize_v2(row(traffic=traffic))

    def test_novelty_and_provenance(self):
        data = normalize_manifest(manifest())
        report = novelty_report(data["scenarios"])["relative_to_train"]["test"]
        self.assertEqual(report["unseen_delay_loss_pairs"], [[250, 25]])
        self.assertEqual(report["unseen_affected_tunnels"], ["satellite"])
        self.assertTrue(report["unseen_traffic_vector"])
        train = resolve_run(data, "train", "train", "online")
        test = resolve_run(data, "test", "held_out_test", "frozen")
        validate_training_provenance(train, test)
        altered = deepcopy(data)
        next(r for r in altered["scenarios"] if r["id"] == "train")["traffic_mbps"]["lte"] = 25
        with self.assertRaises(ValueError):
            validate_training_provenance(resolve_run(altered, "train", "train", "online"), test)

    def test_schema_mixing_rejected(self):
        data = manifest()
        data["schema_version"] = 1
        with self.assertRaises(ValueError):
            normalize_manifest(data)


class RuntimeCommandTests(unittest.TestCase):
    def test_netem_change_keeps_parent_and_queue(self):
        q = {"kind": "netem", "parent": "5:1", "handle": "10:"}
        profile = episode_profiles(row(), 4)["lte"]
        with patch("sdwan_rl.scenario_runtime.netem_child", side_effect=[q, q]), \
                patch("sdwan_rl.scenario_runtime.checked") as checked:
            self.assertEqual(change_netem(object(), "s1-eth1", profile), q)
            args = checked.call_args.args[1]
            self.assertEqual(args[:6], ["tc", "qdisc", "change", "dev", "s1-eth1", "parent"])
            self.assertIn("5:1", args)
            self.assertIn("150ms", args)
            self.assertIn("20%", args)
            self.assertEqual(args[-2:], ["limit", "750"])
            self.assertNotIn("root", args)

    def test_counter_parser_and_missing_rule(self):
        node = type("Node", (), {"name": "s1"})()
        with patch("sdwan_rl.scenario_runtime.checked", return_value="cookie=0xc100, n_packets=42, actions=output:2"):
            self.assertEqual(flow_counter(node, 0xc100), 42)
            with self.assertRaises(RuntimeError):
                flow_counter(node, 0xc101)


if __name__ == "__main__":
    unittest.main()
