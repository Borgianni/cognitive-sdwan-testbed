import unittest
from sdwan_rl.scale_runtime import parse_table, rule_deltas


class ScaleReadbackTests(unittest.TestCase):
    def test_numeric_and_controller_outputs(self):
        lines = ("cookie=0xa0000000, n_packets=2, n_bytes=200, actions=output:3\n"
                 "cookie=0xa00c8000, n_packets=10, n_bytes=500, actions=CONTROLLER:65535\n"
                 "cookie=0x0, n_packets=0, n_bytes=0, actions=drop")
        result = parse_table(lines)
        self.assertEqual(result[0xa0000000]["out_port"], 3)
        self.assertEqual(result[0xa00c8000]["out_port"], "controller")

    def test_deltas_and_wrong_route(self):
        rules = [{"site": 1, "cookie": 0xa0000000, "out_port": 3}]
        before = {1: {0xa0000000: {"packets": 2, "bytes": 200, "out_port": 3}}, 2: {}}
        after = {1: {0xa0000000: {"packets": 5, "bytes": 500, "out_port": 3}}, 2: {}}
        result = rule_deltas(rules, before, after)
        self.assertEqual(result[str(0xa0000000)]["packets"], 3)
        after[1][0xa0000000]["out_port"] = 2
        with self.assertRaises(RuntimeError):
            rule_deltas(rules, before, after)

    def test_unexpected_cookie_and_reset(self):
        before = {1: {1: {"packets": 2, "bytes": 20, "out_port": 2}}}
        rules = [{"site": 1, "cookie": 1, "out_port": 2}]
        with self.assertRaises(RuntimeError):
            rule_deltas(rules, before, {1: {}})
        with self.assertRaises(RuntimeError):
            rule_deltas(rules, before, {1: {1: {"packets": 1, "bytes": 10, "out_port": 2}}})


if __name__ == "__main__":
    unittest.main()
