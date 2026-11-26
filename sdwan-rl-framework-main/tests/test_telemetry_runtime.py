import unittest

from sdwan_rl.telemetry_runtime import flow_evidence


def table(action, packet_count=0, offpath=0):
    result = {}
    for dpid in (1, 3):
        def row(p, n):
            return {"output": p, "packets": n, "bytes": n * 100}
        result[dpid] = {
            0xd100 + dpid: row(action + 1, 0),
            0xd300 + dpid: row(action + 1, packet_count),
            **{0xd200 + dpid * 16 + p: row(3, packet_count if p == action + 1 else offpath)
               for p in (1, 2)},
        }
    for i in range(2):
        c, p = 0xc200 + i * 4, i + 1
        for d, cookie, out in ((1, c, p), (3, c + 1, 3), (3, c + 2, p), (1, c + 3, 3)):
            result[d][cookie] = row(out, packet_count)
    return result


class FlowEvidenceTests(unittest.TestCase):
    def test_selected_path_and_probe_counters(self):
        out = flow_evidence(table(1), table(1, 10), 1)
        self.assertEqual(out["primary"]["3"]["selected_tunnel_receive"]["packets"], 10)
        self.assertEqual(out["probes"]["lte"]["forward_ingress"]["bytes"], 1000)

    def test_missing_or_wrong_rule_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "forwarding rule"):
            flow_evidence(table(1), table(0, 10), 1)

    def test_offpath_and_absent_ingress_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "non-selected"):
            flow_evidence(table(0), table(0, 10, 1), 0)
        with self.assertRaisesRegex(RuntimeError, "No primary"):
            flow_evidence(table(0), table(0), 0)


if __name__ == "__main__":
    unittest.main()
