"""Read-only/mock regression tests; no root or network changes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import subprocess

from sdwan_rl.network_guard import (
    assert_names_free, experiment_lock, inventory, prepare_network,
    stop_network, topology_names,
)


def topology():
    t = Mock()
    t.nodes.return_value = ["h1", "s1", "s3"]
    t.links.return_value = [("h1", "s1", {"intfName1": "h1-s1", "intfName2": "s1-h1"})]
    return t


class NetworkGuardTests(unittest.TestCase):
    def test_unrelated_bridge_allowed(self):
        with patch("sdwan_rl.network_guard.inventory", return_value={"s4", "enp88s0", "tailscale0"}):
            assert_names_free(topology())

    def test_bridge_and_interface_collision_refused_before_mininet(self):
        for existing in ("s1", "h1-s1"):
            with patch("sdwan_rl.network_guard.inventory", return_value={existing}):
                with self.assertRaisesRegex(RuntimeError, existing):
                    prepare_network(topology())

    def test_inventory_includes_unavailable_ovs_port(self):
        output = ['[{"ifname":"enp88s0"}]',
                  '{"headings":["name"],"data":[["s4"]]}',
                  '{"headings":["name"],"data":[["s1-h1"]]}',
                  '{"headings":["name"],"data":[["missing-port"]]}']
        with patch("sdwan_rl.network_guard.subprocess.check_output", side_effect=output):
            self.assertEqual(inventory(), {"enp88s0", "s4", "s1-h1", "missing-port"})

    def test_inventory_failure_refuses_start(self):
        with patch("sdwan_rl.network_guard.inventory", side_effect=subprocess.TimeoutExpired("ip", 10)):
            with self.assertRaises(subprocess.TimeoutExpired):
                prepare_network(topology())

    def test_duplicate_interface_rejected(self):
        t = topology()
        t.links.return_value[0][2]["intfName2"] = "h1-s1"
        with self.assertRaises(ValueError):
            topology_names(t)

    def test_stop_only_owned_network_and_check_residue(self):
        net = Mock(topo=topology())
        with patch("sdwan_rl.network_guard.inventory", return_value={"s4"}):
            stop_network(net)
        net.stop.assert_called_once_with()
        with patch("sdwan_rl.network_guard.inventory", return_value={"s1"}):
            with self.assertRaisesRegex(RuntimeError, "s1"):
                stop_network(net)

    def test_lock_rejects_concurrent_runner_and_releases(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "test.lock")
            with experiment_lock(path):
                with self.assertRaisesRegex(RuntimeError, "host lock"):
                    with experiment_lock(path):
                        self.fail("Unexpected acquisition")
            with experiment_lock(path):
                pass


if __name__ == "__main__":
    unittest.main()
