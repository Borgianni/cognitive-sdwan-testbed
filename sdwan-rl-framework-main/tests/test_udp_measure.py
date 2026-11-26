import json
import subprocess
import sys
import time
import unittest
from pathlib import Path

from sdwan_rl.telemetry_core import stream_measurement


class UDPWorkerTests(unittest.TestCase):
    def test_local_loopback_sequence_echo(self):
        # Loopback validates protocol mechanics, not Mininet/network performance.
        worker = Path(__file__).resolve().parents[1] / "sdwan_rl" / "udp_measure.py"
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        spec = {"token": "01" * 16, "start_s": time.monotonic() + 1.0, "window_s": .5,
                "grace_s": .3, "server_ip": "127.0.0.1", "client_ip": "127.0.0.1",
                "streams": {"primary": {"port": port, "packets": 10, "payload_bytes": 64}}}
        args = [sys.executable, str(worker)]
        server = subprocess.Popen(args + ["server", json.dumps(spec)], text=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertEqual(server.stdout.readline().strip(), "READY")
            client = subprocess.run(args + ["client", json.dumps(spec)], capture_output=True,
                                    text=True, timeout=5, check=True)
            out, err = server.communicate(timeout=5)
            self.assertEqual(server.returncode, 0, err)
            result = stream_measurement(json.loads(client.stdout)["streams"]["primary"],
                                        json.loads(out)["streams"]["primary"], spec["streams"]["primary"],
                                        spec["start_s"], spec["start_s"] + .8, .5)
            self.assertEqual(result["loss_pct"], 0)
            self.assertEqual(result["echo_received_packets"], 10)
            self.assertGreater(result["rtt_ms"], 0)
        finally:
            if server.poll() is None:
                server.kill()
            server.communicate()


if __name__ == "__main__":
    unittest.main()
