import json
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

from sdwan_rl.udp_measure import packet


class ReorderingReceiverTests(unittest.TestCase):
    def test_out_of_order_unique_sequences_and_duplicates(self):
        worker = Path(__file__).resolve().parents[1] / "sdwan_rl" / "udp_measure.py"
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tmp:
            tmp.bind(("127.0.0.1", 0))
            port = tmp.getsockname()[1]
        spec = {"token": "01"*16, "start_s": time.monotonic()+.4, "window_s": .5,
                "grace_s": .3, "server_ip": "127.0.0.1", "client_ip": "127.0.0.1",
                "streams": {"secondary": {"port": port, "packets": 4, "payload_bytes": 64}}}
        proc = subprocess.Popen([sys.executable, str(worker), "server", json.dumps(spec)],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertEqual(proc.stdout.readline().strip(), "READY")
            time.sleep(max(0, spec["start_s"]-time.monotonic()))
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                for seq in (0, 2, 1, 1, 3):
                    sender.sendto(packet(spec["token"], seq, 64), ("127.0.0.1", port))
                    time.sleep(.01)
                sender.sendto(packet("02"*16, 0, 64), ("127.0.0.1", port))
            out, err = proc.communicate(timeout=5)
            self.assertEqual(proc.returncode, 0, err)
            result = json.loads(out)["streams"]["secondary"]
            self.assertEqual(result["received_unique"], 4)
            self.assertEqual(result["late_sequence_unique"], 1)
            self.assertEqual(result["max_sequence_lag"], 1)
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.communicate()


if __name__ == "__main__":
    unittest.main()
