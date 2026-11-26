"""REST control-plane client; acknowledged decisions precede traffic."""
import time
import requests
from .config import ControllerConfig


class ControllerAPI:
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else ControllerConfig()
        self.session = requests.Session()

    def wait_ready(self, timeout_s=15):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                response = self.session.get(self.cfg.base_url + "/ready", timeout=1)
                response.raise_for_status()
                if response.json().get("datapaths") == [1, 3]:
                    return
            except (requests.RequestException, ValueError):
                pass
            time.sleep(.2)
        raise RuntimeError("Ryu did not report both OpenFlow datapaths ready")

    def apply_decision(self, action, decision_id):
        if type(action) is not int or action not in (0, 1):
            raise ValueError("Invalid action")
        response = self.session.post(self.cfg.base_url + "/decision",
                                     json={"action": action, "decision_id": decision_id}, timeout=10)
        response.raise_for_status()
        ack = response.json()
        expected = "lte" if action == 0 else "satellite"
        if (ack.get("decision_id") != decision_id or ack.get("action") != action
                or ack.get("link") != expected or ack.get("datapaths") != [1, 3]
                or ack.get("barrier_acknowledged") is not True):
            raise RuntimeError("Controller acknowledgement does not match requested action")
        return ack

    def send_metrics(self, metrics):
        response = self.session.post(self.cfg.base_url + self.cfg.metrics_endpoint,
                                     json=metrics, timeout=5)
        response.raise_for_status()
        return response.text.strip()

    def get_link_decision(self):
        response = self.session.get(self.cfg.base_url + self.cfg.decision_endpoint, timeout=5)
        response.raise_for_status()
        return response.text.strip()

    def close(self):
        self.session.close()
