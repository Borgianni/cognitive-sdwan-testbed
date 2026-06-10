import requests
from .config import ControllerConfig

class ControllerAPI:
    def __init__(self, cfg: ControllerConfig = ControllerConfig()):
        self.cfg = cfg

    def send_metrics(self, metrics: dict) -> None:
        url = self.cfg.base_url + self.cfg.metrics_endpoint
        r = requests.post(url, json=metrics, timeout=5)
        r.raise_for_status()

    def get_link_decision(self) -> str:
        url = self.cfg.base_url + self.cfg.decision_endpoint
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        # controller returns "lte" or "satellite"
        return r.text.strip()
