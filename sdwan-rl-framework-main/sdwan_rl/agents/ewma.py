"""Causal latency-only reference: EWMA of measured RTT, no learning."""
import math


class EWMALatencyAgent:
    """Use only the last available measurement of each tunnel.

    alpha weights the new sample. Reset estimates at episode boundaries
    to match the DQN runner's observation reset. Missing/invalid RTTs are
    excluded from the current comparison, rather than interpreted as zero.
    """
    name = "EWMA latency-only"
    tunnels = ("lte", "satellite")

    def __init__(self, alpha=0.3):
        if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
                or not math.isfinite(alpha) or not 0 < alpha <= 1):
            raise ValueError("EWMA alpha must be finite and in (0, 1]")
        self.alpha = float(alpha)
        self.reset()

    def reset(self):
        self.estimates = [None, None]
        self.last_action = 0

    @staticmethod
    def _rtt(metrics, tunnel):
        if not isinstance(metrics, dict):
            return None
        entry = metrics.get(tunnel)
        if not isinstance(entry, dict):
            return None
        value = entry.get("delay")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            return None
        return float(value)

    def decide(self, metrics):
        valid = [False, False]
        for i, tunnel in enumerate(self.tunnels):
            sample = self._rtt(metrics, tunnel)
            if sample is None:
                continue
            valid[i] = True
            old = self.estimates[i]
            self.estimates[i] = sample if old is None else self.alpha * sample + (1 - self.alpha) * old
        candidates = [i for i in range(2) if valid[i]]
        if candidates:
            # Explicit tie-break consistent with DQN argmax.
            action = min(candidates, key=lambda i: (self.estimates[i], i))
            reason = "minimum_smoothed_rtt" if len(candidates) == 2 else "only_valid_rtt"
        else:
            action = self.last_action
            reason = "no_valid_rtt_hold"
        self.last_action = action
        return action, {"ewma_rtt_ms": list(self.estimates),
                        "valid_rtt": valid, "reason": reason}
