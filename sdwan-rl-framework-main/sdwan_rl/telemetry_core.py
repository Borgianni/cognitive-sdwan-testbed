"""Measurement semantics and causal observation timing, independent of Mininet."""
import math
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TelemetryConfig:
    window_s: float = 2.0
    grace_s: float = 1.5
    probe_packets: int = 100
    probe_payload_bytes: int = 128
    primary_packets: int = 500
    primary_payload_bytes: int = 512
    observation_lag_ticks: int = 0
    carry_observations_across_episodes: bool = False

    def __post_init__(self):
        for key in ("window_s", "grace_s"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("probe_packets", "primary_packets"):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 100000:
                raise ValueError(f"{key} must be in [1,100000]")
        for key in ("probe_payload_bytes", "primary_payload_bytes"):
            value = getattr(self, key)
            if type(value) is not int or not 20 <= value <= 1400:
                raise ValueError(f"{key} must be in [20,1400]")
        if type(self.observation_lag_ticks) is not int or self.observation_lag_ticks < 0:
            raise ValueError("Observation lag must be a nonnegative number of ticks")

        if type(self.carry_observations_across_episodes) is not bool:
            raise ValueError("carry_observations_across_episodes must be boolean")

    def measurement_settings(self):
        return {k: v for k, v in asdict(self).items()
                if k not in ("observation_lag_ticks", "carry_observations_across_episodes")}


PORTS = {"lte": 5201, "satellite": 5202, "primary": 5300}
TUNNELS = ("lte", "satellite")


def stream_specs(cfg, probes=True):
    result = {"primary": {"port": PORTS["primary"], "packets": cfg.primary_packets,
                          "payload_bytes": cfg.primary_payload_bytes}}
    if probes:
        for tunnel in TUNNELS:
            result[tunnel] = {"port": PORTS[tunnel], "packets": cfg.probe_packets,
                              "payload_bytes": cfg.probe_payload_bytes}
    return result


def throughput_mbps(before, after):
    """Decimal Mbps of total TX interface bytes over actual monotonic interval."""
    t0, b0 = before["time_s"], before["tx_bytes"]
    t1, b1 = after["time_s"], after["tx_bytes"]
    if (type(b0) is not int or type(b1) is not int or b0 < 0 or b1 < b0
            or not math.isfinite(t0) or not math.isfinite(t1) or t1 <= t0):
        raise ValueError("Invalid or reset interface counter")
    return 8.0 * (b1 - b0) / (1e6 * (t1 - t0))


def stream_measurement(client, server, stream, start_s, end_s, window_s):
    sent, received = client["sent"], server["received_unique"]
    echoes, echo_sent = client["echo_received_unique"], server["echo_sent"]
    if (sent != stream["packets"] or not 0 <= echoes <= echo_sent <= received <= sent
            or client["sent_payload_bytes"] != sent * stream["payload_bytes"]
            or server["received_payload_bytes"] != received * stream["payload_bytes"]
            or server["echo_payload_bytes"] != echo_sent * stream["payload_bytes"]):
        raise ValueError("Incomplete send schedule or inconsistent probe counts")
    if (client["first_send_s"] is None or client["last_send_s"] is None
            or not start_s <= client["first_send_s"] <= client["last_send_s"] < start_s + window_s):
        raise ValueError("Send timestamps outside measurement window")
    delay = client["rtt_ms"]
    if echoes == 0:
        if delay is not None:
            raise ValueError("RTT without a matched echo")
    elif not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay <= 0:
        raise ValueError("Invalid echo RTT")
    return {"rtt_ms": delay, "loss_pct": 100.0 * (sent - received) / sent,
            "roundtrip_missing_pct": 100.0 * (sent - echoes) / sent,
            "sent_packets": sent, "received_packets": received, "echo_sent_packets": echo_sent,
            "echo_received_packets": echoes, "forward_payload_bytes": client["sent_payload_bytes"],
            "reverse_payload_bytes": server["echo_payload_bytes"],
            "received_payload_bytes": server["received_payload_bytes"],
            "offered_payload_mbps": client["sent_payload_bytes"] * 8 / (1e6 * window_s),
            # Denominator includes drain/grace: window-level delivered payload rate.
            "delivered_payload_mbps": server["received_payload_bytes"] * 8 / (1e6 * (end_s - start_s)),
            "send_start_s": client["first_send_s"], "send_end_s": client["last_send_s"],
            "first_receive_s": server["first_receive_s"], "last_receive_s": server["last_receive_s"],
            "first_echo_s": client["first_echo_s"], "last_echo_s": client["last_echo_s"],
            "max_send_lateness_s": client["max_send_lateness_s"],
            "receive_deadline_s": end_s}


def observation_index(tick, lag):
    """At tick zero there is no prior sample; later use only past ticks.

    Clamp early ticks to the first available sample, rather than fabricate
    older measurements. Report the effective lag and exclude startup in
    steady-state age sensitivity.
    """
    if type(tick) is not int or type(lag) is not int or min(tick, lag) < 0:
        raise ValueError("Invalid tick or lag")
    return None if tick == 0 else max(0, tick - 1 - lag)


def observation_source(episode, tick, ticks_per_episode, cfg):
    """Return the zero-based global sample index, or None for cold start.

    The optional continuous-history stress test can observe samples from
    before episode-boundary impairments. Standard experiments keep resets.
    """
    if (type(episode) is not int or episode < 1 or type(tick) is not int
            or type(ticks_per_episode) is not int or not 0 <= tick < ticks_per_episode):
        raise ValueError("Invalid episode/tick")
    offset = (episode - 1) * ticks_per_episode
    if cfg.carry_observations_across_episodes:
        return observation_index(offset + tick, cfg.observation_lag_ticks)
    source = observation_index(tick, cfg.observation_lag_ticks)
    return None if source is None else offset + source


def observation_age(windows, decision_s):
    if windows is None:
        return None
    spans = [span for tunnel in TUNNELS for span in windows[tunnel].values()]
    if not spans or any(not math.isfinite(s["start_s"]) or not math.isfinite(s["end_s"])
                        or not s["start_s"] <= s["end_s"] <= decision_s for s in spans):
        raise ValueError("Measurement support must precede policy decision")
    starts, ends = [s["start_s"] for s in spans], [s["end_s"] for s in spans]
    return {"oldest_support_age_s": decision_s - min(starts),
            "oldest_completion_age_s": decision_s - min(ends),
            "newest_completion_age_s": decision_s - max(ends),
            "completion_skew_s": max(ends) - min(ends),
            "support_overlap_s": max(0.0, min(ends) - max(starts)),
            "support_span_s": max(ends) - min(starts)}
