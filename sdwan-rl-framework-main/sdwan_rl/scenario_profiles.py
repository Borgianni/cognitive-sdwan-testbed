"""Version-2 scenario schema and novelty diagnostics (no Mininet dependency)."""
import math

from .config import TopologyConfig

TUNNELS = ("lte", "satellite")
SPLITS = ("train", "calibration", "held_out_test")
FIELDS = {"id", "split", "episodes", "ticks_per_episode", "events", "traffic_mbps"}
EVENT_FIELDS = {"tunnel", "start_episode", "end_episode", "delay_ms", "loss_pct"}


def nominal_profiles():
    cfg = TopologyConfig()
    return {t: {"delay_ms": float(link.delay_ms), "loss_pct": float(link.loss_pct),
                "jitter_ms": float(link.jitter_ms), "max_queue": int(link.max_queue),
                "capacity_mbps": float(link.bw_mbps)}
            for t, link in (("lte", cfg.lte), ("satellite", cfg.sat))}


def number(value, lower, upper):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not lower <= value <= upper):
        raise ValueError(f"Expected finite number in [{lower}, {upper}]")
    value = float(value)
    return 0.0 if value == 0.0 else value


def normalize_v2(row):
    if not isinstance(row, dict) or set(row) != FIELDS:
        raise ValueError("Invalid v2 scenario fields")
    if not isinstance(row["id"], str) or not row["id"].strip() or row["split"] not in SPLITS:
        raise ValueError("Invalid scenario id or split")
    for key in ("episodes", "ticks_per_episode"):
        if type(row[key]) is not int or row[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    nominal = nominal_profiles()
    traffic = row["traffic_mbps"]
    if not isinstance(traffic, dict) or set(traffic) != set(TUNNELS):
        raise ValueError("Specify background UDP offered rate for each tunnel")
    traffic = {t: number(traffic[t], 0, nominal[t]["capacity_mbps"]) for t in TUNNELS}
    # Shared access interface: reserve some nominal headroom for probes.
    if sum(traffic.values()) > TopologyConfig().host_link.bw_mbps - 10:
        raise ValueError("Combined background load exceeds access budget")
    if not isinstance(row["events"], list):
        raise ValueError("Expected impairment event list")
    events = []
    for event in row["events"]:
        if not isinstance(event, dict) or set(event) != EVENT_FIELDS:
            raise ValueError("Invalid impairment event")
        t = event["tunnel"]
        start, end = event["start_episode"], event["end_episode"]
        if (t not in TUNNELS or type(start) is not int or type(end) is not int
                or not 1 <= start <= end <= row["episodes"]):
            raise ValueError("Invalid tunnel or event range")
        delay = number(event["delay_ms"], nominal[t]["delay_ms"], 10000)
        loss = number(event["loss_pct"], nominal[t]["loss_pct"], 100)
        if delay == nominal[t]["delay_ms"] and loss == nominal[t]["loss_pct"]:
            raise ValueError("An impairment event must change the nominal profile")
        events.append({**event, "delay_ms": delay, "loss_pct": loss})
    merged = []
    for event in sorted(events, key=lambda e: (e["tunnel"], e["start_episode"], e["end_episode"])):
        previous = merged[-1] if merged and merged[-1]["tunnel"] == event["tunnel"] else None
        if previous:
            same = all(previous[k] == event[k] for k in ("delay_ms", "loss_pct"))
            if event["start_episode"] <= previous["end_episode"] and not same:
                raise ValueError("Conflicting overlapping impairments on one tunnel")
            if same and event["start_episode"] <= previous["end_episode"] + 1:
                previous["end_episode"] = max(previous["end_episode"], event["end_episode"])
                continue
        merged.append(event)
    return {**row, "events": merged, "traffic_mbps": traffic}


def episode_profiles(scenario, episode):
    scenario = normalize_v2(scenario)
    if type(episode) is not int or not 1 <= episode <= scenario["episodes"]:
        raise ValueError("Episode outside scenario")
    profiles = nominal_profiles()
    for event in scenario["events"]:
        if event["start_episode"] <= episode <= event["end_episode"]:
            profiles[event["tunnel"]].update(delay_ms=event["delay_ms"], loss_pct=event["loss_pct"])
    return profiles


def _signatures(rows):
    severities = {(e["delay_ms"], e["loss_pct"]) for r in rows for e in r["events"]}
    affected = {e["tunnel"] for r in rows for e in r["events"]}
    traffic = {tuple(r["traffic_mbps"][t] for t in TUNNELS) for r in rows}
    return severities, affected, traffic


def novelty_report(rows):
    """Compare severity pairs, impaired tunnels and offered-rate vectors.

    Distinguish novelty relative to training from novelty relative to all
    development data. Different pairs need not be outside the training range.
    """
    rows = [normalize_v2(r) for r in rows]
    result = {}
    for label, splits in (("relative_to_train", {"train"}),
                          ("relative_to_development", {"train", "calibration"})):
        seen = _signatures([r for r in rows if r["split"] in splits])
        tests = {}
        for row in rows:
            if row["split"] != "held_out_test":
                continue
            current = _signatures([row])
            tests[row["id"]] = {
                "unseen_delay_loss_pairs": [list(v) for v in sorted(current[0] - seen[0])],
                "unseen_affected_tunnels": sorted(current[1] - seen[1]),
                "unseen_traffic_vector": bool(current[2] - seen[2]),
            }
        result[label] = tests
    return result
