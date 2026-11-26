"""Measure immediate-reward decision loss induced by observation aliasing."""
import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, replace
from statistics import fmean

from .config import RLConfig
from .env import SDWANEnv

ACTIONS = ("LTE", "LEO")
MODES = ("legacy_code", "legacy_paper", "revised_binary", "continuous")
TIE_TOL = 1e-9


def encoding_config(cfg, mode):
    if mode == "continuous":
        return replace(cfg, state_representation="continuous",
                       bandwidth_state_mode="residual_threshold")
    if mode == "revised_binary":
        return replace(cfg, state_representation="binary",
                       bandwidth_state_mode="residual_threshold")
    if mode not in MODES:
        raise ValueError(f"Unknown encoding: {mode}")
    return replace(cfg, state_representation="binary",
                   lte_delay_threshold_ms=700.0, sat_delay_threshold_ms=1200.0,
                   lte_loss_threshold_pct=0.0, sat_loss_threshold_pct=0.0,
                   bandwidth_state_mode=("legacy_throughput" if mode == "legacy_code"
                                         else "positive_residual"))


def evaluate_snapshot(metrics, cfg):
    for tunnel in ("lte", "satellite"):
        for name in ("delay", "packet_loss", "throughput"):
            value = metrics[tunnel][name]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid {tunnel} {name}")
        m = metrics[tunnel]
        if m["delay"] <= 0 or not 0 <= m["packet_loss"] <= 100 or m["throughput"] < 0:
            raise ValueError(f"Invalid telemetry for {tunnel}")
    env = SDWANEnv(cfg)
    env.update_from_metrics(metrics)
    utilities = tuple(env.reward(a) for a in range(2))
    if not all(math.isfinite(u) for u in utilities):
        raise ValueError("Non-finite immediate reward")
    return tuple(float(v) for v in env.make_state().vec), utilities


def audit(snapshots, cfg):
    """Empirical hindsight benchmark; not a learned policy or return bound."""
    groups = defaultdict(list)
    for index, metrics in enumerate(snapshots):
        try:
            state, rewards = evaluate_snapshot(metrics, cfg)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid snapshot {index}: {exc}") from exc
        groups[state].append(rewards)
    n = sum(len(rows) for rows in groups.values())
    if not n:
        raise ValueError("At least one valid snapshot is required")
    details = []
    for state, rows in sorted(groups.items()):
        means = [fmean(row[a] for row in rows) for a in range(2)]
        unrestricted = fmean(max(row) for row in rows)
        restricted = max(means)
        lte = sum(row[0] - row[1] > TIE_TOL for row in rows)
        leo = sum(row[1] - row[0] > TIE_TOL for row in rows)
        details.append({
            "state": list(state), "samples": len(rows),
            "strict_lte_preference": lte, "strict_leo_preference": leo,
            "ties_within_tolerance": len(rows) - lte - leo,
            "conflicting_preferences": bool(lte and leo),
            "best_fixed_action": ACTIONS[0 if means[0] >= means[1] else 1],
            "mean_reward_by_action": dict(zip(ACTIONS, means)),
            "mean_snapshot_oracle_reward": unrestricted,
            "mean_best_fixed_action_reward": restricted,
            "mean_aliasing_gap": max(0.0, unrestricted - restricted),
        })
    weighted = lambda key: fmean(
        g[key] for g in details for _ in range(g["samples"])
    )
    ambiguous = [g for g in details if g["conflicting_preferences"]]
    return {
        "samples": n, "distinct_observations": len(details),
        "singleton_observations": sum(g["samples"] == 1 for g in details),
        "fraction_samples_in_singleton_observations": sum(g["samples"] == 1 for g in details) / n,
        "grouping": "exact_float32_observation_equality",
        "interpretation": ("Empirical hindsight gap only; singleton groups can make "
                           "the gap zero without demonstrating generalization."),
        "conflicting_observations": len(ambiguous),
        "fraction_samples_in_conflicting_observations": sum(g["samples"] for g in ambiguous) / n,
        "mean_snapshot_oracle_reward": weighted("mean_snapshot_oracle_reward"),
        "mean_best_observation_only_reward": weighted("mean_best_fixed_action_reward"),
        "mean_aliasing_gap": weighted("mean_aliasing_gap"),
        "groups": details,
    }


def compare_encodings(snapshots, cfg):
    snapshots = list(snapshots)
    return {mode: audit(snapshots, encoding_config(cfg, mode)) for mode in MODES}


def snapshots_from_trace(path, required_split=None):
    """Read decision-time telemetry, never the post-decision measurement."""
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or rows[0].get("kind") != "run" or rows[-1].get("kind") != "complete":
        raise ValueError("Expected one complete policy trace")
    if any(row.get("kind") != "decision" for row in rows[1:-1]):
        raise ValueError("Unexpected records inside policy trace")
    if rows[0].get("mode") != "frozen":
        raise ValueError("Use a completed frozen evaluation trace")
    if (rows[0]["initial_weights_sha256"] != rows[-1]["weights_sha256"]
            or rows[-1]["epsilon"] != 0 or rows[-1]["replay_size"] != 0):
        raise ValueError("Frozen trace integrity check failed")
    if required_split is not None:
        protocol = rows[0].get("protocol") or {}
        if protocol.get("split") != required_split:
            raise ValueError(f"Trace must originate from the {required_split} split")
    cfg = RLConfig(**rows[0]["rl_config"])
    snapshots = []
    skipped = 0
    for row in rows[1:-1]:
        metrics = row["decision_metrics"]
        if row["tick"] == 0 and metrics is None:
            skipped += 1
            continue
        if metrics is None:
            raise ValueError("Missing non-reset decision telemetry")
        snapshots.append(metrics)
    return snapshots, cfg, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshots", help="JSON array of metric snapshots")
    source.add_argument("--trace", help="Completed frozen JSONL policy trace")
    parser.add_argument("--config", help="JSON RLConfig fields; for snapshots only")
    args = parser.parse_args()
    if args.trace:
        if args.config:
            parser.error("--config cannot override a trace configuration")
        snapshots, cfg, skipped = snapshots_from_trace(args.trace)
    else:
        with open(args.snapshots, encoding="utf-8") as stream:
            snapshots = json.load(stream)
        if not isinstance(snapshots, list):
            raise ValueError("Expected a JSON array of metric snapshots")
        cfg = RLConfig()
        if args.config:
            with open(args.config, encoding="utf-8") as stream:
                cfg = RLConfig(**json.load(stream))
        skipped = 0
    result = {"source": args.trace or args.snapshots, "reset_samples_excluded": skipped,
              "base_config": asdict(cfg), "encodings": compare_encodings(snapshots, cfg)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
