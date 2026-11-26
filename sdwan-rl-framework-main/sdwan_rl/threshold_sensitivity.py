"""Offline threshold sensitivity and a non-executing policy experiment plan."""
import argparse
import itertools
import json
import math
from dataclasses import asdict, replace

from .aliasing_audit import audit, encoding_config, snapshots_from_trace
from .config import RLConfig

FIELDS = (
    "lte_delay_threshold_ms", "sat_delay_threshold_ms",
    "lte_loss_threshold_pct", "sat_loss_threshold_pct",
    "lte_residual_threshold", "sat_residual_threshold",
)
FEATURES = ("lte_delay", "leo_delay", "lte_loss", "leo_loss",
            "lte_residual", "leo_residual")
DEFAULT_GRID = {
    "lte_delay_threshold_ms": [100.0, 175.0, 350.0],
    "sat_delay_threshold_ms": [150.0, 240.0, 480.0],
    "lte_loss_threshold_pct": [5.0, 10.0, 20.0],
    "sat_loss_threshold_pct": [5.0, 10.0, 20.0],
    "lte_residual_threshold": [0.10, 0.20, 0.40],
    "sat_residual_threshold": [0.10, 0.20, 0.40],
}
DEFAULT_SEEDS = (1, 2, 3, 4, 5)


def thresholds(cfg):
    return {name: getattr(cfg, name) for name in FIELDS}


def validate_grid(grid, cfg):
    if set(grid) != set(FIELDS):
        raise ValueError("Grid must specify all six observation thresholds")
    clean = {}
    for name in FIELDS:
        levels = grid[name]
        if not isinstance(levels, (list, tuple)) or len(levels) < 2:
            raise ValueError(f"{name} needs at least two levels")
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in levels):
            raise ValueError(f"Invalid level for {name}")
        vals = sorted(set(float(v) for v in levels))
        if len(vals) != len(levels) or getattr(cfg, name) not in vals:
            raise ValueError(f"{name} needs unique levels including the baseline")
        for value in vals:
            replace(cfg, **{name: value})
        clean[name] = vals
    return clean


def configurations(cfg, grid, design="ofat"):
    """Yield deterministic identifiers and configs; do not modify the input."""
    cfg = encoding_config(cfg, "revised_binary")
    grid = validate_grid(grid, cfg)
    if design == "ofat":
        yield "baseline", cfg
        for name in FIELDS:
            for i, value in enumerate(grid[name]):
                if value != getattr(cfg, name):
                    yield f"{name}_level{i}", replace(cfg, **{name: value})
    elif design == "factorial":
        for indices in itertools.product(*(range(len(grid[k])) for k in FIELDS)):
            changes = {k: grid[k][i] for k, i in zip(FIELDS, indices)}
            identifier = "grid_" + "_".join(map(str, indices))
            yield identifier, replace(cfg, **changes)
    else:
        raise ValueError("Unknown sensitivity design")


def summarize(snapshots, cfg):
    report = audit(snapshots, cfg)
    n = report["samples"]
    groups = report["groups"]
    return {
        "samples": n,
        "distinct_observations": report["distinct_observations"],
        "fraction_samples_in_conflicting_observations":
            report["fraction_samples_in_conflicting_observations"],
        "mean_aliasing_gap": report["mean_aliasing_gap"],
        "mean_snapshot_oracle_reward": report["mean_snapshot_oracle_reward"],
        "mean_best_observation_only_reward": report["mean_best_observation_only_reward"],
        "feature_mean": {
            name: sum(g["samples"] * g["state"][i] for g in groups) / n
            for i, name in enumerate(FEATURES)
        },
        "fraction_samples_in_singleton_observations":
            report["fraction_samples_in_singleton_observations"],
    }


def sweep(snapshots, cfg, grid, design="factorial"):
    """Use identical input snapshots and reward for every threshold configuration."""
    snapshots = list(snapshots)
    rows = []
    for identifier, candidate in configurations(cfg, grid, design):
        rows.append({"id": identifier, "thresholds": thresholds(candidate),
                     **summarize(snapshots, candidate)})
    return rows


def make_plan(cfg, grid, seeds=DEFAULT_SEEDS, include_references=True):
    """Build a manifest, not a launcher; split/seed execution is pending C09-C12."""
    seeds = list(seeds)
    if (not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0
                         for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("Seeds must be unique nonnegative integers")
    candidates = list(configurations(cfg, grid, "ofat"))
    if include_references:
        candidates.extend((mode, encoding_config(cfg, mode))
                          for mode in ("legacy_code", "legacy_paper", "continuous"))
    entries = [{"id": identifier, "seed": seed, "rl_config": asdict(candidate),
                "training_split": "train", "evaluation_split": "held_out_test",
                "evaluation_mode": "frozen"}
               for identifier, candidate in candidates for seed in seeds]
    return {
        "status": "planned_not_executable",
        "design": "one_factor_at_a_time",
        "seeds": seeds, "configurations": len(candidates),
        "training_runs": len(entries), "frozen_evaluations": len(entries),
        "requires": ["validated telemetry and forwarding", "disjoint scenario splits",
                     "runner seed support", "fixed training budget and calibration protocol"],
        "entries": entries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", help="JSON object of six lists of candidate levels")
    parser.add_argument("--config", help="JSON RLConfig fields; cannot override trace config")
    sub = parser.add_subparsers(dest="command", required=True)
    offline = sub.add_parser("offline", help="Evaluate calibration snapshots; no training")
    source = offline.add_mutually_exclusive_group(required=True)
    source.add_argument("--snapshots", help="JSON array of calibration telemetry snapshots")
    source.add_argument("--trace", help="Complete frozen trace from a calibration scenario")
    offline.add_argument("--design", choices=["ofat", "factorial"], default="factorial")
    offline.add_argument("--split", choices=["calibration"], required=True,
                         help="Explicit declaration: do not use final held-out data for tuning")
    plan = sub.add_parser("plan", help="Print experiment manifest; launch nothing")
    plan.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    plan.add_argument("--no-references", action="store_true")
    args = parser.parse_args()
    cfg = RLConfig()
    grid = DEFAULT_GRID
    if args.config:
        with open(args.config, encoding="utf-8") as stream:
            cfg = RLConfig(**json.load(stream))
    if args.grid:
        with open(args.grid, encoding="utf-8") as stream:
            grid = json.load(stream)
    if args.command == "plan":
        report = make_plan(cfg, grid, args.seeds, not args.no_references)
    else:
        skipped = 0
        if args.trace:
            if args.config:
                parser.error("--config cannot override trace configuration")
            snapshots, cfg, skipped = snapshots_from_trace(args.trace, required_split="calibration")
        else:
            with open(args.snapshots, encoding="utf-8") as stream:
                snapshots = json.load(stream)
            if not isinstance(snapshots, list):
                raise ValueError("Expected a JSON array of telemetry snapshots")
        report = {"declared_split": args.split,
                  "source": args.trace or args.snapshots,
                  "reset_samples_excluded": skipped,
                  "base_config": asdict(cfg), "grid": grid,
                  "design": args.design,
                  "rows": sweep(snapshots, cfg, grid, args.design),
                  "references": {mode: summarize(snapshots, encoding_config(cfg, mode))
                                 for mode in ("legacy_code", "legacy_paper", "continuous")},
                  "interpretation": "Hindsight observation diagnostics, not trained policy results. "
                                    "Continuous singleton groups can mechanically yield zero gap. "
                                    "Raw snapshot split is user-declared; trace split must be calibration."}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
