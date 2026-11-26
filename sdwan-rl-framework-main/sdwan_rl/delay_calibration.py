"""Calibrate binary RTT observation thresholds, not contractual SLA limits."""
import argparse
import json
import math
from collections import defaultdict
from statistics import fmean

from .threshold_sensitivity import DEFAULT_GRID

TUNNELS = ("lte", "satellite")
CONDITIONS = ("nominal", "degraded")
HISTORICAL_THRESHOLDS = {"lte": 700.0, "satellite": 1200.0}


def default_candidates():
    return {
        "lte": sorted(set(DEFAULT_GRID["lte_delay_threshold_ms"] + [700.0])),
        "satellite": sorted(set(DEFAULT_GRID["sat_delay_threshold_ms"] + [1200.0])),
    }


def _positive_finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def calibrate(dataset, candidates=None):
    """Score candidate thresholds on externally labelled calibration scenarios.

    Average error rates within scenarios first, then equally across scenarios
    of the same class. Do not infer labels from RTT or an existing threshold.
    Return all tied candidates; never change runtime configuration.
    """
    if not isinstance(dataset, dict) or dataset.get("split") != "calibration":
        raise ValueError("Only explicitly labelled calibration data are accepted")
    samples = dataset.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Expected a nonempty samples array")
    candidates = default_candidates() if candidates is None else candidates
    if not isinstance(candidates, dict) or set(candidates) != set(TUNNELS):
        raise ValueError("Provide candidate RTT thresholds for LTE and satellite")
    groups = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(samples):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid sample {i}")
        tunnel, label, scenario, rtt = (row.get(k) for k in
                                        ("tunnel", "condition", "scenario_id", "rtt_ms"))
        if (tunnel not in TUNNELS or label not in CONDITIONS
                or not isinstance(scenario, str) or not scenario.strip()
                or not _positive_finite(rtt)):
            raise ValueError(f"Invalid labelled RTT sample {i}")
        groups[(tunnel, label)][scenario].append(float(rtt))

    report = {}
    for tunnel in TUNNELS:
        nominal, degraded = (groups[(tunnel, label)] for label in CONDITIONS)
        if not nominal or not degraded:
            raise ValueError(f"{tunnel} requires nominal and degraded calibration samples")
        values = candidates[tunnel]
        if (not isinstance(values, list) or not values
                or not all(_positive_finite(v) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"Invalid candidate thresholds for {tunnel}")
        rows = []
        for threshold in sorted(values):
            # Same inclusive acceptance convention as SDWANEnv.make_state.
            false_alarm = fmean(fmean(rtt > threshold for rtt in rtts)
                                for rtts in nominal.values())
            missed = fmean(fmean(rtt <= threshold for rtt in rtts)
                           for rtts in degraded.values())
            rows.append({"threshold_ms": float(threshold),
                         "nominal_false_alarm_rate": false_alarm,
                         "degradation_miss_rate": missed,
                         "balanced_error": (false_alarm + missed) / 2.0})
        minimum = min(row["balanced_error"] for row in rows)
        best = [row["threshold_ms"] for row in rows
                if math.isclose(row["balanced_error"], minimum, rel_tol=0, abs_tol=1e-12)]
        report[tunnel] = {
            "nominal_scenarios": len(nominal), "degraded_scenarios": len(degraded),
            "nominal_samples": sum(map(len, nominal.values())),
            "degraded_samples": sum(map(len, degraded.values())),
            "max_nominal_rtt_ms": max(max(v) for v in nominal.values()),
            "min_degraded_rtt_ms": min(min(v) for v in degraded.values()),
            "perfect_separation_on_calibration_possible":
                max(max(v) for v in nominal.values()) < min(min(v) for v in degraded.values()),
            "minimum_candidate_balanced_error": minimum,
            "best_candidate_thresholds_ms": best,
            "candidates": rows,
        }
    return {
        "status": "calibration_diagnostic_only",
        "declared_split": "calibration",
        "objective": "scenario_macro_averaged_balanced_error",
        "tunnels": report,
        "limitations": [
            "Scenario labels and split are supplied externally, not independently verified.",
            "RTT samples must come from validated measurement paths.",
            "Minimum calibration error is not held-out performance or an SLA.",
            "No thresholds are deployed automatically; retraining and held-out evaluation are required.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Labelled calibration JSON")
    parser.add_argument("--candidates", help="JSON object of LTE and satellite RTT candidate lists")
    args = parser.parse_args()
    with open(args.dataset, encoding="utf-8") as stream:
        dataset = json.load(stream)
    candidates = None
    if args.candidates:
        with open(args.candidates, encoding="utf-8") as stream:
            candidates = json.load(stream)
    print(json.dumps(calibrate(dataset, candidates), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
