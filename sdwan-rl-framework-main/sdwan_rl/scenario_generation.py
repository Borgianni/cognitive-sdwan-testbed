"""Generate reproducible v2 impairment schedules before training starts."""
import argparse
import itertools
import json
import random
from pathlib import Path

from .experiment_protocol import digest, normalize_manifest, read_manifest, scenario_digest
from .scenario_profiles import SPLITS, normalize_v2

DEFAULT_SEEDS = {"train": 1101, "calibration": 2202, "held_out_test": 3303}
DEFAULT_REPLICAS = {"train": 1, "calibration": 1, "held_out_test": 3}
DEFAULT_DURATIONS = {"train": [5, 15], "calibration": [3, 9], "held_out_test": [3, 9]}
GENERATOR_VERSION = 1


def temporal_signature(row):
    """Ignore names, severity, tunnel and load when comparing timing alone."""
    intervals = sorted((e["start_episode"], e["end_episode"]) for e in row["events"])
    union = []
    for start, end in intervals:
        if union and start <= union[-1][1] + 1:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([start, end])
    return row["episodes"], tuple(map(tuple, union))


def overlaps(a, b):
    return max(a["start_episode"], b["start_episode"]) <= min(a["end_episode"], b["end_episode"])


def valid_layout(original, sampled, gap):
    for i, j in itertools.combinations(range(len(original)), 2):
        a, b = sampled[i], sampled[j]
        if a["tunnel"] == b["tunnel"]:
            # Canonical template order is chronological on each tunnel.
            if b["start_episode"] - a["end_episode"] - 1 < gap:
                return False
        elif overlaps(original[i], original[j]) != overlaps(a, b):
            # Preserve the intended overlap/non-overlap of combined impairments.
            return False
    return True


def validate_settings(seeds, replicas, durations, nominal, gap, max_attempts):
    if any(set(mapping) != set(SPLITS) for mapping in (seeds, replicas, durations)):
        raise ValueError("Specify independent settings for all three splits")
    if (any(type(v) is not int or v < 0 for v in seeds.values())
            or len(set(seeds.values())) != len(SPLITS)):
        raise ValueError("Use distinct nonnegative integer scenario seeds")
    if any(type(v) is not int or v <= 0 for v in replicas.values()):
        raise ValueError("Replica counts must be positive integers")
    for bounds in durations.values():
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(v) is not int for v in bounds)
                or not 1 <= bounds[0] < bounds[1]):
            raise ValueError("Duration bounds must contain at least two positive integers")
    if any(type(v) is not int or v <= 0 for v in (nominal, gap, max_attempts)):
        raise ValueError("Nominal guard, event gap and attempt limit must be positive integers")


def generate(template, seeds=None, replicas=None, durations=None,
             nominal=2, gap=1, max_attempts=2000):
    """Return canonical executable manifest and separate generation receipt.

    Local RNG streams are independent of the agent and global Python RNG.
    Candidates are sampled uniformly then rejected for layout/collision
    constraints; the accepted joint distribution is not independently uniform.
    """
    seeds = dict(DEFAULT_SEEDS if seeds is None else seeds)
    replicas = dict(DEFAULT_REPLICAS if replicas is None else replicas)
    durations = {k: list(v) for k, v in (DEFAULT_DURATIONS if durations is None else durations).items()}
    validate_settings(seeds, replicas, durations, nominal, gap, max_attempts)
    template = normalize_manifest(template)
    if template["schema_version"] != 2:
        raise ValueError("Randomization requires a v2 scenario template")
    if any(not row["events"] for row in template["scenarios"]):
        raise ValueError("No-event control scenarios must be specified separately")
    used = {temporal_signature(row) for row in template["scenarios"]}
    rows, receipt_rows = [], []
    # Process development first so changing the test seed cannot change training.
    for split in SPLITS:
        for base in (r for r in template["scenarios"] if r["split"] == split):
            low, high = durations[split]
            latest = base["episodes"] - nominal
            if high > base["episodes"] - 2 * nominal:
                raise ValueError(f"{base['id']}: duration range does not fit nominal guards")
            for replica in range(1, replicas[split] + 1):
                scenario_id = f"{base['id']}_r{replica:03d}"
                stream_seed = int(digest({"version": GENERATOR_VERSION, "split": split,
                                          "seed": seeds[split], "template_id": base["id"],
                                          "replica": replica}), 16)
                rng = random.Random(stream_seed)
                for attempt in range(1, max_attempts + 1):
                    events = []
                    for event in base["events"]:
                        duration = rng.randint(low, high)
                        start = rng.randint(nominal + 1, latest - duration + 1)
                        events.append({**event, "start_episode": start,
                                       "end_episode": start + duration - 1})
                    if not valid_layout(base["events"], events, gap):
                        continue
                    candidate = normalize_v2({**base, "id": scenario_id, "events": events})
                    signature = temporal_signature(candidate)
                    if signature in used:
                        continue
                    used.add(signature)
                    rows.append(candidate)
                    receipt_rows.append({"id": scenario_id, "template_id": base["id"],
                                         "split": split, "replica": replica,
                                         "stream_seed": str(stream_seed),
                                         "attempts": attempt,
                                         "scenario_sha256": scenario_digest(candidate)})
                    break
                else:
                    raise ValueError(f"{scenario_id}: no distinct feasible schedule after "
                                     f"{max_attempts} attempts; revise constraints")
    manifest = normalize_manifest({"schema_version": 2, "scenarios": rows})
    receipt = {"generator_version": GENERATOR_VERSION,
               "template_sha256": digest(template), "manifest_sha256": digest(manifest),
               "split_seeds": seeds, "replicas_per_template": replicas,
               "duration_bounds_episodes": durations, "nominal_guard_episodes": nominal,
               "same_tunnel_gap_episodes": gap, "max_attempts": max_attempts,
               "generated_scenarios": receipt_rows,
               "status": "planned_not_executed"}
    return manifest, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True, help="New v2 manifest JSON")
    parser.add_argument("--receipt", required=True, help="New generation receipt JSON")
    for split, label in (("train", "train"), ("calibration", "calibration"), ("held_out_test", "test")):
        parser.add_argument(f"--{label}-seed", type=int, default=DEFAULT_SEEDS[split])
        parser.add_argument(f"--{label}-replicas", type=int, default=DEFAULT_REPLICAS[split])
        parser.add_argument(f"--{label}-duration", type=int, nargs=2, default=DEFAULT_DURATIONS[split])
    parser.add_argument("--nominal-guard", type=int, default=2)
    parser.add_argument("--same-tunnel-gap", type=int, default=1)
    args = parser.parse_args()
    output, receipt_path = Path(args.output), Path(args.receipt)
    if output.resolve() == receipt_path.resolve() or output.exists() or receipt_path.exists():
        parser.error("Use distinct new output and receipt paths")
    seeds = {"train": args.train_seed, "calibration": args.calibration_seed, "held_out_test": args.test_seed}
    replicas = {"train": args.train_replicas, "calibration": args.calibration_replicas,
                "held_out_test": args.test_replicas}
    durations = {"train": args.train_duration, "calibration": args.calibration_duration,
                 "held_out_test": args.test_duration}
    manifest, receipt = generate(read_manifest(args.template), seeds, replicas, durations,
                                 args.nominal_guard, args.same_tunnel_gap)
    for path, value in ((output, manifest), (receipt_path, receipt)):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(f"Generated {len(manifest['scenarios'])} scenarios; no emulation launched.")


if __name__ == "__main__":
    main()
