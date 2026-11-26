"""Explicit scenario splits and checkpoint provenance for C09."""
import argparse
import hashlib
import json

from .scenario_profiles import normalize_v2, novelty_report

SPLITS = ("train", "calibration", "held_out_test")
SCENARIO_FIELDS = {"id", "split", "episodes", "ticks_per_episode", "degrade_lte_ranges"}


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_scenario(row):
    if isinstance(row, dict) and "events" in row:
        return normalize_v2(row)
    if not isinstance(row, dict) or set(row) != SCENARIO_FIELDS:
        raise ValueError("Invalid scenario fields")
    if not isinstance(row["id"], str) or not row["id"].strip() or row["split"] not in SPLITS:
        raise ValueError("Invalid scenario id or split")
    for key in ("episodes", "ticks_per_episode"):
        if type(row[key]) is not int or row[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    ranges = row["degrade_lte_ranges"]
    if not isinstance(ranges, (list, tuple)):
        raise ValueError("Expected degradation ranges")
    checked = []
    for bounds in ranges:
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(v) is not int for v in bounds)
                or not 1 <= bounds[0] <= bounds[1] <= row["episodes"]):
            raise ValueError("Invalid inclusive degradation range")
        checked.append(list(bounds))
    merged = []
    for lo, hi in sorted(checked):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    return {**row, "degrade_lte_ranges": merged}


def scenario_digest(row):
    row = normalize_scenario(row)
    return digest({k: v for k, v in row.items() if k not in ("id", "split")})


def normalize_manifest(manifest):
    if (not isinstance(manifest, dict) or set(manifest) != {"schema_version", "scenarios"}
            or type(manifest["schema_version"]) is not int or manifest["schema_version"] not in (1, 2)
            or not isinstance(manifest["scenarios"], list)):
        raise ValueError("Expected scenario manifest schema version 1 or 2")
    rows = [normalize_scenario(row) for row in manifest["scenarios"]]
    if any(("events" in row) != (manifest["schema_version"] == 2) for row in rows):
        raise ValueError("Scenario fields must match manifest schema version")
    if {row["split"] for row in rows} != set(SPLITS):
        raise ValueError("Manifest must contain train, calibration and held_out_test scenarios")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate scenario id")
    if len({scenario_digest(row) for row in rows}) != len(rows):
        raise ValueError("Duplicate effective scenario schedule, including across splits")
    return {"schema_version": manifest["schema_version"], "scenarios": sorted(rows, key=lambda row: row["id"])}


def read_manifest(path):
    with open(path, encoding="utf-8") as stream:
        return normalize_manifest(json.load(stream))


def resolve_run(manifest, scenario_id, split, mode):
    if mode not in ("online", "frozen", "heuristic"):
        raise ValueError("Unknown experiment mode")
    if split == "diagnostic":
        if manifest is not None or scenario_id is not None:
            raise ValueError("Diagnostic mode does not accept a formal scenario")
        return {"split": "diagnostic"}
    if split not in SPLITS or manifest is None or not scenario_id:
        raise ValueError("Formal runs require a manifest, scenario id and split")
    if (split == "train" and mode != "online") or (split != "train" and mode not in ("frozen", "heuristic")):
        raise ValueError("Training must be online; calibration/test use frozen or non-learning heuristic mode")
    manifest = normalize_manifest(manifest)
    scenario = next((row for row in manifest["scenarios"] if row["id"] == scenario_id), None)
    if scenario is None or scenario["split"] != split:
        raise ValueError("Scenario does not belong to the requested split")
    return {"split": split, "scenario_id": scenario_id,
            "scenario_sha256": scenario_digest(scenario), "scenario": scenario,
            "manifest_sha256": digest(manifest), "manifest": manifest}


def validate_training_provenance(training, evaluation):
    if evaluation["split"] not in ("calibration", "held_out_test"):
        raise ValueError("Expected calibration or test provenance")
    expected_eval = resolve_run(evaluation["manifest"], evaluation["scenario_id"],
                                evaluation["split"], "frozen")
    if expected_eval != evaluation:
        raise ValueError("Evaluation provenance does not match its manifest")
    if not isinstance(training, dict) or training.get("split") != "train":
        raise ValueError("Formal evaluation requires a checkpoint from a formal training run")
    if training.get("manifest_sha256") != evaluation["manifest_sha256"]:
        raise ValueError("Training and evaluation manifests differ")
    expected_train = resolve_run(evaluation["manifest"], training.get("scenario_id"), "train", "online")
    if expected_train != training:
        raise ValueError("Training provenance does not match the manifest")
    if training["scenario_sha256"] == evaluation["scenario_sha256"]:
        raise ValueError("Training and evaluation schedules coincide")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest = read_manifest(args.manifest)
    result = {"manifest_sha256": digest(manifest), "scenarios": [
        {"id": row["id"], "split": row["split"], "scenario_sha256": scenario_digest(row)}
        for row in manifest["scenarios"]]}
    if manifest["schema_version"] == 2:
        result["novelty"] = novelty_report(manifest["scenarios"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
