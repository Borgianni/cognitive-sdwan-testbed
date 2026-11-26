"""Checkpoint provenance and state-conditioned frozen-policy diagnostics."""
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import RLConfig
from .env import SDWANEnv
from .dqn_settings import CURRENT_DQN, LEGACY_DQN, training_settings
from .runtime_environment import runtime_environment
from .experiment_protocol import validate_training_provenance


def weights_digest(agent):
    digest = hashlib.sha256()
    for weight in agent.model.get_weights():
        arr = np.ascontiguousarray(weight)
        digest.update(str(arr.shape).encode())
        digest.update(arr.dtype.str.encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_path(path):
    return Path(str(path) + ".json")


def validate_run_seed(seed, required=False):
    if seed is None and not required:
        return
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Use an explicit integer run seed in [0, 2**32-1]")


def save_checkpoint(agent, cfg, path, protocol=None, run_seed=None, measurement_settings=None):
    validate_run_seed(run_seed, required=protocol is not None and protocol["split"] == "train")
    if cfg.dqn_version != CURRENT_DQN or agent.settings() != training_settings(cfg):
        raise ValueError("Checkpoint training settings differ from declared DQN configuration")
    path = Path(path)
    if path.suffix != ".keras":
        raise ValueError("Use a .keras checkpoint filename")
    if path.exists() or metadata_path(path).exists():
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path))
    meta = {"observation_schema": 2, "rl_config": asdict(cfg), "checkpoint_sha256": file_digest(path),
            "weights_sha256": weights_digest(agent),
            "training_protocol": protocol, "training_seed": run_seed,
            "dqn_training_state": agent.training_state(),
            "runtime_environment": runtime_environment(),
            "checkpoint_purpose": "frozen_inference_only",
            "measurement_settings": measurement_settings}
    with metadata_path(path).open("x", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2)


def checkpoint_config(path):
    path = Path(path)
    with metadata_path(path).open(encoding="utf-8") as stream:
        meta = json.load(stream)
    if file_digest(path) != meta["checkpoint_sha256"]:
        raise ValueError("Checkpoint file does not match its metadata")
    fields = meta["rl_config"]
    # C04/C05 checkpoints predate the continuous encoder and were binary.
    # Require explicit encoding fields for checkpoints saved by C06 onwards.
    schema = meta.get("observation_schema", 1)
    if schema == 1:
        if "state_representation" in fields or "continuous_delay_scale_ms" in fields:
            raise ValueError("Unexpected observation fields in legacy metadata")
        fields = {**fields, "state_representation": "binary",
                  "continuous_delay_scale_ms": 100.0}
    elif schema == 2:
        if not {"state_representation", "continuous_delay_scale_ms"} <= fields.keys():
            raise ValueError("Checkpoint is missing observation configuration")
    else:
        raise ValueError("Unsupported observation schema")
    return RLConfig.from_record(fields), meta


def load_frozen(agent, cfg, path, protocol=None, run_seed=None, measurement_settings=None):
    saved_cfg, meta = checkpoint_config(path)
    if asdict(saved_cfg) != asdict(cfg):
        raise ValueError("Evaluation configuration differs from training checkpoint")
    if protocol is not None and protocol["split"] != "diagnostic":
        if cfg.dqn_version != CURRENT_DQN:
            raise ValueError("Legacy DQN checkpoints are diagnostic-only")
        validate_training_provenance(meta.get("training_protocol"), protocol)
        if measurement_settings is not None and meta.get("measurement_settings") != measurement_settings:
            raise ValueError("Measurement configuration differs from training checkpoint")
        validate_run_seed(run_seed, required=True)
        if type(meta.get("training_seed")) is not int or meta["training_seed"] != run_seed:
            raise ValueError("Evaluation run seed must match the training checkpoint seed")
    if agent.settings() != training_settings(cfg):
        raise ValueError("Agent implementation settings differ from checkpoint configuration")
    agent.load(str(path))
    if weights_digest(agent) != meta["weights_sha256"]:
        raise ValueError("Loaded weights differ from checkpoint metadata")
    agent.freeze()
    return meta


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def append_trace(path, record, mode="a"):
    with open(path, mode, encoding="utf-8") as stream:
        stream.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")


def compare_observations(agent, cfg, nominal, degraded):
    """Compare two metric snapshots at one frozen checkpoint, without rewards."""
    if not agent.frozen:
        raise ValueError("Policy must be frozen before comparing observations")
    before = weights_digest(agent)
    rows = {}
    for label, metrics in (("nominal", nominal), ("degraded", degraded)):
        for key in ("lte", "satellite"):
            for metric in ("delay", "packet_loss", "throughput"):
                value = metrics[key][metric]
                if value is None or not math.isfinite(value):
                    raise ValueError("Use finite, valid telemetry snapshots")
            if (metrics[key]["delay"] <= 0 or metrics[key]["throughput"] < 0
                    or not 0 <= metrics[key]["packet_loss"] <= 100):
                raise ValueError("Use finite, valid telemetry snapshots")
        env = SDWANEnv(cfg)
        env.update_from_metrics(metrics)
        state = env.make_state().vec
        q = np.asarray(agent.q_values(state))
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError("Expected two finite Q-values")
        rows[label] = {"state": state.tolist(), "q_values": q.tolist(),
                       "greedy_action": int(np.argmax(q)),
                       "q_margin_lte_minus_leo": float(q[0] - q[1])}
    after = weights_digest(agent)
    if before != after:
        raise RuntimeError("Weights changed during frozen probe")
    a, b = rows["nominal"], rows["degraded"]
    return {**rows, "aliased": a["state"] == b["state"],
            "greedy_action_changed": a["greedy_action"] != b["greedy_action"],
            "weights_sha256": before}


def main():
    import argparse
    from .agents.dqn import DQNAgent

    parser = argparse.ArgumentParser(description="Compare observations at a frozen DQN checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--snapshots", required=True,
                        help="JSON containing nominal and degraded metric dictionaries")
    args = parser.parse_args()
    cfg, _ = checkpoint_config(args.checkpoint)
    # A seed is required only if deterministic TensorFlow operations were requested.
    agent = DQNAgent.from_config(cfg, seed=0 if cfg.dqn_deterministic_ops else None)
    load_frozen(agent, cfg, args.checkpoint)
    with open(args.snapshots, encoding="utf-8") as stream:
        snapshots = json.load(stream)
    result = compare_observations(agent, cfg, snapshots["nominal"], snapshots["degraded"])
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
