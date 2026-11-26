"""Variable-N continuous encoder and shared normalized reward.

Only the candidate catalog changes. The standard DQN class is reused.
A path's equivalent load is a representation of its bottleneck residual,
NOT an observed end-to-end throughput.
"""
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import numpy as np

from .config import RLConfig
from .env import SDWANEnv
from .dqn_settings import CURRENT_DQN, training_settings
from .policy_audit import file_digest, metadata_path, weights_digest, validate_run_seed
from .scale_cases import digest, architecture, path_residual, validate_scenario
from .telemetry_core import TelemetryConfig
from .runtime_environment import runtime_environment


class PathEnvironment:
    def __init__(self, graph, cfg):
        if cfg.state_representation != "continuous" or cfg.reward_mode != "normalized":
            raise ValueError("Scalability study uses continuous observations and normalized reward")
        self.graph, self.cfg = graph, cfg
        self.paths = graph["paths"]
        self.envs = []
        for p in self.paths:
            capacity = path_residual(graph, p, {e["id"]: 0.0 for e in graph["edges"]})["nominal_bottleneck_mbps"]
            self.envs.append(SDWANEnv(replace(cfg, lte_capacity_mbps=capacity, lte_cost_index=p["cost_index"])))
        self.reset()

    @property
    def n(self):
        return len(self.paths)

    def reset(self):
        self.metrics = None
        for env in self.envs:
            env.reset_metrics()

    def update(self, metrics):
        if set(metrics) != {p["id"] for p in self.paths}:
            raise ValueError("Missing or unknown path measurements")
        self.metrics = metrics
        for path, env in zip(self.paths, self.envs):
            m = metrics[path["id"]]
            # Reuse exact C01-C20 encoding/reward on the first slot of
            # SDWANEnv; the second slot is irrelevant in this adapter.
            values = {"delay": m["delay"], "packet_loss": m["packet_loss"],
                      "throughput": m["equivalent_load_mbps"]}
            env.update_from_metrics({"lte": values, "satellite": values})

    def state(self):
        per_path = [env.make_state().vec[[0, 2, 4]] for env in self.envs]
        # Group by feature: all delays, all losses, all residual fractions.
        return np.asarray(per_path, dtype=np.float32).T.reshape(3 * self.n)

    def terms(self, action):
        if type(action) is not int or not 0 <= action < self.n:
            raise ValueError("Unknown action")
        return self.envs[action].reward_terms(0)

    def greedy(self, last_action=0):
        values = []
        for i in range(self.n):
            try:
                values.append(self.terms(i)["reward"])
            except ValueError:
                values.append(None)
        available = [i for i, value in enumerate(values) if value is not None and math.isfinite(value)]
        return (max(available, key=lambda i: (values[i], -i)) if available else last_action), values


def policy_signature(scenario, cfg, telemetry_cfg):
    validate_scenario(scenario)
    return {"graph_sha256": digest(scenario["graph"]),
            "ordered_paths": [p["id"] for p in scenario["graph"]["paths"]],
            "architecture": architecture(len(scenario["graph"]["paths"])),
            "rl_config": asdict(cfg), "telemetry_config": asdict(telemetry_cfg)}


def save_scale(agent, scenario, cfg, telemetry_cfg, seed, path):
    validate_run_seed(seed, required=True)
    signature = policy_signature(scenario, cfg, telemetry_cfg)
    n = signature["architecture"]["output"]
    if (scenario["split"] != "train" or cfg.dqn_version != CURRENT_DQN
            or agent.settings() != training_settings(cfg) or agent.frozen
            or agent.state_size != 3*n or agent.action_size != n):
        raise ValueError("Expected a freshly trained matching scalability DQN")
    path = Path(path)
    if path.suffix != ".keras" or path.exists() or metadata_path(path).exists():
        raise ValueError("Use a new .keras checkpoint and metadata")
    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path))
    meta = {"scale_checkpoint_schema": 1, "signature": signature, "training_scenario": scenario,
            "training_seed": seed, "checkpoint_sha256": file_digest(path),
            "weights_sha256": weights_digest(agent), "training_state": agent.training_state(),
            "runtime_environment": runtime_environment(), "purpose": "frozen_inference_only"}
    with metadata_path(path).open("x", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2, allow_nan=False)


def load_scale(agent, scenario, cfg, telemetry_cfg, seed, path):
    validate_run_seed(seed, required=True)
    signature = policy_signature(scenario, cfg, telemetry_cfg)
    with metadata_path(path).open(encoding="utf-8") as stream:
        meta = json.load(stream)
    if (meta.get("scale_checkpoint_schema") != 1 or meta["signature"] != signature
            or meta["training_seed"] != seed or scenario["split"] != "held_out_test"
            or meta["checkpoint_sha256"] != file_digest(path)):
        raise ValueError("Checkpoint provenance/configuration differs from this fixed catalog and seed")
    train = validate_scenario(meta["training_scenario"])
    if (train["split"] != "train" or digest(train["graph"]) != digest(scenario["graph"])
            or digest(train) == digest(scenario)):
        raise ValueError("Invalid training/test separation")
    n = signature["architecture"]["output"]
    if agent.settings() != training_settings(cfg) or agent.state_size != 3*n or agent.action_size != n:
        raise ValueError("Agent dimensions/settings differ from declared signature")
    agent.load(str(path))
    if weights_digest(agent) != meta["weights_sha256"]:
        raise ValueError("Loaded weights differ from saved checkpoint")
    agent.freeze()
    return meta
