"""Shared learner selection and algorithm-specific checkpoint provenance."""
import json
from dataclasses import asdict
from pathlib import Path

from .experiment_protocol import validate_training_provenance
from .policy_audit import (file_digest,metadata_path,weights_digest,validate_run_seed,
                          load_frozen,save_checkpoint)
from .runtime_environment import runtime_environment

LEARNING_AGENTS = ("dqn","expected_sarsa")


def checkpoint_suffix(kind):
    if kind not in LEARNING_AGENTS:
        raise ValueError("Unknown learning agent")
    return ".keras" if kind=="dqn" else ".npz"


def save_learner(kind,agent,cfg,path,protocol,seed,measurement_settings):
    if kind=="dqn":
        return save_checkpoint(agent,cfg,path,protocol=protocol,run_seed=seed,
                               measurement_settings=measurement_settings)
    validate_run_seed(seed,required=protocol["split"]=="train")
    path=Path(path)
    if kind!="expected_sarsa" or path.suffix!=".npz" or path.exists() or metadata_path(path).exists() or agent.frozen or agent.memory:
        raise ValueError("Use a new Expected SARSA checkpoint after an online update")
    path.parent.mkdir(parents=True,exist_ok=True)
    agent.save(str(path))
    meta={"learner_schema":1,"algorithm":kind,"settings":agent.settings(),"rl_config":asdict(cfg),
          "checkpoint_sha256":file_digest(path),"weights_sha256":weights_digest(agent),
          "training_protocol":protocol,"training_seed":seed,"measurement_settings":measurement_settings,
          "training_state":agent.training_state(),"runtime_environment":runtime_environment(),
          "checkpoint_purpose":"frozen_inference_only"}
    with metadata_path(path).open("x",encoding="utf-8") as stream:
        json.dump(meta,stream,indent=2,allow_nan=False)


def load_learner(kind,agent,cfg,path,protocol,seed,measurement_settings):
    if kind=="dqn":
        return load_frozen(agent,cfg,path,protocol=protocol,run_seed=seed,
                           measurement_settings=measurement_settings)
    if kind!="expected_sarsa" or Path(path).suffix!=".npz":
        raise ValueError("Expected SARSA requires a .npz checkpoint")
    with metadata_path(path).open(encoding="utf-8") as stream:
        meta=json.load(stream)
    if (meta.get("learner_schema")!=1 or meta.get("algorithm")!=kind or
            meta.get("settings")!=agent.settings() or meta.get("rl_config")!=asdict(cfg)
            or meta.get("measurement_settings")!=measurement_settings
            or meta.get("checkpoint_sha256")!=file_digest(path)):
        raise ValueError("Expected SARSA checkpoint/configuration mismatch")
    if protocol["split"]!="diagnostic":
        validate_training_provenance(meta["training_protocol"],protocol)
        validate_run_seed(seed,required=True)
        if type(meta.get("training_seed")) is not int or meta["training_seed"]!=seed:
            raise ValueError("Expected SARSA seed differs from training checkpoint")
    agent.load(str(path))
    if weights_digest(agent)!=meta["weights_sha256"]:
        raise ValueError("Loaded parameter digest differs")
    agent.freeze()
    return meta
