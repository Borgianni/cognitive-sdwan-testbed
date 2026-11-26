"""Versioned DQN training settings, independent of TensorFlow."""
import math

CURRENT_DQN = "target_batch_v1"
LEGACY_DQN = "legacy_online_v0"
DQN_FIELDS = ("dqn_version", "dqn_replay_warmup", "dqn_target_update_steps",
              "dqn_gradient_clipnorm", "dqn_deterministic_ops")
TRAINING_FIELDS = DQN_FIELDS + ("gamma", "replay_size", "batch_size", "eps_start",
                                "eps_end", "eps_decay", "lr")


def training_settings(cfg):
    return {name: getattr(cfg, name) for name in TRAINING_FIELDS}


def validate_settings(cfg):
    if cfg.dqn_version not in (CURRENT_DQN, LEGACY_DQN):
        raise ValueError("Unknown DQN implementation")
    for name in ("replay_size", "batch_size", "dqn_replay_warmup", "dqn_target_update_steps"):
        value = getattr(cfg, name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not cfg.batch_size <= cfg.dqn_replay_warmup <= cfg.replay_size:
        raise ValueError("Require batch_size <= replay warm-up <= replay capacity")
    for name in ("gamma", "eps_start", "eps_end", "eps_decay", "lr", "dqn_gradient_clipnorm"):
        value = getattr(cfg, name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0 <= cfg.gamma < 1:
        raise ValueError("gamma must be in [0, 1)")
    if not 0 <= cfg.eps_end <= cfg.eps_start <= 1 or not 0 < cfg.eps_decay <= 1:
        raise ValueError("Invalid epsilon schedule")
    if cfg.lr <= 0 or cfg.dqn_gradient_clipnorm <= 0:
        raise ValueError("Learning rate and global gradient clip norm must be positive")
    if type(cfg.dqn_deterministic_ops) is not bool:
        raise ValueError("dqn_deterministic_ops must be boolean")


def add_dqn_arguments(parser):
    from .config import RLConfig
    parser.add_argument("--dqn-version", choices=[CURRENT_DQN, LEGACY_DQN],
                        default=RLConfig.dqn_version,
                        help="Legacy version supports old checkpoint inference only")
    parser.add_argument("--dqn-replay-warmup", type=int, default=RLConfig.dqn_replay_warmup)
    parser.add_argument("--dqn-target-update-steps", type=int, default=RLConfig.dqn_target_update_steps)
    parser.add_argument("--dqn-gradient-clipnorm", type=float, default=RLConfig.dqn_gradient_clipnorm)
    parser.add_argument("--dqn-deterministic-ops", action="store_true",
                        help="Request deterministic TensorFlow operations; requires seed")


def dqn_kwargs(args):
    return {name: getattr(args, name) for name in DQN_FIELDS}
