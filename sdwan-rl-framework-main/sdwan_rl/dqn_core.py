"""Numerical DQN helpers, testable without TensorFlow or Mininet."""
import numpy as np


def td_targets(rewards, next_q, dones, gamma):
    """Vanilla DQN target: max over the *target* network, zero bootstrap if done."""
    rewards = np.asarray(rewards, dtype=np.float32)
    next_q = np.asarray(next_q, dtype=np.float32)
    dones = np.asarray(dones, dtype=bool)
    if (rewards.ndim != 1 or dones.shape != rewards.shape or next_q.ndim != 2
            or next_q.shape[0] != len(rewards) or next_q.shape[1] == 0
            or not np.isfinite(rewards).all() or not np.isfinite(next_q).all()
            or not 0 <= gamma < 1):
        raise ValueError("Invalid TD target inputs")
    return rewards + np.float32(gamma) * np.where(dones, 0.0, next_q.max(axis=1))


def replay_ready(size, warmup, frozen):
    return not frozen and size >= warmup


def target_sync_due(updates, interval):
    return updates > 0 and updates % interval == 0


def decay_epsilon(epsilon, minimum, factor):
    return max(minimum, epsilon * factor)
