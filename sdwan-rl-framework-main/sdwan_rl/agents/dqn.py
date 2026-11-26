"""Vanilla minibatch DQN with a hard-updated target network."""
import hashlib
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from .base import Agent
from ..config import RLConfig
from ..dqn_settings import CURRENT_DQN, LEGACY_DQN, training_settings
from ..dqn_core import td_targets, replay_ready, target_sync_due, decay_epsilon


class DQNAgent(Agent):
    def __init__(self, state_size: int, action_size: int, lr=1e-4,
                 gamma=0.95, replay_size=10_000, batch_size=32,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                 warmup=None, target_update_steps=100, gradient_clipnorm=10.0,
                 deterministic_ops=False, seed=None, version=CURRENT_DQN):
        self.cfg = RLConfig(lr=lr, gamma=gamma, replay_size=replay_size,
                            batch_size=batch_size, eps_start=eps_start, eps_end=eps_end,
                            eps_decay=eps_decay, dqn_version=version,
                            dqn_replay_warmup=batch_size if warmup is None else warmup,
                            dqn_target_update_steps=target_update_steps,
                            dqn_gradient_clipnorm=gradient_clipnorm,
                            dqn_deterministic_ops=deterministic_ops)
        if type(state_size) is not int or type(action_size) is not int or min(state_size, action_size) <= 0:
            raise ValueError("Positive integer state/action dimensions required")
        if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**32 - 1):
            raise ValueError("Invalid DQN seed")
        if deterministic_ops and seed is None:
            raise ValueError("Deterministic TensorFlow operations require a seed")
        if seed is not None:
            tf.keras.utils.set_random_seed(seed)
        if deterministic_ops:
            tf.config.experimental.enable_op_determinism()
        # Separate RNG streams for action exploration and replay sampling.
        def stream(label):
            return random.Random(None if seed is None else int(
                hashlib.sha256(f"{seed}:{label}".encode()).hexdigest(), 16))
        self.exploration_rng = stream("exploration")
        self.replay_rng = stream("replay")
        self.state_size, self.action_size = state_size, action_size
        self.gamma, self.batch_size = gamma, batch_size
        self.eps, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.memory = deque(maxlen=replay_size)
        self.hidden_units = (64, 32) if version == CURRENT_DQN else (24, 24)
        self.model = self._build()
        self.target_model = None
        self.optimizer = None
        if version == CURRENT_DQN:
            self.target_model = self._build()
            self.target_model.set_weights(self.model.get_weights())
            self.target_model.trainable = False
            self.optimizer = optimizers.Adam(learning_rate=lr, global_clipnorm=gradient_clipnorm)
            self.model.compile(loss="mse", optimizer=self.optimizer)
        self.updates = 0
        self.target_syncs = 0  # excludes initialization
        self.frozen = False
        if version == LEGACY_DQN:
            self.freeze()

    @classmethod
    def from_config(cls, cfg, state_size=6, action_size=2, seed=None):
        return cls(state_size, action_size, lr=cfg.lr, gamma=cfg.gamma,
                   replay_size=cfg.replay_size, batch_size=cfg.batch_size,
                   eps_start=cfg.eps_start, eps_end=cfg.eps_end, eps_decay=cfg.eps_decay,
                   warmup=cfg.dqn_replay_warmup, target_update_steps=cfg.dqn_target_update_steps,
                   gradient_clipnorm=cfg.dqn_gradient_clipnorm,
                   deterministic_ops=cfg.dqn_deterministic_ops, seed=seed, version=cfg.dqn_version)

    def _build(self):
        return models.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(self.hidden_units[0], activation="relu",
                         kernel_initializer="glorot_uniform", bias_initializer="zeros"),
            layers.Dense(self.hidden_units[1], activation="relu",
                         kernel_initializer="glorot_uniform", bias_initializer="zeros"),
            layers.Dense(self.action_size, activation="linear",
                         kernel_initializer="glorot_uniform", bias_initializer="zeros"),
        ])

    def q_values(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_size,) or not np.isfinite(state).all():
            raise ValueError("Invalid DQN observation")
        q = self.model(state[np.newaxis, :], training=False).numpy()[0]
        if not np.isfinite(q).all():
            raise ValueError("Non-finite Q-values")
        return q

    def freeze(self):
        self.frozen = True
        self.eps = 0.0

    def act(self, state: np.ndarray) -> int:
        return self.traced_action(state)[0]

    def traced_action(self, state: np.ndarray):
        q = self.q_values(state)
        exploratory = not self.frozen and self.exploration_rng.random() < self.eps
        action = self.exploration_rng.randrange(self.action_size) if exploratory else int(np.argmax(q))
        return action, q, exploratory

    def remember(self, s, a, r, s2, done):
        if self.frozen:
            return
        s, s2 = np.asarray(s, dtype=np.float32), np.asarray(s2, dtype=np.float32)
        if (s.shape != (self.state_size,) or s2.shape != s.shape
                or not np.isfinite(s).all() or not np.isfinite(s2).all()
                or not isinstance(a, (int, np.integer)) or not 0 <= a < self.action_size
                or not np.isfinite(r)):
            raise ValueError("Invalid replay transition")
        self.memory.append((s.copy(), int(a), float(r), s2.copy(), bool(done)))

    def training_state(self):
        return {"optimizer_updates": self.updates, "target_syncs": self.target_syncs,
                "replay_size": len(self.memory), "epsilon": self.eps}

    def settings(self):
        return training_settings(self.cfg)

    def train_step(self):
        if not replay_ready(len(self.memory), self.cfg.dqn_replay_warmup, self.frozen):
            return None
        batch = self.replay_rng.sample(list(self.memory), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        next_q = self.target_model(next_states, training=False).numpy()
        targets = tf.convert_to_tensor(td_targets(rewards, next_q, dones, self.gamma))
        indices = tf.convert_to_tensor(np.column_stack((np.arange(self.batch_size), actions)),
                                       dtype=tf.int32)
        with tf.GradientTape() as tape:
            q = self.model(states, training=True)
            selected = tf.gather_nd(q, indices)
            loss = tf.reduce_mean(tf.square(targets - selected))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        if any(g is None for g in gradients):
            raise RuntimeError("Missing DQN gradient")
        norm = tf.linalg.global_norm(gradients)
        if not np.isfinite(float(loss.numpy())) or not np.isfinite(float(norm.numpy())):
            raise RuntimeError("Non-finite DQN training loss or gradient norm")
        # Adam applies global_clipnorm once, across all gradients.
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.updates += 1
        sync = target_sync_due(self.updates, self.cfg.dqn_target_update_steps)
        if sync:
            self.target_model.set_weights(self.model.get_weights())
            self.target_syncs += 1
        self.eps = decay_epsilon(self.eps, self.eps_end, self.eps_decay)
        return {**self.training_state(), "td_mse": float(loss.numpy()),
                "unclipped_gradient_global_norm": float(norm.numpy()), "target_synced": sync}

    def save(self, path: str):
        if self.cfg.dqn_version != CURRENT_DQN:
            raise ValueError("Legacy model is inference-only; use historical commit for training")
        self.model.save(path)

    def load(self, path: str):
        # Checkpoint API deliberately supports inference, not training resumption.
        model = tf.keras.models.load_model(path, compile=False)
        expected_units = [*self.hidden_units, self.action_size]
        if (tuple(model.input_shape[1:]) != (self.state_size,)
                or tuple(model.output_shape[1:]) != (self.action_size,)
                or len(model.layers) != len(expected_units)
                or any(not isinstance(layer, layers.Dense) for layer in model.layers)
                or [layer.units for layer in model.layers] != expected_units
                or [layer.activation.__name__ for layer in model.layers] != ["relu", "relu", "linear"]):
            raise ValueError("Checkpoint architecture differs from declared DQN version")
        self.model = model
        if self.target_model is not None:
            self.target_model.set_weights(model.get_weights())
        self.memory.clear()
        self.updates = self.target_syncs = 0
        self.optimizer = None
        self.freeze()
