import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from .base import Agent

class DQNAgent(Agent):
    def __init__(self, state_size: int, action_size: int, lr=1e-4,
                 gamma=0.95, replay_size=10_000, batch_size=32,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.memory = deque(maxlen=replay_size)
        self.model = self._build(lr)
        self.frozen = False

    def _build(self, lr):
        model = models.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(24, activation="relu"),
            layers.Dense(24, activation="relu"),
            layers.Dense(self.action_size, activation="linear"),
        ])
        model.compile(loss="mse", optimizer=optimizers.Adam(learning_rate=lr))
        return model

    def q_values(self, state: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(state, dtype=np.float32)[np.newaxis, :],
                                  verbose=0)[0]

    def freeze(self):
        """Disable exploration, replay insertion and optimizer updates."""
        self.frozen = True
        self.eps = 0.0

    def act(self, state: np.ndarray) -> int:
        if not self.frozen and np.random.rand() < self.eps:
            return random.randrange(self.action_size)
        return int(np.argmax(self.q_values(state)))

    def traced_action(self, state: np.ndarray):
        """Capture Q-values and exploration before any training update."""
        q = self.q_values(state)
        exploratory = not self.frozen and np.random.rand() < self.eps
        action = random.randrange(self.action_size) if exploratory else int(np.argmax(q))
        return action, q, exploratory

    def remember(self, s, a, r, s2, done):
        if not self.frozen:
            self.memory.append((s, a, r, s2, done))

    def train_step(self):
        if self.frozen or len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        for s, a, r, s2, done in batch:
            target = r if done else r + self.gamma * np.max(self.model.predict(s2[np.newaxis, :], verbose=0)[0])
            qvals = self.model.predict(s[np.newaxis, :], verbose=0)
            qvals[0][a] = target
            self.model.fit(s[np.newaxis, :], qvals, verbose=0)
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path: str): self.model.save(path)
    def load(self, path: str): self.model = tf.keras.models.load_model(path)
