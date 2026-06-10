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

    def _build(self, lr):
        model = models.Sequential([
            layers.Input(shape=(self.state_size,)),
            layers.Dense(24, activation="relu"),
            layers.Dense(24, activation="relu"),
            layers.Dense(self.action_size, activation="linear"),
        ])
        model.compile(loss="mse", optimizer=optimizers.Adam(learning_rate=lr))
        return model

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.eps:
            return random.randrange(self.action_size)
        q = self.model.predict(state[np.newaxis, :], verbose=0)[0]
        return int(np.argmax(q))

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def train_step(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        for s, a, r, s2, done in batch:
            target = r if done else r + self.gamma * np.max(self.model.predict(s2[np.newaxis, :], verbose=0)[0])
            qvals = self.model.predict(s[np.newaxis, :], verbose=0)
            qvals[0][a] = target
            self.model.fit(s[np.newaxis, :], qvals, verbose=0)
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path: str): self.model.save(path)
    def load(self, path: str): self.model = tf.keras.models.load_model(path)
