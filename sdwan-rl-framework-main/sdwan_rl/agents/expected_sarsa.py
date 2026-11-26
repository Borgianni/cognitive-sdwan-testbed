"""Online linear Expected SARSA, using the same discrete-action Agent API."""
import math
import random
import numpy as np

from .base import Agent
from ..dqn_core import decay_epsilon


class ExpectedSarsaAgent(Agent):
    def __init__(self, cfg, seed=None, alpha=.01, state_size=6, action_size=2):
        if type(alpha) not in (int,float) or not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Expected SARSA alpha must be in (0,1]")
        if type(state_size) is not int or type(action_size) is not int or min(state_size,action_size) <= 0:
            raise ValueError("Invalid state/action dimensions")
        self.state_size, self.action_size = state_size, action_size
        self.alpha, self.gamma = float(alpha), cfg.gamma
        self.eps, self.eps_end, self.eps_decay = cfg.eps_start, cfg.eps_end, cfg.eps_decay
        self.eps_start = cfg.eps_start
        self.rng = random.Random(seed)
        # Small independently seeded initial weights; affine features include bias.
        self.weights = np.random.default_rng(seed).uniform(-.01,.01,(state_size+1,action_size))
        self.memory = []  # no replay; one pending transition at most
        self.updates, self.frozen = 0, False

    def settings(self):
        return {"algorithm":"linear_expected_sarsa_v1","alpha":self.alpha,"gamma":self.gamma,
                "eps_start":self.eps_start,"eps_end":self.eps_end,"eps_decay":self.eps_decay,
                "state_size":self.state_size,"action_size":self.action_size,
                "initialization":"uniform_-0.01_0.01","replay":False}

    def parameter_arrays(self):
        return [self.weights]

    def features(self,state):
        x = np.asarray(state,dtype=np.float64)
        if x.shape != (self.state_size,) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):
            raise ValueError("Expected finite normalized state features in [0,1]")
        return np.concatenate((x,[1.]))

    def q_values(self,state):
        q = self.features(state) @ self.weights
        if not np.isfinite(q).all():
            raise RuntimeError("Non-finite Expected SARSA Q values")
        return q

    def traced_action(self,state):
        q = self.q_values(state)
        exploratory = not self.frozen and self.rng.random() < self.eps
        return (self.rng.randrange(self.action_size) if exploratory else int(np.argmax(q))),q,exploratory

    def act(self,state):
        return self.traced_action(state)[0]

    def remember(self,s,a,r,s2,done):
        if self.frozen:
            return
        if self.memory:
            raise RuntimeError("Previous transition has not been consumed")
        if type(a) is not int or not 0 <= a < self.action_size or not math.isfinite(r):
            raise ValueError("Invalid transition")
        self.memory.append((self.features(s),a,float(r),self.features(s2),bool(done)))

    def train_step(self):
        if self.frozen or not self.memory:
            return None
        x,a,r,x2,done = self.memory.pop()
        q2 = x2 @ self.weights
        # Expected next value under the current epsilon-greedy policy
        # (not a maximization backup). Terminal transitions do not bootstrap.
        expected = (1-self.eps)*float(np.max(q2))+self.eps*float(np.mean(q2))
        target = r if done else r+self.gamma*expected
        error = target-float(x @ self.weights[:,a])
        updated = self.weights[:,a]+self.alpha*error*x
        if not np.isfinite(updated).all() or not math.isfinite(error):
            raise RuntimeError("Non-finite Expected SARSA update")
        self.weights[:,a] = updated
        self.updates += 1
        self.eps = decay_epsilon(self.eps,self.eps_end,self.eps_decay)
        return {**self.training_state(),"td_error":error,"td_target":target}

    def training_state(self):
        return {"optimizer_updates":self.updates,"replay_size":len(self.memory),"epsilon":self.eps}

    def freeze(self):
        self.frozen = True
        self.eps = 0.
        self.memory.clear()

    def save(self,path):
        with open(path,"xb") as stream:
            np.savez(stream,weights=self.weights)

    def load(self,path):
        with np.load(path,allow_pickle=False) as data:
            if set(data.files)!={"weights"}:
                raise ValueError("Unexpected checkpoint fields")
            w = data["weights"]
        if w.shape != self.weights.shape or w.dtype != np.float64 or not np.isfinite(w).all():
            raise ValueError("Invalid Expected SARSA checkpoint")
        self.weights = w.copy()
        self.updates = 0
        self.freeze()
