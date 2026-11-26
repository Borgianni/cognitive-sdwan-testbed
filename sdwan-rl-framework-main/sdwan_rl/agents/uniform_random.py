"""Uniform stochastic reference policy, independent of telemetry."""
import random


class UniformRandomAgent:
    name = "Uniform random"

    def __init__(self, seed=None, action_size=2):
        if type(action_size) is not int or action_size <= 0:
            raise ValueError("Positive integer action_size required")
        if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**32 - 1):
            raise ValueError("Invalid run seed")
        self.rng = random.Random(seed)
        self.action_size = action_size

    @property
    def configuration(self):
        return {"distribution": "uniform", "action_count": self.action_size,
                "rng": "python_random", "reset_each_episode": False}

    def reset(self):
        # Deliberately do not reseed at episode boundaries.
        pass

    def decide(self, metrics):
        action = self.rng.randrange(self.action_size)
        return action, {"action_probabilities": [1 / self.action_size] * self.action_size,
                        "reason": "uniform_random"}
