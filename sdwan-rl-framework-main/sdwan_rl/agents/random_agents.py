import random
from .base import Agent

class RandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space
        self.name = "Random"
        
    def act(self, state, reward, done):
        # state può essere anche None, non serve
        return random.randrange(self.action_space.n)
    
    def remember(self, state, action, reward, next_state, done):
        pass
    
    def train_step(self):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass