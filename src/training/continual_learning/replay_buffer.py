"""Experience replay buffer for continual learning."""
import random, torch

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = []
        self.capacity = capacity

    def add(self, sample):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(sample)

    def sample(self, n):
        return random.sample(self.buffer, min(n, len(self.buffer)))
