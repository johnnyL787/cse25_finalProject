import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import random

class Agent:
    def __init__(self, action_space_n, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.action_space_n = action_space_n
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.Q = {}  

    def _key(self, state):
        return tuple(int(x > 0.5) for x in state)

    def _ensure_state(self, s_key):
        if s_key not in self.Q:
            self.Q[s_key] = np.zeros(self.action_space_n, dtype=np.float32)

    def act(self, state):
        s_key = self._key(state)
        self._ensure_state(s_key)

        if random.random() < self.epsilon:
            return random.randrange(self.action_space_n)
        return int(np.argmax(self.Q[s_key]))

    def learn(self, state, action, reward, next_state, done):
        s_key = self._key(state)
        ns_key = self._key(next_state)
        self._ensure_state(s_key)
        self._ensure_state(ns_key)

        q_sa = self.Q[s_key][action]
        target = reward if done else reward + self.gamma * np.max(self.Q[ns_key])
        self.Q[s_key][action] += self.alpha * (target - q_sa)

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
