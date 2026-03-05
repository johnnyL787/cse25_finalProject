import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import random

class SnakeEnv(gym.Env):
    """
    Minimal Snake environment (work-in-progress).
    Actions: 0=Straight, 1=Left, 2=Right
    Observation: 11-dim feature vector
    """

    
    REWARD_EAT = 10.0
    PENALTY_DEATH = -10.0
    STEP_PENALTY = -0.1

    def __init__(self, grid_size=10, max_steps=500):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(11,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        mid = self.grid_size // 2
        self.direction = (1, 0)  # right
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        self._spawn_food()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.steps += 1

       
        self.direction = self._turn(self.direction, action)

    
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        reward = self.STEP_PENALTY
        terminated = False
        truncated = False

        # Collision with wall
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            reward = self.PENALTY_DEATH
            terminated = True

        # Collision with self
        elif new_head in self.snake:
            reward = self.PENALTY_DEATH
            terminated = True

        else:
            # Advance snake
            self.snake.insert(0, new_head)

            # Eat food
            if new_head == self.food:
                reward = self.REWARD_EAT
                self._spawn_food()
            else:
                self.snake.pop()  

        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {"length": len(self.snake)}
        return obs, reward, terminated, truncated, info

    def _spawn_food(self):
        while True:
            fx = self.np_random.integers(0, self.grid_size)
            fy = self.np_random.integers(0, self.grid_size)
            if (fx, fy) not in self.snake:
                self.food = (fx, fy)
                return

    def _turn(self, direction, action):
        dx, dy = direction
        if action == 0:      # straight
            return (dx, dy)
        if action == 1:      # left
            return (-dy, dx)
        else:                # right
            return (dy, -dx)

    def _danger(self, direction):
        head_x, head_y = self.snake[0]
        dx, dy = direction
        nxt = (head_x + dx, head_y + dy)
        if not (0 <= nxt[0] < self.grid_size and 0 <= nxt[1] < self.grid_size):
            return 1.0
        if nxt in self.snake:
            return 1.0
        return 0.0

    def _get_obs(self):
        head_x, head_y = self.snake[0]
        fx, fy = self.food
        dx, dy = self.direction

        straight = (dx, dy)
        left = (-dy, dx)
        right = (dy, -dx)

        danger_straight = self._danger(straight)
        danger_left = self._danger(left)
        danger_right = self._danger(right)

        food_left = 1.0 if fx < head_x else 0.0
        food_right = 1.0 if fx > head_x else 0.0
        food_up = 1.0 if fy < head_y else 0.0
        food_down = 1.0 if fy > head_y else 0.0

        moving_left = 1.0 if (dx, dy) == (-1, 0) else 0.0
        moving_right = 1.0 if (dx, dy) == (1, 0) else 0.0
        moving_up = 1.0 if (dx, dy) == (0, -1) else 0.0
        moving_down = 1.0 if (dx, dy) == (0, 1) else 0.0

        return np.array([
            danger_straight, danger_left, danger_right,
            food_left, food_right, food_up, food_down,
            moving_left, moving_right, moving_up, moving_down
        ], dtype=np.float32)