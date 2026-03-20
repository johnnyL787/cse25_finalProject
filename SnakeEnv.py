import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import math

from Game import Game
import time

class SnakeEnv(gym.Env):
    """
    Minimal Snake environment (work-in-progress).
    Actions: 0=Straight, 1=Left, 2=Right
    Observation: 11-dim feature vector

    Variables:
        snake = [(x0, y0), (x1, y1), (x2, y2), ...]
            array of xy coordinates, first coordinate is the head of the snake

        steps
            number of times the state has been updated

        direction
            which way the snake is facing
            up - (0, 1)
            down - (0, -1)
            left - (-1, 0)
            right - (1, 0)

        action
            0 - straight
            1 - left
            2 - right

        food
            coordinate (x, y) of current food

    """

    
    REWARD_EAT = 10.0
    PENALTY_DEATH = -10.0
    STEP_PENALTY = -0.2
    FURTHER_PENALTY = -0.1
    CLOSER_REWARD = 0.1

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
            previous_distance = self._distance()
            # Advance snake
            self.snake.insert(0, new_head)

            # Eat food
            if new_head == self.food:
                reward = self.REWARD_EAT
                self._spawn_food()
            else:
                self.snake.pop()
                
                if self._distance() < previous_distance:
                    reward += self.CLOSER_REWARD
                else:
                    reward += self.FURTHER_PENALTY

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

    def _danger_distance(self, direction):
        x, y = self.snake[0]
        dx, dy = direction
        steps = 0

        while True:
            x += dx
            y += dy
            steps += 1

            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                break
            if (x, y) in self.snake:
                break

        return steps / self.grid_size
    
    def _flood_fill(self, start):
        visited = set()
        stack = [start]

        while stack:
            x, y = stack.pop()

            if (x, y) in visited:
                continue
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                continue
            if (x, y) in self.snake:
                continue

            visited.add((x, y))

            stack.append((x + 1, y))
            stack.append((x - 1, y))
            stack.append((x, y + 1))
            stack.append((x, y - 1))

        return len(visited)
    
    def _space_after_move(self, direction):
        hx, hy = self.snake[0]
        dx, dy = direction
        new_head = (hx + dx, hy + dy)

        if new_head in self.snake:
            return 0.0
        
        space = self._flood_fill(new_head)
        return space / (self.grid_size ** 2)

    def _get_obs(self):
        head_x, head_y = self.snake[0]
        fx, fy = self.food
        dx, dy = self.direction

        straight = (dx, dy)
        left = (-dy, dx)
        right = (dy, -dx)

        danger_straight = self._danger_distance(straight)
        danger_left = self._danger_distance(left)
        danger_right = self._danger_distance(right)

        food_dx = (fx - head_x) / self.grid_size
        food_dy = (fy - head_y) / self.grid_size

        space_straight = self._space_after_move(straight)
        space_left = self._space_after_move(left)
        space_right = self._space_after_move(right)

        #food_left = 1.0 if fx < head_x else 0.0
        #food_right = 1.0 if fx > head_x else 0.0
        #food_up = 1.0 if fy < head_y else 0.0
        #food_down = 1.0 if fy > head_y else 0.0

        moving_left = 1.0 if (dx, dy) == (-1, 0) else 0.0
        moving_right = 1.0 if (dx, dy) == (1, 0) else 0.0
        moving_up = 1.0 if (dx, dy) == (0, -1) else 0.0
        moving_down = 1.0 if (dx, dy) == (0, 1) else 0.0

        return np.array([
            danger_straight, danger_left, danger_right,
            food_dx, food_dy,
            moving_left, moving_right, moving_up, moving_down,
            space_straight, space_left, space_right,
        ], dtype=np.float32)
    
    def visualize(self):
        for i in range(self.grid_size):
            print(self.grid_size * "[_]")

    def _distance(self):
        head = self.snake[0]
        return abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
