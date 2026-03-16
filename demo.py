import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import random
from collections import namedtuple, deque
from itertools import count
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from SnakeEnv import SnakeEnv
from ReplayMemory import ReplayMemory
from DQN import DQN

from Board import Board
from Game import Game
import time
import pygame

def select_action(state):
    with torch.no_grad():
        return policy_net(state).argmax(dim=1)

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

env = SnakeEnv()

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load("snake_dqn.pth", map_location=device))
policy_net.eval()

game = Game()

state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
done = False

while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            
    action = select_action(state)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    if not done:
        state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    game.update(env.snake, tuple(map(int, env.food)), reward)
    pygame.display.update()
    time.sleep(0.2)
