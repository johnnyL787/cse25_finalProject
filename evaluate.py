import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

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

episodes = 2000
scores = []
steps = []

for _ in range(episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False
    score = 0
    step = 0

    while not done:
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        step += 1

        if reward == env.REWARD_EAT:
            score += 1

        if not done:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    scores.append(score)
    steps.append(step)


print(f"Max score: {max(scores)}")
print(f"Average score: {sum(scores) / len(scores)}")
print(f"Average step: {sum(steps) / len(steps)}")

