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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import SnakeEnv
from DQN import DQN
from ReplayMemory import ReplayMemory, Transition

#Hyperparameters:
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
LR = 0.001

#Setup:
env = SnakeEnv.SnakeEnv()
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]

policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    done_batch = torch.cat(batch.done)

    # Predicted Q-values for actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Expected Q-values from target network
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (1 - done_batch))

    # Loss calculation
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#Training
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print("Actions: 0=Straight, 1=Turn Left, 2=Turn Right")

num_episodes = 500
for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor([state], dtype=torch.float32)
    
    for t in count():
        action = select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        reward = torch.tensor([reward], dtype=torch.float32)
        next_state_tensor = torch.tensor([next_state], dtype=torch.float32)
        done_flag = torch.tensor([float(done)], dtype=torch.float32)

        #Savingto memory
        memory.push(state, action, next_state_tensor, reward, done_flag)
        
        # Moveing to next state
        state = next_state_tensor

        optimize_model()
        
        if done:
            break
            
    # Periodically update target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if i_episode % 10 == 0:
        print(f"Episode {i_episode} complete. Last length: {info['length']}")

print("Training Complete")
