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

env = SnakeEnv.SnakeEnv()


print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print("Actions: 0=Straight, 1=Turn Left, 2=Turn Right")