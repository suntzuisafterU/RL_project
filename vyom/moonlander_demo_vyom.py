from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions.normal import Normal
from tqdm import tqdm

import gymnasium as gym

torch.manual_seed(42)
np.random.seed(42)

plt.rcParams["figure.figsize"] = (10, 5)


# Initialise the environment
env = gym.make("LunarLander-v3")


class PolicyNetwork(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int, n_layers: int = 2):
        super().__init__()
        self.l1 = nn.Linear(obs_space_dims, 64)
        self.layers = self.make_feature_layers(n_layers)
        self.l2 = nn.Linear(64, action_space_dims)
        self.policy = nn.Sequential(self.l1, nn.ReLU(), *self.layers, nn.ReLU(), self.l2, nn.Softmax(dim=-1))

    def make_feature_layers(self, n_layers: int) -> nn.Sequential:
        layers = []
        for i in range(n_layers):
            layers += [nn.Linear(64, 64), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities"""
        return self.policy(x)

    def sample_action(self, state: torch.Tensor) -> tuple[float, torch.Tensor]:
        """Sample an action from the policy"""
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.item()
        return action, log_prob


def episode(env, policy, optimizer):
    finished = False
    observation, info = env.reset()
    rewards = []
    log_probs = []
    gamma = 0.999

    while not finished:
        action, log_prob = policy.sample_action(torch.tensor(observation))
        observation, reward, terminated, truncated, info = env.step(action)
        finished = terminated or truncated
        rewards.append(reward)
        log_probs.append(log_prob)

    log_probs = torch.tensor(log_probs, requires_grad=True)
    total_reward = sum(rewards)

    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, requires_grad=True)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    loss = torch.sum(-log_probs * returns)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return total_reward, loss


policy = PolicyNetwork(obs_space_dims=8, action_space_dims=4, n_layers=10)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
n_episodes = 1000

for i in (range(n_episodes)):
    
    total_reward, loss = episode(env, policy, optimizer)

    print(f"episode {i}, total_reward = {round(total_reward, 2)}, loss = {round(loss.item(), 2)}")


env.close()
