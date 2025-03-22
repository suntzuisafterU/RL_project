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
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()

        hidden_space1 = 512
        hidden_space2 = 256

        # cross entropy loss for discrete action outputs.
        # NOTE: Torch initializes Linear layers to U(-1/sqrt(n), 1/sqrt(n)) by default, biases to 0.
        self.policy = nn.Sequential(
                nn.Linear(obs_space_dims, hidden_space1),
                nn.ReLU(),
                nn.Linear(hidden_space1, hidden_space2),
                nn.ReLU(),
                nn.Linear(hidden_space2, action_space_dims),
                nn.Softmax(dim=-1),
        )

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
    gamma = 0.99

    while not finished:
        action, log_prob = policy.sample_action(torch.tensor(observation))
        observation, reward, terminated, truncated, info = env.step(action)
        finished = terminated or truncated
        rewards.append(reward)
        log_probs.append(log_prob)

    # log_probs = torch.tensor(log_probs, requires_grad=True)
    total_reward = sum(rewards)

    returns = []
    G = 0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, requires_grad=True)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    # loss = torch.sum(-log_probs * returns)

    policy_loss = []
    for log_prob, G in zip(log_probs, returns):
        policy_loss.append(-log_prob * G)  # Negative for gradient ascent

    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()

    return total_reward, loss


policy = PolicyNetwork(obs_space_dims=8, action_space_dims=4)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
n_episodes = 1000

for i in (range(n_episodes)):
    
    total_reward, loss = episode(env, policy, optimizer)

    print(f"episode {i}, total_reward = {round(total_reward, 2)}, loss = {round(loss.item(), 2)}")


env.close()
