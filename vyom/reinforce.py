from __future__ import annotations
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions.normal import Normal
from tqdm import tqdm
import gymnasium as gym

torch.manual_seed(42)
np.random.seed(42)

plt.rcParams["figure.figsize"] = (10, 5)


class PolicyNetwork(nn.Module):
    def __init__(self, env, obs_space_dims: int, action_space_dims: int, gamma=0.99, n_layers: int = 2, layer_size: int = 128):
        super().__init__()
        self.env = env
        self.gamma = gamma
        self.n_layers = n_layers
        self.layer_size = layer_size

        self.input_layer = nn.Linear(obs_space_dims, layer_size)
        hidden = []
        for _ in range(n_layers):
            hidden += [nn.Linear(layer_size, layer_size), nn.ReLU(inplace=True)]
        self.feats = nn.Sequential(*hidden)

        self.policy_head = nn.Sequential(
            nn.Linear(layer_size, action_space_dims),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities"""
        x = F.relu(self.input_layer(x))
        x = self.feats(x)
        return self.policy_head(x)

    def sample_action(self, state: torch.Tensor) -> tuple[float, torch.Tensor]:
        """Sample an action from the policy"""
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def play_episode(self, discount=False):
        finished = False
        state, _ = self.env.reset()
        rewards = []
        log_probs = []

        while not finished:
            action, log_prob = self.sample_action(torch.tensor(state))
            state, reward, terminated, truncated, _ = self.env.step(action)
            finished = terminated or truncated
            rewards.append(reward)
            log_probs.append(log_prob)

        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        if discount:
            returns *= (self.gamma ** torch.arange(len(returns)))
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        log_probs = torch.stack(log_probs)

        return returns, log_probs, sum(rewards)


def main():
    env = gym.make("LunarLander-v3")
    policy = PolicyNetwork(env, obs_space_dims=8, action_space_dims=4, n_layers=3, layer_size=128)
    policy = torch.compile(policy)
    optimizer = optim.Adam(policy.parameters(), lr=5e-4)
    n_episodes = 1000
    batch_size = 1

    for epi in range(n_episodes):
        batch_returns, batch_rewards, batch_log_probs = [], [], []
        for i in range(batch_size):
            returns, log_probs, reward = policy.play_episode(discount=True)
            batch_returns.append(returns)
            batch_rewards.append(reward)
            batch_log_probs.append(log_probs)

        batch_returns, batch_log_probs = torch.cat(batch_returns), torch.cat(batch_log_probs)
        batch_rewards = torch.tensor(batch_rewards)

        loss = torch.sum(-batch_log_probs * batch_returns) / batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Episode {epi} \t Loss: {loss.item():.3f} \t Reward: {batch_rewards.mean().item():.2f}")

    env.close()


if __name__ == "__main__":
    main()