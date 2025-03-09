from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gymnasium as gym


plt.rcParams["figure.figsize"] = (10, 5)



# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")


def simple_rule_based_policy(observation):
    x, y, vel_x, vel_y, angle, angular_vel, right_leg, left_leg = observation
    
    # Convert angular velocity to rad/sec (as per documentation note)
    angular_vel *= 2.5
    
    # Default action is do nothing
    action = 0
    
    # If tilting too much right, fire left engine
    if angle > 0.2 or angular_vel > 0.5:
        action = 1
    # If tilting too much left, fire right engine
    elif angle < -0.2 or angular_vel < -0.5:
        action = 3
    # If falling too fast, fire main engine
    elif vel_y < -0.5:
        action = 2
        
    return action


#####################################################
########## REINFORCE ################################
#####################################################

class PolicyNetwork(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()

        hidden_space1 = 16
        hidden_space2 = 32

        # cross entropy loss for discrete action outputs.
        # NOTE: Torch initializes Linear layers to U(-1/sqrt(n), 1/sqrt(n)) by default, biases to 0.
        self.cat_net = nn.Sequential([
            nn.Linear(obs_space_dims, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, action_space_dims),
        ])
        self.last_layer = nn.Sequential([
            nn.Softmax(dim=-1) # Converts outputs to  probabilities that sum to 1 ie: exp(p_i)/sum_j(exp(p_j))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Returns action probabilities """
        x = self.cat_net(x)
        return self.last_layer(x)

    def sample_action(self, state: torch.Tensor) -> tuple[float, torch.Tensor]:
        """ Sample an action from the policy """
        probs = self.forward()
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        # TODO: Why did Claude want log_probs here??
        log_prob = dist.log_prob(action)
        action = action.numpy()
        return action, log_prob


    # STOPPING HERE. TODO:
    #                       The rest of the RIENFORCE algo, mixing the article (at faram docs, the tutorial)
    #                       and input from Claude.

# REINFORCE:
#   Initialize theta random.
#   for each episode {s_1, a_1, r_2, ..., s_T-1, a_T-1, r_T} ~ pie_theta do
#       for t = 1 to T-1 do
#           theta <- theta + alpha * grad_theta(log(pie_theta(s_t, a_t) * v_t))
#           # where pie_theta is the policy (the network)
#           # s_t states, a_t actions, v_t is ??? maybe values or something ???
#   return theta




#####################################################
#####################################################
#####################################################




# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)

policy = PolicyNetwork(obs_space_dims=8, action_space_dims=4)

for _ in range(1000):
    # this is where you would insert your policy
    # Action space: 0: nothing; 1: left; 2: main; 3: right;
    # action = env.action_space.sample()
    # action = simple_rule_based_policy(observation)
    action, log_prob = policy.sample_action(observation)

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode
    # has terminated or truncated. Us 'terminated' to determine 
    # if bootstrapping is appropriate.
    observation, reward, terminated, truncated, info = env.step(action)

    # observation 8-dim: (lander x, lander y, velocity x, velocity y, 
    #                     lander angle, lander angular velocity,
    #                     right leg contact ground?, left leg contact ground?) # Last 2 bools
    # NOTE: angular velocity is in units of 0.4 rads/second. So have to multiply by 2.5 to get rads/sec...

    ## Update the policy parameters. Must be where we make use of the log_prob???
    ##                               Why log_prob??? Why not just prob???






    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
