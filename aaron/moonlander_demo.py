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

        hidden_space1 = 64
        hidden_space2 = 128

        # cross entropy loss for discrete action outputs.
        # NOTE: Torch initializes Linear layers to U(-1/sqrt(n), 1/sqrt(n)) by default, biases to 0.
        self.cat_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, action_space_dims),
        )
        self.last_layer = nn.Sequential(
            nn.Softmax(dim=-1) # Converts outputs to  probabilities that sum to 1 ie: exp(p_i)/sum_j(exp(p_j))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Returns action probabilities """
        x = self.cat_net(x)
        return self.last_layer(x)

    def sample_action(self, state: torch.Tensor) -> tuple[float, torch.Tensor]:
        """ Sample an action from the policy """
        probs = self.forward(torch.Tensor(state))
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        # TODO: Why did Claude want log_probs here??
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def save_params(self, path: str):
        """Save the model parameters to a file"""
        torch.save(self.state_dict(), path)

    def load_params(self, path: str):
        """Load model parameters from a file"""
        try:
            self.load_state_dict(torch.load(path))
        except:
            pass


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

class REINFORCE(object):

    def __init__(self, policy):
        self.policy = policy
        self.optim = torch.optim.Adam(
                self.policy.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
            )

    def sample_action(self, obs):
        action, log_prob = self.policy.sample_action(obs)
        return action, log_prob
    
    def episode_update(self, log_probs, rewards):
        """ Update at the end of an episode... """
        # How do we contextualize the reward at this point? We have the log_prob...
        # Want parameter update based on reward and learning rate.
        # That's the job of an optimizer like Adam.

        # Rewards are returns no discounting.
        final_reward = rewards[-1]
        policy_loss = []
        for R, log_prob in zip(rewards, log_probs):
            if abs(final_reward) == 100:
                R += final_reward
            policy_loss.append(-log_prob * R) # negative because grad ascent

        policy_loss = torch.stack(policy_loss).sum()
        self.optim.zero_grad()
        policy_loss.backward()
        self.optim.step()
        return policy_loss





#####################################################
#####################################################
#####################################################


import signal
import sys

def clean_up():
    policy.save_params('params_file')
    # Plot final rewards
    plt.figure()
    plt.plot(episode_final_rewards, alpha=0.5, label='Episode Rewards')

    win_size = 200

    # Calculate and plot moving average
    if len(episode_final_rewards) >= win_size:
        windowing = pd.Series(episode_final_rewards).rolling(window=win_size)
        moving_avg = windowing.mean()
        moving_std = windowing.std()
        plt.plot(moving_avg, label=f'{win_size}-episode Moving Average', linewidth=2)
        plt.fill_between(moving_avg.index,
                         moving_avg - moving_std,
                         moving_avg + moving_std,
                         alpha=0.2, label='±1 std')

    plt.plot(episode_final_rewards_binary, alpha=0.5, label='Episode Rewards Binary')

    # Calculate and plot moving average
    if len(episode_final_rewards_binary) >= win_size:
        windowing = pd.Series(episode_final_rewards_binary).rolling(window=win_size)
        moving_avg = windowing.mean()
        moving_std = windowing.std()
        plt.plot(moving_avg, label=f'{win_size}-episode Moving Average Binary', linewidth=2)
        plt.fill_between(moving_avg.index,
                         moving_avg - moving_std,
                         moving_avg + moving_std,
                         alpha=0.2, label='±1 std binary')
    
    plt.xlabel('Episode')
    plt.ylabel('Final Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig('training_progress.png')
    plt.close()
    env.close()

# Register keyboard interrupt handler
def signal_handler(sig, frame):
    print('\nExiting gracefully...')
    clean_up()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)




# Reset the environment to generate the first observation
num_episodes = 10000

lr=1e-3
betas=(0.9, 0.999)
eps=1e-8

policy = PolicyNetwork(obs_space_dims=8, action_space_dims=4)
reinforce = REINFORCE(policy)

# Load parameters if they exist
try:
    policy.load_params('params_file')
    print("Loaded existing parameters from params_file")
except FileNotFoundError:
    print("No existing parameters found, starting with random initialization")





################################################################
########### wandb init #########################################
# Initialize wandb for experiment tracking
import wandb

# Configure wandb settings
wandb.init(
    project="lunar-lander-reinforce",
    config={
        "lr": lr,  # Add actual learning rate from REINFORCE
        "num_episodes": num_episodes, # TODO: Save/load this too. 
        # And keep track of this when saving/loading the network, should have all the same params.
        "max_steps": 1000,
        "architecture": "REINFORCE",
        "env_name": "LunarLander-v3"
    }
)

# Create dictionary to track metrics
metrics = {
    'episode_reward': 0,
    'episode_length': 0,
    'episode_final_reward': 0,
    'episode_binary_reward': 0,
    'moving_avg_reward': 0
}



################################################################
########### env ################################################
plt.rcParams["figure.figsize"] = (10, 5)

# Initialise the environment
# env = gym.make("LunarLander-v3", render_mode="human")
env = gym.make("LunarLander-v3")

obs, info = env.reset(seed=42)

###############################################################
##### start training loop #####################################
episode_final_rewards = []
episode_final_rewards_binary = []
for episode in range(num_episodes):
    log_probs = []
    rewards = []
    for _ in range(1000):
        # this is where you would insert your policy
        # Action space: 0: nothing; 1: left; 2: main; 3: right;
        # action = env.action_space.sample()
        # action = simple_rule_based_policy(obs)
        action, log_prob = reinforce.sample_action(obs)
        log_probs.append(log_prob)

        # Convert back to numpy since that's what env.step expects.
        action = action.numpy()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode
        # has terminated or truncated. Us 'terminated' to determine 
        # if bootstrapping is appropriate.
        obs, reward, terminated, truncated, info = env.step(action)

        # # apply baseline to action.
        # # Penalize being away from the center line of the state.
        x_pos, y_pos, x_v, y_v, angle, angular_velocity, left_bool, right_bool = obs
        # if not abs(reward) == 100:
        #     reward -= abs(y_pos) * 0.3
        #     reward -= abs(angular_velocity) * 5.0
        #     # boost reward if moving down with angle and momentum low
        #     if y_v < 0 and abs(x_v) < .1:
        #         reward += abs(y_v) * 5
        #         reward -= x_v * 1

        rewards.append(reward)

        # observation 8-dim: (lander x, lander y, velocity x, velocity y, 
        #                     lander angle, lander angular velocity,
        #                     right leg contact ground?, left leg contact ground?) # Last 2 bools
        # NOTE: angular velocity is in units of 0.4 rads/second. So have to multiply by 2.5 to get rads/sec...

        ## Update the policy parameters. Must be where we make use of the log_prob???
        ##                               Why log_prob??? Why not just prob???

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            if abs(y_pos) > 0.1 and abs(rewards[-1]) == 100:
                # May have landed, but not between flags!! Penalize heavily.
                rewards[-1] -= 50 * abs(y_pos)

            policy_loss = reinforce.episode_update(log_probs, rewards)

            print(episode, rewards[-2:])
            if abs(rewards[-1]) == 100:
                episode_final_rewards.append(rewards[-2] + rewards[-1])
                episode_final_rewards_binary.append(rewards[-1])
            else:
                episode_final_rewards.append(rewards[-1])
                episode_final_rewards_binary.append(0)


            ###################################################################
            #################### wandb ########################################
            # Log gradients and weights histograms
            for name, param in policy.named_parameters():
                if param.requires_grad:
                    wandb.log({
                        f"gradients/{name}": wandb.Histogram(param.grad.detach().cpu().numpy()),
                        f"weights/{name}": wandb.Histogram(param.data.detach().cpu().numpy())
                    }, step=episode)
            
            # Log performance metrics
            wandb.log({
                "episode": episode,
                "reward": episode_final_rewards[-1],
                "policy_loss": policy_loss.item(),
                "episode_length": len(rewards)
            }, step=episode)
            
            # Optional: Log parameter statistics
            for name, param in policy.named_parameters():
                wandb.log({
                    f"param_mean/{name}": param.data.mean().item(),
                    f"param_std/{name}": param.data.std().item(),
                    f"grad_mean/{name}": param.grad.mean().item() if param.grad is not None else 0,
                    f"grad_std/{name}": param.grad.std().item() if param.grad is not None else 0
                }, step=episode)
            
            obs, info = env.reset()
            break # new episode
    else:
        clean_up()
        raise RuntimeError('Did not expect 1000 steps in episode')

clean_up()
