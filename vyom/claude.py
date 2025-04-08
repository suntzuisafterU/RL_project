import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.policy(x)
    
    def sample_action(self, state):
        state = state.float()
        probs = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

# REINFORCE algorithm implementation
def reinforce(env, policy, optimizer, n_episodes=1000, gamma=0.99, print_every=1):
    episode_rewards = []
    
    for episode in range(1, n_episodes + 1):
        # Initialize episode
        state, _ = env.reset()
        done = False
        rewards = []
        log_probs = []
        
        # Run episode
        while not done:
            state_tensor = torch.FloatTensor(state)
            action, log_prob = policy.sample_action(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Store rewards and log probabilities
            rewards.append(reward)
            log_probs.append(log_prob)
            
            done = terminated or truncated
            state = next_state
        
        # Calculate total episode reward
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)
        
        # Calculate returns (discounted future rewards)
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)
        
        # Normalize returns for stability (optional)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        log_probs = torch.stack(log_probs)

        optimizer.zero_grad()
        policy_loss = torch.sum(-log_probs * returns)
        policy_loss.backward()
        optimizer.step()
        
        # Print progress
        print(f'Episode {episode}, Total Reward: {episode_reward:.2f}, Policy Loss: {policy_loss:.2f}')
    
    return episode_rewards

# Function to plot the learning curve
def plot_learning_curve(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Learning Curve: REINFORCE on Lunar Lander')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Add rolling average
    window_size = 50
    if len(rewards) >= window_size:
        rolling_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), rolling_avg, 'r-', label=f'{window_size}-Episode Moving Average')
    
    plt.legend()
    plt.show()

# Main function to run the training
def main():
    # Create environment
    env = gym.make("LunarLander-v3")

    
    # Get environment dimensions
    input_dim = env.observation_space.shape[0]  # 8 for LunarLander
    output_dim = env.action_space.n  # 4 for LunarLander
    
    # Create policy network and optimizer
    policy = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    
    # Train the agent
    print("Starting training...")
    rewards = reinforce(env, policy, optimizer, n_episodes=1000, gamma=0.99, print_every=50)
    print("Training complete!")
    
    # Plot learning curve
    plot_learning_curve(rewards)
    
    # Test the trained policy
    print("Testing trained policy...")
    test_rewards = []
    for _ in range(10):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state)
            action, _ = policy.sample_action(state_tensor)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            # Render the environment (optional)
            # env.render()
        
        test_rewards.append(episode_reward)
    
    print(f"Average test reward: {np.mean(test_rewards):.2f}")
    
    # Close the environment
    env.close()

# Run the main function
if __name__ == "__main__":
    main()