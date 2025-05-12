from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import wandb

class Actor(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        super().__init__()
        
        hidden_space1 = 64
        hidden_space2 = 128
        
        self.network = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, action_space_dims)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns action probabilities and sampled action"""
        logits = self.forward(state)
        probs = nn.functional.softmax(logits, dim=-1)
        action = torch.multinomial(probs, 1)
        return probs, action
    
    def save_params(self, path: str):
        torch.save(self.state_dict(), path)

    def load_params(self, path: str):
        self.load_state_dict(torch.load(path))

class Critic(nn.Module):
    def __init__(self, obs_space_dims: int):
        super().__init__()
        
        hidden_space1 = 64
        hidden_space2 = 128
        
        self.network = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.ReLU(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, hidden_space2),
            nn.ReLU(),
            nn.Linear(hidden_space2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def save_params(self, path: str):
        torch.save(self.state_dict(), path)

    def load_params(self, path: str):
        self.load_state_dict(torch.load(path))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append((state, action, reward, next_state, done, log_prob))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones, log_probs = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), np.array(log_probs))
    
    def __len__(self):
        return len(self.buffer)

class ActorCriticAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Initialize networks
        self.actor = Actor(state_size, action_size).to(device)
        self.critic = Critic(state_size).to(device)
        
        # Hyperparameters
        self.gamma = 0.99  # discount factor
        self.learning_rate = 1e-3
        self.buffer_size = 100000
        self.batch_size = 64
        
        # Optimizers (why do we need two optimizers???)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # Is a ReplayBuffer strictly part of the A2C algo??
        self.memory = ReplayBuffer(self.buffer_size)
        
    def act(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs, action = self.actor.get_action(state)
        log_prob = torch.log(probs.gather(1, action))
        return action.item(), log_prob
    
    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return 0, 0
        
        states, actions, rewards, next_states, dones, log_probs = self.memory.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        log_probs = torch.FloatTensor(log_probs).to(self.device)
        
        # Calculate value estimates
        values = self.critic(states)
        next_values = self.critic(next_states)
        
        # Calculate TD error
        td_target = rewards + (1 - dones) * self.gamma * next_values.detach()
        td_error = td_target - values
        
        # Actor loss (policy gradient)
        actor_loss = -(log_probs * td_error.detach()).mean()
        
        # Critic loss (value function)
        critic_loss = nn.MSELoss()(values, td_target.detach())
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

def main():
    # Initialize environment
    env = gym.make("LunarLander-v3", render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = ActorCriticAgent(state_size, action_size, device)
    
    # Training parameters
    n_episodes = 1000
    max_steps = 1000
    
    # Initialize wandb
    wandb.init(
        project="lunar-lander-actor-critic",
        config={
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "buffer_size": agent.buffer_size,
            "batch_size": agent.batch_size
        }
    )
    
    # Training loop
    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Get action and log probability
            action, log_prob = agent.act(state)
            
            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done, log_prob.item())
            
            # Train agent
            actor_loss, critic_loss = agent.train(agent.batch_size)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Log metrics
        wandb.log({
            "episode": episode,
            "reward": total_reward,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "steps": step
        })
        
        print(f"Episode: {episode}, Total Reward: {total_reward:.2f}")
        
        # Save model periodically
        if episode % 100 == 0:
            agent.actor.save_params(f'actor_model_episode_{episode}.pth')
            agent.critic.save_params(f'critic_model_episode_{episode}.pth')
    
    agent.actor.save_params(f'actor_model_episode_{episode}.pth')
    agent.critic.save_params(f'critic_model_episode_{episode}.pth')
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main() 