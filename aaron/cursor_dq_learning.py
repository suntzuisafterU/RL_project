from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import wandb

class QNetwork(nn.Module):
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
    
    def save_params(self, path: str):
        torch.save(self.state_dict(), path)

    def load_params(self, path: str):
        self.load_state_dict(torch.load(path))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Q-Network and target network
        self.qnetwork = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.qnetwork.state_dict())
        
        # Hyperparameters
        self.gamma = 0.99  # discount factor
        self.learning_rate = 1e-3
        self.buffer_size = 100000
        self.batch_size = 64
        self.target_update = 10  # update target network every N episodes
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=self.learning_rate)
        self.memory = ReplayBuffer(self.buffer_size)
        
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_values = self.qnetwork(state)
            return action_values.argmax().item()
    
    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q_values = self.qnetwork(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.qnetwork.state_dict())
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def main():
    # Initialize environment
    # env = gym.make("LunarLander-v3")
    env = gym.make("LunarLander-v3", render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DQNAgent(state_size, action_size, device)
    starting_episode = 1300
    agent.qnetwork.load_params(f'dqn_model_episode_{starting_episode}.pth')
    agent.target_network.load_state_dict(agent.qnetwork.state_dict())
    for _ in range(starting_episode):
        agent.decay_epsilon()
    
    # Training parameters
    n_episodes = 1000
    max_steps = 1000
    
    # Initialize wandb
    wandb.init(
        project="lunar-lander-dqn",
        config={
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "epsilon_decay": agent.epsilon_decay,
            "buffer_size": agent.buffer_size,
            "batch_size": agent.batch_size,
            "target_update": agent.target_update
        }
    )
    
    # Training loop
    for episode in range(starting_episode, n_episodes + starting_episode):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train(agent.batch_size)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update target network periodically
        if episode % agent.target_update == 0:
            agent.update_target_network()
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Log metrics
        wandb.log({
            "episode": episode,
            "reward": total_reward,
            "epsilon": agent.epsilon,
            "steps": step
        })
        
        print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # Save model periodically
        if episode % 100 == 0:
            agent.qnetwork.save_params(f'dqn_model_episode_{episode}.pth')
    
    agent.qnetwork.save_params(f'dqn_model_episode_{episode}.pth')
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()