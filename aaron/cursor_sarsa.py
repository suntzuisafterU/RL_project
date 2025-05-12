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
    """
    MODIFIED FROM DQN:
    - Saves the next action as well (hence SARSA)
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, next_action, done):
        self.buffer.append((state, action, reward, next_state, next_action, done))
    
    def sample(self, batch_size):
        states, actions, rewards, next_states, next_actions, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(next_actions), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class SARSAAgent:
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Q-Network
        self.qnetwork = QNetwork(state_size, action_size).to(device)
        
        # Hyperparameters
        self.gamma = 0.99  # discount factor
        self.learning_rate = 1e-3
        self.buffer_size = 100000
        self.batch_size = 64
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
        
        states, actions, rewards, next_states, next_actions, dones = self.memory.sample(batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_actions = torch.LongTensor(next_actions).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Get current Q values
        current_q_values = self.qnetwork(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values for the actual next action taken
        with torch.no_grad():
            # Why no_grad??
            qs = self.qnetwork(next_states)
            # qs.shape == (64, 4)
            acts = next_actions.unsqueeze(1)
            # next_actions.shape == (64,) ==> .unsqueeze(1) == (64,1) ie: inserts a dim at location 1
            # next_actions was sampled 
            next_q_values = qs.gather(1, acts)
            # qs is (64, 4); acts is (64, 1). So we're doing something here where the rows match up and
            # we're doing something that broadcasts the rows together.
            ## next_q_values.shape == (64, 1)
            ## So for each sample row, we have computed the next step Q value.
            ## This is indexing by acts. self.qnetwork outputs the Q-value estimates for each state.
            ## States are sent in batches, with each row being 1 instance from the batch (0 is batch dim)
            ## That is, second tensor is index tensor.
            # next_q_values = self.qnetwork(next_states).gather(1, next_actions.unsqueeze(1))
        
        # Compute target Q values using SARSA update
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def main():
    # Initialize environment
    env = gym.make("LunarLander-v3", render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SARSAAgent(state_size, action_size, device)
    
    # Training parameters
    n_episodes = 1000
    max_steps = 1000
    
    # Initialize wandb
    wandb.init(
        project="lunar-lander-sarsa",
        config={
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "epsilon_decay": agent.epsilon_decay,
            "buffer_size": agent.buffer_size,
            "batch_size": agent.batch_size
        }
    )
    
    # Training loop
    for episode in range(n_episodes):
        state, _ = env.reset()
        action = agent.act(state)
        total_reward = 0
        
        for step in range(max_steps):
            # Take action and observe next state and reward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # MODIFIED: SARSA takes second step look ahead to get the Q-value. 
            #           As opposed to taking the average over the possible actions
            #           which is what QL does.
            # Choose next action
            next_action = agent.act(next_state)
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, next_action, done)
            
            # Train agent
            loss = agent.train(agent.batch_size)
            
            state = next_state
            action = next_action
            total_reward += reward
            
            if done:
                break
        
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
            agent.qnetwork.save_params(f'sarsa_model_episode_{episode}.pth')
    
    agent.qnetwork.save_params(f'sarsa_model_episode_{episode}.pth')
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main() 