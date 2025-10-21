import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Dirichlet
import numpy as np


class ActorCritic(nn.Module):
    """
    Actor-Critic network with shared features.
    Actor outputs parameters for Dirichlet distribution over portfolio weights.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # Ensure positive concentration parameters
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Actor output (add small constant to avoid zero concentration)
        alpha = self.actor(features) + 0.1
        
        # Critic output (value)
        value = self.critic(features)
        
        return alpha, value


class A2CAgent:
    """
    Advantage Actor-Critic (A2C) agent for portfolio optimization.
    """
    
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99, entropy_coef=0.01, value_coef=0.5, device='cpu'):
        """
        Initialize the A2C agent.
        
        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output (number of assets).
            lr (float): Learning rate.
            gamma (float): Discount factor.
            entropy_coef (float): Entropy regularization coefficient.
            value_coef (float): Value loss coefficient.
            device (str): Device to use for tensor operations.
        """
        self.device = device
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Initialize network and optimizer
        self.network = ActorCritic(input_dim, output_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # For storing episode history
        self.reset_episode()
    
    def reset_episode(self):
        """
        Reset the episode history.
        """
        self.values = []
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        self.actions = []
    
    def select_action(self, state, evaluate=False):
        """
        Select an action using the actor-critic network.
        
        Args:
            state (np.array): Current state.
            evaluate (bool): If True, use the mode of the distribution.
            
        Returns:
            np.array: Selected action (portfolio weights).
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get policy parameters and value
        alpha, value = self.network(state)
        alpha = alpha.squeeze()
        
        if evaluate:
            # During evaluation, use mean of Dirichlet distribution
            action = alpha / alpha.sum()
            return action.detach().cpu().numpy()
        
        # Sample from Dirichlet distribution
        m = Dirichlet(alpha)
        action = m.sample()
        
        # Store log probability and entropy for training
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        
        # Store for training
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)
        self.actions.append(action)
        
        return action.detach().cpu().numpy()
    
    def update(self, next_value=0):
        """
        Update the actor-critic network.
        
        Args:
            next_value (float): Value estimate for the final state.
            
        Returns:
            float: Total loss.
            float: Actor loss.
            float: Critic loss.
            float: Entropy loss.
        """
        rewards = self.rewards
        values = self.values + [torch.tensor([[next_value]], device=self.device)]
        
        # Calculate returns and advantages using Generalized Advantage Estimation (GAE)
        returns = []
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1].item() - values[i].item()
            gae = delta + self.gamma * 0.95 * gae  # 0.95 is the GAE lambda
            advantage = gae
            
            returns.append(gae + values[i].item())
            advantages.append(advantage)
        
        # Reverse the lists to match the original order
        returns = returns[::-1]
        advantages = advantages[::-1]
        
        # Convert to tensors
        returns = torch.tensor(returns, device=self.device).unsqueeze(1)
        advantages = torch.tensor(advantages, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate actor (policy) loss
        actor_loss = 0
        for log_prob, advantage in zip(self.log_probs, advantages):
            actor_loss += -log_prob * advantage
        actor_loss = actor_loss / len(self.log_probs)
        
        # Calculate critic (value) loss
        critic_loss = F.mse_loss(torch.cat(self.values), returns)
        
        # Calculate entropy loss (for exploration)
        entropy_loss = -torch.stack(self.entropies).mean()
        
        # Total loss
        total_loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)  # Gradient clipping
        self.optimizer.step()
        
        # Reset episode history
        self.reset_episode()
        
        return (
            total_loss.item(),
            actor_loss.item(),
            critic_loss.item(),
            entropy_loss.item()
        )
    
    def store_reward(self, reward):
        """
        Store a reward from the environment.
        
        Args:
            reward (float): Reward from the environment.
        """
        self.rewards.append(reward)
    
    def save(self, path):
        """
        Save the agent's model.
        
        Args:
            path (str): Path to save the model.
        """
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """
        Load the agent's model.
        
        Args:
            path (str): Path to load the model from.
        """
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
