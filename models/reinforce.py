import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Dirichlet


class PolicyNetwork(nn.Module):
    """
    Policy network for the REINFORCE algorithm.
    Outputs parameters for a Dirichlet distribution over portfolio weights.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # Ensure positive concentration parameters for Dirichlet
        )
    
    def forward(self, x):
        # Add a small constant to avoid zero concentration parameters
        return self.network(x) + 0.1

class ValueNetwork(nn.Module):
    """
    Value network for baseline in REINFORCE algorithm.
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class REINFORCEAgent:
    """
    REINFORCE agent with baseline for portfolio optimization.
    """
    
    def __init__(self, input_dim, output_dim, lr_policy=0.001, lr_value=0.001, gamma=0.99, device='cpu'):
        """
        Initialize the REINFORCE agent.
        
        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output (number of assets).
            lr_policy (float): Learning rate for policy network.
            lr_value (float): Learning rate for value network.
            gamma (float): Discount factor.
            device (str): Device to use for tensor operations.
        """
        self.device = device
        self.gamma = gamma
        
        # Initialize networks
        self.policy_net = PolicyNetwork(input_dim, output_dim).to(device)
        self.value_net = ValueNetwork(input_dim).to(device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        
        # For storing episode history
        self.reset_episode()
    
    def reset_episode(self):
        """
        Reset the episode history.
        """
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
        self.actions = []
        
    def select_action(self, state, evaluate=False):
        """
        Select an action using the policy network.
        
        Args:
            state (np.array): Current state.
            evaluate (bool): If True, use the mode of the distribution.
            
        Returns:
            np.array: Selected action (portfolio weights).
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Dirichlet concentration parameters
        alpha = self.policy_net(state).squeeze()
        
        if evaluate:
            # During evaluation, use mean of Dirichlet distribution
            # E[Dir(alpha)] = alpha / sum(alpha)
            action = alpha / alpha.sum()
            return action.detach().cpu().numpy()
        
        # Sample from Dirichlet distribution
        m = Dirichlet(alpha)
        action = m.sample()
        
        # Store log probability and entropy for training
        log_prob = m.log_prob(action)
        entropy = m.entropy()
        
        # Store value estimate
        value = self.value_net(state)
        
        # Store for training
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.values.append(value)
        self.actions.append(action)
        
        return action.detach().cpu().numpy()
    
    def update(self, last_value=0):
        """
        Update the policy and value networks.
        
        Args:
            last_value (float): Value estimate for the final state.
            
        Returns:
            float: Policy loss.
            float: Value loss.
        """
        rewards = self.rewards
        values = self.values + [torch.tensor([[last_value]], device=self.device)]
        
        # Calculate returns and advantages
        returns = []
        advantages = []
        R = last_value
        
        for i in reversed(range(len(rewards))):
            R = rewards[i] + self.gamma * R
            advantage = R - values[i].item()
            
            returns.append(R)
            advantages.append(advantage)
        
        # Reverse the lists to match the original order
        returns = returns[::-1]
        advantages = advantages[::-1]
        
        # Convert to tensors
        returns = torch.tensor(returns, device=self.device).unsqueeze(1)
        advantages = torch.tensor(advantages, device=self.device)
        
        # Update policy network
        policy_loss = 0
        for log_prob, advantage, entropy in zip(self.log_probs, advantages, self.entropies):
            policy_loss += -log_prob * advantage - 0.01 * entropy  # Add entropy regularization
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)  # Gradient clipping
        self.policy_optimizer.step()
        
        # Update value network
        value_loss = F.mse_loss(torch.cat(self.values), returns)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)  # Gradient clipping
        self.value_optimizer.step()
        
        # Reset episode history
        self.reset_episode()
        
        return policy_loss.item(), value_loss.item()
    
    def store_reward(self, reward):
        """
        Store a reward from the environment.
        
        Args:
            reward (float): Reward from the environment.
        """
        self.rewards.append(reward)
    
    def save(self, path):
        """
        Save the agent's models.
        
        Args:
            path (str): Path to save the models.
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """
        Load the agent's models.
        
        Args:
            path (str): Path to load the models from.
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
