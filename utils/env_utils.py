import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

class PortfolioEnv(gym.Env):
    """
    A reinforcement learning environment for portfolio management.
    """
    
    def __init__(self, returns, features=None, window_size=10, transaction_cost=0.001, 
                 risk_aversion=1.0, initial_amount=1.0, reward_mode='risk_adjusted'):
        """
        Initialize the environment.
        
        Args:
            returns (pd.DataFrame): DataFrame with asset returns.
            features (pd.DataFrame, optional): DataFrame with additional features.
            window_size (int): Size of the observation window.
            transaction_cost (float): Transaction cost as a fraction of traded amount.
            risk_aversion (float): Risk aversion parameter for CVaR penalty.
            initial_amount (float): Initial portfolio value.
            reward_mode (str): Reward calculation mode ('return', 'sharpe', 'risk_adjusted').
        """
        super(PortfolioEnv, self).__init__()
        
        self.returns = returns.values
        self.dates = returns.index
        self.asset_names = returns.columns
        self.num_assets = returns.shape[1]
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.risk_aversion = risk_aversion
        self.initial_amount = initial_amount
        self.reward_mode = reward_mode
        
        # Set additional features if provided
        if features is not None:
            self.features = features.values
            assert features.shape[0] == returns.shape[0], "Features and returns must have same number of time steps"
            self.feature_dim = features.shape[1]
        else:
            self.features = None
            self.feature_dim = 0
        
        # Define action and observation spaces
        # Action space: portfolio weights (continuous) that sum to 1
        self.action_space = spaces.Box(low=0, high=1, shape=(self.num_assets,), dtype=np.float32)
        
        # Observation space: window of past returns and features
        obs_dim = self.window_size * self.num_assets
        if self.features is not None:
            obs_dim += self.feature_dim
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _get_observation(self):
        """
        Construct the observation from the current state.
        
        Returns:
            np.array: The observation.
        """
        # Get the returns window
        returns_window = self.returns[self.current_step - self.window_size:self.current_step]
        obs = returns_window.flatten()
        
        # Add features if available
        if self.features is not None:
            current_features = self.features[self.current_step]
            obs = np.concatenate([obs, current_features])
        
        return obs
    
    def _calculate_reward(self, portfolio_return, previous_weights, new_weights):
        """
        Calculate the reward based on portfolio return and risk.
        
        Args:
            portfolio_return (float): Portfolio return.
            previous_weights (np.array): Previous portfolio weights.
            new_weights (np.array): New portfolio weights.
            
        Returns:
            float: The reward.
        """
        # Calculate transaction cost
        turnover = np.sum(np.abs(new_weights - previous_weights))
        cost = self.transaction_cost * turnover
        
        # Apply transaction cost to return
        net_return = portfolio_return - cost
        
        if self.reward_mode == 'return':
            return net_return
        
        # For risk-adjusted rewards, we need a window of recent portfolio returns
        if self.current_step >= self.window_size + 20:  # Need enough history for meaningful risk calculation
            # Calculate portfolio returns over a window
            hist_portfolio_returns = np.zeros(20)
            for i in range(20):
                step = self.current_step - 20 + i
                hist_portfolio_returns[i] = np.dot(self.returns[step], new_weights)
            
            # Calculate CVaR for the window
            sorted_returns = np.sort(hist_portfolio_returns)
            cvar_threshold_idx = int(0.05 * len(sorted_returns))
            cvar = np.mean(sorted_returns[:cvar_threshold_idx+1]) if cvar_threshold_idx >= 0 else sorted_returns[0]
            
            if self.reward_mode == 'risk_adjusted':
                # Risk-adjusted return (return - λ * CVaR)
                return net_return - self.risk_aversion * abs(cvar)
            elif self.reward_mode == 'sharpe':
                # Approximate Sharpe ratio over the window
                mean_return = np.mean(hist_portfolio_returns)
                std_return = np.std(hist_portfolio_returns)
                return mean_return / (std_return + 1e-6)  # Avoid division by zero
        
        # Default to net return if not enough history
        return net_return
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (np.array): Portfolio weights.
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Ensure action is valid (weights sum to 1)
        action = np.clip(action, 0, 1)
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        
        # Store previous weights and portfolio value
        prev_weights = self.weights.copy()
        prev_portfolio_value = self.portfolio_value
        
        # Move to the next time step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.returns) - 1
        truncated = False
        
        # Update weights
        self.weights = action
        
        # Calculate portfolio return
        if not terminated:
            step_returns = self.returns[self.current_step]
            portfolio_return = np.dot(step_returns, self.weights)
            self.portfolio_value = prev_portfolio_value * (1 + portfolio_return)
            self.portfolio_returns.append(portfolio_return)
            
            # Calculate reward
            reward = self._calculate_reward(portfolio_return, prev_weights, self.weights)
            
            # Get new observation
            observation = self._get_observation()
            
            # Store info for logging
            info = {
                'portfolio_value': self.portfolio_value,
                'portfolio_return': portfolio_return,
                'weights': self.weights,
                'date': self.dates[self.current_step] if isinstance(self.dates, np.ndarray) else self.dates.iloc[self.current_step]
            }
        else:
            # If done, return the final observation and zero reward
            observation = self._get_observation()
            reward = 0
            info = {
                'portfolio_value': self.portfolio_value,
                'portfolio_return': 0,
                'weights': self.weights,
                'date': self.dates[self.current_step] if isinstance(self.dates, np.ndarray) else self.dates.iloc[self.current_step]
            }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None):
        """
        Reset the environment to the initial state.
        
        Returns:
            np.array: Initial observation.
        """
        super().reset(seed=seed)
        
        # Initialize state variables
        self.current_step = self.window_size
        self.portfolio_value = self.initial_amount
        self.portfolio_returns = []
        
        # Initialize with equal weights
        self.weights = np.ones(self.num_assets) / self.num_assets
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {}
        return observation, info
    
    def render(self):
        """
        Render the environment.
        """
        # This environment doesn't need rendering
        pass
    
    def get_portfolio_history(self):
        """
        Get the portfolio value history.
        
        Returns:
            tuple: (dates, portfolio_values)
        """
        portfolio_values = [self.initial_amount]
        for r in self.portfolio_returns:
            portfolio_values.append(portfolio_values[-1] * (1 + r))
        
        dates = self.dates[self.window_size:self.window_size + len(portfolio_values)]
        
        return dates, portfolio_values
