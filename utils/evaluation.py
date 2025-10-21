import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .data_utils import calculate_sharpe_ratio, calculate_cvar

def evaluate_portfolio(env, agent, returns_df, plot=True, title=None):
    """
    Evaluate a trained agent on a portfolio environment.
    
    Args:
        env: The portfolio environment.
        agent: The trained agent.
        returns_df (pd.DataFrame): DataFrame with returns data.
        plot (bool): Whether to plot the performance.
        title (str): Title for the plot.
        
    Returns:
        dict: Performance metrics.
    """
    obs, _ = env.reset()
    done = False
    truncated = False
    
    # Track portfolio values and weights
    portfolio_values = [env.portfolio_value]
    weights_history = []
    dates = []
    
    while not done and not truncated:
        action = agent.select_action(obs, evaluate=True)
        obs, _, done, truncated, info = env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        weights_history.append(info['weights'])
        dates.append(info['date'])
    
    # Convert to arrays and DataFrames
    portfolio_values = np.array(portfolio_values)
    weights_history = np.array(weights_history)
    
    # Calculate performance metrics
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
    sharpe = calculate_sharpe_ratio(portfolio_returns)
    cvar = calculate_cvar(portfolio_returns)
    volatility = np.std(portfolio_returns) * np.sqrt(252)
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    # Calculate turnover
    turnover = 0
    for i in range(1, len(weights_history)):
        turnover += np.sum(np.abs(weights_history[i] - weights_history[i-1])) / 2.0
    avg_turnover = turnover / (len(weights_history) - 1) if len(weights_history) > 1 else 0
    
    # Create metrics dictionary
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'cvar': cvar,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'avg_turnover': avg_turnover
    }
    
    if plot:
        # Plot portfolio performance
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot portfolio value
        ax1.plot(dates, portfolio_values, label='Portfolio Value')
        
        if title:
            ax1.set_title(title)
        else:
            ax1.set_title('Portfolio Performance')
            
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot asset weights over time
        if len(weights_history) > 0:
            df_weights = pd.DataFrame(weights_history, columns=env.asset_names, index=dates)
            df_weights.plot(kind='area', stacked=True, ax=ax2)
            ax2.set_title('Asset Allocation Over Time')
            ax2.set_ylabel('Weight')
            ax2.set_xlabel('Date')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Print metrics
        print("\nPerformance Metrics:")
        print(f"Total Return: {total_return:.4f} ({annualized_return:.4f} annualized)")
        print(f"Sharpe Ratio: {sharpe:.4f}")
        print(f"CVaR (5%): {cvar:.4f}")
        print(f"Volatility (annualized): {volatility:.4f}")
        print(f"Maximum Drawdown: {max_drawdown:.4f}")
        print(f"Average Turnover: {avg_turnover:.4f}")
    
    return metrics

def calculate_max_drawdown(portfolio_values):
    """
    Calculate the maximum drawdown of a portfolio.
    
    Args:
        portfolio_values (np.array): Array of portfolio values.
        
    Returns:
        float: Maximum drawdown.
    """
    # Calculate the maximum drawdown
    peak = portfolio_values[0]
    max_drawdown = 0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown

def compare_agents(envs, agents, agent_names, returns_df, plot=True):
    """
    Compare multiple agents on the same environment.
    
    Args:
        envs (list): List of portfolio environments.
        agents (list): List of trained agents.
        agent_names (list): List of agent names.
        returns_df (pd.DataFrame): DataFrame with returns data.
        plot (bool): Whether to plot the comparison.
        
    Returns:
        pd.DataFrame: Performance metrics for each agent.
    """
    all_metrics = []
    portfolio_values_dict = {}
    
    for i, (env, agent, name) in enumerate(zip(envs, agents, agent_names)):
        metrics = evaluate_portfolio(env, agent, returns_df, plot=False)
        metrics['agent'] = name
        all_metrics.append(metrics)
        
        # Track portfolio values
        obs, _ = env.reset()
        done = False
        truncated = False
        portfolio_values = [env.portfolio_value]
        dates = []
        
        while not done and not truncated:
            action = agent.select_action(obs, evaluate=True)
            obs, _, done, truncated, info = env.step(action)
            portfolio_values.append(info['portfolio_value'])
            if i == 0:  # Only need to save dates once
                dates.append(info['date'])
                
        portfolio_values_dict[name] = portfolio_values
    
    # Create DataFrame with metrics
    metrics_df = pd.DataFrame(all_metrics)
    
    if plot:
        # Plot portfolio values
        plt.figure(figsize=(12, 6))
        
        for name, values in portfolio_values_dict.items():
            plt.plot(dates[:len(values)], values, label=name)
            
        plt.title('Portfolio Performance Comparison')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot metrics comparison
        metrics_for_plot = ['annualized_return', 'sharpe_ratio', 'cvar', 'volatility', 'max_drawdown', 'avg_turnover']
        metrics_df_for_plot = metrics_df.set_index('agent')[metrics_for_plot]
        
        plt.figure(figsize=(14, 7))
        sns.heatmap(metrics_df_for_plot, annot=True, cmap='coolwarm', fmt='.4f', cbar=True)
        plt.title('Performance Metrics Comparison')
        plt.tight_layout()
        plt.show()
    
    return metrics_df

def analyze_weights_transition(env, agent):
    """
    Analyze the transitions in portfolio weights.
    
    Args:
        env: The portfolio environment.
        agent: The trained agent.
        
    Returns:
        pd.DataFrame: DataFrame with weight transitions.
    """
    obs, _ = env.reset()
    done = False
    truncated = False
    
    weights_history = []
    dates = []
    
    while not done and not truncated:
        action = agent.select_action(obs, evaluate=True)
        obs, _, done, truncated, info = env.step(action)
        
        weights_history.append(info['weights'])
        dates.append(info['date'])
    
    # Create DataFrame with weights
    weights_df = pd.DataFrame(weights_history, columns=env.asset_names, index=dates)
    
    # Calculate weight changes
    weight_changes = weights_df.diff().abs().sum(axis=1).dropna()
    
    # Plot weight transitions
    plt.figure(figsize=(12, 6))
    plt.bar(weight_changes.index, weight_changes.values)
    plt.title('Portfolio Turnover Over Time')
    plt.xlabel('Date')
    plt.ylabel('Weight Change (Turnover)')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    # Calculate average weight for each asset
    avg_weights = weights_df.mean()
    
    # Plot average weights
    plt.figure(figsize=(12, 6))
    avg_weights.plot(kind='bar')
    plt.title('Average Portfolio Allocation')
    plt.xlabel('Asset')
    plt.ylabel('Average Weight')
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
    
    return weights_df

def stress_test_performance(env, agent, crisis_periods, returns_df):
    """
    Evaluate agent performance during crisis periods.
    
    Args:
        env: The portfolio environment.
        agent: The trained agent.
        crisis_periods (dict): Dictionary with crisis period date ranges.
        returns_df (pd.DataFrame): DataFrame with returns data.
        
    Returns:
        dict: Performance metrics during crisis periods.
    """
    crisis_metrics = {}
    
    for period_name, period_dates in crisis_periods.items():
        start_date = pd.Timestamp(period_dates['start'])
        end_date = pd.Timestamp(period_dates['end'])
        
        # Filter returns for the crisis period
        crisis_returns = returns_df.loc[start_date:end_date]
        
        if len(crisis_returns) == 0:
            print(f"No data available for period: {period_name}")
            continue
        
        # Create a new environment with crisis period data
        from .env_utils import PortfolioEnv
        crisis_env = PortfolioEnv(
            returns=crisis_returns,
            window_size=env.window_size,
            transaction_cost=env.transaction_cost,
            risk_aversion=env.risk_aversion,
            initial_amount=env.initial_amount,
            reward_mode=env.reward_mode
        )
        
        # Evaluate agent during the crisis period
        metrics = evaluate_portfolio(
            crisis_env, agent, crisis_returns, 
            plot=True, 
            title=f'Portfolio Performance During {period_name}'
        )
        
        crisis_metrics[period_name] = metrics
    
    return crisis_metrics
