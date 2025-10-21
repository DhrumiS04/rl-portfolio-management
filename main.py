import os
import argparse
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Import custom modules
from utils.data_utils import (
    download_stock_data, calculate_returns, split_data,
    prepare_market_features, get_crisis_periods
)
from utils.env_utils import PortfolioEnv
from utils.training import train_reinforce, train_a2c
from utils.evaluation import (
    evaluate_portfolio, compare_agents, analyze_weights_transition,
    stress_test_performance
)
from models.reinforce import REINFORCEAgent
from models.actor_critic import A2CAgent


def parse_args():
    parser = argparse.ArgumentParser(description='Risk-Sensitive Portfolio Management with Policy Gradients')
    
    # Data parameters
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'GLD', 'TLT', 'VNQ', 'BND', 'VWO'], 
                        help='Ticker symbols to use')
    parser.add_argument('--start-date', type=str, default='2010-01-01', 
                        help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-01-01', 
                        help='End date for data (YYYY-MM-DD)')
    parser.add_argument('--window-size', type=int, default=20, 
                        help='Size of observation window')
    
    # Environment parameters
    parser.add_argument('--transaction-cost', type=float, default=0.001, 
                        help='Transaction cost as fraction of traded amount')
    parser.add_argument('--risk-aversion', type=float, default=1.0, 
                        help='Risk aversion parameter for CVaR penalty')
    parser.add_argument('--reward-mode', type=str, default='risk_adjusted', 
                        choices=['return', 'sharpe', 'risk_adjusted'], 
                        help='Reward calculation mode')
    
    # Training parameters
    parser.add_argument('--algorithm', type=str, default='reinforce', 
                        choices=['reinforce', 'a2c', 'both'], 
                        help='RL algorithm to use')
    parser.add_argument('--episodes', type=int, default=200, 
                        help='Number of episodes to train for')
    parser.add_argument('--eval-interval', type=int, default=20, 
                        help='Evaluation interval during training')
    parser.add_argument('--lr-policy', type=float, default=0.001, 
                        help='Learning rate for policy network')
    parser.add_argument('--lr-value', type=float, default=0.001, 
                        help='Learning rate for value network')
    parser.add_argument('--gamma', type=float, default=0.99, 
                        help='Discount factor')
    parser.add_argument('--entropy-coef', type=float, default=0.01, 
                        help='Entropy coefficient')
    parser.add_argument('--value-coef', type=float, default=0.5, 
                        help='Value loss coefficient (A2C only)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', 
                        choices=['cpu', 'cuda'], 
                        help='Device to use for PyTorch')
    parser.add_argument('--save-dir', type=str, default='saved_models', 
                        help='Directory to save models')
    parser.add_argument('--data-dir', type=str, default='data', 
                        help='Directory to save data')
    parser.add_argument('--load-data', action='store_true', 
                        help='Load data from saved file instead of downloading')
    parser.add_argument('--eval-only', action='store_true', 
                        help='Only run evaluation on saved models')
    parser.add_argument('--stress-test', action='store_true', 
                        help='Run stress test on crisis periods')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories if they don't exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Data file path
    data_file = os.path.join(args.data_dir, 'stock_data.csv')
    
    # Load or download data
    if args.load_data and os.path.exists(data_file):
        print(f"Loading data from {data_file}")
        price_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        print(f"Downloading data for {args.tickers}")
        price_data = download_stock_data(
            args.tickers, args.start_date, args.end_date, save_path=data_file
        )
    
    # Calculate returns
    returns_data = calculate_returns(price_data, log_returns=False)
    
    # Split data into train, validation, and test sets
    train_data, val_data, test_data = split_data(returns_data)
    
    print(f"Data shapes - Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")
    
    # Create environments
    train_env = PortfolioEnv(
        returns=train_data,
        window_size=args.window_size,
        transaction_cost=args.transaction_cost,
        risk_aversion=args.risk_aversion,
        reward_mode=args.reward_mode
    )
    
    val_env = PortfolioEnv(
        returns=val_data,
        window_size=args.window_size,
        transaction_cost=args.transaction_cost,
        risk_aversion=args.risk_aversion,
        reward_mode=args.reward_mode
    )
    
    test_env = PortfolioEnv(
        returns=test_data,
        window_size=args.window_size,
        transaction_cost=args.transaction_cost,
        risk_aversion=args.risk_aversion,
        reward_mode=args.reward_mode
    )
    
    # Get dimensions for neural networks
    input_dim = train_env.observation_space.shape[0]
    output_dim = train_env.action_space.shape[0]
    
    print(f"State dimension: {input_dim}, Action dimension: {output_dim}")
    
    # Define agent save paths
    reinforce_path = os.path.join(args.save_dir, 'reinforce_agent.pth')
    a2c_path = os.path.join(args.save_dir, 'a2c_agent.pth')
    
    agents = {}
    
    # Train or load REINFORCE agent
    if args.algorithm in ['reinforce', 'both']:
        if args.eval_only and os.path.exists(reinforce_path):
            print("Loading REINFORCE agent from saved model")
            reinforce_agent = REINFORCEAgent(
                input_dim=input_dim,
                output_dim=output_dim,
                lr_policy=args.lr_policy,
                lr_value=args.lr_value,
                gamma=args.gamma,
                device=device
            )
            reinforce_agent.load(reinforce_path)
        else:
            print("Training REINFORCE agent")
            reinforce_agent = REINFORCEAgent(
                input_dim=input_dim,
                output_dim=output_dim,
                lr_policy=args.lr_policy,
                lr_value=args.lr_value,
                gamma=args.gamma,
                device=device
            )
            reinforce_agent, _, _ = train_reinforce(
                env=train_env,
                agent=reinforce_agent,
                num_episodes=args.episodes,
                eval_interval=args.eval_interval,
                save_path=reinforce_path
            )
        
        agents['REINFORCE'] = reinforce_agent
    
    # Train or load A2C agent
    if args.algorithm in ['a2c', 'both']:
        if args.eval_only and os.path.exists(a2c_path):
            print("Loading A2C agent from saved model")
            a2c_agent = A2CAgent(
                input_dim=input_dim,
                output_dim=output_dim,
                lr=args.lr_policy,
                gamma=args.gamma,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                device=device
            )
            a2c_agent.load(a2c_path)
        else:
            print("Training A2C agent")
            a2c_agent = A2CAgent(
                input_dim=input_dim,
                output_dim=output_dim,
                lr=args.lr_policy,
                gamma=args.gamma,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                device=device
            )
            a2c_agent, _, _ = train_a2c(
                env=train_env,
                agent=a2c_agent,
                num_episodes=args.episodes,
                eval_interval=args.eval_interval,
                save_path=a2c_path
            )
        
        agents['A2C'] = a2c_agent
    
    # Evaluate agents on test data
    print("\nEvaluating agents on test data")
    envs = []
    agent_list = []
    agent_names = []
    
    for name, agent in agents.items():
        envs.append(test_env)
        agent_list.append(agent)
        agent_names.append(name)
    
    # If multiple agents, compare them
    if len(agents) > 1:
        metrics_df = compare_agents(envs, agent_list, agent_names, test_data, plot=True)
        print("\nComparison of agents:")
        print(metrics_df)
    else:
        # If only one agent, evaluate it
        name = list(agents.keys())[0]
        agent = list(agents.values())[0]
        print(f"\nEvaluating {name} agent:")
        metrics = evaluate_portfolio(test_env, agent, test_data, plot=True, title=f"{name} Performance on Test Set")
        
        # Analyze weight transitions
        analyze_weights_transition(test_env, agent)
    
    # Run stress test if requested
    if args.stress_test:
        print("\nRunning stress tests on crisis periods")
        crisis_periods = get_crisis_periods()
        
        for name, agent in agents.items():
            print(f"\nStress testing {name} agent")
            crisis_metrics = stress_test_performance(
                test_env, agent, crisis_periods, returns_data
            )


if __name__ == "__main__":
    main()
