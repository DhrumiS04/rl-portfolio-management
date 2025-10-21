# Risk-Sensitive Portfolio Management with Reinforcement Learning

This repository contains an implementation of Policy Gradient and Actor-Critic methods for risk-sensitive portfolio management. The project applies reinforcement learning algorithms to optimize portfolio allocation while considering risk measures like Conditional Value-at-Risk (CVaR).

## Project Overview

The project explores how reinforcement learning can be used for portfolio optimization with a focus on managing downside risk. It implements:

- REINFORCE algorithm with baseline for portfolio weight optimization
- Actor-Critic (A2C) method for improved sample efficiency
- Risk-adjusted reward functions incorporating CVaR
- Stress testing during crisis periods (2008 Financial Crisis, 2020 COVID crash)

## Project Structure

```
.
├── data/                  # Directory for storing market data
├── models/                # Model implementations
│   ├── reinforce.py       # REINFORCE with baseline implementation
│   └── actor_critic.py    # Actor-Critic (A2C) implementation
├── utils/                 # Utility functions
│   ├── data_utils.py      # Data handling and financial metrics
│   ├── env_utils.py       # Portfolio environment implementation
│   ├── evaluation.py      # Performance evaluation tools
│   └── training.py        # Training procedures for RL agents
├── frontend/              # Streamlit web application
│   ├── app.py             # Main Streamlit application
│   └── run_app.sh         # Script to launch the app
├── saved_models/          # Directory for storing trained models
├── main.py                # Main script to run experiments
└── portfolio_management.ipynb # Jupyter notebook demonstrating the project
```

## Installation and Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd rl-portfolio-management
```

2. Install the required packages:

```bash
pip install numpy pandas matplotlib torch gymnasium scipy yfinance seaborn tqdm streamlit plotly
```

## Data

The project uses historical price data for multiple assets downloaded from Yahoo Finance. Default assets include:

- SPY (S&P 500 ETF)
- QQQ (NASDAQ ETF)
- GLD (Gold ETF)
- TLT (Long-term Treasury ETF)
- VNQ (Real Estate ETF)
- BND (Bond ETF)
- VWO (Emerging Markets ETF)

## Usage

### Training models

To train the models, run:

```bash
python main.py --algorithm both --episodes 200 --risk-aversion 1.0 --reward-mode risk_adjusted
```

Available options:
- `--algorithm`: Choose from 'reinforce', 'a2c', or 'both'
- `--episodes`: Number of episodes to train
- `--risk-aversion`: Risk aversion parameter for CVaR penalty
- `--reward-mode`: Choose from 'return', 'sharpe', or 'risk_adjusted'

### Evaluating models

To evaluate trained models on test data:

```bash
python main.py --algorithm both --eval-only
```

To run stress testing on crisis periods:

```bash
python main.py --algorithm both --eval-only --stress-test
```

### Web Interface

The project includes a Streamlit web interface for interactive visualization and experimentation:

```bash
# Run the Streamlit app
streamlit run frontend/app.py

# Or use the provided shell script
./frontend/run_app.sh
```

The web interface provides:
1. **Data Explorer**: Visualize historical asset price data and correlations
2. **Portfolio Backtesting**: Backtest portfolios with different allocation strategies
3. **Algorithm Comparison**: Compare performance of different RL algorithms
4. **Stress Testing**: Analyze portfolio performance during crisis periods
```

## Key Components

### Portfolio Environment

The environment represents a portfolio allocation problem with:
- State space: Historical returns over a window period
- Action space: Portfolio weights (continuous)
- Reward function: Risk-adjusted returns (return - λ * CVaR)

### RL Algorithms

1. **REINFORCE with Baseline**:
   - Policy network outputs parameters for a Dirichlet distribution over portfolio weights
   - Value network estimates expected returns for variance reduction

2. **Actor-Critic (A2C)**:
   - Shared feature extraction for actor and critic
   - Actor outputs Dirichlet concentration parameters
   - Critic estimates state value function

### Risk Adjustment

The risk adjustment is done by penalizing the Conditional Value-at-Risk (CVaR) in the reward function:

```
Reward = Portfolio Return - λ * CVaR
```

where λ is the risk aversion parameter.

## Results and Performance Metrics

The performance of the trained agents is evaluated using:

1. Return metrics:
   - Total return
   - Annualized return

2. Risk metrics:
   - Sharpe ratio
   - CVaR (5%)
   - Volatility
   - Maximum drawdown

3. Portfolio characteristics:
   - Average turnover
   - Asset allocation over time

## Acknowledgments

This project draws inspiration from academic work on reinforcement learning for portfolio management, particularly:

- Jiang, Z., Xu, D., & Liang, J. (2017). A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem.
- Moody, J., & Saffell, M. (2001). Learning to Trade via Direct Reinforcement.

## License

[MIT License](LICENSE)
