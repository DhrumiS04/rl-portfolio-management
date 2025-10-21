import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import torch
from datetime import datetime

# Add the parent directory to the path so we can import our project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.reinforce import REINFORCEAgent
from models.actor_critic import A2CAgent
from utils.data_utils import download_stock_data, calculate_returns, calculate_cvar, calculate_sharpe_ratio, split_data
from utils.env_utils import PortfolioEnv

# Set page configuration
st.set_page_config(
    page_title="RL Portfolio Management",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define the main title
st.title("Risk-Sensitive Portfolio Management with Reinforcement Learning")

# Sidebar for navigation and parameters
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a mode",
    ["Home", "Data Explorer", "Portfolio Backtesting", "Algorithm Comparison", "Stress Testing"]
)

# Add project information in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Project Info")
st.sidebar.markdown(
    "This application demonstrates reinforcement learning for portfolio optimization with risk-sensitivity."
)
st.sidebar.markdown("Source code: [GitHub Repository](https://github.com/DhrumiS04/rl-portfolio-management)")

# Define global parameters
@st.cache_data
def get_default_tickers():
    return ['SPY', 'QQQ', 'GLD', 'TLT', 'VNQ', 'BND', 'VWO']

@st.cache_data
def load_data(tickers, start_date, end_date):
    """
    Load and cache price data and return data
    """
    data_path = os.path.join('data', 'stock_data.csv')
    if os.path.exists(data_path):
        price_data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # Filter by date if needed
        price_data = price_data.loc[start_date:end_date]
    else:
        price_data = download_stock_data(tickers, start_date, end_date, save_path=data_path)
    
    # Calculate returns
    returns_data = calculate_returns(price_data, log_returns=False)
    
    return price_data, returns_data

# Define device for PyTorch models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Home page content
def home_page():
    st.write("""
    ## Welcome to the RL Portfolio Management Dashboard
    
    This dashboard provides an interactive interface to explore and evaluate reinforcement learning
    approaches for portfolio management with a focus on risk-sensitivity.
    
    ### Main Features:
    
    1. **Data Explorer**: Visualize asset prices, returns, and correlations
    2. **Portfolio Backtesting**: Backtest RL-based portfolio strategies
    3. **Algorithm Comparison**: Compare different RL algorithms
    4. **Stress Testing**: Test portfolio strategies during crisis periods
    
    ### How to Use:
    
    Use the navigation menu on the left to switch between different sections.
    Each section provides controls to customize parameters and visualize results.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### RL Algorithms Implemented
        - **REINFORCE with Baseline**: Policy gradient method with value function baseline
        - **Actor-Critic (A2C)**: Advantage Actor-Critic with shared network architecture
        """)
        
    with col2:
        st.info("""
        ### Risk-Sensitivity Features
        - **CVaR Penalty**: Conditional Value-at-Risk penalty in reward function
        - **Risk-Adjusted Returns**: Combining returns with risk metrics
        - **Stress Period Analysis**: Testing on historical crisis periods
        """)
    
    # Check if models are available
    reinforce_path = os.path.join('saved_models', 'reinforce_agent.pth')
    a2c_path = os.path.join('saved_models', 'a2c_agent.pth')
    
    reinforce_available = os.path.exists(reinforce_path)
    a2c_available = os.path.exists(a2c_path)
    
    model_status = f"REINFORCE Model: {'Available ✅' if reinforce_available else 'Not Available ❌'}\n"
    model_status += f"A2C Model: {'Available ✅' if a2c_available else 'Not Available ❌'}"
    
    st.sidebar.markdown("### Model Status")
    st.sidebar.code(model_status)

# Data Explorer page
def data_explorer():
    st.header("Data Explorer")
    
    # Parameters in sidebar
    st.sidebar.header("Data Parameters")
    default_tickers = get_default_tickers()
    selected_tickers = st.sidebar.multiselect(
        "Select Assets",
        default_tickers,
        default=default_tickers
    )
    
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=pd.to_datetime("2010-01-01")
    )
    
    end_date = st.sidebar.date_input(
        "End Date", 
        value=pd.to_datetime("2023-01-01")
    )
    
    # Load data
    if not selected_tickers:
        st.warning("Please select at least one asset to display data.")
        return
    
    price_data, returns_data = load_data(selected_tickers, start_date, end_date)
    
    # Tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Price Data", "Return Statistics", "Correlations"])
    
    with tab1:
        st.subheader("Asset Prices")
        fig = px.line(
            price_data, 
            title='Historical Prices (Adjusted Close)',
            labels={"value": "Price", "variable": "Asset"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Normalized prices
        st.subheader("Normalized Asset Prices")
        normalized_prices = price_data.div(price_data.iloc[0])
        fig_norm = px.line(
            normalized_prices,
            title='Normalized Prices (First Day = 1.0)',
            labels={"value": "Normalized Price", "variable": "Asset"}
        )
        st.plotly_chart(fig_norm, use_container_width=True)
    
    with tab2:
        st.subheader("Return Statistics")
        
        # Calculate stats
        stats = pd.DataFrame(index=selected_tickers)
        stats['Annual Return'] = [np.mean(returns_data[ticker]) * 252 for ticker in selected_tickers]
        stats['Annual Volatility'] = [np.std(returns_data[ticker]) * np.sqrt(252) for ticker in selected_tickers]
        stats['Sharpe Ratio'] = [calculate_sharpe_ratio(returns_data[ticker].values) for ticker in selected_tickers]
        stats['CVaR (5%)'] = [calculate_cvar(returns_data[ticker].values) for ticker in selected_tickers]
        stats['Max Drawdown'] = [calculate_max_drawdown(price_data[ticker].values) for ticker in selected_tickers]
        
        st.dataframe(stats.style.format('{:.4f}'), use_container_width=True)
        
        # Return distribution plots
        st.subheader("Return Distributions")
        fig = make_subplots(rows=1, cols=1)
        
        for ticker in selected_tickers:
            fig.add_trace(go.Histogram(
                x=returns_data[ticker],
                name=ticker,
                opacity=0.7,
                histnorm='probability density',
                nbinsx=50
            ))
        
        fig.update_layout(
            title_text="Return Distributions",
            xaxis_title="Daily Return",
            yaxis_title="Probability Density",
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Correlation Analysis")
        
        # Correlation matrix
        corr_matrix = returns_data[selected_tickers].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix of Asset Returns"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Rolling correlation
        st.subheader("Rolling Correlation")
        ticker1 = st.selectbox("Select first asset", selected_tickers, index=0)
        ticker2 = st.selectbox("Select second asset", selected_tickers, index=min(1, len(selected_tickers)-1))
        window = st.slider("Rolling window (days)", 20, 252, 60)
        
        rolling_corr = returns_data[ticker1].rolling(window).corr(returns_data[ticker2])
        fig = px.line(
            rolling_corr,
            title=f'{window}-day Rolling Correlation: {ticker1} vs {ticker2}',
            labels={"value": "Correlation", "index": "Date"}
        )
        fig.update_layout(yaxis_range=[-1, 1])
        st.plotly_chart(fig, use_container_width=True)

# Helper function to calculate drawdown
def calculate_max_drawdown(prices):
    peak = prices[0]
    max_drawdown = 0
    
    for value in prices:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown

# Portfolio Backtesting page
def portfolio_backtesting():
    st.header("Portfolio Backtesting")
    
    # Parameters in sidebar
    st.sidebar.header("Model Parameters")
    model_type = st.sidebar.selectbox(
        "Select Algorithm",
        ["REINFORCE", "A2C"],
    )
    
    window_size = st.sidebar.slider("Window Size", 5, 60, 20)
    transaction_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 2.0, 0.1) / 100
    risk_aversion = st.sidebar.slider("Risk Aversion", 0.0, 5.0, 1.0)
    
    # Load data
    default_tickers = get_default_tickers()
    price_data, returns_data = load_data(default_tickers, "2010-01-01", "2023-01-01")
    
    # Split data into train, validation, and test sets
    train_data, val_data, test_data = split_data(returns_data)
    
    # Check if models are available
    reinforce_path = os.path.join('saved_models', 'reinforce_agent.pth')
    a2c_path = os.path.join('saved_models', 'a2c_agent.pth')
    
    if (model_type == "REINFORCE" and not os.path.exists(reinforce_path)) or \
       (model_type == "A2C" and not os.path.exists(a2c_path)):
        st.warning(f"No trained {model_type} model found. Please train the model first using the main script.")
        return
    
    # Dataset selection
    dataset = st.radio(
        "Select Dataset",
        ["Training Set", "Validation Set", "Test Set"],
        horizontal=True
    )
    
    if dataset == "Training Set":
        data = train_data
    elif dataset == "Validation Set":
        data = val_data
    else:
        data = test_data
    
    # Create environment
    env = PortfolioEnv(
        returns=data,
        window_size=window_size,
        transaction_cost=transaction_cost,
        risk_aversion=risk_aversion,
        reward_mode='risk_adjusted'
    )
    
    # Create agent
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    
    if model_type == "REINFORCE":
        agent = REINFORCEAgent(input_dim=input_dim, output_dim=output_dim, device=device)
        agent.load(reinforce_path)
    else:
        agent = A2CAgent(input_dim=input_dim, output_dim=output_dim, device=device)
        agent.load(a2c_path)
    
    # Run backtest
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Running backtest...")
    
    obs, _ = env.reset()
    done = False
    truncated = False
    
    # Track portfolio values and weights
    portfolio_values = [env.portfolio_value]
    weights_history = []
    dates = []
    
    step = 0
    total_steps = len(data) - env.window_size - 1
    
    while not done and not truncated:
        action = agent.select_action(obs, evaluate=True)
        obs, _, done, truncated, info = env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        weights_history.append(info['weights'])
        dates.append(info['date'])
        
        step += 1
        progress = min(step / total_steps, 1.0)
        progress_bar.progress(progress)
    
    status_text.text("Backtest complete!")
    
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
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Portfolio Performance")
        
        # Portfolio value chart
        fig = px.line(
            x=dates, y=portfolio_values,
            title=f'{model_type} Portfolio Value',
            labels={"x": "Date", "y": "Portfolio Value"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add comparison to benchmark if desired
        benchmark_ticker = 'SPY'
        benchmark_prices = price_data[benchmark_ticker].loc[dates[0]:dates[-1]]
        benchmark_norm = benchmark_prices / benchmark_prices.iloc[0]
        portfolio_norm = portfolio_values / portfolio_values[0]
        
        compare_df = pd.DataFrame({
            'Date': dates,
            f'{model_type} Portfolio': portfolio_norm,
            benchmark_ticker: benchmark_norm.values[:len(dates)]
        }).set_index('Date')
        
        st.subheader("Performance vs Benchmark")
        fig = px.line(
            compare_df,
            title='Normalized Performance Comparison',
            labels={"value": "Normalized Value", "variable": "Asset"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Performance Metrics")
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Total Return", f"{total_return:.2%}")
            st.metric("Annualized Return", f"{annualized_return:.2%}")
            st.metric("Sharpe Ratio", f"{sharpe:.4f}")
            
        with metrics_col2:
            st.metric("Volatility (Ann.)", f"{volatility:.2%}")
            st.metric("CVaR (5%)", f"{cvar:.2%}")
            st.metric("Max Drawdown", f"{max_drawdown:.2%}")
        
        st.metric("Avg. Turnover", f"{avg_turnover:.4f}")
        
        st.subheader("Asset Allocation Over Time")
        
        # Asset allocation chart
        if len(weights_history) > 0:
            df_weights = pd.DataFrame(weights_history, columns=env.asset_names, index=dates)
            
            fig = go.Figure()
            
            # Create a stacked area chart
            for i, col in enumerate(df_weights.columns):
                fig.add_trace(go.Scatter(
                    x=df_weights.index, 
                    y=df_weights[col],
                    mode='lines',
                    name=col,
                    stackgroup='one',
                ))
            
            fig.update_layout(
                title="Asset Allocation Over Time",
                yaxis_title="Weight",
                xaxis_title="Date",
                legend=dict(x=1.0, y=1.0),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    # Show rolling metrics
    st.subheader("Rolling Performance Metrics")
    rolling_window = st.slider("Rolling Window (days)", 20, 252, 60)
    
    portfolio_returns_df = pd.DataFrame({'Returns': portfolio_returns}, index=dates[1:])
    
    rolling_returns = portfolio_returns_df['Returns'].rolling(rolling_window).mean() * 252  # Annualized
    rolling_vol = portfolio_returns_df['Returns'].rolling(rolling_window).std() * np.sqrt(252)
    rolling_sharpe = rolling_returns / rolling_vol
    
    # Calculate rolling CVaR
    rolling_cvar = portfolio_returns_df['Returns'].rolling(rolling_window).apply(
        lambda x: calculate_cvar(x.values), raw=True
    )
    
    rolling_df = pd.DataFrame({
        'Rolling Return (Ann.)': rolling_returns,
        'Rolling Volatility (Ann.)': rolling_vol,
        'Rolling Sharpe': rolling_sharpe,
        'Rolling CVaR (5%)': rolling_cvar
    })
    
    # Plot rolling metrics
    tab1, tab2 = st.tabs(["Risk/Return Metrics", "Rolling Asset Weights"])
    
    with tab1:
        fig = px.line(
            rolling_df,
            title=f'{rolling_window}-day Rolling Metrics',
            labels={"value": "Metric Value", "variable": "Metric"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if len(weights_history) > 0:
            df_weights = pd.DataFrame(weights_history, columns=env.asset_names, index=dates)
            
            # Create selectbox for assets
            selected_asset = st.selectbox("Select Asset", env.asset_names)
            
            rolling_weight = df_weights[selected_asset].rolling(rolling_window).mean()
            
            fig = px.line(
                rolling_weight,
                title=f'{rolling_window}-day Rolling Weight: {selected_asset}',
                labels={"value": "Weight", "index": "Date"}
            )
            st.plotly_chart(fig, use_container_width=True)

# Algorithm Comparison page
def algorithm_comparison():
    st.header("Algorithm Comparison")
    
    # Parameters in sidebar
    st.sidebar.header("Comparison Parameters")
    window_size = st.sidebar.slider("Window Size", 5, 60, 20, key="comp_window")
    transaction_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 2.0, 0.1, key="comp_tc") / 100
    risk_aversion = st.sidebar.slider("Risk Aversion", 0.0, 5.0, 1.0, key="comp_risk")
    
    # Load data
    default_tickers = get_default_tickers()
    price_data, returns_data = load_data(default_tickers, "2010-01-01", "2023-01-01")
    
    # Split data into train, validation, and test sets
    train_data, val_data, test_data = split_data(returns_data)
    
    # Dataset selection
    dataset = st.radio(
        "Select Dataset",
        ["Training Set", "Validation Set", "Test Set"],
        horizontal=True,
        key="comp_dataset"
    )
    
    if dataset == "Training Set":
        data = train_data
    elif dataset == "Validation Set":
        data = val_data
    else:
        data = test_data
    
    # Check if models are available
    reinforce_path = os.path.join('saved_models', 'reinforce_agent.pth')
    a2c_path = os.path.join('saved_models', 'a2c_agent.pth')
    
    if not os.path.exists(reinforce_path) or not os.path.exists(a2c_path):
        st.warning("Both REINFORCE and A2C models are needed for comparison. Please train the models first using the main script.")
        return
    
    # Set up environments
    reinforce_env = PortfolioEnv(
        returns=data,
        window_size=window_size,
        transaction_cost=transaction_cost,
        risk_aversion=risk_aversion,
        reward_mode='risk_adjusted'
    )
    
    a2c_env = PortfolioEnv(
        returns=data,
        window_size=window_size,
        transaction_cost=transaction_cost,
        risk_aversion=risk_aversion,
        reward_mode='risk_adjusted'
    )
    
    # Create agents
    input_dim = reinforce_env.observation_space.shape[0]
    output_dim = reinforce_env.action_space.shape[0]
    
    reinforce_agent = REINFORCEAgent(input_dim=input_dim, output_dim=output_dim, device=device)
    reinforce_agent.load(reinforce_path)
    
    a2c_agent = A2CAgent(input_dim=input_dim, output_dim=output_dim, device=device)
    a2c_agent.load(a2c_path)
    
    # Run backtests
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Running comparison backtests...")
    
    # Dictionary to store results
    results = {}
    
    for model_name, (env, agent) in zip(['REINFORCE', 'A2C'], 
                                        [(reinforce_env, reinforce_agent), 
                                         (a2c_env, a2c_agent)]):
        obs, _ = env.reset()
        done = False
        truncated = False
        
        portfolio_values = [env.portfolio_value]
        weights_history = []
        dates = []
        
        step = 0
        total_steps = len(data) - env.window_size - 1
        
        while not done and not truncated:
            action = agent.select_action(obs, evaluate=True)
            obs, _, done, truncated, info = env.step(action)
            
            portfolio_values.append(info['portfolio_value'])
            weights_history.append(info['weights'])
            if model_name == 'REINFORCE':  # Only save dates once
                dates.append(info['date'])
            
            step += 1
            progress = min((step + (0 if model_name == 'REINFORCE' else total_steps)) / (total_steps * 2), 1.0)
            progress_bar.progress(progress)
        
        # Calculate portfolio returns
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calculate metrics
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
        
        results[model_name] = {
            'portfolio_values': portfolio_values,
            'weights_history': weights_history,
            'portfolio_returns': portfolio_returns,
            'metrics': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe': sharpe,
                'cvar': cvar,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'avg_turnover': avg_turnover
            }
        }
    
    status_text.text("Comparison complete!")
    
    # Display comparison results
    st.subheader("Performance Comparison")
    
    # Portfolio value comparison
    portfolio_compare_df = pd.DataFrame({
        'Date': dates,
        'REINFORCE': results['REINFORCE']['portfolio_values'],
        'A2C': results['A2C']['portfolio_values'],
    }).set_index('Date')
    
    fig = px.line(
        portfolio_compare_df,
        title='Portfolio Value Comparison',
        labels={"value": "Portfolio Value", "variable": "Model"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Normalized comparison
    normalized_compare_df = pd.DataFrame({
        'Date': dates,
        'REINFORCE': np.array(results['REINFORCE']['portfolio_values']) / results['REINFORCE']['portfolio_values'][0],
        'A2C': np.array(results['A2C']['portfolio_values']) / results['A2C']['portfolio_values'][0],
    }).set_index('Date')
    
    fig = px.line(
        normalized_compare_df,
        title='Normalized Portfolio Value Comparison',
        labels={"value": "Normalized Value", "variable": "Model"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics comparison
    metrics_df = pd.DataFrame({
        'Metric': [
            'Total Return',
            'Annualized Return',
            'Sharpe Ratio',
            'CVaR (5%)',
            'Volatility (Ann.)',
            'Max Drawdown',
            'Avg. Turnover'
        ],
        'REINFORCE': [
            f"{results['REINFORCE']['metrics']['total_return']:.2%}",
            f"{results['REINFORCE']['metrics']['annualized_return']:.2%}",
            f"{results['REINFORCE']['metrics']['sharpe']:.4f}",
            f"{results['REINFORCE']['metrics']['cvar']:.2%}",
            f"{results['REINFORCE']['metrics']['volatility']:.2%}",
            f"{results['REINFORCE']['metrics']['max_drawdown']:.2%}",
            f"{results['REINFORCE']['metrics']['avg_turnover']:.4f}"
        ],
        'A2C': [
            f"{results['A2C']['metrics']['total_return']:.2%}",
            f"{results['A2C']['metrics']['annualized_return']:.2%}",
            f"{results['A2C']['metrics']['sharpe']:.4f}",
            f"{results['A2C']['metrics']['cvar']:.2%}",
            f"{results['A2C']['metrics']['volatility']:.2%}",
            f"{results['A2C']['metrics']['max_drawdown']:.2%}",
            f"{results['A2C']['metrics']['avg_turnover']:.4f}"
        ]
    }).set_index('Metric')
    
    st.table(metrics_df)
    
    # Asset allocation comparison
    st.subheader("Asset Allocation Comparison")
    
    comparison_tabs = st.tabs(["REINFORCE Allocation", "A2C Allocation", "Allocation Differences"])
    
    for i, (tab, model_name) in enumerate(zip(comparison_tabs[:2], ['REINFORCE', 'A2C'])):
        with tab:
            weights_df = pd.DataFrame(
                results[model_name]['weights_history'], 
                columns=reinforce_env.asset_names, 
                index=dates
            )
            
            fig = go.Figure()
            
            # Create a stacked area chart
            for col in weights_df.columns:
                fig.add_trace(go.Scatter(
                    x=weights_df.index, 
                    y=weights_df[col],
                    mode='lines',
                    name=col,
                    stackgroup='one',
                ))
            
            fig.update_layout(
                title=f"{model_name} Asset Allocation Over Time",
                yaxis_title="Weight",
                xaxis_title="Date",
                legend=dict(x=1.0, y=1.0),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Show allocation differences
    with comparison_tabs[2]:
        reinforce_weights = pd.DataFrame(
            results['REINFORCE']['weights_history'], 
            columns=reinforce_env.asset_names, 
            index=dates
        )
        
        a2c_weights = pd.DataFrame(
            results['A2C']['weights_history'], 
            columns=a2c_env.asset_names, 
            index=dates
        )
        
        diff_weights = a2c_weights - reinforce_weights
        
        fig = px.line(
            diff_weights,
            title='Weight Differences (A2C - REINFORCE)',
            labels={"value": "Weight Difference", "variable": "Asset"}
        )
        
        # Add a horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mean absolute weight differences
        mean_abs_diff = diff_weights.abs().mean()
        
        fig = px.bar(
            mean_abs_diff,
            title='Mean Absolute Weight Differences by Asset',
            labels={"value": "Mean Abs. Weight Difference", "index": "Asset"}
        )
        st.plotly_chart(fig, use_container_width=True)

# Stress Testing page
def stress_testing():
    st.header("Crisis Period Stress Testing")
    
    # Parameters in sidebar
    st.sidebar.header("Stress Test Parameters")
    model_type = st.sidebar.selectbox(
        "Select Algorithm",
        ["REINFORCE", "A2C"],
        key="stress_model"
    )
    
    window_size = st.sidebar.slider("Window Size", 5, 60, 20, key="stress_window")
    transaction_cost = st.sidebar.slider("Transaction Cost (%)", 0.0, 2.0, 0.1, key="stress_tc") / 100
    risk_aversion = st.sidebar.slider("Risk Aversion", 0.0, 5.0, 1.0, key="stress_risk")
    
    # Define crisis periods
    crisis_periods = {
        'Financial Crisis (2008)': {
            'start': '2008-01-01',
            'end': '2009-06-30',
            'description': 'The global financial crisis triggered by the collapse of Lehman Brothers'
        },
        'COVID-19 Crash (2020)': {
            'start': '2020-02-01',
            'end': '2020-05-31',
            'description': 'The rapid market decline due to the COVID-19 pandemic'
        },
    }
    
    # Allow user to select a crisis period
    crisis_names = list(crisis_periods.keys())
    selected_crisis = st.radio(
        "Select Crisis Period",
        crisis_names,
        horizontal=True
    )
    
    crisis_info = crisis_periods[selected_crisis]
    st.info(f"**{selected_crisis}**: {crisis_info['description']}\n\n"
            f"Period: {crisis_info['start']} to {crisis_info['end']}")
    
    # Check if models are available
    reinforce_path = os.path.join('saved_models', 'reinforce_agent.pth')
    a2c_path = os.path.join('saved_models', 'a2c_agent.pth')
    
    if (model_type == "REINFORCE" and not os.path.exists(reinforce_path)) or \
       (model_type == "A2C" and not os.path.exists(a2c_path)):
        st.warning(f"No trained {model_type} model found. Please train the model first using the main script.")
        return
    
    # Load data for the crisis period
    default_tickers = get_default_tickers()
    
    # Need to download data for a wider range to ensure we have enough for the window
    start_date = pd.to_datetime(crisis_info['start']) - pd.Timedelta(days=window_size*2)
    end_date = pd.to_datetime(crisis_info['end'])
    
    with st.spinner("Loading crisis period data..."):
        price_data, returns_data = load_data(default_tickers, start_date, end_date)
    
    # Filter to just the crisis period for display
    crisis_price_data = price_data.loc[crisis_info['start']:crisis_info['end']]
    crisis_returns_data = returns_data.loc[crisis_info['start']:crisis_info['end']]
    
    if len(crisis_returns_data) == 0:
        st.error(f"No data available for the selected crisis period: {crisis_info['start']} to {crisis_info['end']}")
        return
    
    # Show crisis period market overview
    st.subheader("Market Overview During Crisis")
    
    fig = px.line(
        crisis_price_data / crisis_price_data.iloc[0],
        title=f'Normalized Asset Prices During {selected_crisis}',
        labels={"value": "Normalized Price", "variable": "Asset"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate crisis period stats
    stats = pd.DataFrame(index=default_tickers)
    stats['Total Return'] = [(crisis_price_data[ticker].iloc[-1] / crisis_price_data[ticker].iloc[0]) - 1 
                            for ticker in default_tickers]
    stats['Volatility'] = [np.std(crisis_returns_data[ticker]) * np.sqrt(252) 
                          for ticker in default_tickers]
    stats['Maximum Drawdown'] = [calculate_max_drawdown(crisis_price_data[ticker].values) 
                                for ticker in default_tickers]
    
    st.dataframe(stats.style.format('{:.2%}'), use_container_width=True)
    
    # Create environment for the crisis period
    env = PortfolioEnv(
        returns=returns_data,  # Use full returns including pre-crisis window
        window_size=window_size,
        transaction_cost=transaction_cost,
        risk_aversion=risk_aversion,
        reward_mode='risk_adjusted'
    )
    
    # Load agent
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]
    
    if model_type == "REINFORCE":
        agent = REINFORCEAgent(input_dim=input_dim, output_dim=output_dim, device=device)
        agent.load(reinforce_path)
    else:
        agent = A2CAgent(input_dim=input_dim, output_dim=output_dim, device=device)
        agent.load(a2c_path)
    
    # Run backtest
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Running crisis period backtest...")
    
    obs, _ = env.reset()
    done = False
    truncated = False
    
    # Track portfolio values and weights
    portfolio_values = [env.portfolio_value]
    weights_history = []
    dates = []
    
    step = 0
    total_steps = len(returns_data) - env.window_size - 1
    
    while not done and not truncated:
        action = agent.select_action(obs, evaluate=True)
        obs, _, done, truncated, info = env.step(action)
        
        current_date = info['date']
        
        # Only include data within the crisis period
        if crisis_info['start'] <= current_date.strftime('%Y-%m-%d') <= crisis_info['end']:
            portfolio_values.append(info['portfolio_value'])
            weights_history.append(info['weights'])
            dates.append(current_date)
        
        step += 1
        progress = min(step / total_steps, 1.0)
        progress_bar.progress(progress)
    
    status_text.text("Crisis backtest complete!")
    
    # Check if we have any portfolio values within the crisis period
    if len(portfolio_values) <= 1:
        st.warning(f"No portfolio values within the crisis period: {crisis_info['start']} to {crisis_info['end']}")
        return
    
    # Convert to arrays and DataFrames
    portfolio_values = np.array(portfolio_values)
    weights_history = np.array(weights_history)
    
    # Calculate performance metrics
    portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
    annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1 if len(portfolio_returns) > 0 else 0
    sharpe = calculate_sharpe_ratio(portfolio_returns) if len(portfolio_returns) > 0 else 0
    cvar = calculate_cvar(portfolio_returns) if len(portfolio_returns) > 0 else 0
    volatility = np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 0 else 0
    max_drawdown = calculate_max_drawdown(portfolio_values)
    
    # Calculate turnover
    turnover = 0
    for i in range(1, len(weights_history)):
        turnover += np.sum(np.abs(weights_history[i] - weights_history[i-1])) / 2.0
    avg_turnover = turnover / (len(weights_history) - 1) if len(weights_history) > 1 else 0
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Crisis Period Portfolio Performance")
        
        # Portfolio value chart
        fig = px.line(
            x=dates, y=portfolio_values,
            title=f'{model_type} Portfolio During {selected_crisis}',
            labels={"x": "Date", "y": "Portfolio Value"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add comparison to benchmark
        benchmark_ticker = 'SPY'
        benchmark_prices = crisis_price_data[benchmark_ticker].loc[dates[0]:dates[-1]]
        benchmark_norm = benchmark_prices / benchmark_prices.iloc[0]
        portfolio_norm = portfolio_values / portfolio_values[0]
        
        compare_df = pd.DataFrame({
            'Date': dates,
            f'{model_type} Portfolio': portfolio_norm,
            benchmark_ticker: benchmark_norm.values[:len(dates)]
        }).set_index('Date')
        
        st.subheader("Crisis Performance vs Benchmark")
        fig = px.line(
            compare_df,
            title=f'Normalized Performance During {selected_crisis}',
            labels={"value": "Normalized Value", "variable": "Asset"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Crisis Performance Metrics")
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Total Return", f"{total_return:.2%}")
            st.metric("Annualized Return", f"{annualized_return:.2%}")
            st.metric("Sharpe Ratio", f"{sharpe:.4f}")
            
        with metrics_col2:
            st.metric("Volatility (Ann.)", f"{volatility:.2%}")
            st.metric("CVaR (5%)", f"{cvar:.2%}")
            st.metric("Max Drawdown", f"{max_drawdown:.2%}")
        
        st.metric("Avg. Turnover", f"{avg_turnover:.4f}")
        
        st.subheader("Asset Allocation During Crisis")
        
        # Asset allocation chart
        if len(weights_history) > 0:
            df_weights = pd.DataFrame(weights_history, columns=env.asset_names, index=dates)
            
            fig = go.Figure()
            
            # Create a stacked area chart
            for i, col in enumerate(df_weights.columns):
                fig.add_trace(go.Scatter(
                    x=df_weights.index, 
                    y=df_weights[col],
                    mode='lines',
                    name=col,
                    stackgroup='one',
                ))
            
            fig.update_layout(
                title=f"Asset Allocation During {selected_crisis}",
                yaxis_title="Weight",
                xaxis_title="Date",
                legend=dict(x=1.0, y=1.0),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Display drawdown analysis
    st.subheader("Drawdown Analysis")
    
    # Calculate drawdowns
    peak = portfolio_values[0]
    drawdowns = np.zeros_like(portfolio_values)
    
    for i, value in enumerate(portfolio_values):
        if value > peak:
            peak = value
        drawdowns[i] = (peak - value) / peak
    
    drawdown_df = pd.DataFrame({
        'Date': dates,
        'Drawdown': drawdowns
    }).set_index('Date')
    
    fig = px.area(
        drawdown_df,
        title=f'Portfolio Drawdown During {selected_crisis}',
        labels={"Drawdown": "Drawdown"},
        color_discrete_sequence=["red"]
    )
    
    fig.update_layout(
        yaxis=dict(
            tickformat=".0%",
            autorange="reversed"  # Invert y-axis for better visualization
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Main app logic
if app_mode == "Home":
    home_page()
elif app_mode == "Data Explorer":
    data_explorer()
elif app_mode == "Portfolio Backtesting":
    portfolio_backtesting()
elif app_mode == "Algorithm Comparison":
    algorithm_comparison()
elif app_mode == "Stress Testing":
    stress_testing()
