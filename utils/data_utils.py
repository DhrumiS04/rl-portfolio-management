import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os

def download_stock_data(tickers, start_date, end_date, save_path=None):
    """
    Download historical price data for a list of stocks.
    
    Args:
        tickers (list): List of stock ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        save_path (str, optional): Path to save the data. If None, data won't be saved.
        
    Returns:
        pd.DataFrame: DataFrame with the adjusted close prices of the stocks.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # If only one ticker, yfinance doesn't return a DataFrame with ticker columns
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data, columns=[tickers[0]])
    
    # Fill missing values using forward fill then backward fill
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path)
    
    return data

def calculate_returns(prices, log_returns=False):
    """
    Calculate daily returns from price data.
    
    Args:
        prices (pd.DataFrame): DataFrame with price data.
        log_returns (bool): If True, calculate log returns, otherwise simple returns.
        
    Returns:
        pd.DataFrame: DataFrame with returns.
    """
    if log_returns:
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices / prices.shift(1) - 1
    
    # Drop the first row which will have NaN values
    return returns.dropna()

def calculate_cvar(returns, alpha=0.05):
    """
    Calculate the Conditional Value-at-Risk (CVaR) for a series of returns.
    
    Args:
        returns (np.array): Array of returns.
        alpha (float): The significance level (e.g., 0.05 for 95% CVaR).
        
    Returns:
        float: The CVaR value.
    """
    # Sort returns in ascending order
    sorted_returns = np.sort(returns)
    
    # Determine the VaR threshold
    var_threshold_idx = int(np.ceil(alpha * len(sorted_returns))) - 1
    if var_threshold_idx < 0:
        var_threshold_idx = 0
    
    # Calculate CVaR as the mean of returns beyond VaR
    cvar = np.mean(sorted_returns[:var_threshold_idx+1])
    
    return cvar

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns (np.array): Array of returns.
        risk_free_rate (float): The risk-free rate.
        periods_per_year (int): Number of periods per year (e.g., 252 for daily returns).
        
    Returns:
        float: The Sharpe ratio.
    """
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    annualized_sharpe = sharpe * np.sqrt(periods_per_year)
    
    return annualized_sharpe

def calculate_portfolio_stats(weights, returns):
    """
    Calculate various portfolio statistics.
    
    Args:
        weights (np.array): Portfolio weights.
        returns (np.array): Asset returns matrix (time x assets).
        
    Returns:
        tuple: (portfolio_return, portfolio_volatility, sharpe_ratio, cvar)
    """
    # Calculate portfolio returns
    portfolio_returns = np.dot(returns, weights)
    
    # Calculate statistics
    mean_return = np.mean(portfolio_returns)
    std_return = np.std(portfolio_returns)
    sharpe = calculate_sharpe_ratio(portfolio_returns)
    cvar = calculate_cvar(portfolio_returns)
    
    return mean_return, std_return, sharpe, cvar

def calculate_turnover(old_weights, new_weights):
    """
    Calculate portfolio turnover.
    
    Args:
        old_weights (np.array): Previous portfolio weights.
        new_weights (np.array): New portfolio weights.
        
    Returns:
        float: Portfolio turnover.
    """
    return np.sum(np.abs(new_weights - old_weights)) / 2.0

def prepare_market_features(returns, lookback_window=10):
    """
    Prepare market features from returns data.
    
    Args:
        returns (pd.DataFrame): DataFrame with returns data.
        lookback_window (int): Number of previous time steps to include as features.
        
    Returns:
        np.array: Array with features (time, assets, lookback).
    """
    num_periods, num_assets = returns.shape
    features = np.zeros((num_periods - lookback_window, lookback_window, num_assets))
    
    for t in range(lookback_window, num_periods):
        features[t - lookback_window] = returns.iloc[t - lookback_window:t].values
        
    return features

def split_data(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Split data into train, validation, and test sets.
    
    Args:
        data (pd.DataFrame): DataFrame to split.
        train_ratio (float): Ratio of data to use for training.
        val_ratio (float): Ratio of data to use for validation.
        test_ratio (float): Ratio of data to use for testing.
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data.iloc[:train_end]
    val_data = data.iloc[train_end:val_end]
    test_data = data.iloc[val_end:]
    
    return train_data, val_data, test_data

def get_crisis_periods():
    """
    Return predefined crisis periods for testing.
    
    Returns:
        dict: Dictionary with crisis periods.
    """
    periods = {
        'financial_crisis_2008': {
            'start': '2008-01-01',
            'end': '2009-06-30'
        },
        'covid_crash_2020': {
            'start': '2020-02-01',
            'end': '2020-05-31'
        }
    }
    
    return periods
