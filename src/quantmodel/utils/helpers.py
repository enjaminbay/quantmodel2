"""Utility functions for the trading system."""
from typing import Optional, Union, Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from quantmodel.utils.logger import get_logger
from quantmodel.utils.exceptions import DataError
from typing import Dict

logger = get_logger(__name__)


def get_current_price(ticker: str) -> Union[float, bool]:
    """
    Get current stock price using Yahoo Finance.

    Args:
        ticker: Stock symbol

    Returns:
        float or bool: Current price if successful, False if failed
    """
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.info['regularMarketPrice']
        return float(current_price)
    except Exception as e:
        logger.error(f"Error fetching current price for {ticker}: {str(e)}")
        return False


def get_stock_beta(ticker: str) -> Optional[float]:
    """
    Get stock beta using Yahoo Finance.

    Args:
        ticker: Stock symbol

    Returns:
        float or None: Stock beta if available
    """
    try:
        stock = yf.Ticker(ticker)
        beta = stock.info['beta']
        return float(beta)
    except Exception as e:
        logger.error(f"Error fetching beta for {ticker}: {str(e)}")
        return None


def format_date(date_str: str) -> str:
    """
    Format date string to consistent format.

    Args:
        date_str: Input date string

    Returns:
        str: Formatted date string (YYYY-MM-DD)
    """
    try:
        return pd.to_datetime(date_str).strftime('%Y-%m-%d')
    except Exception as e:
        logger.error(f"Error formatting date {date_str}: {str(e)}")
        raise DataError(f"Error formatting date: {str(e)}")


def calculate_returns(
        prices: Union[pd.Series, np.ndarray, List[float]],
        method: str = 'simple'
) -> np.ndarray:
    """
    Calculate returns from price series.

    Args:
        prices: Price series
        method: Return calculation method ('simple' or 'log')

    Returns:
        np.ndarray: Calculated returns
    """
    try:
        prices = np.array(prices)
        if method == 'simple':
            returns = (prices[1:] - prices[:-1]) / prices[:-1]
        elif method == 'log':
            returns = np.log(prices[1:] / prices[:-1])
        else:
            raise ValueError(f"Unknown return calculation method: {method}")

        return returns
    except Exception as e:
        logger.error(f"Error calculating returns: {str(e)}")
        raise DataError(f"Error calculating returns: {str(e)}")


def calculate_metrics(returns: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Array of returns

    Returns:
        dict: Dictionary of calculated metrics
    """
    try:
        metrics = {
            'total_return': float((1 + returns).prod() - 1),
            'annualized_return': float(((1 + returns).prod() ** (252 / len(returns))) - 1),
            'volatility': float(returns.std() * np.sqrt(252)),
            'sharpe_ratio': float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() != 0 else 0,
            'max_drawdown': float(_calculate_max_drawdown(returns)),
            'win_rate': float((returns > 0).sum() / len(returns)),
            'avg_win': float(returns[returns > 0].mean()) if len(returns[returns > 0]) > 0 else 0,
            'avg_loss': float(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0,
        }

        # Add advanced metrics
        metrics.update(_calculate_advanced_metrics(returns))

        return metrics
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise DataError(f"Error calculating metrics: {str(e)}")


def _calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from returns."""
    cum_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / running_max - 1
    return float(drawdowns.min())


def _calculate_advanced_metrics(returns: np.ndarray) -> Dict[str, float]:
    """Calculate additional advanced metrics."""
    try:
        # Calculate daily values
        daily_cum_returns = (1 + returns).cumprod()

        # Sortino Ratio (using 0 as minimum acceptable return)
        downside_returns = returns[returns < 0]
        sortino_ratio = (returns.mean() * 252) / (np.std(downside_returns) * np.sqrt(252)) if len(
            downside_returns) > 0 else 0

        # Calmar Ratio
        max_dd = _calculate_max_drawdown(returns)
        calmar_ratio = (returns.mean() * 252) / abs(max_dd) if max_dd != 0 else 0

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)

        # Expected Shortfall
        es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

        return {
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'var_95': float(var_95),
            'expected_shortfall_95': float(es_95),
            'kurtosis': float(pd.Series(returns).kurtosis()),
            'skewness': float(pd.Series(returns).skew())
        }
    except Exception as e:
        logger.error(f"Error calculating advanced metrics: {str(e)}")
        return {}


def validate_ticker_data(
        data: Dict,
        required_fields: List[str],
        min_periods: int = 100
) -> bool:
    """
    Validate ticker data meets requirements.

    Args:
        data: Ticker data dictionary
        required_fields: List of required fields
        min_periods: Minimum number of periods required

    Returns:
        bool: True if validation passes
    """
    try:
        # Check basic structure
        if not isinstance(data, dict):
            logger.error("Data is not a dictionary")
            return False

        # Check required fields
        missing_fields = set(required_fields) - set(data.keys())
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return False

        # Check data length
        first_field = next(iter(data.values()))
        if len(first_field) < min_periods:
            logger.error(f"Insufficient data periods: {len(first_field)} < {min_periods}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating ticker data: {str(e)}")
        return False


def time_series_split(
        data: Union[pd.DataFrame, np.ndarray],
        train_size: float = 0.8,
        min_samples: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split time series data preserving temporal order.

    Args:
        data: Input data
        train_size: Proportion of data to use for training
        min_samples: Minimum samples required

    Returns:
        tuple: (train_data, test_data)
    """
    try:
        if isinstance(data, pd.DataFrame):
            data = data.values

        if len(data) < min_samples:
            raise ValueError(f"Insufficient samples: {len(data)} < {min_samples}")

        split_idx = int(len(data) * train_size)
        return data[:split_idx], data[split_idx:]

    except Exception as e:
        logger.error(f"Error splitting time series: {str(e)}")
        raise DataError(f"Error splitting time series: {str(e)}")




