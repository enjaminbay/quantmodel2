"""
Risk metrics calculation module.

Provides comprehensive risk analytics including:
- Beta and Alpha
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Maximum Drawdown
- Sharpe, Sortino, and Calmar ratios
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from quantmodel.utils.logger import get_logger

logger = get_logger(__name__)


class RiskMetrics:
    """
    Calculate comprehensive risk metrics for portfolios and strategies.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate

    def beta(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate portfolio beta relative to benchmark.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns

        Returns:
            Beta coefficient
        """
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        covariance = np.cov(returns, benchmark_returns)[0][1]
        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def alpha(
        self,
        returns: pd.Series,
        benchmark_returns: pd.Series,
        beta: Optional[float] = None
    ) -> float:
        """
        Calculate Jensen's alpha.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            beta: Pre-calculated beta (optional)

        Returns:
            Alpha value
        """
        if beta is None:
            beta = self.beta(returns, benchmark_returns)

        portfolio_return = returns.mean() * 252
        benchmark_return = benchmark_returns.mean() * 252

        alpha = portfolio_return - (
            self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate)
        )

        return alpha

    def sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Series of returns
            periods_per_year: Number of periods in a year (252 for daily)

        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()

        return sharpe

    def sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized Sortino ratio.

        Args:
            returns: Series of returns
            periods_per_year: Number of periods in a year

        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0

        sortino = (
            np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
        )

        return sortino

    def max_drawdown(self, portfolio_values: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """
        Calculate maximum drawdown and its dates.

        Args:
            portfolio_values: Series of portfolio values

        Returns:
            Tuple of (max_drawdown, peak_date, trough_date)
        """
        if len(portfolio_values) == 0:
            return 0.0, None, None

        cumulative_max = portfolio_values.cummax()
        drawdown = (portfolio_values - cumulative_max) / cumulative_max

        max_dd = drawdown.min()
        trough_date = drawdown.idxmin()

        peak_date = portfolio_values[:trough_date].idxmax()

        return max_dd, peak_date, trough_date

    def calmar_ratio(
        self,
        returns: pd.Series,
        portfolio_values: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (return / max drawdown).

        Args:
            returns: Series of returns
            portfolio_values: Series of portfolio values
            periods_per_year: Number of periods in a year

        Returns:
            Calmar ratio
        """
        annualized_return = returns.mean() * periods_per_year
        max_dd, _, _ = self.max_drawdown(portfolio_values)

        if max_dd == 0:
            return 0.0

        return annualized_return / abs(max_dd)

    def omega_ratio(
        self,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega ratio.

        Args:
            returns: Series of returns
            threshold: Minimum acceptable return

        Returns:
            Omega ratio
        """
        if len(returns) == 0:
            return 0.0

        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())

        if losses == 0:
            return np.inf if gains > 0 else 0.0

        return gains / losses

    def tail_ratio(self, returns: pd.Series, percentile: float = 95) -> float:
        """
        Calculate tail ratio (right tail / left tail).

        Args:
            returns: Series of returns
            percentile: Percentile for tail definition

        Returns:
            Tail ratio
        """
        if len(returns) == 0:
            return 0.0

        right_tail = np.percentile(returns, percentile)
        left_tail = abs(np.percentile(returns, 100 - percentile))

        if left_tail == 0:
            return 0.0

        return right_tail / left_tail

    def calculate_all_metrics(
        self,
        returns: pd.Series,
        portfolio_values: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate all risk metrics.

        Args:
            returns: Portfolio returns
            portfolio_values: Portfolio values over time
            benchmark_returns: Benchmark returns (optional)

        Returns:
            Dictionary of all risk metrics
        """
        metrics = {
            'sharpe_ratio': self.sharpe_ratio(returns),
            'sortino_ratio': self.sortino_ratio(returns),
            'calmar_ratio': self.calmar_ratio(returns, portfolio_values),
            'omega_ratio': self.omega_ratio(returns),
            'tail_ratio': self.tail_ratio(returns),
        }

        max_dd, peak_date, trough_date = self.max_drawdown(portfolio_values)
        metrics['max_drawdown'] = max_dd
        metrics['max_drawdown_peak'] = peak_date
        metrics['max_drawdown_trough'] = trough_date

        if benchmark_returns is not None and len(benchmark_returns) > 0:
            beta_value = self.beta(returns, benchmark_returns)
            metrics['beta'] = beta_value
            metrics['alpha'] = self.alpha(returns, benchmark_returns, beta_value)

        metrics['volatility'] = returns.std() * np.sqrt(252)
        metrics['downside_deviation'] = returns[returns < 0].std() * np.sqrt(252)

        return metrics

    def rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling risk metrics.

        Args:
            returns: Series of returns
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: self.sharpe_ratio(x) if len(x) == window else np.nan
        )

        rolling_vol = returns.rolling(window).std() * np.sqrt(252)

        rolling_df = pd.DataFrame({
            'Rolling Sharpe': rolling_sharpe,
            'Rolling Volatility': rolling_vol
        })

        return rolling_df
