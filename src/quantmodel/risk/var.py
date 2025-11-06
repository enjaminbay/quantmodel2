"""
Value at Risk (VaR) and Conditional Value at Risk (CVaR) calculations.

Supports multiple VaR calculation methods:
- Historical VaR
- Parametric VaR (Variance-Covariance)
- Monte Carlo VaR
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats
from quantmodel.utils.logger import get_logger

logger = get_logger(__name__)


class VaRCalculator:
    """
    Calculate Value at Risk and Conditional Value at Risk.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator.

        Args:
            confidence_level: Confidence level for VaR (default 95%)
        """
        self.confidence_level = confidence_level

    def historical_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1000000.0
    ) -> Tuple[float, float]:
        """
        Calculate VaR using historical simulation.

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value

        Returns:
            Tuple of (VaR, CVaR) in dollars
        """
        if len(returns) == 0:
            return 0.0, 0.0

        var_percentile = (1 - self.confidence_level) * 100
        var_return = np.percentile(returns, var_percentile)

        var_dollar = abs(var_return * portfolio_value)

        returns_below_var = returns[returns <= var_return]
        if len(returns_below_var) > 0:
            cvar_return = returns_below_var.mean()
            cvar_dollar = abs(cvar_return * portfolio_value)
        else:
            cvar_dollar = var_dollar

        logger.debug(f"Historical VaR ({self.confidence_level:.0%}): ${var_dollar:,.2f}")
        logger.debug(f"Historical CVaR: ${cvar_dollar:,.2f}")

        return var_dollar, cvar_dollar

    def parametric_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1000000.0
    ) -> Tuple[float, float]:
        """
        Calculate VaR using parametric method (assumes normal distribution).

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value

        Returns:
            Tuple of (VaR, CVaR) in dollars
        """
        if len(returns) == 0:
            return 0.0, 0.0

        mu = returns.mean()
        sigma = returns.std()

        z_score = stats.norm.ppf(1 - self.confidence_level)

        var_return = mu + (z_score * sigma)
        var_dollar = abs(var_return * portfolio_value)

        cvar_return = mu - sigma * (
            stats.norm.pdf(z_score) / (1 - self.confidence_level)
        )
        cvar_dollar = abs(cvar_return * portfolio_value)

        logger.debug(f"Parametric VaR ({self.confidence_level:.0%}): ${var_dollar:,.2f}")
        logger.debug(f"Parametric CVaR: ${cvar_dollar:,.2f}")

        return var_dollar, cvar_dollar

    def monte_carlo_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1000000.0,
        n_simulations: int = 10000,
        time_horizon: int = 1
    ) -> Tuple[float, float]:
        """
        Calculate VaR using Monte Carlo simulation.

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value
            n_simulations: Number of simulations to run
            time_horizon: Time horizon in days

        Returns:
            Tuple of (VaR, CVaR) in dollars
        """
        if len(returns) == 0:
            return 0.0, 0.0

        mu = returns.mean()
        sigma = returns.std()

        simulated_returns = np.random.normal(
            mu * time_horizon,
            sigma * np.sqrt(time_horizon),
            n_simulations
        )

        var_percentile = (1 - self.confidence_level) * 100
        var_return = np.percentile(simulated_returns, var_percentile)
        var_dollar = abs(var_return * portfolio_value)

        returns_below_var = simulated_returns[simulated_returns <= var_return]
        if len(returns_below_var) > 0:
            cvar_return = returns_below_var.mean()
            cvar_dollar = abs(cvar_return * portfolio_value)
        else:
            cvar_dollar = var_dollar

        logger.debug(f"Monte Carlo VaR ({self.confidence_level:.0%}): ${var_dollar:,.2f}")
        logger.debug(f"Monte Carlo CVaR: ${cvar_dollar:,.2f}")

        return var_dollar, cvar_dollar

    def calculate_var(
        self,
        returns: pd.Series,
        portfolio_value: float = 1000000.0,
        method: str = 'historical'
    ) -> Tuple[float, float]:
        """
        Calculate VaR using specified method.

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value
            method: Calculation method ('historical', 'parametric', 'monte_carlo')

        Returns:
            Tuple of (VaR, CVaR) in dollars
        """
        methods = {
            'historical': self.historical_var,
            'parametric': self.parametric_var,
            'monte_carlo': self.monte_carlo_var
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")

        return methods[method](returns, portfolio_value)

    def marginal_var(
        self,
        asset_returns: pd.DataFrame,
        portfolio_weights: np.ndarray,
        portfolio_value: float = 1000000.0
    ) -> pd.Series:
        """
        Calculate marginal VaR for each asset in the portfolio.

        Args:
            asset_returns: DataFrame of individual asset returns
            portfolio_weights: Array of portfolio weights
            portfolio_value: Current portfolio value

        Returns:
            Series of marginal VaR for each asset
        """
        if len(asset_returns) == 0:
            return pd.Series()

        portfolio_returns = (asset_returns * portfolio_weights).sum(axis=1)
        portfolio_var, _ = self.historical_var(portfolio_returns, portfolio_value)

        marginal_vars = {}
        epsilon = 0.01

        for i, asset in enumerate(asset_returns.columns):
            perturbed_weights = portfolio_weights.copy()
            perturbed_weights[i] += epsilon

            perturbed_weights = perturbed_weights / perturbed_weights.sum()

            perturbed_returns = (asset_returns * perturbed_weights).sum(axis=1)
            perturbed_var, _ = self.historical_var(perturbed_returns, portfolio_value)

            marginal_var = (perturbed_var - portfolio_var) / epsilon
            marginal_vars[asset] = marginal_var

        return pd.Series(marginal_vars)

    def component_var(
        self,
        asset_returns: pd.DataFrame,
        portfolio_weights: np.ndarray,
        portfolio_value: float = 1000000.0
    ) -> pd.Series:
        """
        Calculate component VaR for each asset.

        Args:
            asset_returns: DataFrame of individual asset returns
            portfolio_weights: Array of portfolio weights
            portfolio_value: Current portfolio value

        Returns:
            Series of component VaR for each asset
        """
        marginal_vars = self.marginal_var(
            asset_returns, portfolio_weights, portfolio_value
        )

        component_vars = marginal_vars * portfolio_weights * portfolio_value

        return component_vars
