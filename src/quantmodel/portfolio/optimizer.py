"""
Portfolio optimization using Modern Portfolio Theory.

This module implements various portfolio optimization strategies including:
- Mean-Variance Optimization (Markowitz)
- Minimum Variance
- Maximum Sharpe Ratio
- Risk Parity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from quantmodel.utils.logger import get_logger

logger = get_logger(__name__)


class PortfolioOptimizer:
    """
    Portfolio optimizer using various optimization methods.

    Supports multiple optimization objectives and constraints.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        risk_free_rate: float = 0.02
    ):
        """
        Initialize the portfolio optimizer.

        Args:
            returns: DataFrame of asset returns (columns are assets)
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns.columns)
        self.asset_names = list(returns.columns)

        self.mean_returns = returns.mean() * 252
        self.cov_matrix = returns.cov() * 252

    def max_sharpe_ratio(
        self,
        max_weight: float = 0.30,
        min_weight: float = 0.0
    ) -> Dict:
        """
        Find portfolio weights that maximize the Sharpe ratio.

        Args:
            max_weight: Maximum weight for any single asset
            min_weight: Minimum weight for any single asset

        Returns:
            Dictionary with optimal weights and metrics
        """
        def negative_sharpe(weights):
            portfolio_return = np.sum(self.mean_returns * weights)
            portfolio_std = np.sqrt(
                np.dot(weights.T, np.dot(self.cov_matrix, weights))
            )
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        initial_guess = np.array([1.0 / self.n_assets] * self.n_assets)

        result = minimize(
            negative_sharpe,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        return self._create_result_dict(optimal_weights)

    def min_variance(
        self,
        max_weight: float = 0.30,
        min_weight: float = 0.0
    ) -> Dict:
        """
        Find minimum variance portfolio.

        Args:
            max_weight: Maximum weight for any single asset
            min_weight: Minimum weight for any single asset

        Returns:
            Dictionary with optimal weights and metrics
        """
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        initial_guess = np.array([1.0 / self.n_assets] * self.n_assets)

        result = minimize(
            portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        return self._create_result_dict(optimal_weights)

    def risk_parity(
        self,
        max_weight: float = 0.30,
        min_weight: float = 0.05
    ) -> Dict:
        """
        Risk parity portfolio - equal risk contribution from each asset.

        Args:
            max_weight: Maximum weight for any single asset
            min_weight: Minimum weight for any single asset

        Returns:
            Dictionary with optimal weights and metrics
        """
        def risk_contribution_variance(weights):
            portfolio_var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
            marginal_contrib = np.dot(self.cov_matrix, weights)
            contrib = weights * marginal_contrib
            risk_target = portfolio_var / self.n_assets

            return np.sum((contrib - risk_target) ** 2)

        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        initial_guess = np.array([1.0 / self.n_assets] * self.n_assets)

        result = minimize(
            risk_contribution_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        return self._create_result_dict(optimal_weights)

    def efficient_frontier(
        self,
        n_portfolios: int = 50,
        max_weight: float = 0.30,
        min_weight: float = 0.0
    ) -> pd.DataFrame:
        """
        Generate efficient frontier portfolios.

        Args:
            n_portfolios: Number of portfolios to generate
            max_weight: Maximum weight for any single asset
            min_weight: Minimum weight for any single asset

        Returns:
            DataFrame with portfolio returns, risks, and Sharpe ratios
        """
        min_return = self.mean_returns.min()
        max_return = self.mean_returns.max()
        target_returns = np.linspace(min_return, max_return, n_portfolios)

        frontier_portfolios = []

        for target_return in target_returns:
            result = self._efficient_portfolio(
                target_return, max_weight, min_weight
            )
            if result is not None:
                frontier_portfolios.append(result)

        return pd.DataFrame(frontier_portfolios)

    def _efficient_portfolio(
        self,
        target_return: float,
        max_weight: float,
        min_weight: float
    ) -> Optional[Dict]:
        """Find portfolio for a given target return on the efficient frontier."""
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) - target_return}
        ]
        bounds = tuple((min_weight, max_weight) for _ in range(self.n_assets))
        initial_guess = np.array([1.0 / self.n_assets] * self.n_assets)

        result = minimize(
            portfolio_variance,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            return self._create_result_dict(result.x)
        return None

    def _create_result_dict(self, weights: np.ndarray) -> Dict:
        """Create a result dictionary from optimal weights."""
        portfolio_return = np.sum(self.mean_returns * weights)
        portfolio_std = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix, weights))
        )
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std

        weights_dict = {
            asset: float(weight)
            for asset, weight in zip(self.asset_names, weights)
            if weight > 0.001
        }

        return {
            'weights': weights_dict,
            'expected_return': float(portfolio_return),
            'volatility': float(portfolio_std),
            'sharpe_ratio': float(sharpe_ratio)
        }

    def get_allocation(
        self,
        method: str = 'max_sharpe',
        total_value: float = 100000.0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Get dollar allocation for each asset.

        Args:
            method: Optimization method ('max_sharpe', 'min_variance', 'risk_parity')
            total_value: Total portfolio value in dollars
            **kwargs: Additional arguments for the optimization method

        Returns:
            DataFrame with asset allocations
        """
        methods = {
            'max_sharpe': self.max_sharpe_ratio,
            'min_variance': self.min_variance,
            'risk_parity': self.risk_parity
        }

        if method not in methods:
            raise ValueError(f"Unknown method: {method}")

        result = methods[method](**kwargs)
        weights = result['weights']

        allocations = []
        for asset, weight in weights.items():
            allocations.append({
                'Asset': asset,
                'Weight': weight,
                'Dollar Amount': weight * total_value
            })

        df = pd.DataFrame(allocations)
        logger.info(f"\nPortfolio Allocation ({method}):")
        logger.info(f"Expected Return: {result['expected_return']:.2%}")
        logger.info(f"Volatility: {result['volatility']:.2%}")
        logger.info(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")

        return df
