"""
Unit tests for portfolio optimization module.
"""

import pytest
import numpy as np
import pandas as pd
from quantmodel.portfolio.optimizer import PortfolioOptimizer
from quantmodel.portfolio.allocator import PositionAllocator


@pytest.fixture
def sample_returns():
    """Generate sample return data for testing."""
    np.random.seed(42)
    n_days = 252
    n_assets = 3

    returns_dict = {}
    for i in range(n_assets):
        returns_dict[f'ASSET{i+1}'] = np.random.normal(0.0005, 0.015, n_days)

    return pd.DataFrame(returns_dict)


class TestPortfolioOptimizer:
    """Test cases for PortfolioOptimizer class."""

    def test_initialization(self, sample_returns):
        """Test optimizer initialization."""
        optimizer = PortfolioOptimizer(sample_returns, risk_free_rate=0.02)
        assert optimizer.n_assets == 3
        assert len(optimizer.asset_names) == 3
        assert optimizer.risk_free_rate == 0.02

    def test_max_sharpe_ratio(self, sample_returns):
        """Test maximum Sharpe ratio optimization."""
        optimizer = PortfolioOptimizer(sample_returns)
        result = optimizer.max_sharpe_ratio(max_weight=0.50, min_weight=0.10)

        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        assert 'sharpe_ratio' in result

        total_weight = sum(result['weights'].values())
        assert abs(total_weight - 1.0) < 0.01

    def test_min_variance(self, sample_returns):
        """Test minimum variance optimization."""
        optimizer = PortfolioOptimizer(sample_returns)
        result = optimizer.min_variance(max_weight=0.50, min_weight=0.10)

        assert 'weights' in result
        assert result['volatility'] >= 0

        total_weight = sum(result['weights'].values())
        assert abs(total_weight - 1.0) < 0.01

    def test_risk_parity(self, sample_returns):
        """Test risk parity optimization."""
        optimizer = PortfolioOptimizer(sample_returns)
        result = optimizer.risk_parity(max_weight=0.50, min_weight=0.10)

        assert 'weights' in result
        total_weight = sum(result['weights'].values())
        assert abs(total_weight - 1.0) < 0.01


class TestPositionAllocator:
    """Test cases for PositionAllocator class."""

    def test_initialization(self):
        """Test allocator initialization."""
        allocator = PositionAllocator(
            portfolio_value=100000,
            max_position_size=0.20
        )
        assert allocator.portfolio_value == 100000
        assert allocator.max_position_size == 0.20

    def test_fixed_fractional(self):
        """Test fixed fractional position sizing."""
        allocator = PositionAllocator(portfolio_value=100000)

        size = allocator.fixed_fractional(signal_strength=2)
        assert size > 0
        assert size <= allocator.portfolio_value * allocator.max_position_size

    def test_kelly_criterion(self):
        """Test Kelly criterion position sizing."""
        allocator = PositionAllocator(portfolio_value=100000)

        size = allocator.kelly_criterion(
            win_rate=0.60,
            avg_win=0.02,
            avg_loss=0.01,
            signal_strength=2
        )
        assert size >= 0
        assert size <= allocator.portfolio_value * allocator.max_position_size

    def test_calculate_shares(self):
        """Test share calculation."""
        allocator = PositionAllocator(portfolio_value=100000)

        shares = allocator.calculate_shares(
            position_size=10000,
            price=100,
            signal_strength=1
        )
        assert shares == 100

        short_shares = allocator.calculate_shares(
            position_size=10000,
            price=100,
            signal_strength=-1
        )
        assert short_shares == -100
