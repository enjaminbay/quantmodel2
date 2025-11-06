"""
Unit tests for risk management module.
"""

import pytest
import numpy as np
import pandas as pd
from quantmodel.risk.metrics import RiskMetrics
from quantmodel.risk.var import VaRCalculator


@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.02, 252))


@pytest.fixture
def portfolio_values():
    """Generate sample portfolio values."""
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    values = 100000 * (1 + pd.Series(returns)).cumprod()
    return values


class TestRiskMetrics:
    """Test cases for RiskMetrics class."""

    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        metrics = RiskMetrics(risk_free_rate=0.02)
        sharpe = metrics.sharpe_ratio(sample_returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        metrics = RiskMetrics(risk_free_rate=0.02)
        sortino = metrics.sortino_ratio(sample_returns)
        assert isinstance(sortino, float)

    def test_max_drawdown(self, portfolio_values):
        """Test maximum drawdown calculation."""
        metrics = RiskMetrics()
        max_dd, peak, trough = metrics.max_drawdown(portfolio_values)
        assert max_dd <= 0
        assert isinstance(peak, (pd.Timestamp, type(None)))

    def test_beta_alpha(self, sample_returns):
        """Test beta and alpha calculations."""
        metrics = RiskMetrics(risk_free_rate=0.02)

        benchmark = pd.Series(np.random.normal(0.0008, 0.015, len(sample_returns)))

        beta = metrics.beta(sample_returns, benchmark)
        assert isinstance(beta, float)

        alpha = metrics.alpha(sample_returns, benchmark)
        assert isinstance(alpha, float)


class TestVaRCalculator:
    """Test cases for VaRCalculator class."""

    def test_historical_var(self, sample_returns):
        """Test historical VaR calculation."""
        var_calc = VaRCalculator(confidence_level=0.95)
        var, cvar = var_calc.historical_var(sample_returns, portfolio_value=100000)

        assert var >= 0
        assert cvar >= var

    def test_parametric_var(self, sample_returns):
        """Test parametric VaR calculation."""
        var_calc = VaRCalculator(confidence_level=0.95)
        var, cvar = var_calc.parametric_var(sample_returns, portfolio_value=100000)

        assert var >= 0
        assert cvar >= 0

    def test_monte_carlo_var(self, sample_returns):
        """Test Monte Carlo VaR calculation."""
        var_calc = VaRCalculator(confidence_level=0.95)
        var, cvar = var_calc.monte_carlo_var(
            sample_returns,
            portfolio_value=100000,
            n_simulations=1000
        )

        assert var >= 0
        assert cvar >= 0

    def test_different_methods(self, sample_returns):
        """Test that different VaR methods return reasonable results."""
        var_calc = VaRCalculator(confidence_level=0.95)

        hist_var, _ = var_calc.calculate_var(sample_returns, method='historical')
        param_var, _ = var_calc.calculate_var(sample_returns, method='parametric')
        mc_var, _ = var_calc.calculate_var(sample_returns, method='monte_carlo')

        assert hist_var > 0
        assert param_var > 0
        assert mc_var > 0
