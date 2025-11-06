#!/usr/bin/env python3
"""
Portfolio Optimization Example

This example demonstrates how to use QuantModel's portfolio optimization
and risk management features.
"""

import pandas as pd
import numpy as np
from quantmodel.utils.config import Config
from quantmodel.utils.logger import setup_logger
from quantmodel.portfolio.optimizer import PortfolioOptimizer
from quantmodel.portfolio.allocator import PositionAllocator
from quantmodel.risk.metrics import RiskMetrics
from quantmodel.risk.var import VaRCalculator


def generate_sample_returns(n_days=252, n_assets=5):
    """Generate sample return data for demonstration."""
    np.random.seed(42)

    mean_returns = np.random.uniform(0.0001, 0.001, n_assets)
    volatilities = np.random.uniform(0.01, 0.03, n_assets)

    returns_dict = {}
    for i in range(n_assets):
        ticker = f"STOCK{i+1}"
        returns = np.random.normal(
            mean_returns[i],
            volatilities[i],
            n_days
        )
        returns_dict[ticker] = returns

    return pd.DataFrame(returns_dict)


def main():
    config = Config.create_default()
    logger = setup_logger('portfolio_example', config)

    logger.info("="*60)
    logger.info("Portfolio Optimization Example")
    logger.info("="*60)

    returns = generate_sample_returns(n_days=252, n_assets=5)
    portfolio_value = 100000.0

    logger.info(f"\nGenerated returns for {len(returns.columns)} assets over {len(returns)} days")

    logger.info("\n" + "="*60)
    logger.info("1. PORTFOLIO OPTIMIZATION")
    logger.info("="*60)

    optimizer = PortfolioOptimizer(returns, risk_free_rate=0.02)

    logger.info("\n--- Maximum Sharpe Ratio Portfolio ---")
    max_sharpe = optimizer.max_sharpe_ratio(max_weight=0.30, min_weight=0.05)
    logger.info(f"Expected Return: {max_sharpe['expected_return']:.2%}")
    logger.info(f"Volatility: {max_sharpe['volatility']:.2%}")
    logger.info(f"Sharpe Ratio: {max_sharpe['sharpe_ratio']:.2f}")
    logger.info("\nOptimal Weights:")
    for asset, weight in max_sharpe['weights'].items():
        logger.info(f"  {asset}: {weight:.2%}")

    logger.info("\n--- Minimum Variance Portfolio ---")
    min_var = optimizer.min_variance(max_weight=0.30, min_weight=0.05)
    logger.info(f"Expected Return: {min_var['expected_return']:.2%}")
    logger.info(f"Volatility: {min_var['volatility']:.2%}")
    logger.info(f"Sharpe Ratio: {min_var['sharpe_ratio']:.2f}")

    logger.info("\n--- Risk Parity Portfolio ---")
    risk_parity = optimizer.risk_parity(max_weight=0.30, min_weight=0.05)
    logger.info(f"Expected Return: {risk_parity['expected_return']:.2%}")
    logger.info(f"Volatility: {risk_parity['volatility']:.2%}")
    logger.info(f"Sharpe Ratio: {risk_parity['sharpe_ratio']:.2f}")

    logger.info("\n--- Dollar Allocation (Max Sharpe) ---")
    allocation = optimizer.get_allocation(
        method='max_sharpe',
        total_value=portfolio_value,
        max_weight=0.30
    )
    logger.info("\n" + allocation.to_string(index=False))

    logger.info("\n" + "="*60)
    logger.info("2. POSITION SIZING")
    logger.info("="*60)

    allocator = PositionAllocator(
        portfolio_value=portfolio_value,
        max_position_size=0.20,
        min_position_size=0.05
    )

    logger.info("\n--- Fixed Fractional Method ---")
    for signal in [-2, -1, 0, 1, 2]:
        size = allocator.fixed_fractional(signal_strength=signal, base_size=0.10)
        logger.info(f"Signal {signal:+d}: ${size:,.2f}")

    logger.info("\n--- Kelly Criterion Method ---")
    kelly_size = allocator.kelly_criterion(
        win_rate=0.55,
        avg_win=0.02,
        avg_loss=0.01,
        signal_strength=2
    )
    shares = allocator.calculate_shares(kelly_size, price=150.00, signal_strength=2)
    logger.info(f"Win Rate: 55%")
    logger.info(f"Avg Win: 2%")
    logger.info(f"Avg Loss: 1%")
    logger.info(f"Position Size: ${kelly_size:,.2f}")
    logger.info(f"Shares at $150/share: {shares}")

    logger.info("\n--- Volatility-Based Sizing ---")
    vol_size = allocator.volatility_based(
        volatility=0.25,
        target_risk=0.02,
        signal_strength=2
    )
    logger.info(f"Asset Volatility: 25%")
    logger.info(f"Target Risk: 2%")
    logger.info(f"Position Size: ${vol_size:,.2f}")

    logger.info("\n" + "="*60)
    logger.info("3. RISK METRICS")
    logger.info("="*60)

    portfolio_returns = (returns * list(max_sharpe['weights'].values())).sum(axis=1)

    cumulative_value = portfolio_value * (1 + portfolio_returns).cumprod()

    risk_metrics = RiskMetrics(risk_free_rate=0.02)

    benchmark_returns = np.random.normal(0.0003, 0.012, len(returns))

    logger.info("\n--- Risk Metrics ---")
    sharpe = risk_metrics.sharpe_ratio(portfolio_returns)
    sortino = risk_metrics.sortino_ratio(portfolio_returns)
    max_dd, peak, trough = risk_metrics.max_drawdown(cumulative_value)
    calmar = risk_metrics.calmar_ratio(portfolio_returns, cumulative_value)
    omega = risk_metrics.omega_ratio(portfolio_returns)

    logger.info(f"Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"Sortino Ratio: {sortino:.2f}")
    logger.info(f"Calmar Ratio: {calmar:.2f}")
    logger.info(f"Omega Ratio: {omega:.2f}")
    logger.info(f"Maximum Drawdown: {max_dd:.2%}")

    beta = risk_metrics.beta(portfolio_returns, pd.Series(benchmark_returns))
    alpha = risk_metrics.alpha(portfolio_returns, pd.Series(benchmark_returns))
    logger.info(f"\nBeta: {beta:.2f}")
    logger.info(f"Alpha: {alpha:.2%}")

    logger.info("\n" + "="*60)
    logger.info("4. VALUE AT RISK (VaR)")
    logger.info("="*60)

    var_calc = VaRCalculator(confidence_level=0.95)

    logger.info("\n--- Historical VaR ---")
    hist_var, hist_cvar = var_calc.historical_var(
        portfolio_returns,
        portfolio_value
    )
    logger.info(f"95% VaR: ${hist_var:,.2f}")
    logger.info(f"95% CVaR (Expected Shortfall): ${hist_cvar:,.2f}")

    logger.info("\n--- Parametric VaR ---")
    param_var, param_cvar = var_calc.parametric_var(
        portfolio_returns,
        portfolio_value
    )
    logger.info(f"95% VaR: ${param_var:,.2f}")
    logger.info(f"95% CVaR: ${param_cvar:,.2f}")

    logger.info("\n--- Monte Carlo VaR ---")
    mc_var, mc_cvar = var_calc.monte_carlo_var(
        portfolio_returns,
        portfolio_value,
        n_simulations=10000
    )
    logger.info(f"95% VaR: ${mc_var:,.2f}")
    logger.info(f"95% CVaR: ${mc_cvar:,.2f}")

    logger.info("\n" + "="*60)
    logger.info("5. EFFICIENT FRONTIER")
    logger.info("="*60)

    logger.info("\nGenerating efficient frontier with 20 portfolios...")
    frontier = optimizer.efficient_frontier(n_portfolios=20, max_weight=0.30)

    if len(frontier) > 0:
        logger.info("\nSample Efficient Frontier Portfolios:")
        logger.info(f"{'Return':<10} {'Volatility':<12} {'Sharpe':<8}")
        logger.info("-" * 30)
        for _, row in frontier.head(5).iterrows():
            logger.info(
                f"{row['expected_return']:>8.2%}  "
                f"{row['volatility']:>10.2%}  "
                f"{row['sharpe_ratio']:>6.2f}"
            )

    logger.info("\n" + "="*60)
    logger.info("Example completed successfully!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
