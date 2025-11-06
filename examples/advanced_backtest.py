#!/usr/bin/env python3
"""
Advanced Backtest Example

Demonstrates a complete workflow:
1. Data fetching
2. Indicator pair analysis
3. Signal generation
4. Backtesting with the new features
5. Risk analysis
"""

from quantmodel.utils.config import Config
from quantmodel.utils.logger import setup_logger
from quantmodel.data.fetcher import DataFetcher
from quantmodel.analysis.pair_analyzer import PairAnalyzer
from quantmodel.signals.generator import SignalGenerator
from quantmodel.backtest.engine import BacktestEngine
from quantmodel.risk.metrics import RiskMetrics
from quantmodel.risk.var import VaRCalculator
import pandas as pd


def main():
    """Run advanced backtest example."""

    config = Config.create_default()
    logger = setup_logger('advanced_backtest', config)

    TICKER = 'AAPL'
    DAYS = 500
    TIMEFRAME = 'daily'
    INITIAL_CAPITAL = 100000

    logger.info("="*70)
    logger.info(f"ADVANCED BACKTEST EXAMPLE - {TICKER}")
    logger.info("="*70)

    logger.info("\n[1/5] Fetching market data...")
    fetcher = DataFetcher(config)
    data = fetcher.STRATTEST(TICKER, DAYS, TIMEFRAME)

    if not data:
        logger.error("Failed to fetch data. Exiting.")
        return

    price_changes = data['Other']['TotalChanges']
    prices = data['Other']['TotalPrices']['5. adjusted close']
    indicator_data = {k: v['rawData'] for k, v in data['Data'].items()}

    logger.info(f"  ✓ Fetched {len(prices)} price points")
    logger.info(f"  ✓ Loaded {len(indicator_data)} indicators")

    logger.info("\n[2/5] Analyzing indicator pairs...")
    analyzer = PairAnalyzer(config)
    pair_results = analyzer.compare_indicator_pairs(indicator_data, price_changes)
    logger.info(f"  ✓ Analyzed {len(pair_results)} indicator pairs")

    logger.info("\n[3/5] Generating trading signals...")
    signal_gen = SignalGenerator(config)

    df = pd.DataFrame({
        'Stock Price': prices,
        'Stock Pct Change': price_changes
    })

    for ind_name, ind_values in indicator_data.items():
        df[ind_name] = pd.Series(ind_values)

    signals = signal_gen.generate_signals(pair_results, df)
    logger.info(f"  ✓ Generated {len(signals)} signals")

    signal_counts = {}
    for sig_data in signals.values():
        strength = sig_data.get('strength', 0)
        signal_counts[strength] = signal_counts.get(strength, 0) + 1

    logger.info("\n  Signal Distribution:")
    for strength in sorted(signal_counts.keys()):
        count = signal_counts[strength]
        pct = (count / len(signals)) * 100
        label = {-2: 'Strong Sell', -1: 'Sell', 0: 'Neutral', 1: 'Buy', 2: 'Strong Buy'}
        logger.info(f"    {label.get(strength, strength):12s}: {count:4d} ({pct:5.1f}%)")

    logger.info("\n[4/5] Running backtest...")
    engine = BacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        ticker=TICKER,
        commission=0.001,
        slippage=0.0005
    )
    engine.add_data(df, signals)
    results = engine.run()

    metrics = engine.get_metrics()

    logger.info("\n[5/5] Calculating risk metrics...")
    returns = results['Portfolio Value'].pct_change().dropna()
    portfolio_values = results['Portfolio Value']

    risk_metrics = RiskMetrics(risk_free_rate=config.TRADING.risk_free_rate)
    var_calc = VaRCalculator(confidence_level=config.BACKTEST.var_confidence)

    hist_var, hist_cvar = var_calc.historical_var(returns, INITIAL_CAPITAL)

    logger.info("\n" + "="*70)
    logger.info("COMPREHENSIVE PERFORMANCE REPORT")
    logger.info("="*70)

    logger.info("\n📊 RETURNS")
    logger.info(f"  Total Return:       {metrics['total_return']:>10.2%}")
    logger.info(f"  Final Value:        ${metrics['final_value']:>10,.2f}")
    logger.info(f"  Initial Capital:    ${INITIAL_CAPITAL:>10,.2f}")

    logger.info("\n⚖️  RISK METRICS")
    logger.info(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>10.2f}")
    logger.info(f"  Sortino Ratio:      {metrics['sortino_ratio']:>10.2f}")
    logger.info(f"  Calmar Ratio:       {metrics['calmar_ratio']:>10.2f}")
    logger.info(f"  Max Drawdown:       {metrics['max_drawdown']:>10.2%}")

    logger.info("\n💼 TRADING ACTIVITY")
    logger.info(f"  Total Trades:       {metrics['total_trades']:>10,}")
    logger.info(f"  Win Rate:           {metrics['win_rate']:>10.2%}")
    logger.info(f"  Avg Trade Size:     {metrics['avg_trade_size']:>10.2f} shares")
    logger.info(f"  Total Costs:        ${metrics['total_costs']:>10,.2f}")

    logger.info("\n🎯 VALUE AT RISK")
    logger.info(f"  95% VaR:            ${hist_var:>10,.2f}")
    logger.info(f"  95% CVaR:           ${hist_cvar:>10,.2f}")
    logger.info(f"  VaR as % of Capital: {(hist_var/INITIAL_CAPITAL):>9.2%}")

    logger.info("\n" + "="*70)

    if metrics['sharpe_ratio'] > 1.0:
        logger.info("✓ Strategy shows promising risk-adjusted returns (Sharpe > 1.0)")
    elif metrics['sharpe_ratio'] > 0.5:
        logger.info("⚠ Strategy shows moderate performance (0.5 < Sharpe < 1.0)")
    else:
        logger.info("✗ Strategy may need improvement (Sharpe < 0.5)")

    if metrics['max_drawdown'] < -0.20:
        logger.warning(f"⚠ Large drawdown detected: {metrics['max_drawdown']:.2%}")

    logger.info("\n" + "="*70)
    logger.info("Backtest completed successfully!")
    logger.info(f"Results saved to: backtest_results/{TICKER}_backtest_*.xlsx")
    logger.info("="*70)


if __name__ == '__main__':
    main()
