#!/usr/bin/env python3
"""
Basic backtest example using QuantModel.

This example demonstrates:
1. Fetching historical data
2. Analyzing indicator pairs
3. Generating signals
4. Running a backtest
"""

from quantmodel.utils.config import Config
from quantmodel.utils.logger import setup_logger
from quantmodel.data.fetcher import DataFetcher
from quantmodel.analysis.pair_analyzer import PairAnalyzer
from quantmodel.signals.generator import SignalGenerator
from quantmodel.backtest.engine import BacktestEngine
import pandas as pd


def main():
    # Initialize configuration
    config = Config.create_default()
    logger = setup_logger('backtest_example', config)

    # Configuration
    TICKER = 'AAPL'
    DAYS = 500
    TIMEFRAME = 'daily'
    INITIAL_CAPITAL = 100000

    logger.info(f"Starting backtest for {TICKER}")

    # Step 1: Fetch data
    logger.info("Fetching market data...")
    fetcher = DataFetcher(config)
    data = fetcher.STRATTEST(TICKER, DAYS, TIMEFRAME)

    if not data:
        logger.error("Failed to fetch data")
        return

    # Step 2: Prepare data for analysis
    price_changes = data['Other']['TotalChanges']
    prices = data['Other']['TotalPrices']['5. adjusted close']
    indicator_data = {k: v['rawData'] for k, v in data['Data'].items()}

    logger.info(f"Fetched {len(prices)} price points and {len(indicator_data)} indicators")

    # Step 3: Analyze indicator pairs
    logger.info("Analyzing indicator pairs...")
    analyzer = PairAnalyzer(config)
    pair_results = analyzer.compare_indicator_pairs(indicator_data, price_changes)

    logger.info(f"Found {len(pair_results)} pair relationships")

    # Step 4: Generate signals
    logger.info("Generating trading signals...")
    signal_gen = SignalGenerator(config)

    # Create historical DataFrame for signal generation
    df = pd.DataFrame({
        'Stock Price': prices,
        'Stock Pct Change': price_changes
    })

    # Add indicators to dataframe
    for ind_name, ind_values in indicator_data.items():
        df[ind_name] = pd.Series(ind_values)

    # Generate signals for all dates
    signals = signal_gen.generate_signals(pair_results, df)

    logger.info(f"Generated {len(signals)} signals")

    # Step 5: Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(initial_capital=INITIAL_CAPITAL, ticker=TICKER)
    engine.add_data(df, signals)
    results = engine.run()

    # Display results
    final_value = results['Portfolio Value'].iloc[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    logger.info("\n" + "="*50)
    logger.info("BACKTEST RESULTS")
    logger.info("="*50)
    logger.info(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    logger.info(f"Final Value: ${final_value:,.2f}")
    logger.info(f"Total Return: {total_return:.2%}")
    logger.info(f"Max Value: ${results['Portfolio Value'].max():,.2f}")
    logger.info(f"Min Value: ${results['Portfolio Value'].min():,.2f}")
    logger.info("="*50)


if __name__ == '__main__':
    main()
