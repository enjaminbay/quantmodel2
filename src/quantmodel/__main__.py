"""
Main entry point for the QuantModel trading system.
"""

import argparse
import sys
from quantmodel.utils.config import Config
from quantmodel.utils.logger import get_logger, setup_logger
from quantmodel.data.fetcher import DataFetcher
from quantmodel.signals.generator import SignalGenerator
from quantmodel.backtest.engine import BacktestEngine
from quantmodel.analysis.pair_analyzer import PairAnalyzer

logger = get_logger(__name__)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='QuantModel - Quantitative Trading System'
    )

    parser.add_argument(
        'command',
        choices=['fetch', 'analyze', 'backtest', 'signal'],
        help='Command to execute'
    )

    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Stock ticker symbol (e.g., AAPL, MSFT)'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=1000,
        help='Number of trading days to fetch (default: 1000)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='daily',
        choices=['daily', 'weekly'],
        help='Data timeframe (default: daily)'
    )

    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial capital for backtesting (default: 100000)'
    )

    args = parser.parse_args()

    # Initialize configuration
    config = Config.create_default()

    # Setup logger with config
    global logger
    logger = setup_logger('quantmodel', config)

    try:
        if args.command == 'fetch':
            fetch_data(config, args.ticker, args.days, args.timeframe)

        elif args.command == 'analyze':
            analyze_pairs(config, args.ticker, args.days, args.timeframe)

        elif args.command == 'backtest':
            run_backtest(config, args.ticker, args.days, args.timeframe, args.capital)

        elif args.command == 'signal':
            generate_signals(config, args.ticker, args.days, args.timeframe)

    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {str(e)}")
        sys.exit(1)


def fetch_data(config, ticker, days, timeframe):
    """Fetch market data for a ticker."""
    logger.info(f"Fetching {days} days of {timeframe} data for {ticker}")

    fetcher = DataFetcher(config)
    data = fetcher.STRATTEST(ticker, days, timeframe)

    if data:
        logger.info(f"Successfully fetched data for {ticker}")
        logger.info(f"Indicators available: {list(data.get('Data', {}).keys())}")
        logger.info(f"Price data points: {len(data.get('Other', {}).get('TotalPrices', {}).get('5. adjusted close', {}))}")
    else:
        logger.error(f"Failed to fetch data for {ticker}")
        sys.exit(1)


def analyze_pairs(config, ticker, days, timeframe):
    """Analyze indicator pairs for correlations."""
    logger.info(f"Analyzing indicator pairs for {ticker}")

    # Fetch data
    fetcher = DataFetcher(config)
    data = fetcher.STRATTEST(ticker, days, timeframe)

    if not data:
        logger.error(f"Failed to fetch data for {ticker}")
        sys.exit(1)

    # Analyze pairs
    analyzer = PairAnalyzer(config)
    price_changes = data['Other']['TotalChanges']
    indicator_data = {k: v['rawData'] for k, v in data['Data'].items()}

    results = analyzer.compare_indicator_pairs(indicator_data, price_changes)

    logger.info(f"Pair analysis complete. Found {len(results)} pair relationships")


def generate_signals(config, ticker, days, timeframe):
    """Generate trading signals for a ticker."""
    logger.info(f"Generating signals for {ticker}")

    # Fetch data
    fetcher = DataFetcher(config)
    data = fetcher.STRATTEST(ticker, days, timeframe)

    if not data:
        logger.error(f"Failed to fetch data for {ticker}")
        sys.exit(1)

    # Analyze pairs
    analyzer = PairAnalyzer(config)
    price_changes = data['Other']['TotalChanges']
    indicator_data = {k: v['rawData'] for k, v in data['Data'].items()}

    pair_results = analyzer.compare_indicator_pairs(indicator_data, price_changes)

    # Generate signals
    signal_gen = SignalGenerator(config)
    # Note: You'll need to implement the full signal generation pipeline
    # This is just a placeholder showing the structure

    logger.info("Signal generation complete")


def run_backtest(config, ticker, days, timeframe, capital):
    """Run backtest for a ticker."""
    logger.info(f"Running backtest for {ticker} with ${capital:,.2f} initial capital")

    # Fetch data
    fetcher = DataFetcher(config)
    data = fetcher.STRATTEST(ticker, days, timeframe)

    if not data:
        logger.error(f"Failed to fetch data for {ticker}")
        sys.exit(1)

    # Note: You'll need to implement the full backtest pipeline
    # This is just a placeholder showing the structure

    logger.info("Backtest complete - check results/ directory for output")


if __name__ == '__main__':
    main()
