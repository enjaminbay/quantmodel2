"""
Backtesting engine for strategy evaluation.

This module provides a robust backtesting framework with support for:
- Position sizing based on signal strength
- Transaction costs and slippage
- Performance metrics calculation
- Results export to Excel
"""

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from quantmodel.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies.

    Features:
    - Dynamic position sizing based on signals
    - Transaction cost modeling
    - Performance analytics
    - Trade tracking and reporting
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        ticker: str = 'NONE',
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize the backtest engine.

        Args:
            initial_capital: Starting capital in dollars
            ticker: Stock ticker symbol
            commission: Commission rate (default 0.1%)
            slippage: Slippage rate (default 0.05%)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.ticker = ticker
        self.commission = commission
        self.slippage = slippage

        self.data: Optional[pd.DataFrame] = None
        self.signals: Dict = {}
        self.results: Optional[pd.DataFrame] = None

    def add_data(self, df: pd.DataFrame, signals: Dict) -> None:
        """
        Add market data and trading signals.

        Args:
            df: DataFrame with price data and indicators
            signals: Dictionary of trading signals by date
        """
        self.data = df.copy()
        self.signals = signals
        logger.info(f"Loaded {len(df)} days of data with {len(signals)} signals")

    def run(self) -> pd.DataFrame:
        """
        Execute the backtest and return results.

        Returns:
            DataFrame containing backtest results
        """
        if self.data is None or not self.signals:
            raise ValueError("Must add data and signals before running backtest")

        logger.info(f"Running backtest for {self.ticker}")

        results = []
        current_cash = self.initial_capital
        previous_shares = 0.0

        position_map = self._get_position_map()

        for date in self.data.index:
            price = self.data.loc[date, 'Stock Price']

            current_position_value = previous_shares * price
            portfolio_value = current_position_value + current_cash

            signal = self.signals.get(date, {}).get('strength', 0)

            position_pct = position_map.get(signal, 0.75)

            target_position_value = portfolio_value * position_pct
            total_shares = target_position_value / price if price > 0 else 0
            delta_shares = total_shares - previous_shares

            trade_cost = self._calculate_trade_cost(delta_shares, price)
            position_value = total_shares * price
            cash_left = current_cash - (delta_shares * price) - trade_cost

            daily_return = (portfolio_value / self.initial_capital) - 1

            results.append({
                'Date': date,
                'Price': price,
                'Signal': signal,
                'Shares': total_shares,
                'Position Value': position_value,
                'Cash': cash_left,
                'Portfolio Value': position_value + cash_left,
                'Return': daily_return,
                'Trade Cost': trade_cost,
                'Shares Traded': delta_shares
            })

            previous_shares = total_shares
            current_cash = cash_left

        self.results = pd.DataFrame(results)
        self._save_results()
        self._log_summary()

        return self.results

    def _get_position_map(self) -> Dict[int, float]:
        """
        Get position sizing based on signal strength.

        Returns:
            Dictionary mapping signal strength to position percentage
        """
        return {
            -2: 0.50,   # Strong sell = 50% invested
            -1: 0.625,  # Sell = 62.5%
            0: 0.75,    # Neutral = 75%
            1: 0.875,   # Buy = 87.5%
            2: 1.0      # Strong buy = 100%
        }

    def _calculate_trade_cost(self, shares: float, price: float) -> float:
        """
        Calculate transaction costs including commission and slippage.

        Args:
            shares: Number of shares traded
            price: Price per share

        Returns:
            Total transaction cost
        """
        if shares == 0:
            return 0.0

        trade_value = abs(shares * price)
        commission_cost = trade_value * self.commission
        slippage_cost = trade_value * self.slippage

        return commission_cost + slippage_cost

    def _save_results(self):
        """Save backtest results to Excel file."""
        if self.results is None:
            return

        output_dir = Path('backtest_results')
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f"{self.ticker}_backtest_{timestamp}.xlsx"

        try:
            self.results.to_excel(output_path, index=False, engine='openpyxl')
            logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save Excel file: {e}")
            csv_path = output_path.with_suffix('.csv')
            self.results.to_csv(csv_path, index=False)
            logger.info(f"Results saved as CSV to: {csv_path}")

    def _log_summary(self):
        """Log backtest performance summary."""
        if self.results is None or len(self.results) == 0:
            return

        final_value = self.results['Portfolio Value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        max_value = self.results['Portfolio Value'].max()
        min_value = self.results['Portfolio Value'].min()

        # Calculate drawdown
        cumulative_max = self.results['Portfolio Value'].cummax()
        drawdown = (self.results['Portfolio Value'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # Calculate Sharpe ratio (annualized)
        returns = self.results['Portfolio Value'].pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0

        # Total costs
        total_costs = self.results['Trade Cost'].sum()

        logger.info("\n" + "="*60)
        logger.info("BACKTEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Ticker: {self.ticker}")
        logger.info(f"Period: {self.results['Date'].iloc[0]} to {self.results['Date'].iloc[-1]}")
        logger.info(f"Trading Days: {len(self.results)}")
        logger.info(f"\nInitial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"Final Value: ${final_value:,.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"\nMax Value: ${max_value:,.2f}")
        logger.info(f"Min Value: ${min_value:,.2f}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"\nSharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Total Transaction Costs: ${total_costs:,.2f}")
        logger.info(f"Cost as % of Returns: {(total_costs/abs(final_value-self.initial_capital)*100):.2f}%")
        logger.info("="*60)

    def get_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if self.results is None or len(self.results) == 0:
            return {}

        final_value = self.results['Portfolio Value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Calculate returns
        returns = self.results['Portfolio Value'].pct_change().dropna()

        # Drawdown
        cumulative_max = self.results['Portfolio Value'].cummax()
        drawdown = (self.results['Portfolio Value'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        # Sharpe and Sortino ratios
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0
        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * returns.mean() / downside_returns.std() if len(downside_returns) > 0 else 0

        # Win rate
        winning_days = (returns > 0).sum()
        total_trading_days = len(returns)
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0

        # Calmar ratio
        calmar_ratio = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return {
            'total_return': total_return,
            'final_value': final_value,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'calmar_ratio': calmar_ratio,
            'total_trades': (self.results['Shares Traded'] != 0).sum(),
            'avg_trade_size': self.results['Shares Traded'].abs().mean(),
            'total_costs': self.results['Trade Cost'].sum()
        }
