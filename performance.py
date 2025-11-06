from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
from utils.logger import get_logger
import os

logger = get_logger(__name__)


class PerformanceTracker:
    def __init__(self):
        self.daily_returns: List[float] = []
        self.portfolio_values: List[float] = []
        self.position_snapshots: List[Dict] = []
        self.trades: List[Dict] = []
        self.metrics: Dict = {}

    def add_daily_data(self, portfolio_value: float, positions: Dict[str, float],
                       cash: float, date: datetime,
                       stock_price: float, signal: Dict, cumulative_return: float) -> None:
        """Record daily portfolio state"""
        self.portfolio_values.append(portfolio_value)

        # Calculate daily return
        if len(self.portfolio_values) > 1:
            daily_return = (portfolio_value / self.portfolio_values[-2]) - 1
            self.daily_returns.append(daily_return)

        # Simplify positions to only number of shares held
        simplified_positions = {ticker: position.size for ticker, position in positions.items()}

        # Record position snapshot
        snapshot = {
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': cash,
            'positions': simplified_positions,
            'stock_price': stock_price,
            'signal': signal,
            'cumulative_return': cumulative_return
        }
        self.position_snapshots.append(snapshot)

    def add_trade(self, trade: Dict) -> None:
        """Record a completed trade"""
        self.trades.append(trade)

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        try:
            # Check if we have any data
            if not self.portfolio_values:
                logger.warning("No portfolio values recorded - returning default metrics")
                return {
                    'total_return': 0.0,
                    'cagr': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': 0.0,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0
                }

            returns = np.array(self.daily_returns)
            values = np.array(self.portfolio_values)

            logger.debug(f"Calculating metrics with {len(values)} portfolio values")
            logger.debug(f"First value: {values[0] if len(values) > 0 else 'N/A'}")
            logger.debug(f"Last value: {values[-1] if len(values) > 0 else 'N/A'}")

            # Return metrics
            total_return = (values[-1] / values[0]) - 1 if len(values) > 1 else 0.0
            cagr = (values[-1] / values[0]) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0.0

            # Risk metrics
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
            sharpe_ratio = (np.mean(returns) * 252) / volatility if volatility != 0 else 0.0

            # Drawdown analysis
            running_max = np.maximum.accumulate(values)
            drawdowns = values / running_max - 1
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0

            # Trade analysis
            winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in self.trades if t.get('pnl', 0) <= 0]

            win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0.0

            logger.info(f"Performance calculation complete:")
            logger.info(f"Total Return: {total_return:.2%}")
            logger.info(f"Number of Trades: {len(self.trades)}")
            logger.info(f"Win Rate: {win_rate:.2%}")

            return {
                'total_return': total_return,
                'cagr': cagr,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else np.inf,
                'total_trades': len(self.trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades)
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def generate_report(self, output_path: str) -> str:
        """Generate Excel report with performance data"""
        try:
            # Ensure we have an absolute path
            output_dir = os.path.abspath(os.path.dirname(output_path))

            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # If output_path doesn't end with .xlsx, add it
            if not output_path.endswith('.xlsx'):
                output_path = os.path.join(
                    output_dir,
                    f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                )

            # Create DataFrames
            position_df = pd.DataFrame(self.position_snapshots)
            trades_df = pd.DataFrame(self.trades)
            metrics_df = pd.DataFrame([self.metrics])

            # Calculate daily P&L and cumulative returns
            returns_df = pd.DataFrame({
                'date': [s['date'] for s in self.position_snapshots],
                'portfolio_value': self.portfolio_values,
                'daily_return': [0] + self.daily_returns,
                'cumulative_return': (np.array(self.portfolio_values) / self.portfolio_values[0]) - 1
            })

            # Save to Excel with error handling
            try:
                with pd.ExcelWriter(output_path) as writer:
                    position_df.to_excel(writer, sheet_name='Positions', index=False)
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
                    returns_df.to_excel(writer, sheet_name='Returns', index=False)
                    metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)

                if os.path.exists(output_path):
                    logger.info(f"Performance report saved to {output_path}")
                    return output_path
                else:
                    logger.error(f"Failed to verify file creation at {output_path}")
                    return ""

            except Exception as e:
                logger.error(f"Error saving Excel file: {e}")
                # Try CSV fallback
                csv_path = output_path.replace('.xlsx', '.csv')
                returns_df.to_csv(csv_path, index=False)
                logger.info(f"Saved fallback CSV to {csv_path}")
                return csv_path

        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return ""
