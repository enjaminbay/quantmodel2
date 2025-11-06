from typing import Dict
import pandas as pd
import os
from datetime import datetime
from quantmodel.utils.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0, ticker: str ='NONE'):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.ticker = ticker

    def add_data(self, df: pd.DataFrame, signals: Dict) -> None:
        """Add ticker data and signals"""
        self.data = df
        self.signals = signals
        logger.info(f"Loaded {len(df)} days of data")
        logger.info(f"Loaded {len(signals)} signals")

    def run(self) -> pd.DataFrame:
        """Run backtest and return results DataFrame"""
        ticker = self.ticker
        results = []
        current_cash = self.initial_capital
        previous_shares = 0

        position_map = {
            -2: 0.50,  # Strong sell = 50%
            -1: 0.625,  # Sell = 62.5%
            0: 0.75,  # Neutral = 75%
            1: 0.875,  # Buy = 87.5%
            2: 1.0  # Strong buy = 100%
        }
        # position_map = {
        #     -2: 0.0,  # Strong sell = 50%
        #     -1: 0.25,  # Sell = 62.5%
        #     0: 0.5,  # Neutral = 75%
        #     1: 0.75,  # Buy = 87.5%
        #     2: 1.0  # Strong buy = 100%
        # }

        for date in self.data.index:
            price = self.data.loc[date, 'Stock Price']

            # Calculate current portfolio value FIRST using new price
            current_position_value = previous_shares * price
            portfolio_value = current_position_value + current_cash

            # Get signal for this date
            try:
                signal = self.signals[date]['strength']
            except:
                logger.warning("Signal for date: " + str(date) + ' set to 0')
                signal = 0

            # Get position percentage based on signal
            position_pct = position_map.get(signal, 0.75)

            # Calculate target position based on CURRENT portfolio value
            target_position_value = portfolio_value * position_pct
            total_shares = target_position_value / price
            delta_shares = total_shares - previous_shares

            # Execute trade
            position_value = total_shares * price
            cash_left = current_cash - (delta_shares * price)

            # Store results
            results.append({
                'Date': date,
                'Price': price,
                'Signal': signal,
                'Shares': total_shares,
                'Position Value': position_value,
                'Cash': cash_left,
                'Portfolio Value': position_value + cash_left,
                'Return': ((position_value + cash_left) / self.initial_capital) - 1
            })

            # Update for next iteration
            previous_shares = total_shares
            current_cash = cash_left

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Save to Excel
        output_dir = os.path.join(os.getcwd(), 'backtest_results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{ticker}_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        results_df.to_excel(output_path, index=False)

        # Log summary
        logger.info(f"\nBacktest Summary:")
        logger.info(f"Total days: {len(results)}")
        logger.info(f"Final portfolio value: ${results[-1]['Portfolio Value']:.2f}")
        logger.info(f"Total return: {results[-1]['Return']:.2%}")
        logger.info(f"Max portfolio value: ${results_df['Portfolio Value'].max():.2f}")
        logger.info(f"Min portfolio value: ${results_df['Portfolio Value'].min():.2f}")
        logger.info(f"Results saved to: {output_path}")

        return results_df
    #
    # def add_data(self, df: pd.DataFrame, signals: Dict) -> None:
    #     """Add ticker data and signals"""
    #     self.data = df
    #     # Convert all signal keys to date strings without time
    #     self.signals = {
    #         pd.Timestamp(k).strftime('%Y-%m-%d'): v
    #         for k, v in signals.items()
    #     }
    #
    #     # Debug logging
    #     logger.info(f"=== Data and Signals after formatting ===")
    #     logger.info(f"First few signal dates: {list(self.signals.keys())[:5]}")
    #     logger.info(f"First few DataFrame dates: {[d.strftime('%Y-%m-%d') for d in self.data.index[:5]]}")
    #
    # def run(self) -> pd.DataFrame:
    #     """Run backtest and return results DataFrame"""
    #     results = []
    #
    #     # Create output directory if it doesn't exist
    #     output_dir = os.path.join(os.getcwd(), 'backtest_results')
    #     os.makedirs(output_dir, exist_ok=True)
    #
    #     for date in self.data.index:
    #         # Convert DataFrame date to string format
    #         date_str = date.strftime('%Y-%m-%d')
    #         price = self.data.loc[date, 'Stock Price']
    #
    #         # Get signal for this date
    #         signal = self.signals.get(date_str, {'strength': 0})
    #
    #         # Debug first few dates
    #         if len(results) < 5:
    #             logger.debug(f"Processing date {date_str}:")
    #             logger.debug(f"  Signal found: {signal.get('strength', 0)}")
    #             logger.debug(f"  Price: {price}")
    #
    #         # Calculate position size using signal
    #         shares = self.calculate_position_size(signal, price)
    #
    #         # Calculate portfolio value
    #         portfolio_value = shares * price + (self.current_capital - shares * price)
    #
    #         # Calculate cumulative return
    #         cumulative_return = (portfolio_value / self.initial_capital) - 1
    #
    #         # Store results
    #         results.append({
    #             'Date': date,
    #             'Stock Price': price,
    #             'Shares Held': shares,
    #             'Signal': signal.get('strength', 0),
    #             'Portfolio Value': portfolio_value,
    #             'Cumulative Return': cumulative_return,
    #             'Signal Confidence': signal.get('confidence', {}).get('overall', 0)
    #         })
    #
    #         # Update capital for next iteration
    #         self.current_capital = portfolio_value
    #
    #     # Convert to DataFrame
    #     results_df = pd.DataFrame(results)
    #
    #     # Save to Excel in backtest_results directory
    #     output_path = os.path.join(output_dir, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    #     results_df.to_excel(output_path, index=False)
    #     logger.info(f"Results saved to: {output_path}")
    #
    #     # Log statistics
    #     if len(results) > 0:
    #         logger.info(f"\nBacktest Statistics:")
    #         logger.info(f"Total days processed: {len(results)}")
    #         logger.info(f"Days with non-zero signals: {sum(1 for r in results if r['Signal'] != 0)}")
    #         logger.info(f"Final portfolio value: ${results[-1]['Portfolio Value']:,.2f}")
    #         logger.info(f"Total return: {results[-1]['Cumulative Return']:.2%}")
    #
    #     # Just return the DataFrame
    #     return results_df

    def calculate_position_size(self, signal: Dict, price: float) -> float:
        """Calculate position size based on signal strength"""
        signal_scale = {
            2: 1.00,  # Strong Buy  = 100% of capital
            1: 0.75,  # Buy         = 75% of capital
            0: 0.50,  # Neutral     = 50% of capital
            -1: 0.25,  # Sell        = 25% of capital
            -2: 0.00  # Strong Sell = 0% of capital
        }

        strength = signal.get('strength', 0)
        allocation = signal_scale.get(strength, 0.0)
        shares = (self.current_capital * allocation) / price if price > 0 else 0

        if shares > 0:
            logger.debug(f"Calculated position: Signal={strength}, Allocation={allocation}, Shares={shares:.2f}")

        return shares