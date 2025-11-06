from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
from .config import BacktestConfig
from .strategy import Position
from utils.logger import get_logger

logger = get_logger(__name__)


class Portfolio:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.cash: float = config.INITIAL_CAPITAL
        self.equity: float = config.INITIAL_CAPITAL
        self.high_water_mark: float = config.INITIAL_CAPITAL

        # Performance tracking
        self.daily_returns: List[float] = []
        self.portfolio_values: List[float] = []
        self.position_history: List[Dict] = []
        self.trades_history: List[Dict] = []

    def get_total_value(self) -> float:
        """Calculate total portfolio value including cash"""
        position_value = sum(pos.calculate_pnl() for pos in self.positions.values())
        return self.cash + position_value

    def get_current_allocation(self) -> float:
        """Calculate current portfolio allocation percentage"""
        total_value = self.get_total_value()
        if total_value == 0:
            return 0.0
        position_value = sum(pos.last_price * pos.size
                             for pos in self.positions.values()
                             if pos.last_price)
        return position_value / total_value

    def can_take_position(self, size: float, price: float) -> bool:
        """Check if new position meets allocation constraints"""
        total_value = self.get_total_value()
        new_position_value = size * price

        # Check if we have enough cash
        if new_position_value > self.cash:
            return False

        # Check if this would exceed our maximum portfolio allocation
        current_allocation = self.get_current_allocation()
        new_allocation = new_position_value / total_value

        if (current_allocation + new_allocation >
                self.config.position_config.MAX_PORTFOLIO_ALLOCATION):
            return False

        return True

    def open_position(self, ticker: str, size: float, price: float,
                      signal_strength: int, signal_confidence: float,
                      date: datetime) -> bool:
        """Open a new position"""
        try:
            # Calculate transaction costs
            commission = price * size * self.config.COMMISSION_RATE
            slippage = price * size * self.config.SLIPPAGE
            total_cost = (price * size) + commission + slippage

            if total_cost > self.cash:
                logger.warning(f"Insufficient cash to open position in {ticker}")
                return False

            # Create position
            position = Position(
                ticker=ticker,
                size=size,
                entry_price=price,
                entry_date=date,
                initial_stop=price * (1 - self.config.risk_config.STOP_LOSS),
                target_price=price * (1 + self.config.risk_config.PROFIT_TARGET),
                signal_strength=signal_strength,
                signal_confidence=signal_confidence
            )

            # Update portfolio
            self.positions[ticker] = position
            self.cash -= total_cost

            # Record trade
            self.trades_history.append({
                'date': date,
                'ticker': ticker,
                'action': 'BUY',
                'size': size,
                'price': price,
                'cost': total_cost,
                'commission': commission,
                'slippage': slippage,
                'signal_strength': signal_strength,
                'signal_confidence': signal_confidence
            })

            logger.info(f"Opened position in {ticker}: {size} shares at {price}")
            return True

        except Exception as e:
            logger.error(f"Error opening position in {ticker}: {str(e)}")
            return False

    def close_position(self, ticker: str, price: float, date: datetime) -> bool:
        """Close an existing position"""
        try:
            if ticker not in self.positions:
                return False

            position = self.positions[ticker]

            # Calculate transaction costs
            commission = price * position.size * self.config.COMMISSION_RATE
            slippage = price * position.size * self.config.SLIPPAGE

            # Calculate proceeds
            gross_proceeds = position.size * price
            net_proceeds = gross_proceeds - commission - slippage

            # Update portfolio
            self.cash += net_proceeds

            # Record trade
            self.trades_history.append({
                'date': date,
                'ticker': ticker,
                'action': 'SELL',
                'size': position.size,
                'price': price,
                'proceeds': net_proceeds,
                'commission': commission,
                'slippage': slippage,
                'pnl': net_proceeds - (position.size * position.entry_price),
                'return': (price - position.entry_price) / position.entry_price
            })

            # Remove position
            del self.positions[ticker]

            logger.info(f"Closed position in {ticker} at {price}")
            return True

        except Exception as e:
            logger.error(f"Error closing position in {ticker}: {str(e)}")
            return False

    def update_state(self, current_prices: Dict[str, float], date: datetime) -> None:
        """Update portfolio state with current prices"""
        try:
            # Update positions
            for ticker, position in self.positions.items():
                if ticker in current_prices:
                    position.update(current_prices[ticker])

            # Calculate daily return
            current_value = self.get_total_value()
            if self.portfolio_values:
                daily_return = (current_value / self.portfolio_values[-1]) - 1
                self.daily_returns.append(daily_return)

            # Update high water mark
            self.high_water_mark = max(self.high_water_mark, current_value)

            # Record portfolio value
            self.portfolio_values.append(current_value)

            # Record position snapshot
            self._record_position_snapshot(date)

        except Exception as e:
            logger.error(f"Error updating portfolio state: {str(e)}")

    def _record_position_snapshot(self, date: datetime) -> None:
        """Record current position states for historical tracking"""
        snapshot = {
            'date': date,
            'portfolio_value': self.get_total_value(),
            'cash': self.cash,
            'allocation': self.get_current_allocation(),
            'positions': {
                ticker: {
                    'size': pos.size,
                    'entry_price': pos.entry_price,
                    'current_price': pos.last_price,
                    'pnl': pos.calculate_pnl(),
                    'return': pos.calculate_return()
                }
                for ticker, pos in self.positions.items()
            }
        }
        self.position_history.append(snapshot)