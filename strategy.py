from typing import Dict, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from .config import BacktestConfig
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents a single position in the portfolio"""
    ticker: str
    size: float
    entry_price: float
    entry_date: datetime
    initial_stop: Optional[float] = None
    trailing_stop: Optional[float] = None
    target_price: Optional[float] = None
    last_price: Optional[float] = None
    highest_price: Optional[float] = None
    signal_strength: Optional[int] = None
    signal_confidence: Optional[float] = None

    def update(self, current_price: float, current_signal: Optional[Dict] = None) -> None:
        """Update position with current price and signal information"""
        self.last_price = current_price
        self.highest_price = max(current_price, self.highest_price or current_price)

        if current_signal:
            self.signal_strength = current_signal.get('strength')
            self.signal_confidence = current_signal.get('confidence', {}).get('overall')

    def calculate_pnl(self) -> float:
        """Calculate current P&L for the position"""
        if self.last_price is None:
            return 0.0
        return (self.last_price - self.entry_price) * self.size

    def calculate_return(self) -> float:
        """Calculate current return percentage"""
        if self.last_price is None or self.entry_price == 0:
            return 0.0
        return (self.last_price - self.entry_price) / self.entry_price


class Strategy:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.cash: float = config.INITIAL_CAPITAL
        self.equity: float = config.INITIAL_CAPITAL
        self.high_water_mark: float = config.INITIAL_CAPITAL

    def calculate_position_size(self, signal: Dict, available_capital: float) -> float:
        """Calculate position size based on signal and available capital"""
        if not signal or 'strength' not in signal:
            return 0.0

        # Get scale factor based on signal strength
        scale_factor = self.config.signal_config.SIGNAL_POSITION_SCALE.get(
            signal['strength'], 0.0
        )

        # Calculate base allocation
        base_allocation = (available_capital *
                           self.config.position_config.MAX_PORTFOLIO_ALLOCATION *
                           self.config.position_config.MAX_TICKER_ALLOCATION)

        # Scale by confidence if available
        confidence = signal.get('confidence', {}).get('overall', 1.0)
        if confidence < self.config.signal_config.MIN_CONFIDENCE:
            return 0.0

        return base_allocation * scale_factor * confidence

    def evaluate_entry(self, ticker: str, signal: Dict, current_price: float) -> Optional[Dict]:
        """Evaluate whether to enter a new position"""
        try:
            # Skip if we already have a position
            if ticker in self.positions:
                return None

            # Validate signal
            if not signal or 'strength' not in signal:
                return None

            # Calculate position size
            position_size = self.calculate_position_size(signal, self.cash + self.equity)
            if position_size <= 0:
                return None

            # Direction based on signal strength
            is_long = signal['strength'] > 0

            # Calculate entry details
            entry = {
                'ticker': ticker,
                'size': position_size * (1 if is_long else -1),  # Negative size for shorts
                'price': current_price,
                'initial_stop': current_price * (
                    (1 - self.config.risk_config.STOP_LOSS) if is_long
                    else (1 + self.config.risk_config.STOP_LOSS)
                ),
                'target_price': current_price * (
                    (1 + self.config.risk_config.PROFIT_TARGET) if is_long
                    else (1 - self.config.risk_config.PROFIT_TARGET)
                )
            }

            logger.info(f"Generated entry for {ticker}: {entry}")
            return entry

        except Exception as e:
            logger.error(f"Error evaluating entry: {e}")
            return None

    def evaluate_exit(self, position: Position, current_signal: Dict) -> bool:
        """Evaluate whether to exit an existing position"""
        if not position.last_price:
            return False

        # Check stop loss
        if (position.initial_stop and
                position.last_price <= position.initial_stop):
            logger.info(f"Stop loss triggered for {position.ticker}")
            return True

        # Check trailing stop
        if (position.trailing_stop and
                position.last_price <= position.trailing_stop):
            logger.info(f"Trailing stop triggered for {position.ticker}")
            return True

        # Check profit target
        if (position.target_price and
                position.last_price >= position.target_price):
            logger.info(f"Profit target reached for {position.ticker}")
            return True

        # Check signal-based exit
        if current_signal and current_signal.get('strength') == -2:  # Strong sell
            logger.info(f"Signal-based exit triggered for {position.ticker}")
            return True

        return False

    def update_position(self, position: Position, current_price: float) -> None:
        """Update position with current price and adjust stops"""
        position.update(current_price)

        # Update trailing stop if enabled
        if self.config.risk_config.TRAILING_STOP > 0:
            new_stop = position.highest_price * (1 - self.config.risk_config.TRAILING_STOP)
            if position.trailing_stop is None or new_stop > position.trailing_stop:
                position.trailing_stop = new_stop