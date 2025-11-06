"""
Position sizing and allocation strategies.

This module provides various position sizing methods for individual trades
and portfolio-level allocation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from quantmodel.utils.logger import get_logger

logger = get_logger(__name__)


class PositionAllocator:
    """
    Position sizing calculator for individual trades.

    Supports multiple position sizing methods including:
    - Fixed fractional
    - Kelly Criterion
    - Risk parity
    - Volatility-based
    """

    def __init__(
        self,
        portfolio_value: float,
        max_position_size: float = 0.20,
        min_position_size: float = 0.05
    ):
        """
        Initialize the position allocator.

        Args:
            portfolio_value: Total portfolio value
            max_position_size: Maximum position size as fraction of portfolio
            min_position_size: Minimum position size as fraction of portfolio
        """
        self.portfolio_value = portfolio_value
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size

    def fixed_fractional(
        self,
        signal_strength: int,
        base_size: float = 0.10
    ) -> float:
        """
        Calculate position size using fixed fractional method.

        Args:
            signal_strength: Signal strength (-2 to 2)
            base_size: Base position size

        Returns:
            Position size in dollars
        """
        strength_multiplier = {
            -2: 0.5,
            -1: 0.75,
            0: 0.0,
            1: 1.0,
            2: 1.5
        }

        multiplier = strength_multiplier.get(abs(signal_strength), 0.0)
        position_fraction = min(
            base_size * multiplier,
            self.max_position_size
        )

        return self.portfolio_value * position_fraction

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        signal_strength: int = 1
    ) -> float:
        """
        Calculate position size using Kelly Criterion.

        Args:
            win_rate: Historical win rate (0 to 1)
            avg_win: Average winning trade size
            avg_loss: Average losing trade size
            signal_strength: Signal strength (used for scaling)

        Returns:
            Position size in dollars
        """
        if avg_loss == 0:
            return 0.0

        win_loss_ratio = abs(avg_win / avg_loss)
        kelly_fraction = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        kelly_fraction = max(0, min(kelly_fraction, 1))

        kelly_fraction *= 0.5

        strength_scale = abs(signal_strength) / 2.0
        position_fraction = kelly_fraction * strength_scale

        position_fraction = np.clip(
            position_fraction,
            self.min_position_size,
            self.max_position_size
        )

        return self.portfolio_value * position_fraction

    def volatility_based(
        self,
        volatility: float,
        target_risk: float = 0.02,
        signal_strength: int = 1
    ) -> float:
        """
        Calculate position size based on volatility targeting.

        Args:
            volatility: Asset volatility (standard deviation)
            target_risk: Target risk per trade
            signal_strength: Signal strength for scaling

        Returns:
            Position size in dollars
        """
        if volatility == 0:
            return 0.0

        base_fraction = target_risk / volatility

        strength_scale = abs(signal_strength) / 2.0
        position_fraction = base_fraction * strength_scale

        position_fraction = np.clip(
            position_fraction,
            self.min_position_size,
            self.max_position_size
        )

        return self.portfolio_value * position_fraction

    def risk_parity(
        self,
        asset_volatilities: Dict[str, float],
        total_allocation: float = 1.0
    ) -> Dict[str, float]:
        """
        Calculate risk parity allocation across multiple assets.

        Args:
            asset_volatilities: Dictionary of asset volatilities
            total_allocation: Total allocation as fraction of portfolio

        Returns:
            Dictionary of position sizes in dollars
        """
        if not asset_volatilities:
            return {}

        inv_volatilities = {
            asset: 1.0 / vol if vol > 0 else 0
            for asset, vol in asset_volatilities.items()
        }

        total_inv_vol = sum(inv_volatilities.values())

        if total_inv_vol == 0:
            equal_weight = 1.0 / len(asset_volatilities)
            return {
                asset: self.portfolio_value * total_allocation * equal_weight
                for asset in asset_volatilities
            }

        allocations = {}
        for asset, inv_vol in inv_volatilities.items():
            weight = (inv_vol / total_inv_vol) * total_allocation
            weight = np.clip(weight, self.min_position_size, self.max_position_size)
            allocations[asset] = self.portfolio_value * weight

        return allocations

    def calculate_shares(
        self,
        position_size: float,
        price: float,
        signal_strength: int
    ) -> int:
        """
        Calculate number of shares to trade.

        Args:
            position_size: Position size in dollars
            price: Current price per share
            signal_strength: Signal strength (-2 to 2)

        Returns:
            Number of shares (negative for short positions)
        """
        if price <= 0:
            return 0

        shares = int(position_size / price)

        if signal_strength < 0:
            shares = -shares

        return shares
