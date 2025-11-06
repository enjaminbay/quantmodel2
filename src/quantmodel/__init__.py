"""
QuantModel - Quantitative Trading System

A comprehensive trading system for backtesting, signal generation, and portfolio management.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .utils.config import Config
from .utils.logger import get_logger, setup_logger
from .utils.exceptions import TradingSystemException

__all__ = [
    "Config",
    "get_logger",
    "setup_logger",
    "TradingSystemException",
]
