# utils/exceptions.py

class TradingSystemException(Exception):
    """Base exception class for the trading system."""
    pass

class DataError(TradingSystemException):
    """Base class for data-related errors."""
    pass

class DataFetchError(DataError):
    """Raised when API requests or data fetching fails."""
    pass

class DataProcessingError(DataError):
    """Raised when data processing or transformation fails."""
    pass

class DataSyncError(DataError):
    """Raised when data synchronization fails."""
    pass

class ValidationError(TradingSystemException):
    """Raised when validation fails."""
    pass

class ConfigurationError(TradingSystemException):
    """Raised when there are configuration issues."""
    pass

class AnalysisError(TradingSystemException):
    """Raised when analysis operations fail."""
    pass

class BacktestError(TradingSystemException):
    """Raised when backtesting operations fail."""
    pass

class SignalError(TradingSystemException):
    """Raised when there are issues with signal generation."""
    pass

class PositionError(TradingSystemException):
    """Raised when there are issues with position management."""
    pass

class RiskError(TradingSystemException):
    """Raised when risk limits are violated or calculations fail."""
    pass

class BinningError(TradingSystemException):
    """Raised when binning stuff goes bad."""
    pass

class PairAnalysisError(TradingSystemException):
    """Raised when pairing stuff goes bad."""
    pass

class StatisticsError(TradingSystemException):
    """Raised when binning statistics goes bad"""
    pass
