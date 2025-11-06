"""Configuration management for the trading system."""
from dataclasses import dataclass, field
from typing import Dict, List
import os
import yaml
from .exceptions import ConfigurationError
from .logger import get_logger

logger = get_logger(__name__)

@dataclass
class APIConfig:
    alpha_vantage_key: str = 'HZ0SGS5ODH8JP1K2'  # Existing API key
    max_retries: int = 3
    retry_delay: float = 5.0

@dataclass
class PathConfig:
    base_directory: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_directory: str = field(init=False)
    results_directory: str = field(init=False)
    log_directory: str = field(init=False)

    def __post_init__(self):
        self.data_directory = os.path.join(self.base_directory, "data")
        self.results_directory = os.path.join(self.base_directory, "results")
        self.log_directory = os.path.join(self.base_directory, "logs")

@dataclass
class TradingConfig:
    initial_capital: float = 10000.0
    position_size_limit: float = 0.95
    slippage: float = 0.0005
    commission: float = 0.001

    # Risk management
    max_drawdown: float = 0.20
    stop_loss: float = 0.05
    take_profit: float = 0.15
    max_positions: int = 5
    correlation_threshold: float = 0.7
    risk_free_rate: float = 0.02

    # Position sizing
    position_sizing_method: str = 'risk_parity'
    max_position_size: float = 0.25

@dataclass
class BacktestConfig:
    num_days: int = 50
    iterations: int = 500
    lookback_window: int = 100
    amount_of_trading_days: int = 10000
    time_frame: str = 'daily'
    benchmark: str = 'SPY'
    rebalance_frequency: str = 'daily'

@dataclass
class MLConfig:
    model_type: str = 'random_forest'
    min_confidence: float = 0.6
    lookback_window: int = 50
    train_size: float = 0.8
    feature_columns: List[str] = field(default_factory=lambda: [
        'Bin_Predicted_Move_mean',
        'Bin_Predicted_Move_std',
        'Bin_Confidence_mean'
    ])
    # Updated model_params to match RandomForestClassifier parameters
    model_params: Dict = field(default_factory=lambda: {
        'n_estimators': 150,
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'random_state': 42
    })

    min_samples: int = 15
    min_correlation: float = 0.05  # Increased from 0.1 for stronger signals
    max_p_value: float = 0.075  # Statistical significance threshold
    confidence_level: float = 0.90

def get_model_params(self) -> Dict:
        """Get the appropriate parameters for the selected model type"""
        base_params = {
            'random_forest': {
                'n_estimators': self.model_params.get('n_estimators', 150),
                'max_depth': self.model_params.get('max_depth', 5),
                'min_samples_split': self.model_params.get('min_samples_split', 10),
                'min_samples_leaf': self.model_params.get('min_samples_leaf', 5),
                'class_weight': 'balanced',
                'random_state': 42
            },
            'gradient_boost': {
                'n_estimators': self.model_params.get('n_estimators', 100),
                'learning_rate': self.model_params.get('learning_rate', 0.1),
                'max_depth': self.model_params.get('max_depth', 3),
                'min_samples_split': self.model_params.get('min_samples_split', 10),
                'min_samples_leaf': self.model_params.get('min_samples_leaf', 5),
                'random_state': 42
            },
            'logistic': {
                'C': self.model_params.get('C', 1.0),
                'class_weight': 'balanced',
                'max_iter': self.model_params.get('max_iter', 1000),
                'random_state': 42
            },
            'svm': {
                'C': self.model_params.get('C', 1.0),
                'kernel': self.model_params.get('kernel', 'rbf'),
                'class_weight': 'balanced',
                'random_state': 42
            }
        }
        return base_params.get(self.model_type, base_params['random_forest'])

@dataclass
class LogConfig:
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_name: str = 'trading_system.log'
    console_output: bool = True

@dataclass
class Config:
    API: APIConfig = field(default_factory=APIConfig)
    PATHS: PathConfig = field(default_factory=PathConfig)
    TRADING: TradingConfig = field(default_factory=TradingConfig)
    BACKTEST: BacktestConfig = field(default_factory=BacktestConfig)
    ML: MLConfig = field(default_factory=MLConfig)
    LOG: LogConfig = field(default_factory=LogConfig)

    # Define indicators as a dictionary with tuple values
    INDICATORS: Dict = field(default_factory=lambda: {
        'SMA20': ('SMA', 20),
        'SMA50': ('SMA', 50),
        'RSI': ('RSI', 14),
        'MACD': ('MACD', None),
    })

    # Base indicators list
    BASE_INDICATORS: List[str] = field(default_factory=lambda: ['SMA20', 'SMA50', 'RSI', 'MACD'])

    def __post_init__(self):
        """Initialize and validate configuration"""
        self._create_directories()
        self._validate()

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.PATHS.data_directory,
            self.PATHS.results_directory,
            self.PATHS.log_directory
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")

    def _validate(self):
        """Validate configuration settings"""
        if not self.API.alpha_vantage_key:
            raise ConfigurationError("Alpha Vantage API key is required")

    @classmethod
    def create_default(cls) -> 'Config':
        """Create a default configuration instance"""
        config = cls()
        # Override with environment variables if they exist
        if api_key := os.getenv('ALPHA_VANTAGE_KEY'):
            config.API.alpha_vantage_key = api_key
        return config

    def get(self, key: str, default=None):
        """Get a configuration value"""
        return getattr(self, key, default)