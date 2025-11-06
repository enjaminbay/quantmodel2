"""
Configuration management for the QuantModel trading system.

This module provides a comprehensive configuration system using dataclasses
for type safety and easy validation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import yaml
from pathlib import Path
from .exceptions import ConfigurationError
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class APIConfig:
    """API configuration for external data sources."""
    alpha_vantage_key: str = 'HZ0SGS5ODH8JP1K2'
    max_retries: int = 3
    retry_delay: float = 5.0
    timeout: int = 30
    cache_enabled: bool = True
    cache_expiry_hours: int = 24


@dataclass
class PathConfig:
    """Path configuration for data, results, and logs."""
    base_directory: str = field(default_factory=lambda: str(Path(__file__).parent.parent.parent.parent))
    data_directory: str = field(init=False)
    results_directory: str = field(init=False)
    log_directory: str = field(init=False)
    models_directory: str = field(init=False)
    cache_directory: str = field(init=False)

    def __post_init__(self):
        base = Path(self.base_directory)
        self.data_directory = str(base / "data")
        self.results_directory = str(base / "results")
        self.log_directory = str(base / "logs")
        self.models_directory = str(base / "models")
        self.cache_directory = str(base / ".cache")


@dataclass
class TradingConfig:
    """Trading and risk management configuration."""
    initial_capital: float = 100000.0
    position_size_limit: float = 0.95
    slippage: float = 0.0005
    commission: float = 0.001

    # Risk management
    max_drawdown: float = 0.20
    stop_loss: float = 0.05
    take_profit: float = 0.15
    trailing_stop: float = 0.03
    max_positions: int = 10
    correlation_threshold: float = 0.7
    risk_free_rate: float = 0.02
    max_portfolio_heat: float = 0.06

    # Position sizing
    position_sizing_method: str = 'risk_parity'
    max_position_size: float = 0.20
    min_position_size: float = 0.05

    # Leverage and margin
    max_leverage: float = 1.0
    margin_requirement: float = 0.25


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    num_days: int = 50
    iterations: int = 500
    lookback_window: int = 100
    amount_of_trading_days: int = 1000
    time_frame: str = 'daily'
    benchmark: str = 'SPY'
    rebalance_frequency: str = 'daily'
    walk_forward_analysis: bool = False
    monte_carlo_simulations: int = 1000

    # Performance metrics
    calculate_var: bool = True
    var_confidence: float = 0.95
    calculate_beta: bool = True
    calculate_alpha: bool = True


@dataclass
class MLConfig:
    """Machine learning configuration."""
    model_type: str = 'random_forest'
    min_confidence: float = 0.6
    lookback_window: int = 50
    train_size: float = 0.8

    feature_columns: List[str] = field(default_factory=lambda: [
        'Bin_Predicted_Move_mean',
        'Bin_Predicted_Move_std',
        'Bin_Confidence_mean'
    ])

    model_params: Dict = field(default_factory=lambda: {
        'n_estimators': 150,
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'random_state': 42
    })

    min_samples: int = 15
    min_correlation: float = 0.05
    max_p_value: float = 0.075
    confidence_level: float = 0.90

    def get_model_params(self) -> Dict:
        """Get the appropriate parameters for the selected model type."""
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
    """Logging configuration."""
    level: str = 'INFO'
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_name: str = 'trading_system.log'
    console_output: bool = True


@dataclass
class PortfolioConfig:
    """Portfolio optimization configuration."""
    optimization_method: str = 'mean_variance'
    target_return: Optional[float] = None
    target_risk: Optional[float] = None
    max_weight: float = 0.30
    min_weight: float = 0.05
    rebalance_threshold: float = 0.05

    # Optimization constraints
    allow_short: bool = False
    leverage: float = 1.0

    # Risk models
    risk_model: str = 'sample_cov'
    returns_model: str = 'mean_historical_return'


@dataclass
class RiskConfig:
    """Risk management and metrics configuration."""
    var_method: str = 'historical'
    cvar_enabled: bool = True
    var_lookback: int = 252
    stress_test_scenarios: List[str] = field(default_factory=lambda: [
        'market_crash',
        'volatility_spike',
        'correlation_breakdown'
    ])

    # Risk limits
    max_sector_exposure: float = 0.30
    max_single_position_var: float = 0.02
    portfolio_var_limit: float = 0.05


@dataclass
class Config:
    """Main configuration class for the trading system."""
    API: APIConfig = field(default_factory=APIConfig)
    PATHS: PathConfig = field(default_factory=PathConfig)
    TRADING: TradingConfig = field(default_factory=TradingConfig)
    BACKTEST: BacktestConfig = field(default_factory=BacktestConfig)
    ML: MLConfig = field(default_factory=MLConfig)
    LOG: LogConfig = field(default_factory=LogConfig)
    PORTFOLIO: PortfolioConfig = field(default_factory=PortfolioConfig)
    RISK: RiskConfig = field(default_factory=RiskConfig)

    INDICATORS: Dict = field(default_factory=lambda: {
        'SMA20': ('SMA', 20),
        'SMA50': ('SMA', 50),
        'SMA200': ('SMA', 200),
        'EMA12': ('EMA', 12),
        'EMA26': ('EMA', 26),
        'RSI': ('RSI', 14),
        'MACD': ('MACD', None),
        'BBANDS': ('BBANDS', 20),
        'ATR': ('ATR', 14),
        'STOCH': ('STOCH', 14),
    })

    BASE_INDICATORS: List[str] = field(default_factory=lambda: [
        'SMA20', 'SMA50', 'RSI', 'MACD', 'BBANDS', 'ATR'
    ])

    def __post_init__(self):
        """Initialize and validate configuration."""
        self._create_directories()
        self._validate()

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.PATHS.data_directory,
            self.PATHS.results_directory,
            self.PATHS.log_directory,
            self.PATHS.models_directory,
            self.PATHS.cache_directory
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _validate(self):
        """Validate configuration settings."""
        if not self.API.alpha_vantage_key:
            raise ConfigurationError("Alpha Vantage API key is required")

        if self.TRADING.initial_capital <= 0:
            raise ConfigurationError("Initial capital must be positive")

        if not 0 < self.TRADING.max_position_size <= 1:
            raise ConfigurationError("Max position size must be between 0 and 1")

    @classmethod
    def create_default(cls) -> 'Config':
        """Create a default configuration instance."""
        config = cls()
        if api_key := os.getenv('ALPHA_VANTAGE_KEY'):
            config.API.alpha_vantage_key = api_key
        return config

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from a YAML file."""
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            config = cls()

            for section, values in data.items():
                if hasattr(config, section):
                    section_obj = getattr(config, section)
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)

            return config
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {yaml_path}: {e}")

    def to_yaml(self, yaml_path: str):
        """Save configuration to a YAML file."""
        try:
            config_dict = {}
            for field_name in ['API', 'PATHS', 'TRADING', 'BACKTEST', 'ML', 'LOG', 'PORTFOLIO', 'RISK']:
                section = getattr(self, field_name)
                config_dict[field_name] = {
                    k: v for k, v in section.__dict__.items()
                    if not k.startswith('_')
                }

            with open(yaml_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)

            logger.info(f"Configuration saved to {yaml_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save config to {yaml_path}: {e}")

    def get(self, key: str, default=None):
        """Get a configuration value."""
        return getattr(self, key, default)
