# QuantModel - Quantitative Trading System

A comprehensive quantitative trading system for backtesting, signal generation, and portfolio management using machine learning and technical analysis.

## Features

- **Data Fetching**: Automated data retrieval from Alpha Vantage API
- **Technical Indicators**: Support for 20+ technical indicators (SMA, RSI, MACD, Bollinger Bands, etc.)
- **Signal Generation**: ML-based signal generation using:
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  - Support Vector Machines
- **Pair Analysis**: Statistical analysis of indicator pairs for pattern detection
- **Backtesting Engine**: Robust backtesting with customizable strategies
- **Portfolio Management**: Position sizing, risk management, and performance tracking
- **Export Capabilities**: Results exported to Excel for easy analysis

## Installation

### Prerequisites

- Python 3.9 or higher
- Alpha Vantage API key (free at [alphavantage.co](https://www.alphavantage.co/))

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantmodel2.git
cd quantmodel2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

3. Set up your Alpha Vantage API key:
   - Edit `src/quantmodel/utils/config.py` and replace the API key
   - Or set the environment variable:
```bash
export ALPHA_VANTAGE_KEY="your_api_key_here"
```

## Project Structure

```
quantmodel2/
├── src/quantmodel/          # Main package
│   ├── data/                # Data fetching and management
│   ├── analysis/            # Pair analysis and statistics
│   ├── signals/             # Signal generation and ML models
│   ├── backtest/            # Backtesting engine
│   ├── processing/          # Data processing and features
│   └── utils/               # Configuration and utilities
├── tests/                   # Unit tests
├── examples/                # Example scripts
├── data/                    # Data storage (created automatically)
├── results/                 # Backtest results (created automatically)
├── logs/                    # Log files (created automatically)
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Package configuration
└── README.md               # This file
```

## Usage

### Basic Example

```python
from quantmodel.utils.config import Config
from quantmodel.data.fetcher import DataFetcher
from quantmodel.signals.generator import SignalGenerator
from quantmodel.backtest.engine import BacktestEngine

# Initialize configuration
config = Config.create_default()

# Fetch data
fetcher = DataFetcher(config)
data = fetcher.STRATTEST(
    ticker='AAPL',
    amount_of_trading_days=1000,
    time_frame='daily'
)

# Generate signals
signal_gen = SignalGenerator(config)
# ... (add your signal generation logic)

# Run backtest
engine = BacktestEngine(initial_capital=100000, ticker='AAPL')
# ... (add your backtesting logic)
```

### Configuration

The system uses a comprehensive configuration system in `src/quantmodel/utils/config.py`:

- **API Configuration**: API keys and retry settings
- **Trading Configuration**: Capital, position sizing, risk management
- **Backtest Configuration**: Timeframes, lookback windows
- **ML Configuration**: Model parameters and thresholds

### Available Indicators

The system supports the following technical indicators:

- **Moving Averages**: SMA, WMA, T3
- **Momentum**: RSI, MFI, MOM, CMO
- **Trend**: MACD, DX, ADXR
- **Volatility**: Bollinger Bands, ATR, TRANGE
- **Volume**: OBV
- **Oscillators**: STOCH, CCI, BOP

## Features Detail

### Signal Generation

The signal generation system uses statistical pair analysis combined with machine learning:

1. **Pair Analysis**: Analyzes relationships between indicator pairs
2. **Binning**: Groups indicator values into quantiles for pattern detection
3. **Statistical Testing**: Validates significance using p-values and correlation
4. **ML Models**: Generates predictions using trained classifiers
5. **Signal Strength**: Outputs signals from -2 (strong sell) to +2 (strong buy)

### Backtesting

The backtesting engine supports:

- **Position Sizing**: Dynamic position sizing based on signal strength
- **Risk Management**: Stop-loss, take-profit, trailing stops
- **Performance Metrics**: Returns, drawdown, Sharpe ratio
- **Excel Export**: Detailed results with charts and statistics

### Risk Management

Built-in risk management features:

- Maximum drawdown limits
- Position size limits
- Correlation-based diversification
- Stop-loss and take-profit levels

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/quantmodel/
```

### Type Checking

```bash
mypy src/quantmodel/
```

## Directory Creation

The system automatically creates the following directories:

- `data/` - Cached market data
- `results/` - Backtest results and exports
- `logs/` - Application logs

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Do not use it for actual trading without proper testing and understanding. Trading carries risk of financial loss.

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## Acknowledgments

- Alpha Vantage for providing the market data API
- scikit-learn for machine learning tools
- pandas for data manipulation
