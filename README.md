# QuantModel - Professional Quantitative Trading System

A comprehensive, production-ready quantitative trading system for backtesting, signal generation, portfolio optimization, and risk management using machine learning and advanced statistical techniques.

## Overview

QuantModel is a sophisticated Python-based trading framework designed for quantitative researchers, algorithmic traders, and portfolio managers. It combines modern portfolio theory, machine learning, and rigorous statistical analysis to provide a complete solution for systematic trading strategy development and evaluation.

## Key Features

### Data Management
- **Multi-Source Support**: Integration with Alpha Vantage API
- **Smart Caching**: Intelligent data caching with configurable expiry
- **Data Quality**: Automatic data validation and cleaning
- **Historical Data**: Support for daily and weekly timeframes

### Technical Analysis
- **20+ Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic, and more
- **Custom Indicators**: Easily add your own technical indicators
- **Indicator Derivatives**: Automatic calculation of velocity and acceleration
- **Multi-Timeframe**: Support for different timeframe analysis

### Signal Generation
- **ML-Based Signals**: Multiple machine learning models:
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  - Support Vector Machines
- **Pair Analysis**: Statistical analysis of indicator pairs for pattern detection
- **Signal Strength**: 5-level signal strength (-2 to +2)
- **Confidence Scoring**: Comprehensive confidence metrics for each signal

### Portfolio Optimization
- **Modern Portfolio Theory**: Implementation of Markowitz optimization
- **Optimization Methods**:
  - Maximum Sharpe Ratio
  - Minimum Variance
  - Risk Parity
  - Efficient Frontier generation
- **Constraints**: Customizable position limits and rebalancing thresholds
- **Smart Allocation**: Dynamic position sizing based on risk and signals

### Risk Management
- **Value at Risk (VaR)**: Multiple calculation methods
  - Historical VaR
  - Parametric VaR
  - Monte Carlo VaR
- **Risk Metrics**:
  - Beta and Alpha (Jensen's)
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Maximum Drawdown
  - Omega Ratio
  - Tail Ratio
- **Component VaR**: Individual asset risk contribution analysis
- **Real-Time Monitoring**: Track portfolio risk in real-time

### Backtesting Engine
- **Realistic Simulation**:
  - Transaction costs and slippage
  - Position sizing based on signals
  - Dynamic rebalancing
- **Performance Analytics**:
  - Comprehensive metrics calculation
  - Drawdown analysis
  - Win rate and trade statistics
- **Excel Reporting**: Detailed results export for analysis

### Position Sizing
- **Multiple Methods**:
  - Fixed Fractional
  - Kelly Criterion
  - Volatility-Based
  - Risk Parity
- **Signal Integration**: Position size scales with signal strength
- **Risk Control**: Configurable position limits

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Alpha Vantage API key (free at [alphavantage.co](https://www.alphavantage.co/))

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/enjaminbay/quantmodel2.git
cd quantmodel2
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

3. **Set up API key**:
```bash
export ALPHA_VANTAGE_KEY="your_api_key_here"
```

## Project Structure

```
quantmodel2/
├── src/quantmodel/              # Main package
│   ├── data/                    # Data fetching and management
│   ├── analysis/                # Statistical analysis and pair analysis
│   ├── signals/                 # Signal generation and ML models
│   ├── backtest/                # Backtesting engine and evaluation
│   ├── portfolio/               # Portfolio optimization and allocation
│   │   ├── optimizer.py         # Modern Portfolio Theory optimization
│   │   └── allocator.py         # Position sizing strategies
│   ├── risk/                    # Risk management and metrics
│   │   ├── metrics.py           # Risk metric calculations
│   │   └── var.py               # VaR and CVaR calculations
│   ├── processing/              # Data processing and features
│   └── utils/                   # Configuration and utilities
├── tests/                       # Unit and integration tests
├── examples/                    # Example scripts and tutorials
├── data/                        # Data storage (auto-created)
├── results/                     # Backtest results (auto-created)
├── models/                      # Saved models (auto-created)
├── logs/                        # Log files (auto-created)
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Package configuration
└── README.md                   # This file
```

## Usage Examples

### Basic Backtest

```python
from quantmodel.utils.config import Config
from quantmodel.data.fetcher import DataFetcher
from quantmodel.analysis.pair_analyzer import PairAnalyzer
from quantmodel.signals.generator import SignalGenerator
from quantmodel.backtest.engine import BacktestEngine
import pandas as pd

# Initialize
config = Config.create_default()

# Fetch data
fetcher = DataFetcher(config)
data = fetcher.STRATTEST(ticker='AAPL', amount_of_trading_days=1000)

# Analyze indicator pairs
analyzer = PairAnalyzer(config)
pair_results = analyzer.compare_indicator_pairs(
    indicator_data, price_changes
)

# Generate signals
signal_gen = SignalGenerator(config)
signals = signal_gen.generate_signals(pair_results, df)

# Run backtest
engine = BacktestEngine(initial_capital=100000, ticker='AAPL')
engine.add_data(df, signals)
results = engine.run()
```

### Portfolio Optimization

```python
from quantmodel.portfolio.optimizer import PortfolioOptimizer
import pandas as pd

# Assume you have returns data for multiple assets
returns = pd.DataFrame({
    'AAPL': apple_returns,
    'MSFT': microsoft_returns,
    'GOOGL': google_returns
})

# Create optimizer
optimizer = PortfolioOptimizer(returns, risk_free_rate=0.02)

# Find maximum Sharpe ratio portfolio
max_sharpe = optimizer.max_sharpe_ratio(max_weight=0.30)
print(f"Optimal weights: {max_sharpe['weights']}")
print(f"Expected return: {max_sharpe['expected_return']:.2%}")
print(f"Sharpe ratio: {max_sharpe['sharpe_ratio']:.2f}")

# Get dollar allocation
allocation = optimizer.get_allocation(
    method='max_sharpe',
    total_value=100000
)
```

### Risk Analysis

```python
from quantmodel.risk.metrics import RiskMetrics
from quantmodel.risk.var import VaRCalculator

# Calculate risk metrics
risk_metrics = RiskMetrics(risk_free_rate=0.02)
metrics = risk_metrics.calculate_all_metrics(
    returns=portfolio_returns,
    portfolio_values=portfolio_values,
    benchmark_returns=spy_returns
)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Beta: {metrics['beta']:.2f}")

# Calculate VaR
var_calc = VaRCalculator(confidence_level=0.95)
var, cvar = var_calc.calculate_var(
    returns=portfolio_returns,
    portfolio_value=1000000,
    method='historical'
)

print(f"95% VaR: ${var:,.2f}")
print(f"95% CVaR: ${cvar:,.2f}")
```

### Position Sizing

```python
from quantmodel.portfolio.allocator import PositionAllocator

allocator = PositionAllocator(
    portfolio_value=100000,
    max_position_size=0.20
)

# Kelly Criterion sizing
position_size = allocator.kelly_criterion(
    win_rate=0.55,
    avg_win=0.02,
    avg_loss=0.01,
    signal_strength=2  # Strong buy
)

shares = allocator.calculate_shares(
    position_size=position_size,
    price=150.00,
    signal_strength=2
)

print(f"Position size: ${position_size:,.2f}")
print(f"Shares to buy: {shares}")
```

## Configuration

The system uses a comprehensive configuration system with support for:

### Configuration Sections

- **API**: API keys, timeouts, caching settings
- **Paths**: Directory structure for data, results, models
- **Trading**: Capital, position limits, transaction costs
- **Backtest**: Timeframes, performance metrics
- **ML**: Model parameters, thresholds
- **Portfolio**: Optimization methods, constraints
- **Risk**: VaR settings, risk limits
- **Log**: Logging levels and formats

### YAML Configuration

Save and load configurations from YAML:

```python
config = Config.create_default()
config.to_yaml('my_config.yaml')

# Load later
config = Config.from_yaml('my_config.yaml')
```

## Performance Metrics

The system calculates comprehensive performance metrics:

### Return Metrics
- Total return
- Annualized return
- Cumulative returns

### Risk Metrics
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Maximum drawdown
- Volatility

### Trading Metrics
- Win rate
- Average trade size
- Total number of trades
- Transaction costs

### Risk-Adjusted Metrics
- Alpha (Jensen's)
- Beta
- Information ratio
- Omega ratio

## Advanced Features

### Walk-Forward Analysis
Test strategy robustness with out-of-sample validation

### Monte Carlo Simulation
Generate thousands of scenarios for stress testing

### Efficient Frontier
Visualize risk-return tradeoffs across portfolios

### Component VaR
Understand individual asset contributions to portfolio risk

## Development

### Running Tests

```bash
pytest tests/ -v --cov=quantmodel
```

### Code Quality

```bash
# Format code
black src/quantmodel/

# Lint
flake8 src/quantmodel/

# Type checking
mypy src/quantmodel/
```

## Best Practices

1. **Always backtest** thoroughly before live trading
2. **Monitor risk metrics** continuously
3. **Use proper position sizing** to manage risk
4. **Diversify** across multiple assets and strategies
5. **Keep transaction costs** in mind
6. **Regular rebalancing** based on your strategy
7. **Document your strategies** and maintain good records

## Roadmap

- [ ] Real-time data streaming
- [ ] Interactive dashboard
- [ ] More data source integrations
- [ ] Advanced order types
- [ ] Paper trading mode
- [ ] Strategy templating system
- [ ] Machine learning model marketplace

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

**IMPORTANT**: This software is for educational and research purposes only.

- Do NOT use for actual trading without extensive testing
- Past performance does not guarantee future results
- Trading carries significant risk of financial loss
- Always consult with a qualified financial advisor
- The authors are not responsible for any financial losses

## Support

- **Issues**: [GitHub Issues](https://github.com/enjaminbay/quantmodel2/issues)
- **Documentation**: See `/examples` directory
- **Discussions**: GitHub Discussions

## Acknowledgments

- **Alpha Vantage** for market data API
- **scikit-learn** for machine learning tools
- **pandas** and **numpy** for data manipulation
- **scipy** for statistical functions
- The quantitative finance community for inspiration

## Citation

If you use QuantModel in your research, please cite:

```bibtex
@software{quantmodel2024,
  title={QuantModel: A Comprehensive Quantitative Trading System},
  author={QuantModel Contributors},
  year={2024},
  url={https://github.com/enjaminbay/quantmodel2}
}
```

---

**Built with passion for quantitative trading** 📈
