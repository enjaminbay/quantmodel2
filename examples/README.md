# QuantModel Examples

This directory contains example scripts demonstrating how to use QuantModel.

## Examples

### basic_backtest.py

A complete example showing:
- Fetching historical data from Alpha Vantage
- Analyzing indicator pairs for correlations
- Generating trading signals
- Running a backtest
- Viewing results

**Usage:**
```bash
python examples/basic_backtest.py
```

**Requirements:**
- Set up your Alpha Vantage API key in `src/quantmodel/utils/config.py`
- Install all dependencies: `pip install -r requirements.txt`

### config_example.yaml

An example configuration file showing all available settings.

**Usage:**
1. Copy to your project root: `cp examples/config_example.yaml config.yaml`
2. Edit with your settings
3. Load in your code:
```python
import yaml
with open('config.yaml') as f:
    config_dict = yaml.safe_load(f)
```

## Creating Your Own Scripts

To create your own trading scripts:

1. **Import the package:**
```python
from quantmodel.utils.config import Config
from quantmodel.data.fetcher import DataFetcher
from quantmodel.signals.generator import SignalGenerator
from quantmodel.backtest.engine import BacktestEngine
```

2. **Initialize configuration:**
```python
config = Config.create_default()
```

3. **Fetch data:**
```python
fetcher = DataFetcher(config)
data = fetcher.STRATTEST('AAPL', amount_of_trading_days=500)
```

4. **Analyze and backtest:**
```python
# Analyze indicator pairs
analyzer = PairAnalyzer(config)
results = analyzer.compare_indicator_pairs(indicators, price_changes)

# Generate signals
signal_gen = SignalGenerator(config)
signals = signal_gen.generate_signals(results, historical_data)

# Run backtest
engine = BacktestEngine(initial_capital=100000, ticker='AAPL')
engine.add_data(df, signals)
results = engine.run()
```

## Tips

- Start with the `basic_backtest.py` example to understand the workflow
- Adjust parameters in `config.py` or use a YAML config file
- Check the `logs/` directory for detailed execution logs
- Results are saved to `results/` in Excel format
- API calls are rate-limited by Alpha Vantage (5 calls/minute for free tier)

## Common Issues

**API Key Error:**
- Make sure you've set your Alpha Vantage API key in the config

**Import Errors:**
- Install the package: `pip install -e .` from project root
- Or add to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:/path/to/quantmodel2/src"`

**Data Fetching Slow:**
- Alpha Vantage free tier has rate limits
- Data is cached in the `data/` directory
- Consider using weekly data instead of daily for faster testing
