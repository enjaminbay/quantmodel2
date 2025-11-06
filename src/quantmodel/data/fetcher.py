# data/fetcher.py
import time

import requests
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union
from abc import ABC, abstractmethod
from quantmodel.utils.logger import get_logger
from quantmodel.utils.config import Config
from quantmodel.utils.exceptions import DataFetchError

logger = get_logger(__name__)


class AlphaVantageAPI:
    """AlphaVantage API implementation"""

    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.API.alpha_vantage_key
        self.base_url = 'https://www.alphavantage.co/query'

    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Make API request with retry logic"""
        try:
            logger.debug(f"Making API request with params: {params}")

            response = requests.get(self.base_url, params=params)
            data = response.json()
            if 'Error Message' in data:
                raise DataFetchError(f"API Error: {data['Error Message']}")

            if 'Note' in data:  # API limit warning
                logger.warning(f"API Note: {data['Note']}")

            return data

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise

    # In fetcher.py - modify the AlphaVantageAPI class

    # In fetcher.py - AlphaVantageAPI class
    def get_historical_data(self, ticker: str, time_frame: str = 'daily') -> Optional[Dict]:
        """Fetch historical stock data with timeframe support"""
        if time_frame == 'weekly':
            function = 'TIME_SERIES_WEEKLY_ADJUSTED'
            response_key = 'Weekly Adjusted Time Series'
        else:
            function = 'TIME_SERIES_DAILY_ADJUSTED'
            response_key = 'Time Series (Daily)'

        params = {
            'function': function,
            'symbol': ticker,
            'outputsize': 'full',
            'apikey': self.api_key
        }

        data = self._make_request(params)

        if data and response_key in data:
            return data[response_key]

        logger.warning(f"No historical {time_frame} data available for {ticker}")
        return None

    def get_technical_indicator(self, ticker: str, function: str,
                                period: Optional[int] = None,
                                series_type: str = 'close',
                                interval: str = 'daily') -> Optional[Dict]:
        """Fetch technical indicator data"""
        params = {
            'function': function,
            'symbol': ticker,
            'interval': interval,
            'series_type': series_type,
            'apikey': self.api_key
        }

        if period:
            params['time_period'] = period

        data = self._make_request(params)
        if data:
            tech_analysis_key = f'Technical Analysis: {function}'
            if tech_analysis_key in data:
                return data[tech_analysis_key]
        print(data)
        logger.warning(f"No data found for {function} indicator for {ticker}")
        return None


class DataFetcher:
    def __init__(self, config: Config):
        self.config = config
        self.data_source = AlphaVantageAPI(config)

    def STRATTEST(self, ticker: str, amount_of_trading_days: int, time_frame: str = 'daily') -> Optional[Dict]:
        """Fetch historical and technical indicator data for backtesting"""
        try:
            logger.info(f"Starting STRATTEST for {ticker}")

            # Fetch historical data
            try:
                price_data = self.data_source.get_historical_data(ticker)
            except:
                time.sleep(10)
                logger.warning(f"NO DATA FOR {ticker}")
                price_data = self.data_source.get_historical_data(ticker)

            if not price_data:
                logger.warning(f"No price data available for {ticker}")
                return None

            # Convert string prices to float and create clean price dictionary
            clean_prices = {}
            for date, daily_data in price_data.items():
                try:
                    clean_prices[date] = float(daily_data['5. adjusted close'])
                except (KeyError, ValueError) as e:
                    logger.warning(f"Error processing price for date {date}: {e}")
                    continue

            if not clean_prices:
                logger.error("No valid price data after cleaning")
                return None

            # Calculate FORWARD-looking percentage changes
            dates = sorted(clean_prices.keys())  # Sort chronologically
            changes = {}
            # Calculate next day's return for each day except the last
            for i in range(len(dates) - 1):
                current_date = dates[i]
                next_date = dates[i + 1]
                current_price = clean_prices[current_date]
                next_price = clean_prices[next_date]
                # Store the next day's return with the current date
                changes[current_date] = (next_price - current_price) / current_price
            changes[dates[-1]] = 0
            # Define indicator configurations
            indicators = {
                'SMA20': ('SMA', 20),
                'SMA50': ('SMA', 50),
                'SMA200': ('SMA', 200),
                'MACD': ('MACD', None),
                'RSI': ('RSI', 14),
                'MFI': ('MFI', 10),
                'BBANDS': ('BBANDS', 20),
                'OBV': ('OBV', None),
                'DX': ('DX', 14),
                'WMA20': ('WMA', 20),
                'WMA50': ('WMA', 50),
                'WMA200': ('WMA', 200),
                'T3': ('T3', 10),
                'STOCH': ('STOCH', None),
                'ADXR': ('ADXR', 10),
                'MOM': ('MOM', 10),
                'BOP': ('BOP', None),
                'CCI10': ('CCI', 10),
                'CCI50': ('CCI', 50),
                'CCI200': ('CCI', 200),
                'CMO': ('CMO', 10),
                'TRANGE': ('TRANGE', None),
                'ATR': ('ATR', 14)
            }

            # Get technical indicators
            indicator_data = {}
            for name, (function, period) in indicators.items():
                logger.debug(f"Fetching {name} indicator")
                data = self.data_source.get_technical_indicator(
                    ticker,
                    function,
                    period,
                    interval=time_frame
                )

                if data is not None:
                    # Clean indicator data
                    clean_indicator_data = {}
                    for date, value in data.items():
                        try:
                            # Handle different indicator data structures
                            if isinstance(value, dict):
                                # Some indicators return multiple values
                                main_value = next(iter(value.values()))
                                clean_indicator_data[date] = float(main_value)
                            else:
                                clean_indicator_data[date] = float(value)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error processing indicator {name} for date {date}: {e}")
                            continue

                    if clean_indicator_data:
                        indicator_data[name] = {'rawData': clean_indicator_data}
                        logger.debug(f"Processed {name} indicator data")

            # Use the amount_of_trading_days parameter to limit the data
            valid_dates = sorted(clean_prices.keys(), reverse=True)[:amount_of_trading_days]

            # Filter the changes to only include dates we have indicator data for
            filtered_changes = {date: changes[date] for date in valid_dates if date in changes}

            return {
                'Data': indicator_data,
                'Other': {
                    'TotalChanges': filtered_changes,
                    'TotalPrices': {'5. adjusted close': clean_prices}
                }
            }

        except Exception as e:
            logger.error(f"Error in STRATTEST for {ticker}: {str(e)}")
            return None