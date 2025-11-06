# data/processor.py
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from utils.logger import get_logger
from utils.exceptions import DataProcessingError

logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, config):
        self.config = config

    def _calculate_derivatives(self, data: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate derivatives and acceleration with validation"""
        try:
            # Calculate first derivative (rate of change)
            derivative = data.diff()

            # Calculate second derivative (acceleration)
            acceleration = derivative.diff()

            # Log some statistics
            logger.debug(f"\nDerivative calculations for {data.name}:")
            logger.debug(f"Original data range: [{data.min():.4f}, {data.max():.4f}]")
            logger.debug(f"Derivative range: [{derivative.min():.4f}, {derivative.max():.4f}]")
            logger.debug(f"Acceleration range: [{acceleration.min():.4f}, {acceleration.max():.4f}]")

            return derivative, acceleration

        except Exception as e:
            logger.error(f"Error calculating derivatives: {str(e)}")
            return pd.Series(), pd.Series()

    # In processor.py - modify the _clean_data method

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate processed data with timeframe awareness"""
        try:
            # Handle differently based on timeframe
            is_weekly = hasattr(self.config.BACKTEST, 'time_frame') and self.config.BACKTEST.time_frame == 'weekly'

            # For weekly data, we expect larger gaps, so we're more careful with filling
            if is_weekly:
                # Only fill gaps up to 2 weeks
                df = df.fillna(method='ffill', limit=2).fillna(method='bfill', limit=2)
            else:
                # Standard handling for daily data
                df = df.fillna(method='ffill').fillna(method='bfill')

            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()

            # Ensure chronological order (oldest to newest)
            df = df.sort_index(ascending=True)

            # Log data cleaning statistics
            logger.debug("\nData cleaning statistics:")
            logger.debug(f"Final shape: {df.shape}")
            logger.debug("Column ranges:")
            for col in df.columns:
                logger.debug(f"{col}: [{df[col].min():.4f}, {df[col].max():.4f}]")

            return df

        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise DataProcessingError(f"Failed to clean data: {str(e)}")

    def process_data(self, market_data: Dict) -> pd.DataFrame:
        """Convert synchronized market data into analysis-ready DataFrame"""
        try:
            # Convert price data to DataFrame
            prices = market_data['Other']['TotalPrices']['5. adjusted close']

            # First, create series of chronological prices
            dates = sorted(prices.keys())  # Oldest to newest
            price_series = pd.Series(prices)[dates]

            # Create DataFrame with chronological order (oldest to newest)
            df = pd.DataFrame(index=pd.to_datetime(dates))

            # Add price data (unshifted)
            df['Stock Price'] = price_series

            # Calculate forward-looking returns (only shift this)
            df['Stock Pct Change'] = price_series.pct_change().shift(-1)  # Only this should be shifted

            # Process each base indicator (do NOT shift these)
            for indicator_name, indicator_data in market_data['Data'].items():
                logger.debug(f"\nProcessing indicator: {indicator_name}")

                # Get base values
                base_values = pd.Series(indicator_data['rawData'])[dates]
                base_values.name = indicator_name  # Set name for logging
                df[indicator_name] = base_values  # These stay aligned with their original dates

                # Calculate and add derivatives with logging
                derivative, acceleration = self._calculate_derivatives(base_values)
                if not derivative.empty:
                    df[f"{indicator_name}_derivative"] = derivative
                    logger.debug(f"Added derivative for {indicator_name}")
                if not acceleration.empty:
                    df[f"{indicator_name}_acceleration"] = acceleration
                    logger.debug(f"Added acceleration for {indicator_name}")

            # Clean and validate the data
            logger.info("Cleaning and validating data...")
            df = self._clean_data(df)

            # Log final dataframe info
            logger.info("\nFinal DataFrame Info:")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            logger.info(f"Number of indicators: {len(df.columns) - 2}")  # Subtract price and returns

            return df

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise DataProcessingError(f"Failed to process data: {str(e)}")
