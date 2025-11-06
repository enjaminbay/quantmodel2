# data/synchronizer.py

from typing import Dict, TypedDict, Tuple
import pandas as pd
from utils.exceptions import DataSyncError
from utils.logger import get_logger

logger = get_logger(__name__)

class IndicatorData(TypedDict):
    rawData: Dict[str, float]

class MarketData(TypedDict):
    Data: Dict[str, IndicatorData]
    Other: Dict[str, Dict[str, float]]

class DataSynchronizer:
    """Handles synchronization of multiple data series."""

    def __init__(self):
        self.date_format = '%Y-%m-%d'

    def synchronize_data(self, data: MarketData) -> MarketData:
        """Synchronize all data series using DataFrame operations."""
        try:
            # Convert to DataFrames for processing
            indicator_df, price_df = self._convert_to_dataframes(data)
            logger.info(f"Initial shapes - Indicators: {indicator_df.shape}, Prices: {price_df.shape}")

            # Combine DataFrames to get common dates
            combined_df = pd.concat([indicator_df, price_df], axis=1)
            combined_df = combined_df.dropna()
            logger.info(f"After synchronization - Shape: {combined_df.shape}")

            if combined_df.empty:
                raise DataSyncError("No common dates found after synchronization")

            # Split back into components and convert to original format
            return self._convert_from_dataframes(
                indicator_df=combined_df[indicator_df.columns],
                price_df=combined_df[price_df.columns]
            )

        except Exception as e:
            logger.error(f"Error in data synchronization: {str(e)}")
            raise DataSyncError(f"Failed to synchronize data: {str(e)}")

    def _convert_to_dataframes(self, data: MarketData) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert dictionary data structures to DataFrames."""
        try:
            # Convert indicator data
            indicator_data = {
                indicator: pd.Series(data['rawData'], name=indicator)
                for indicator, data in data['Data'].items()
            }
            indicator_df = pd.DataFrame(indicator_data)
            indicator_df.index = pd.to_datetime(indicator_df.index)

            # Convert price data
            price_df = pd.DataFrame({
                'Price': pd.Series(data['Other']['TotalPrices']['5. adjusted close']),
                'Returns': pd.Series(data['Other']['TotalChanges'])
            })
            price_df.index = pd.to_datetime(price_df.index)

            return indicator_df, price_df

        except Exception as e:
            logger.error(f"Error converting to DataFrames: {str(e)}")
            raise DataSyncError(f"Failed to convert data to DataFrames: {str(e)}")

    def _convert_from_dataframes(
        self,
        indicator_df: pd.DataFrame,
        price_df: pd.DataFrame
    ) -> MarketData:
        """Convert DataFrames back to the original dictionary structure."""
        try:
            # Convert indicator DataFrame
            indicator_dict = {
                column: {'rawData': series.dropna().apply(float).to_dict()}
                for column, series in indicator_df.items()
            }

            # Convert price DataFrame
            price_dict = {
                'TotalPrices': {
                    '5. adjusted close': price_df['Price'].dropna().apply(float).to_dict()
                },
                'TotalChanges': price_df['Returns'].dropna().apply(float).to_dict()
            }

            return MarketData(
                Data=indicator_dict,
                Other=price_dict
            )

        except Exception as e:
            logger.error(f"Error converting from DataFrames: {str(e)}")
            raise DataSyncError(f"Failed to convert DataFrames to dictionary format: {str(e)}")

    def verify_data_integrity(self, data: MarketData) -> None:
        """Verify data integrity after synchronization."""
        try:
            # Check data lengths
            indicator_lengths = [len(ind['rawData']) for ind in data['Data'].values()]
            price_length = len(data['Other']['TotalChanges'])

            if not all(length == price_length for length in indicator_lengths):
                raise DataSyncError(
                    f"Data length mismatch: Indicators: {indicator_lengths}, Prices: {price_length}"
                )

            # Check date alignment
            indicator_dates = set(next(iter(data['Data'].values()))['rawData'].keys())
            price_dates = set(data['Other']['TotalChanges'].keys())

            if indicator_dates != price_dates:
                raise DataSyncError("Date mismatch between indicators and prices")

        except Exception as e:
            logger.error(f"Error in data integrity verification: {str(e)}")
            raise DataSyncError(f"Failed to verify data integrity: {str(e)}")