"""Base validators for the trading system."""
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from .exceptions import ValidationError
from .logger import get_logger

logger = get_logger(__name__)


def validate_dataframe(
        df: pd.DataFrame,
        required_columns: List[str],
        index_type: str = 'datetime'
) -> bool:
    """
    Validate a DataFrame meets basic requirements.

    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        index_type: Type of index required ('datetime' or 'numeric')

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            raise ValidationError("DataFrame is empty")

        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")

        # Validate index
        if index_type == 'datetime':
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValidationError("DataFrame index must be DatetimeIndex")
        elif index_type == 'numeric':
            if not pd.api.types.is_numeric_dtype(df.index):
                raise ValidationError("DataFrame index must be numeric")

        # Check for infinite values
        if np.any(np.isinf(df.select_dtypes(include=np.number))):
            raise ValidationError("DataFrame contains infinite values")

        return True

    except Exception as e:
        logger.error(f"DataFrame validation failed: {str(e)}")
        raise ValidationError(f"DataFrame validation failed: {str(e)}")


def validate_numeric_data(data: Union[Dict, pd.Series, np.ndarray]) -> bool:
    """
    Validate numeric data is properly formatted and contains valid values.

    Args:
        data: Numeric data to validate

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    try:
        if isinstance(data, dict):
            values = list(data.values())
        elif isinstance(data, (pd.Series, np.ndarray)):
            values = data.tolist()
        else:
            raise ValidationError(f"Unsupported data type: {type(data)}")

        # Check for non-numeric values
        if not all(isinstance(x, (int, float)) for x in values):
            raise ValidationError("Data contains non-numeric values")

        # Check for invalid values
        if any(np.isnan(values)) or any(np.isinf(values)):
            raise ValidationError("Data contains NaN or infinite values")

        return True

    except Exception as e:
        logger.error(f"Numeric data validation failed: {str(e)}")
        raise ValidationError(f"Numeric data validation failed: {str(e)}")


def validate_date_range(
        dates: pd.DatetimeIndex,
        min_periods: int = 2,
        max_gap_days: int = 5
) -> bool:
    """
    Validate a date range meets requirements.

    Args:
        dates: DatetimeIndex to validate
        min_periods: Minimum number of periods required
        max_gap_days: Maximum allowed gap between dates in days

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    try:
        # Check minimum periods
        if len(dates) < min_periods:
            raise ValidationError(f"Insufficient periods. Required: {min_periods}, Got: {len(dates)}")

        # Check for gaps
        date_diffs = dates[1:] - dates[:-1]
        max_gap = date_diffs.max()
        if max_gap.days > max_gap_days:
            raise ValidationError(f"Date gap of {max_gap.days} days exceeds maximum of {max_gap_days}")

        # Check for duplicates
        if dates.duplicated().any():
            raise ValidationError("Duplicate dates found")

        return True

    except Exception as e:
        logger.error(f"Date range validation failed: {str(e)}")
        raise ValidationError(f"Date range validation failed: {str(e)}")


def validate_configuration(config: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate configuration dictionary has required fields.

    Args:
        config: Configuration dictionary to validate
        required_fields: List of required field names

    Returns:
        bool: True if validation passes

    Raises:
        ValidationError: If validation fails
    """
    try:
        # Check required fields
        missing_fields = set(required_fields) - set(config.keys())
        if missing_fields:
            raise ValidationError(f"Missing required configuration fields: {missing_fields}")

        return True

    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise ValidationError(f"Configuration validation failed: {str(e)}")