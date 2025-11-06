"""
Bin processing for analysis module.
"""

import numpy as np
from typing import Dict, Optional
from quantmodel.utils.logger import get_logger

logger = get_logger(__name__)


class BinProcessor:
    """Handles binning of indicator data for analysis."""

    def __init__(self, config):
        self.config = config
        self.min_bin_size = 5
        self.max_bins = 10

    def create_dynamic_bins(self, data: Dict[str, float]) -> Optional[Dict]:
        """
        Create dynamic bins from indicator data.

        Args:
            data: Dictionary mapping dates to indicator values

        Returns:
            Dictionary of bin definitions or None if insufficient data
        """
        try:
            if not data:
                return None

            values = np.array(list(data.values()))

            # Remove NaN values
            values = values[~np.isnan(values)]

            if len(values) < self.min_bin_size:
                logger.debug(f"Insufficient data for binning: {len(values)} values")
                return None

            # Calculate quartile-based bins
            bins = {}
            quartiles = np.percentile(values, [0, 25, 50, 75, 100])

            for i in range(len(quartiles) - 1):
                bins[f'bin_{i}'] = {
                    'bounds': (quartiles[i], quartiles[i + 1]),
                    'lower': quartiles[i],
                    'upper': quartiles[i + 1],
                    'label': f'Q{i + 1}'
                }

            return bins

        except Exception as e:
            logger.error(f"Error creating dynamic bins: {str(e)}")
            return None
