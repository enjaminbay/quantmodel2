from typing import Dict, List
import pandas as pd
import numpy as np
from analysis.binning.statistics import BinStatistics
from scipy import stats


class SignalFeatureGenerator:
    def __init__(self, config):
        self.config = config
        self.bin_statistics = BinStatistics(config)

    def create_feature_vector(self,
                              pair_results: Dict,
                              historical_data: pd.DataFrame) -> pd.DataFrame:
        """Convert pair analysis results into ML features"""
        features = []

        for pair_key, result in pair_results.items():
            # Extract base features from existing analysis
            base_features = self._extract_base_features(result)

            # Add statistical features
            stat_features = self._calculate_statistical_features(result)

            # Combine all features
            combined_features = {**base_features, **stat_features}
            features.append(combined_features)

        return pd.DataFrame(features)

    def _extract_base_features(self, result: Dict) -> Dict:
        """Extract primary features from analysis results"""
        return {
            'signal_strength': result['correlation_weight'],
            'win_rate': result['win_rate'],
            'volatility': result['volatility'],
            'sample_size': result['data_count']
        }

    def _calculate_statistical_features(self, result: Dict) -> Dict:
        """Calculate additional statistical features"""
        returns = np.array(result['returns'])

        return {
            'mean_return': np.mean(returns),
            'return_std': np.std(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        }