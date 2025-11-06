# analysis/binning/pair_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from quantmodel.utils.logger import get_logger
from quantmodel.utils.exceptions import PairAnalysisError
from scipy import stats
from .processor import BinProcessor

logger = get_logger(__name__)


class PairAnalyzer:
    def __init__(self, config):
        self.config = config
        # Reduce minimum samples for weekly data
        if hasattr(config.BACKTEST, 'time_frame') and config.BACKTEST.time_frame == 'weekly':
            self.min_samples = 5  # Reduced from 10 for weekly data
        else:
            self.min_samples = 10

        self.significance_level = 0.05

        # Adjust minimum R² threshold for weekly data
        if hasattr(config.BACKTEST, 'time_frame') and config.BACKTEST.time_frame == 'weekly':
            self.min_rsquared = 0.15  # Increased threshold for weekly data
        else:
            self.min_rsquared = 0.10

        self.bin_processor = BinProcessor(config)

    def compare_indicator_pairs(self, data: Dict[str, Dict[str, float]], price_changes: Dict[str, float]) -> Dict:
        """Compare all pairs of indicators and their relationship with price changes"""
        try:
            results = {}
            indicators = sorted(list(data.keys()))
            pairs_analyzed = 0
            pairs_with_data = 0

            logger.info(f"Starting pair analysis with {len(indicators)} indicators")

            # Create all bins first to avoid duplicate processing
            indicator_bins = {}
            for ind in indicators:
                logger.debug(f"Creating bins for {ind}")
                bins = self.bin_processor.create_dynamic_bins(data[ind])
                if bins:
                    indicator_bins[ind] = bins

            # Process all pairs
            total_pairs = len(indicators) * (len(indicators) - 1) // 2

            for i in range(len(indicators)):
                for j in range(i + 1, len(indicators)):
                    ind1 = indicators[i]
                    ind2 = indicators[j]

                    if pairs_analyzed % 100 == 0:
                        logger.info(f"Processed {pairs_analyzed}/{total_pairs} pairs")

                    pair_key = f"{ind1}_{ind2}"

                    try:
                        if ind1 not in indicator_bins or ind2 not in indicator_bins:
                            logger.debug(f"Skipping pair {pair_key} - missing bins")
                            pairs_analyzed += 1
                            continue

                        # Compare bins and calculate gains
                        pair_results = self._analyze_bin_combination(
                            indicator_bins[ind1],
                            indicator_bins[ind2],
                            price_changes
                        )

                        if pair_results and any(bin1_data for bin1_data in pair_results.values()):
                            results[pair_key] = {
                                'analysis': pair_results,
                                'bins1': indicator_bins[ind1],
                                'bins2': indicator_bins[ind2]
                            }
                            pairs_with_data += 1

                    except Exception as e:
                        logger.error(f"Error processing pair {pair_key}: {str(e)}")

                    pairs_analyzed += 1

            logger.info(f"Analyzed {pairs_analyzed} pairs, {pairs_with_data} had valid data")
            return results

        except Exception as e:
            logger.error(f"Error comparing indicator pairs: {str(e)}")
            raise PairAnalysisError(f"Failed to analyze pairs: {str(e)}")

    def _analyze_bin_combination(self, bins1: Dict, bins2: Dict,
                                 price_changes: Dict[str, float]) -> Dict:
        """Analyze bin combinations with corrected win rate calculation"""
        try:
            analysis = {}

            for bin1_name, bin1_data in bins1.items():
                if not isinstance(bin1_data, dict) or 'values' not in bin1_data:
                    continue

                analysis[bin1_name] = {}

                for bin2_name, bin2_data in bins2.items():
                    if not isinstance(bin2_data, dict) or 'values' not in bin2_data:
                        continue

                    # Find common dates
                    common_dates = set(bin1_data['values'].keys()) & set(bin2_data['values'].keys())
                    common_dates = common_dates & set(price_changes.keys())

                    if len(common_dates) >= self.min_samples:
                        dates = sorted(common_dates)
                        returns = np.array([price_changes[d] for d in dates])

                        # Calculate basic statistics
                        mean_return = np.mean(returns)
                        std_return = np.std(returns)

                        if std_return > 0:
                            # Calculate t-statistic and p-value
                            t_stat = mean_return / (std_return / np.sqrt(len(returns)))
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(returns) - 1))

                            # Calculate win rate based on return direction
                            wins = np.sum(returns > 0)  # Count any positive return as a win
                            losses = np.sum(returns < 0)  # Count any negative return as a loss
                            win_rate = wins / len(returns) if len(returns) > 0 else 0.5

                            # Calculate R-squared based correlation
                            r_squared_weight = self._calculate_correlation_weight(
                                bin1_data['values'],
                                bin2_data['values'],
                                price_changes,
                                dates
                            )

                            # Store confidence stats directly in the analysis
                            confidence_stats = {
                                'mean': float(mean_return),
                                'std': float(std_return),
                                'sample_size': len(returns),
                                't_statistic': float(t_stat),
                                'p_value': float(p_value),
                                'direction': 'positive' if mean_return > 0 else 'negative'
                            }

                            analysis[bin1_name][bin2_name] = {
                                'average_gain': float(mean_return),
                                'volatility': float(std_return),
                                'data_count': len(returns),
                                'win_rate': float(win_rate),
                                'correlation_weight': float(r_squared_weight),
                                'returns': returns.tolist(),
                                'direction': 'negative' if mean_return < 0 else 'positive',
                                'wins': int(wins),
                                'losses': int(losses),
                                'confidence_stats': confidence_stats  # Add confidence stats here
                            }

                return analysis

        except Exception as e:
            logger.error(f"Error in bin combination analysis: {str(e)}")
            return {}

    def _calculate_correlation_weight(self, values1: Dict[str, float],
                                      values2: Dict[str, float],
                                      price_changes: Dict[str, float],
                                      dates: List[str]) -> float:
        """Calculate R-squared based correlation weight"""
        try:
            # Convert to numpy arrays
            arr1 = np.array([values1[d] for d in dates])
            arr2 = np.array([values2[d] for d in dates])
            price_chg = np.array([price_changes[d] for d in dates])

            # Quick validity check
            if len(arr1) < self.min_samples:
                return 0.0

            # Calculate R-squared values
            r_squared_values = []

            # Individual R² for each indicator
            for arr in [arr1, arr2]:
                if np.std(arr) > 0 and np.std(price_chg) > 0:
                    correlation = np.corrcoef(arr, price_chg)[0, 1]
                    if not np.isnan(correlation):
                        r_squared = correlation ** 2
                        if r_squared >= self.min_rsquared:
                            r_squared_values.append(r_squared)

            # Interaction R²
            if np.std(arr1) > 0 and np.std(arr2) > 0:
                # Multiplicative interaction
                interaction = arr1 * arr2
                if np.std(interaction) > 0:
                    corr = np.corrcoef(interaction, price_chg)[0, 1]
                    if not np.isnan(corr):
                        r_squared = corr ** 2
                        if r_squared >= self.min_rsquared:
                            r_squared_values.append(r_squared)

                # Additive interaction
                combined = arr1 + arr2
                if np.std(combined) > 0:
                    corr = np.corrcoef(combined, price_chg)[0, 1]
                    if not np.isnan(corr):
                        r_squared = corr ** 2
                        if r_squared >= self.min_rsquared:
                            r_squared_values.append(r_squared)

            if r_squared_values:
                # Take the maximum R² as our weight
                max_r_squared = max(r_squared_values)

                # Apply sample size adjustment
                sample_factor = min(1.0, len(arr1) / 50)

                # Calculate final weight
                final_weight = max_r_squared * sample_factor

                return float(final_weight)

            return 0.0

        except Exception as e:
            logger.error(f"Error calculating R-squared weight: {str(e)}")
            return 0.0

    def _interpret_rsquared(self, r_squared: float) -> str:
        """Interpret R-squared value"""
        if r_squared >= 0.70:
            return "Very strong relationship"
        elif r_squared >= 0.50:
            return "Strong relationship"
        elif r_squared >= 0.30:
            return "Moderate relationship"
        elif r_squared >= 0.10:
            return "Weak relationship"
        else:
            return "Very weak relationship"

