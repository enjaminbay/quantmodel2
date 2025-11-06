import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import gaussian_kde
from utils.logger import get_logger

logger = get_logger(__name__)


class BinStatistics:
    """Calculate and store statistics for indicator bin pairs"""

    def __init__(self, config):
        self.config = config
        self.min_samples = config.ML.min_samples
        self.min_correlation = config.ML.min_correlation
        self.max_p_value = config.ML.max_p_value
        self.confidence_level = config.ML.confidence_level

    # In statistics.py - modify the calculate_confidence_interval method

    def calculate_confidence_interval(self, returns: np.array) -> Optional[Dict]:
        """Calculate confidence intervals and distribution properties"""
        try:
            if len(returns) < self.min_samples:
                return None

            # Convert and clean data
            returns = np.array(returns, dtype=float)
            returns = returns[~np.isnan(returns)]
            returns = returns[~np.isinf(returns)]

            if len(returns) < self.min_samples:
                return None

            # Basic statistics
            n = len(returns)
            mean = np.mean(returns)
            std = np.std(returns, ddof=1)  # Using ddof=1 for sample standard deviation

            # Determine annualization factor based on timeframe
            if hasattr(self.config.BACKTEST, 'time_frame') and self.config.BACKTEST.time_frame == 'weekly':
                annualization_factor = 52  # 52 weeks in a year
            else:
                annualization_factor = 252  # ~252 trading days in a year

            # Calculate t-statistic and p-value
            if std > 0:
                t_stat = mean / (std / np.sqrt(n))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

                # Confidence interval
                t_value = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
                ci_margin = t_value * (std / np.sqrt(n))
                ci_lower = mean - ci_margin
                ci_upper = mean + ci_margin
            else:
                t_stat = 0.0
                p_value = 1.0
                ci_lower = mean
                ci_upper = mean

            # Rest of the method remains the same...

            # Distribution metrics
            if len(returns) > 2:  # Need at least 3 points for these
                skewness = float(stats.skew(returns))
                kurtosis = float(stats.kurtosis(returns))
            else:
                skewness = 0.0
                kurtosis = 0.0

            # Directional probability and risk metrics
            unique_returns = np.unique(returns)
            if len(unique_returns) > 5 and std > 0:
                try:
                    kde = gaussian_kde(returns)
                    prob_positive = float(1 - kde.integrate_box_1d(-np.inf, 0))

                    # Calculate VaR and ES
                    var_95 = float(np.percentile(returns, 5))
                    es_95 = float(np.mean(returns[returns <= var_95]))
                except Exception as e:
                    logger.warning(f"KDE calculation failed: {str(e)}")
                    prob_positive = float(np.mean(returns > 0))
                    var_95 = -1.645 * std if std > 0 else 0
                    es_95 = -2.062 * std if std > 0 else 0
            else:
                prob_positive = float(np.mean(returns > 0))
                var_95 = -1.645 * std if std > 0 else 0
                es_95 = -2.062 * std if std > 0 else 0

            stats_dict = {
                'mean': float(mean),
                'std': float(std),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'sample_size': int(n),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'prob_positive': float(prob_positive),
                'prob_negative': float(1 - prob_positive),
                'value_at_risk_95': float(var_95),
                'expected_shortfall_95': float(es_95),
                'direction': 'positive' if mean > 0 else 'negative'
            }

            # Validate output - no zeroes where they shouldn't be
            logger.debug(f"Calculated statistics: {stats_dict}")
            return stats_dict

        except Exception as e:
            logger.error(f"Error calculating confidence interval: {str(e)}")
            return None

    def rank_combinations(self, all_stats: Dict[Tuple[str, str], Dict]) -> Dict:
        """Rank bin combinations based on statistical significance and return distribution"""
        try:
            rankings = {}

            for pair, stats in all_stats.items():
                if stats is None or stats['data_count'] < self.min_samples:
                    continue

                # Calculate confidence interval and distribution stats
                ci_stats = self.calculate_confidence_interval(
                    np.array(stats['returns'])) if 'returns' in stats else None

                if ci_stats is None:
                    continue

                # Calculate core score components
                correlation = stats['correlation_weight']

                # Skip pairs with weak correlations (using absolute value)
                if abs(correlation) < self.min_correlation:
                    continue

                correlation_squared = correlation * correlation
                sample_factor = min(1.0, np.log(stats['data_count']) / np.log(50))

                # Calculate final score based on original formula
                score = correlation_squared * sample_factor

                rankings[pair] = {
                    'score': float(score),
                    'statistics': {
                        'correlation': float(correlation),
                        'correlation_squared': float(correlation_squared),
                        'sample_factor': float(sample_factor),
                        'sample_size': int(stats['data_count']),
                        'returns': stats['returns'],
                        'confidence_stats': ci_stats
                    }
                }

            # Sort by score
            sorted_rankings = dict(sorted(
                rankings.items(),
                key=lambda x: x[1]['score'],
                reverse=True
            ))

            return sorted_rankings

        except Exception as e:
            logger.error(f"Error ranking combinations: {str(e)}")
            raise

    def combine_distributions(self, distributions: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """Combine multiple return distributions using weighted KDE"""
        try:
            if not distributions:
                return None

            # Normalize weights if provided
            if weights is None:
                weights = [1 / len(distributions)] * len(distributions)
            else:
                weights = np.array(weights) / np.sum(weights)

            # Combine samples and their weights
            all_returns = []
            combined_weights = []

            for dist, weight in zip(distributions, weights):
                returns = dist['returns']
                n = len(returns)
                all_returns.extend(returns)
                combined_weights.extend([weight] * n)

            # Create weighted KDE
            kde = gaussian_kde(all_returns, weights=combined_weights)

            # Calculate weighted statistics
            weighted_mean = np.average(all_returns, weights=combined_weights)
            weighted_var = np.average((all_returns - weighted_mean) ** 2, weights=combined_weights)
            weighted_std = np.sqrt(weighted_var)

            return {
                'kde': kde,
                'mean': weighted_mean,
                'std': weighted_std,
                'total_samples': len(all_returns)
            }

        except Exception as e:
            logger.error(f"Error combining distributions: {str(e)}")
            return None