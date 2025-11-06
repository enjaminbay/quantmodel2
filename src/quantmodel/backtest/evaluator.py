from typing import Dict, List
import numpy as np
from scipy import stats
from quantmodel.utils.logger import get_logger

logger = get_logger(__name__)


class SignalEvaluator:
    """Evaluates trading signals based on probability distribution characteristics"""

    def __init__(self, config):
        self.config = config
        self.min_confidence = 0.95  # 95% confidence level
        self.min_samples = 30
        self.max_volatility = 2.0  # Maximum acceptable volatility ratio

    def evaluate_signals(self, signals: List[Dict]) -> List[Dict]:
        """Evaluate signal quality based on probability distributions"""
        try:
            evaluations = []

            for signal in signals:
                # Get distribution metrics
                stats = signal['probability_distribution']
                confidence = signal['confidence']

                # Calculate quality metrics
                sample_quality = self._evaluate_sample_quality(
                    signal['metrics']['active_pairs'],
                    signal['metrics'].get('total_samples', 0)
                )

                statistical_quality = self._evaluate_statistical_quality(
                    confidence['metrics']['mean_return'],
                    confidence['metrics'].get('lower_ci'),
                    confidence['metrics'].get('upper_ci'),
                    stats.get('skewness', 0),
                    stats.get('kurtosis', 0)
                )

                distribution_quality = self._evaluate_distribution_quality(stats)

                # Combine metrics into overall quality score
                quality_metrics = {
                    'sample_quality': sample_quality,
                    'statistical_quality': statistical_quality,
                    'distribution_quality': distribution_quality,
                    'overall_quality': self._calculate_overall_quality(
                        sample_quality,
                        statistical_quality,
                        distribution_quality
                    ),
                    'warning_flags': signal.get('metrics', {}).get('warning_flags', [])
                }

                evaluations.append(quality_metrics)

            return evaluations

        except Exception as e:
            logger.error(f"Error in signal evaluation: {str(e)}")
            raise

    def _evaluate_sample_quality(self, n_pairs: int, total_samples: int) -> float:
        """Evaluate quality based on sample size"""
        if total_samples < self.min_samples:
            return 0.0

        # Scale up to 1.0 based on sample size and number of pairs
        sample_score = min(1.0, np.log(total_samples) / np.log(100))
        pair_score = min(1.0, n_pairs / 10)

        return 0.6 * sample_score + 0.4 * pair_score

    def _evaluate_statistical_quality(self, mean: float, ci_lower: float,
                                      ci_upper: float, skewness: float,
                                      kurtosis: float) -> float:
        """Evaluate statistical properties of the signal"""
        try:
            # Check if CI bounds are on same side of zero
            ci_quality = 1.0 if (ci_lower > 0 or ci_upper < 0) else 0.0

            # Penalize extreme skewness and kurtosis
            moment_penalty = 1.0 / (1.0 + abs(skewness) + max(0, kurtosis - 3))

            # Signal strength relative to CI width
            precision = 1.0 / (1.0 + abs(ci_upper - ci_lower))

            return (0.4 * ci_quality +
                    0.3 * moment_penalty +
                    0.3 * precision)

        except Exception as e:
            logger.error(f"Error in statistical evaluation: {str(e)}")
            return 0.0

    def _evaluate_distribution_quality(self, stats: Dict) -> float:
        """Evaluate the quality of the probability distribution"""
        try:
            if not stats or 'kde' not in stats:
                return 0.0

            kde = stats['kde']

            # Evaluate smoothness of distribution
            x = np.linspace(kde.dataset.min(), kde.dataset.max(), 100)
            density = kde(x)
            smoothness = 1.0 / (1.0 + np.std(np.diff(density)))

            # Check for multimodality
            peaks = self._find_peaks(density)
            modality_score = 1.0 if len(peaks) == 1 else 0.5

            return 0.6 * smoothness + 0.4 * modality_score

        except Exception as e:
            logger.error(f"Error in distribution evaluation: {str(e)}")
            return 0.0

    def _find_peaks(self, density: np.ndarray) -> List[int]:
        """Find peaks in density estimation"""
        peaks = []
        for i in range(1, len(density) - 1):
            if density[i - 1] < density[i] > density[i + 1]:
                peaks.append(i)
        return peaks

    def _calculate_overall_quality(self, sample_quality: float,
                                   statistical_quality: float,
                                   distribution_quality: float) -> float:
        """Calculate overall signal quality score"""
        weights = {
            'sample': 0.3,
            'statistical': 0.4,
            'distribution': 0.3
        }

        overall = (weights['sample'] * sample_quality +
                   weights['statistical'] * statistical_quality +
                   weights['distribution'] * distribution_quality)

        return min(1.0, max(0.0, overall))