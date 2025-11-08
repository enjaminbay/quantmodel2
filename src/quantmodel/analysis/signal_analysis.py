from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from quantmodel.utils.logger import get_logger

logger = get_logger(__name__)


class SignalAnalyzer:
    """Analyzes and reports on trading signal performance and statistics."""

    def __init__(self):
        """Initialize the SignalAnalyzer."""
        pass

    def analyze_signals(self, signals: Dict) -> Dict:
        """
        Analyze signal statistics and return metrics.

        Args:
            signals: Dictionary of generated signals

        Returns:
            Dictionary containing analysis metrics
        """
        if not signals:
            logger.warning("No signals to analyze")
            return {}

        metrics = {
            'total_signals': len(signals),
            'active_signals': sum(1 for s in signals.values() if s['strength'] != 0),
            'signal_distribution': self._get_signal_distribution(signals),
            'confidence_metrics': self._get_confidence_metrics(signals),
            'statistical_metrics': self._get_statistical_metrics(signals)
        }

        return metrics

    def analyze_performance(self, df: pd.DataFrame) -> Dict:
        """
        Analyze signal performance against actual returns.

        Args:
            df: DataFrame with 'Signal' and 'Stock Pct Change' columns

        Returns:
            Dictionary containing performance metrics
        """
        if 'Signal' not in df.columns or 'Stock Pct Change' not in df.columns:
            logger.warning("Required columns not found in DataFrame")
            return {}

        performance = {}

        # Calculate performance by signal strength
        for strength in [-2, -1, 0, 1, 2]:
            mask = df['Signal'] == strength
            if mask.any():
                returns = df.loc[mask, 'Stock Pct Change']
                performance[strength] = {
                    'count': len(returns),
                    'mean_return': float(returns.mean()),
                    'std_dev': float(returns.std()),
                    'win_rate': float((returns > 0).mean()),
                    'max_gain': float(returns.max()),
                    'max_loss': float(returns.min())
                }

        return performance

    def print_statistics(self, signals: Dict):
        """Print signal statistics to console."""
        print_signal_statistics(signals)

    def print_performance(self, df: pd.DataFrame):
        """Print performance statistics to console."""
        print_success_statistics(df)

    def _get_signal_distribution(self, signals: Dict) -> Dict:
        """Get distribution of signal strengths."""
        distribution = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
        for s in signals.values():
            strength = s['strength']
            distribution[strength] = distribution.get(strength, 0) + 1
        return distribution

    def _get_confidence_metrics(self, signals: Dict) -> Dict:
        """Calculate confidence-related metrics."""
        confidences = [s['confidence']['overall'] for s in signals.values()]
        return {
            'mean': float(np.mean(confidences)),
            'median': float(np.median(confidences)),
            'std_dev': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }

    def _get_statistical_metrics(self, signals: Dict) -> Dict:
        """Calculate statistical significance metrics."""
        significance_scores = [
            s['confidence']['metrics'].get('statistical_significance', 0)
            for s in signals.values()
        ]
        agreement_scores = [
            s['confidence']['metrics'].get('signal_agreement', 0)
            for s in signals.values()
        ]
        accuracy_scores = [
            s['confidence']['metrics'].get('historical_accuracy', 0)
            for s in signals.values()
        ]

        return {
            'statistical_significance': {
                'mean': float(np.mean(significance_scores)),
                'max': float(np.max(significance_scores)),
                'min': float(np.min(significance_scores))
            },
            'signal_agreement': {
                'mean': float(np.mean(agreement_scores))
            },
            'historical_accuracy': {
                'mean': float(np.mean(accuracy_scores))
            }
        }


def print_signal_statistics(signals: Dict):
    """Enhanced signal statistics with focus on predictive metrics"""
    if not signals:
        print("\nNo signals generated")
        return

    print("\n=== Signal Predictive Analysis ===")

    # Statistical significance metrics
    significance_scores = [s['confidence']['metrics'].get('statistical_significance', 0)
                           for s in signals.values()]
    print(f"\nStatistical Significance:")
    print(f"Average: {np.mean(significance_scores):.4f}")
    print(f"Maximum: {np.max(significance_scores):.4f}")
    print(f"Minimum: {np.min(significance_scores):.4f}")

    # Signal agreement metrics
    agreement_scores = [s['confidence']['metrics'].get('signal_agreement', 0)
                        for s in signals.values()]
    print(f"\nSignal Agreement:")
    print(f"Average: {np.mean(agreement_scores):.4f}")

    # Historical accuracy metrics
    accuracy_scores = [s['confidence']['metrics'].get('historical_accuracy', 0)
                       for s in signals.values()]
    print(f"\nHistorical Accuracy:")
    print(f"Average: {np.mean(accuracy_scores):.4f}")

    # Confidence distribution
    confidences = [s['confidence']['overall'] for s in signals.values()]
    print(f"\nOverall Confidence Distribution:")
    print(f"Average: {np.mean(confidences):.4f}")
    print(f"Median: {np.median(confidences):.4f}")
    print(f"Std Dev: {np.std(confidences):.4f}")

    # Signal strength distribution with predictive metrics
    strength_groups = {}
    for s in signals.values():
        strength = s['strength']
        if strength not in strength_groups:
            strength_groups[strength] = []
        strength_groups[strength].append(s['confidence']['metrics'])

    print("\nPredictive Metrics by Signal Strength:")
    for strength, metrics in strength_groups.items():
        print(f"\nStrength {strength}:")
        print(f"   Count: {len(metrics)}")
        print(
            f"   Avg Statistical Significance: {np.mean([m.get('statistical_significance', 0) for m in metrics]):.4f}")
        print(f"   Avg Historical Accuracy: {np.mean([m.get('historical_accuracy', 0) for m in metrics]):.4f}")

def print_success_statistics(df: pd.DataFrame):
    """Print statistics about trading success based on signals"""
    if 'Signal' not in df.columns or 'Stock Pct Change' not in df.columns:
        print("\nNo signal performance data available")
        return

    print("\n=== Signal Performance Statistics ===")

    # Group by signal strength
    signal_map = {
        2: 'Strong Buy',
        1: 'Buy',
        0: 'Neutral',
        -1: 'Sell',
        -2: 'Strong Sell'
    }

    df['SignalType'] = df['Signal'].map(signal_map)

    # Calculate performance metrics
    performance = df.groupby('SignalType')['Stock Pct Change'].agg([
        ('Count', 'count'),
        ('Mean Return', 'mean'),
        ('Std Dev', 'std'),
        ('Win Rate', lambda x: (x > 0).mean()),
        ('Max Gain', 'max'),
        ('Max Loss', 'min')
    ]).round(4)

    print("\nPerformance by Signal Type:")
    print(performance)

    # Calculate aggregate statistics
    total_signals = len(df[df['Signal'] != 0])
    successful_signals = len(df[(df['Signal'] != 0) & (df['Stock Pct Change'] > 0)])
    avg_return = df[df['Signal'] != 0]['Stock Pct Change'].mean()

    print("\nAggregate Statistics:")
    print(f"Total Active Signals: {total_signals}")
    print(
        f"Overall Success Rate: {(successful_signals / total_signals * 100):.1f}%" if total_signals > 0 else "No active signals")
    print(f"Average Return: {avg_return * 100:.2f}%" if total_signals > 0 else "No returns data")