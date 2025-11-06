import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Optional, Union, Dict, List, Tuple


def analyze_indicator_stats(df: pd.DataFrame, indicator: str):
    """Analyze and print statistics for an indicator"""
    stats = {
        'mean': df[indicator].mean(),
        'std': df[indicator].std(),
        'min': df[indicator].min(),
        'max': df[indicator].max(),
        'null_count': df[indicator].isnull().sum()
    }

    print(f"\nIndicator: {indicator}")
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Std Dev: {stats['std']:.4f}")
    print(f"Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"Null Values: {stats['null_count']}")

def analyze_bin_distribution(bins: Dict):
    """Analyze and print bin distribution in a simplified format"""
    total_points = sum(len(bin_data['values']) for bin_data in bins.values())

    print("\nBin Distribution:")
    for bin_name, bin_data in bins.items():
        count = len(bin_data['values'])
        percentage = (count / total_points) * 100
        lower, upper = bin_data['bounds']
        print(f"{bin_name:12s}: {count:5d} points ({percentage:5.1f}%) [{lower:8.4f}, {upper:8.4f}]")


def analyze_pair_results(pair_results: Dict, top_n: int = 5):
    """Analyze and print pair analysis results"""
    analyzed_pairs = []
    for pair_key, result in pair_results.items():
        if 'analysis' not in result:
            continue

        for bin1, bin2_data in result['analysis'].items():
            for bin2, stats in bin2_data.items():
                # Extract p-value from confidence_stats if available
                p_value = stats.get('confidence_stats', {}).get('p_value', 1.0)

                analyzed_pairs.append({
                    'pair': pair_key,
                    'bin_combo': f"{bin1}-{bin2}",
                    'avg_gain': stats['average_gain'],
                    'volatility': stats['volatility'],
                    'win_rate': stats['win_rate'],
                    'count': stats['data_count'],
                    'correlation': stats['correlation_weight'],
                    'p_value': p_value  # Add p-value to the dictionary
                })

    df_pairs = pd.DataFrame(analyzed_pairs)
    metrics = ['avg_gain', 'win_rate', 'correlation', 'p_value']

    for metric in metrics:
        print(f"\nTop {top_n} pairs by {metric}:")
        # For p-value we want smallest values, for others we want largest
        if metric == 'p_value':
            top_pairs = df_pairs.nsmallest(top_n, metric)
        else:
            top_pairs = df_pairs.nlargest(top_n, metric)
        print(top_pairs[['pair', 'bin_combo', metric, 'count']].to_string())