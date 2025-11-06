import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import os
from utils.logger import get_logger

logger = get_logger(__name__)



def save_analysis_to_excel(processed_df: pd.DataFrame, signals: Dict,
                           pair_results: Dict, ranked_combinations: List,
                           output_dir: str, ticker: str) -> str:
    """Enhanced Excel export with focus on predictive metrics"""
    try:
        output_file = os.path.join(
            output_dir,
            f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        )

        with pd.ExcelWriter(output_file) as writer:
            # Predictive Metrics Sheet (New)
            predictive_metrics = []
            for combo in ranked_combinations:
                stats = combo['statistics']
                pred_comp = stats.get('predictive_components', {})
                conf_stats = stats.get('confidence_stats', {})

                predictive_metrics.append({
                    'Pair': f"{combo['indicators'][0]} - {combo['indicators'][1]}",
                    'Bins': f"{combo['bins'][0]} - {combo['bins'][1]}",
                    'Predictive Score': combo['score'],
                    'Statistical Significance': pred_comp.get('statistical_significance', 0),
                    'Direction Consistency': pred_comp.get('direction_consistency', 0),
                    'Correlation Strength': abs(stats['correlation']),
                    'Sample Reliability': pred_comp.get('sample_reliability', 0),
                    'Sharpe Ratio': pred_comp.get('sharpe_ratio', 0),
                    'P-Value': conf_stats.get('p_value', 1.0),
                    'Sample Size': stats['sample_size'],
                    'Mean Return': conf_stats.get('mean', 0),
                    'Win Rate': conf_stats.get('prob_positive', 0)
                })

            pd.DataFrame(predictive_metrics).to_excel(
                writer,
                sheet_name='Predictive_Metrics',
                index=False
            )

            # Signal Performance Sheet (Enhanced)
            signals_df = pd.DataFrame([{
                'Date': date,
                'Strength': s['strength'],
                'Expected Return': s['expected_return'],
                'Actual Return': s['actual_return'],
                'Overall Confidence': s['confidence']['overall'],
                'Statistical Significance': s['confidence']['metrics'].get('statistical_significance', 0),
                'Signal Agreement': s['confidence']['metrics'].get('signal_agreement', 0),
                'Historical Accuracy': s['confidence']['metrics'].get('historical_accuracy', 0),
                'Active Pairs': s['active_pairs']
            } for date, s in signals.items()])

            signals_df.to_excel(writer, sheet_name='Signal_Performance', index=False)

            # Original data and other sheets
            processed_df.to_excel(writer, sheet_name='Processed_Data')

            # Add summary statistics sheet
            summary_stats = pd.DataFrame([{
                'Metric': 'Average Predictive Score',
                'Value': np.mean([c['score'] for c in ranked_combinations])
            }, {
                'Metric': 'Average Statistical Significance',
                'Value': np.mean([c['statistics'].get('predictive_components', {}).get(
                    'statistical_significance', 0) for c in ranked_combinations])
            }, {
                'Metric': 'Average P-Value',
                'Value': np.mean([c['statistics'].get('confidence_stats', {}).get(
                    'p_value', 1.0) for c in ranked_combinations])
            }])

            summary_stats.to_excel(writer, sheet_name='Summary_Stats', index=False)

        return output_file

    except Exception as e:
        logger.error(f"Error saving analysis to Excel: {str(e)}")
        raise
