# analysis/binning/stats_interface.py

from typing import Dict, List, Tuple
from utils.logger import get_logger
from utils.exceptions import StatisticsError
import numpy as np
import pandas as pd
import os

logger = get_logger(__name__)


class BinStatisticsInterface:
    """Interface for bin statistics calculations"""

    def __init__(self, config, statistics_calculator):
        self.config = config
        self.calculator = statistics_calculator

    def analyze_bin_combination(
            self,
            indicator1: str,
            bin1: str,
            indicator2: str,
            bin2: str,
            bin1_data: Dict[str, float],
            bin2_data: Dict[str, float],
            returns: Dict[str, float]
    ) -> Dict:
        """Analyze a specific bin combination"""
        try:
            stats = self.calculator.calculate_bin_pair_statistics(
                bin1_data,
                bin2_data,
                returns
            )

            if stats is None:
                logger.warning(
                    f"Insufficient data for {indicator1}:{bin1} - {indicator2}:{bin2}"
                )
                return None

            return {
                'indicators': (indicator1, indicator2),
                'bins': (bin1, bin2),
                'statistics': stats
            }

        except Exception as e:
            logger.error(f"Error in bin combination analysis: {str(e)}")
            raise StatisticsError(f"Failed to analyze bin combination: {str(e)}")

    def get_ranked_combinations(
            self,
            all_combinations: Dict[Tuple[str, str, str, str], Dict]
    ) -> List[Dict]:
        """Get ranked list of bin combinations based on directional prediction confidence"""
        try:
            ranked_list = []
            for combination_key, stats in all_combinations.items():
                ind1, ind2, bin1, bin2 = combination_key

                # Get core statistics
                correlation = stats.get('correlation_weight', 0)
                sample_size = stats.get('data_count', 0)
                confidence_stats = stats.get('confidence_stats', {})
                p_value = confidence_stats.get('p_value', 1.0)

                # Apply minimum thresholds
                if (sample_size >= self.calculator.min_samples and
                        abs(correlation) >= self.calculator.min_correlation and
                        p_value <= self.calculator.max_p_value):
                    # Calculate directional confidence score components:
                    statistical_significance = 1 - p_value  # How statistically significant is the relationship
                    win_rate = confidence_stats.get('prob_positive', 0.5)  # Historical accuracy at predicting direction
                    direction_consistency = abs(
                        win_rate - 0.5) * 2  # Convert to 0-1 scale, higher means more consistent direction

                    # Sample size reliability (scaled 0-1)
                    sample_factor = min(1.0, np.log(sample_size) / np.log(50)) if sample_size > 0 else 0

                    # Calculate risk-adjusted returns
                    mean_return = abs(confidence_stats.get('mean', 0))
                    std_dev = confidence_stats.get('std', 1)
                    sharpe = mean_return / std_dev if std_dev > 0 else 0

                    # Combined score weighted by importance
                    predictive_score = (
                            statistical_significance * 0.30 +  # Statistical validity
                            direction_consistency * 0.25 +  # Directional prediction consistency
                            abs(correlation) * 0.20 +  # Strength of relationship
                            sample_factor * 0.15 +  # Sample size reliability
                            sharpe * 0.10  # Risk-adjusted returns
                    )

                    ranked_list.append({
                        'indicators': (ind1, ind2),
                        'bins': (bin1, bin2),
                        'score': predictive_score,
                        'statistics': {
                            'correlation': correlation,
                            'sample_size': sample_size,
                            'returns': stats.get('returns', []),
                            'confidence_stats': confidence_stats,
                            'predictive_components': {
                                'statistical_significance': statistical_significance,
                                'direction_consistency': direction_consistency,
                                'correlation_strength': abs(correlation),
                                'sample_reliability': sample_factor,
                                'sharpe_ratio': sharpe
                            }
                        }
                    })

            # Sort by predictive score in descending order
            ranked_list.sort(key=lambda x: x['score'], reverse=True)

            return ranked_list

        except Exception as e:
            logger.error(f"Error ranking combinations: {str(e)}")
            raise

    def get_top_combinations(
            self,
            ranked_combinations: List[Dict],
            top_n: int = 10
    ) -> List[Dict]:
        """Get top N ranked combinations"""
        return ranked_combinations[:top_n]

    def save_significant_pairs(self, ranked_combinations: List[Dict], output_path: str, ticker: str) -> str:
        """Create and save a detailed report of significant pairs"""
        try:
            significant_pairs = []

            for combo in ranked_combinations:
                stats = combo['statistics']
                predictive_components = stats.get('predictive_components', {})
                confidence_stats = stats.get('confidence_stats', {})

                pair_data = {
                    'Indicators': f"{combo['indicators'][0]} - {combo['indicators'][1]}",
                    'Bins': f"{combo['bins'][0]} - {combo['bins'][1]}",
                    'Predictive Score': combo['score'],
                    'Statistical Significance': predictive_components.get('statistical_significance', 0),
                    'Direction Consistency': predictive_components.get('direction_consistency', 0),
                    'Correlation': stats['correlation'],
                    'Sample Size': stats['sample_size'],
                    'P-Value': confidence_stats.get('p_value', 1.0),
                    'Mean Return': confidence_stats.get('mean', 0),
                    'Std Dev': confidence_stats.get('std', 0),
                    'Sharpe Ratio': predictive_components.get('sharpe_ratio', 0),
                    'Win Rate': confidence_stats.get('prob_positive', 0),
                    'Value at Risk': confidence_stats.get('value_at_risk_95', 0),
                    'Expected Shortfall': confidence_stats.get('expected_shortfall_95', 0)
                }

                significant_pairs.append(pair_data)

            # Convert to DataFrame and sort by predictive score
            df = pd.DataFrame(significant_pairs)
            df = df.sort_values('Predictive Score', ascending=False)

            # Create output file path
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, f"{ticker}_significant_pairs.xlsx")

            # Save to Excel with formatting
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Significant Pairs', index=False)

                # Get workbook and worksheet objects
                workbook = writer.book
                worksheet = writer.sheets['Significant Pairs']

                # Add formats
                percent_format = workbook.add_format({'num_format': '0.00%'})
                score_format = workbook.add_format({'num_format': '0.0000'})

                # Set column formats
                worksheet.set_column('C:C', 12, score_format)  # Score
                worksheet.set_column('D:D', 12, score_format)  # Correlation
                worksheet.set_column('E:E', 12, score_format)  # P-Value
                worksheet.set_column('G:G', 12, percent_format)  # Mean Return
                worksheet.set_column('H:H', 12, percent_format)  # Win Rate
                worksheet.set_column('I:I', 12, percent_format)  # Std Dev
                worksheet.set_column('J:K', 12, percent_format)  # CI Lower/Upper
                worksheet.set_column('L:M', 12, percent_format)  # VaR and ES

                # Adjust column widths
                worksheet.set_column('A:B', 30)  # Indicators and Bins
                worksheet.set_column('C:N', 15)  # Numeric columns

            logger.info(f"Saved significant pairs report to: {output_file}")
            logger.info(f"Total significant pairs after strict filtering: {len(df)}")

            return output_file

        except Exception as e:
            logger.error(f"Error saving significant pairs: {str(e)}")
            raise
