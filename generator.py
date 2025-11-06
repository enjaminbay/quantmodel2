from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy.stats import norm
from utils.logger import get_logger
import os
import pickle
from datetime import datetime

logger = get_logger(__name__)


class SignalGenerator:
    def __init__(self, config):
        self.min_samples = config.ML.min_samples
        self.min_correlation = config.ML.min_correlation
        self.max_p_value = config.ML.max_p_value

        # Adjust thresholds based on timeframe
        if hasattr(config.BACKTEST, 'time_frame') and config.BACKTEST.time_frame == 'weekly':
            self.signal_thresholds = {
                'strong': 0.02,  # 2% for weekly (up from 1%)
                'weak': 0.005  # 0.5% for weekly (up from 0.001%)
            }
        else:
            self.signal_thresholds = {
                'strong': 0.01,
                'weak': 0.00001
            }

    def _calculate_signal_strength(self, expected_return: float) -> int:
            """Calculate signal strength based on expected return thresholds"""
            if abs(expected_return) < self.signal_thresholds['weak']:
                return 0
            elif abs(expected_return) > self.signal_thresholds['strong']:
                return 2 if expected_return > 0 else -2
            else:
                return 1 if expected_return > 0 else -1

    def _calculate_directional_consistency(self, returns: np.array) -> float:
        """Calculate directional consistency from historical returns"""
        if len(returns) < self.min_samples:
            return 0.0

        # Calculate proportion of positive returns
        positive_ratio = np.mean(returns > 0)

        # Convert to -1 to 1 scale where 0.5 is random
        consistency = abs(positive_ratio - 0.5) * 2

        return float(consistency)

    def generate_signal(self, active_pairs: List[Dict]) -> Dict:
        """Generate signal from active pairs using predictive scoring"""
        if not active_pairs:
            return {
                'strength': 0,
                'expected_return': 0.0,
                'confidence': {
                    'overall': 0.0,
                    'metrics': {'total_weight': 0.0, 'active_pairs': 0, 'win_rate': 0.0}
                }
            }

        # Calculate predictive scores for each pair
        pair_scores = []
        for pair in active_pairs:
            # Statistical validity (from p-value)
            statistical_significance = 1 - pair.get('p_value', 1.0)

            # Historical accuracy
            win_rate = pair.get('win_rate', 0.5)

            # Calculate directional consistency
            returns = np.array(pair.get('returns', []))
            direction_consistency = self._calculate_directional_consistency(returns)

            # Correlation strength
            correlation_strength = abs(pair['correlation'])

            # Sample reliability
            sample_size = pair.get('samples', 0)
            sample_reliability = min(1.0, np.log(max(sample_size, 1)) / np.log(50))

            # Calculate composite score
            predictive_score = (
                    statistical_significance * 0.30 +  # Statistical validity
                    direction_consistency * 0.25 +  # Directional prediction accuracy
                    correlation_strength * 0.20 +  # Relationship strength
                    sample_reliability * 0.15 +  # Sample size reliability
                    win_rate * 0.10  # Historical success rate
            )

            pair_scores.append({
                'pair': pair,
                'score': predictive_score,
                'components': {
                    'statistical_significance': statistical_significance,
                    'direction_consistency': direction_consistency,
                    'correlation_strength': correlation_strength,
                    'sample_reliability': sample_reliability,
                    'win_rate': win_rate
                }
            })

        # Calculate weighted expected return
        total_score = sum(ps['score'] for ps in pair_scores)
        if total_score == 0:
            return {'strength': 0, 'expected_return': 0.0}

        weighted_return = sum(
            ps['pair']['expected_return'] * ps['score']
            for ps in pair_scores
        ) / total_score

        # Calculate strength based on weighted return only
        strength = self._calculate_signal_strength(weighted_return)

        # Calculate overall confidence metrics
        avg_components = {
            'statistical_significance': np.mean(
                [ps['components']['statistical_significance'] for ps in pair_scores]),
            'direction_consistency': np.mean([ps['components']['direction_consistency'] for ps in pair_scores]),
            'correlation_strength': np.mean([ps['components']['correlation_strength'] for ps in pair_scores]),
            'sample_reliability': np.mean([ps['components']['sample_reliability'] for ps in pair_scores]),
            'win_rate': np.mean([ps['components']['win_rate'] for ps in pair_scores])
        }

        overall_confidence = (
                avg_components['statistical_significance'] * 0.30 +
                avg_components['direction_consistency'] * 0.25 +
                avg_components['correlation_strength'] * 0.20 +
                avg_components['sample_reliability'] * 0.15 +
                avg_components['win_rate'] * 0.10
        )

        return {
            'strength': strength,
            'expected_return': weighted_return,
            'confidence': {
                'overall': overall_confidence,
                'metrics': {
                    'total_weight': total_score,
                    'active_pairs': len(active_pairs),
                    'components': avg_components
                }
            },
            'pair_scores': [{
                'pair': ps['pair']['pair_key'],
                'score': ps['score'],
                'components': ps['components']
            } for ps in pair_scores]
        }

    def get_latest_signal(self, ticker: str, model_file: str) -> Optional[Dict]:
            try:
                # Load model and validate ticker
                loaded_ticker, significant_pairs = self.load_signal_model(model_file)
                if loaded_ticker.upper() != ticker.upper():
                    logger.warning(f"Model ticker mismatch: {loaded_ticker} != {ticker}")
                    return None

                # Get current active pairs
                active_pairs = self._get_active_pairs(significant_pairs, ticker)
                if not active_pairs:
                    return None

                # Generate signal from active pairs
                signal = self.generate_signal(active_pairs)

                return {
                    'ticker': ticker,
                    'signal': signal,
                    'active_pairs': active_pairs
                }

            except Exception as e:
                logger.error(f"Error getting latest signal: {str(e)}")
                return None

    def _get_significant_pairs(self, pair_results: Dict) -> List[Dict]:
        """Identify pairs with significant correlations and enough samples"""
        significant = []

        for pair_key, result in pair_results.items():
            if 'analysis' not in result or not result['analysis']:
                continue

            bins1 = result.get('bins1', {})
            bins2 = result.get('bins2', {})

            for bin1, bin2_data in result['analysis'].items():
                for bin2, stats in bin2_data.items():
                    # Get p-value from confidence stats
                    confidence_stats = stats.get('confidence_stats', {})
                    p_value = confidence_stats.get('p_value', 1.0)

                    # Add p-value threshold filter
                    if (stats['data_count'] >= self.min_samples and
                            abs(stats['correlation_weight']) >= self.min_correlation and
                            p_value <= self.max_p_value):  # Only include statistically significant pairs

                        win_rate = stats.get('win_rate', 0.5)
                        win_rate_confidence = abs(win_rate - 0.5) * 2
                        statistical_confidence = 1 - (p_value / self.max_p_value)

                        significant.append({
                            'pair_key': pair_key,
                            'bins': (bin1, bin2),
                            'bin_definitions': {
                                'bin1': bins1.get(bin1, {}),
                                'bin2': bins2.get(bin2, {})
                            },
                            'correlation': stats['correlation_weight'],
                            'expected_return': stats['average_gain'],
                            'volatility': stats['volatility'],
                            'samples': stats['data_count'],
                            'win_rate': win_rate,
                            'win_rate_confidence': win_rate_confidence,
                            'p_value': p_value,
                            'statistical_confidence': statistical_confidence
                        })

        # Sort by statistical significance (p-value) and correlation
        significant.sort(key=lambda x: (x['p_value'], -abs(x['correlation'])))
        return significant

    def _calculate_confidence(self, active_signals: List[Dict], total_pairs: int) -> Dict:
        """
        Calculate comprehensive signal confidence using multiple factors

        Confidence Components:
        1. Statistical significance (from p-values)
        2. Signal agreement (directional consistency)
        3. Correlation strength
        4. Historical accuracy
        5. Sample adequacy
        """
        if not active_signals:
            return {
                'overall': 0.0,
                'metrics': {
                    'statistical_significance': 0.0,
                    'signal_agreement': 0.0,
                    'correlation_strength': 0.0,
                    'historical_accuracy': 0.0,
                    'coverage': 0.0
                }
            }

        # 1. Statistical Significance (from p-values)
        # Transform p-values to confidence level (e.g., p=0.05 -> 0.95 confidence)
        statistical_significance = np.mean([
            max(0, 1 - p['p_value']) for p in active_signals
        ])

        # 2. Signal Agreement
        # How well signals agree on direction
        expected_returns = [p['expected_return'] for p in active_signals]
        signal_directions = np.sign(expected_returns)
        signal_agreement = abs(np.mean(signal_directions))  # 1.0 if perfect agreement, 0.0 if split

        # 3. Correlation Strength
        # Use absolute correlation values to measure relationship strength
        correlation_strength = np.mean([
            abs(p['correlation']) for p in active_signals
        ])

        # 4. Historical Accuracy
        # Consider both win rate and consistency
        win_rates = [p['win_rate'] for p in active_signals]
        historical_accuracy = np.mean(win_rates) if win_rates else 0.0

        # 5. Coverage
        # What proportion of significant pairs are active
        coverage = len(active_signals) / total_pairs if total_pairs > 0 else 0.0

        # Weight factors based on importance
        # Prioritize statistical significance and historical accuracy
        weights = {
            'statistical_significance': 0.30,  # Most important - based on p-values
            'historical_accuracy': 0.25,  # Past performance
            'signal_agreement': 0.20,  # Directional consensus
            'correlation_strength': 0.15,  # Strength of relationships
            'coverage': 0.10  # Breadth of signals
        }

        # Calculate weighted overall confidence
        metrics = {
            'statistical_significance': float(statistical_significance),
            'signal_agreement': float(signal_agreement),
            'correlation_strength': float(correlation_strength),
            'historical_accuracy': float(historical_accuracy),
            'coverage': float(coverage)
        }

        overall_confidence = sum(
            metrics[key] * weight
            for key, weight in weights.items()
        )

        # Add detail to metrics
        metrics.update({
            'active_signals': len(active_signals),
            'total_pairs': total_pairs,
            'mean_p_value': float(np.mean([p['p_value'] for p in active_signals])),
            'min_p_value': float(min(p['p_value'] for p in active_signals)),
            'mean_correlation': float(np.mean([p['correlation'] for p in active_signals])),
            'mean_expected_return': float(np.mean(expected_returns)),
            'signal_direction_consensus': float(signal_agreement)
        })

        return {
            'overall': float(overall_confidence),
            'metrics': metrics
        }

    def _generate_daily_signal(self, date: pd.Timestamp, significant_pairs: List[Dict],
                               historical_data: pd.DataFrame, pair_results: Dict) -> Dict:
        """Generate signal for a single day by combining active significant pairs"""
        try:
            active_signals = []
            date_data = historical_data.loc[date]

            for pair in significant_pairs:
                ind1, ind2 = self._split_pair_key(pair['pair_key'])
                if ind1 in date_data and ind2 in date_data:
                    val1, val2 = date_data[ind1], date_data[ind2]
                    # Only consider pairs where individual p-value meets threshold
                    if (self._check_bin_match(val1, val2, pair['bin_definitions']) and
                            pair['p_value'] <= self.max_p_value):
                        active_signals.append(pair)

            if not active_signals:
                return self._create_neutral_signal(date)

            # Calculate weights only for statistically significant signals
            total_weight = sum(
                abs(p['correlation']) * p['win_rate_confidence'] * p['statistical_confidence']
                for p in active_signals
            )

            if total_weight == 0:
                return self._create_neutral_signal(date)

            weighted_return = sum(
                p['expected_return'] * abs(p['correlation']) *
                p['win_rate_confidence'] * p['statistical_confidence']
                for p in active_signals
            ) / total_weight

            strength = self._calculate_signal_strength(
                weighted_return)  # No p-value needed here since we filtered above

            # Calculate comprehensive confidence
            confidence = self._calculate_confidence(
                active_signals=active_signals,
                total_pairs=len(significant_pairs)
            )

            return {
                'date': date,
                'strength': strength,
                'expected_return': weighted_return,
                'actual_return': historical_data.loc[
                    date, 'Stock Pct Change'] if date in historical_data.index else None,
                'confidence': confidence,
                'active_pairs': len(active_signals),
                'active_signals': [
                    {
                        'pair': p['pair_key'],
                        'bins': p['bins'],
                        'correlation': p['correlation'],
                        'expected_return': p['expected_return'],
                        'win_rate': p['win_rate'],
                        'p_value': p['p_value']
                    }
                    for p in active_signals
                ]
            }

        except Exception as e:
            logger.error(f"Error generating daily signal: {str(e)}")
            return self._create_neutral_signal(date)

    def generate_signals(self, pair_results: Dict, historical_data: pd.DataFrame) -> Dict:
        """Generate trading signals based on significant indicator pairs"""
        try:
            logger.info("Starting simplified signal generation")

            # 1. First identify significant pairs and their stats once
            significant_pairs = self._get_significant_pairs(pair_results)
            logger.info(f"Found {len(significant_pairs)} significant pairs")

            if not significant_pairs:
                logger.warning("No significant pairs found")
                return {}  # Return empty dict, not None or float

            # 2. Generate signals for each date
            signals = {}
            for date in historical_data.index:
                signal = self._generate_daily_signal(
                    date,
                    significant_pairs,
                    historical_data,
                    pair_results
                )
                if signal:  # Make sure we got a valid signal dict
                    signals[date] = signal

            # 3. Analyze performance
            performance = self._analyze_signal_performance(signals)

            # Log performance but don't return it
            logger.info("\nSignal Performance Summary:")
            for metric, value in performance.items():
                logger.info(f"{metric}: {value:.2%}")

            return signals  # Always return the signals dictionary

        except Exception as e:
            logger.error(f"Error in signal generation: {str(e)}")
            return {}  # Return empty dict on error, not None or float

    def _create_neutral_signal(self, date: pd.Timestamp) -> Dict:
        """Create a neutral signal"""
        return {
            'date': date,
            'strength': 0,
            'expected_return': 0.0,
            'actual_return': None,
            'confidence': {  # Match the structure of active signals
                'overall': 0.0,
                'metrics': {
                    'total_weight': 0.0,
                    'active_pairs': 0,
                    'win_rate': 0.0
                }
            },
            'active_pairs': 0,
            'active_signals': []
        }

    def _check_bin_match(self, val1: float, val2: float, bin_definitions: Dict) -> bool:
        """Check if values fall within their respective bin ranges"""
        try:
            bin1_def = bin_definitions['bin1']
            bin2_def = bin_definitions['bin2']

            # Get bin boundaries
            if 'bounds' not in bin1_def or 'bounds' not in bin2_def:
                return False

            lower1, upper1 = bin1_def['bounds']
            lower2, upper2 = bin2_def['bounds']

            # Check ranges
            in_bin1 = lower1 <= val1 <= upper1
            in_bin2 = lower2 <= val2 <= upper2

            return in_bin1 and in_bin2

        except Exception as e:
            logger.error(f"Error checking bin match: {str(e)}")
            return False

    def _split_pair_key(self, pair_key: str) -> Tuple[str, str]:
        """
        Split indicator pair key into individual indicators, using derivative/acceleration as delimiters

        Example inputs and outputs:
            'RSI_MACD_derivative' -> ('RSI', 'MACD_derivative')
            'RSI_derivative_MACD' -> ('RSI_derivative', 'MACD')
            'RSI_acceleration_MACD_derivative' -> ('RSI_acceleration', 'MACD_derivative')
            'SMA_50_acceleration_RSI' -> ('SMA_50_acceleration', 'RSI')
        """
        try:
            parts = pair_key.split('_')
            ind1_parts = []
            ind2_parts = []
            found_second = False

            for i, part in enumerate(parts):
                # If we find a base indicator after a derivative/acceleration,
                # start the second indicator
                if part not in {'derivative', 'acceleration'}:
                    if i > 0 and parts[i - 1] in {'derivative', 'acceleration'}:
                        found_second = True

                if found_second:
                    ind2_parts.append(part)
                else:
                    ind1_parts.append(part)

            # If we never found a second indicator, try splitting after first derivative/acceleration
            if not found_second:
                for i, part in enumerate(parts):
                    if part in {'derivative', 'acceleration'}:
                        ind2_parts = parts[i + 1:]
                        ind1_parts = parts[:i + 1]
                        break

            if not ind2_parts:  # If still no split found, take last part as second indicator
                ind2_parts = [parts[-1]]
                ind1_parts = parts[:-1]

            return '_'.join(ind1_parts), '_'.join(ind2_parts)

        except Exception as e:
            logger.error(f"Error splitting pair key '{pair_key}': {str(e)}")
            # Return original string split in half as fallback
            parts = pair_key.split('_')
            mid = len(parts) // 2
            return '_'.join(parts[:mid]), '_'.join(parts[mid:])

    def _analyze_signal_performance(self, signals: Dict) -> Dict:
        """Analyze signal performance"""
        try:
            performance = {
                'total_signals': 0,
                'correct_signals': 0,
                'total_return': 0.0,
                'signal_returns': [],
                'strong_signal_returns': [],
                'buy_returns': [],
                'sell_returns': [],
                'strong_buy_returns': [],
                'strong_sell_returns': []
            }

            for signal in signals.values():
                actual_return = signal.get('actual_return')
                if actual_return is None:
                    continue

                strength = signal.get('strength', 0)

                # Track total return
                performance['total_return'] += actual_return

                if strength != 0:
                    performance['total_signals'] += 1
                    performance['signal_returns'].append(actual_return)

                    # Check if signal was correct
                    if (strength > 0 and actual_return > 0) or \
                            (strength < 0 and actual_return < 0):
                        performance['correct_signals'] += 1

                    # Track returns by signal type
                    if strength > 0:
                        performance['buy_returns'].append(actual_return)
                        if strength == 2:
                            performance['strong_buy_returns'].append(actual_return)
                            performance['strong_signal_returns'].append(actual_return)
                    else:
                        performance['sell_returns'].append(actual_return)
                        if strength == -2:
                            performance['strong_sell_returns'].append(actual_return)
                            performance['strong_signal_returns'].append(actual_return)

            # Calculate final metrics
            result = {
                'total_signals': performance['total_signals'],  # Count, not percentage
                'correct_signals': performance['correct_signals'],  # Count, not percentage
                'total_return': performance['total_return'],
                'signal_return': np.mean(performance['signal_returns']) if performance['signal_returns'] else 0.0,
                'strong_signal_return': np.mean(performance['strong_signal_returns']) if performance[
                    'strong_signal_returns'] else 0.0,
                'buy_return': np.mean(performance['buy_returns']) if performance['buy_returns'] else 0.0,
                'sell_return': np.mean(performance['sell_returns']) if performance['sell_returns'] else 0.0,
                'strong_buy_return': np.mean(performance['strong_buy_returns']) if performance[
                    'strong_buy_returns'] else 0.0,
                'strong_sell_return': np.mean(performance['strong_sell_returns']) if performance[
                    'strong_sell_returns'] else 0.0,
                'accuracy': performance['correct_signals'] / performance['total_signals'] if performance[
                                                                                                 'total_signals'] > 0 else 0.0
            }

            # Log with appropriate formatting
            logger.info("\nSignal Performance Summary:")
            for metric, value in result.items():
                if metric in ['total_signals', 'correct_signals']:
                    logger.info(f"{metric}: {value:,}")  # Format as number
                elif metric in ['accuracy']:
                    logger.info(f"{metric}: {value:.2%}")  # Format as percentage
                else:
                    logger.info(f"{metric}: {value:.2%}")  # Format returns as percentage

            return result

        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            return {
                'total_signals': 0,
                'correct_signals': 0,
                'total_return': 0.0,
                'signal_return': 0.0,
                'strong_signal_return': 0.0,
                'buy_return': 0.0,
                'sell_return': 0.0,
                'strong_buy_return': 0.0,
                'strong_sell_return': 0.0,
                'accuracy': 0.0
            }

    def create_signal_summary(self, signals: Dict, historical_data: pd.DataFrame,
                              output_path: str, ticker: str) -> pd.DataFrame:
        """
        Create and save summary of signal performance

        Args:
            signals: Dictionary of generated signals
            historical_data: DataFrame of historical prices and indicators
            output_path: Directory to save output files
            ticker: Stock ticker symbol

        Returns:
            DataFrame containing signal performance summary
        """
        try:
            # Create summary data
            summary_data = []

            for date, signal in signals.items():
                summary_data.append({
                    'date': date,
                    'stock_price': historical_data.loc[date, 'Stock Price'],
                    'signal': signal['strength'],
                    'confidence': signal['confidence'],
                    'expected_return': signal['expected_return'],
                    'actual_return': signal['actual_return'],
                    'active_pairs': signal['active_pairs']
                })

            df = pd.DataFrame(summary_data)

            # Calculate performance metrics
            performance = {
                'All Returns': df['actual_return'].dropna().mean(),
                'Signal Returns': df.loc[df['signal'] != 0, 'actual_return'].dropna().mean(),
                'Buy Returns': df.loc[df['signal'] > 0, 'actual_return'].dropna().mean(),
                'Sell Returns': df.loc[df['signal'] < 0, 'actual_return'].dropna().mean(),
                'Strong Buy Returns': df.loc[df['signal'] == 2, 'actual_return'].dropna().mean(),
                'Strong Sell Returns': df.loc[df['signal'] == -2, 'actual_return'].dropna().mean()
            }

            # Signal distribution
            distribution = {
                'Total Signals': len(df),
                'Active Signals': len(df[df['signal'] != 0]),
                'Strong Buys': len(df[df['signal'] == 2]),
                'Buys': len(df[df['signal'] == 1]),
                'Sells': len(df[df['signal'] == -1]),
                'Strong Sells': len(df[df['signal'] == -2])
            }

            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, f"{ticker}_signal_summary.xlsx")

            # Save to Excel
            try:
                with pd.ExcelWriter(output_file) as writer:
                    # Main signal data
                    df.to_excel(writer, sheet_name='Signals', index=False)

                    # Performance summary
                    pd.DataFrame([performance]).T.to_excel(
                        writer, sheet_name='Performance')

                    # Signal distribution
                    pd.DataFrame([distribution]).T.to_excel(
                        writer, sheet_name='Distribution')

                logger.info(f"Saved signal summary to {output_file}")

            except Exception as e:
                logger.error(f"Failed to save Excel file: {e}")
                # Fallback to CSV
                csv_file = output_file.replace('.xlsx', '.csv')
                df.to_csv(csv_file, index=False)
                logger.info(f"Saved signal summary as CSV to {csv_file}")

            # Log performance summary
            logger.info("\nSignal Performance Summary:")
            for metric, value in performance.items():
                logger.info(f"{metric}: {value:.4%}")

            logger.info("\nSignal Distribution:")
            for metric, value in distribution.items():
                logger.info(f"{metric}: {value}")

            return df

        except Exception as e:
            logger.error(f"Error creating signal summary: {str(e)}")
            raise

    def save_signal_model(self, pair_results: Dict, output_path: str, ticker: str):
        """Save the analyzed pairs and bin configurations for future use"""
        try:
            # Get significant pairs (this already contains all the needed info)
            significant_pairs = self._get_significant_pairs(pair_results)

            model_data = {
                'ticker': ticker,
                'date_created': datetime.now().isoformat(),
                'parameters': {
                    'min_samples': self.min_samples,
                    'min_correlation': self.min_correlation,
                    'signal_thresholds': self.signal_thresholds
                },
                'significant_pairs': significant_pairs
            }

            # Create directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)

            # Save to file
            model_file = os.path.join(output_path, f"{ticker}_signal_model.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Saved signal model for {ticker} with {len(significant_pairs)} pairs")

            return model_file

        except Exception as e:
            logger.error(f"Error saving signal model: {str(e)}")
            raise

    def load_signal_model(self, model_file: str) -> Tuple[str, List[Dict]]:
        """Load a saved signal model"""
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)

            # Update parameters if needed
            self.min_samples = model_data['parameters']['min_samples']
            self.min_correlation = model_data['parameters']['min_correlation']
            self.signal_thresholds = model_data['parameters']['signal_thresholds']

            return model_data['ticker'], model_data['significant_pairs']

        except Exception as e:
            logger.error(f"Error loading signal model: {str(e)}")
            raise

    def test_model_save_load(self, pair_results: Dict, ticker: str):
        """Test save and load functionality of signal model"""
        try:
            # 1. Save the model
            test_dir = "test_models"
            saved_file = self.save_signal_model(pair_results, test_dir, ticker)
            logger.info(f"Saved model to: {saved_file}")

            # 2. Load the model
            loaded_ticker, loaded_pairs = self.load_signal_model(saved_file)

            # 3. Verify contents
            logger.info(f"\nModel Verification:")
            logger.info(f"Ticker matches: {ticker == loaded_ticker}")
            logger.info(f"Number of significant pairs: {len(loaded_pairs)}")

            # 4. Check one pair in detail
            if loaded_pairs:
                sample_pair = loaded_pairs[0]
                logger.info("\nSample Pair Structure:")
                for key, value in sample_pair.items():
                    logger.info(f"{key}: {type(value)}")

            return True

        except Exception as e:
            logger.error(f"Error testing model save/load: {str(e)}")
            return False