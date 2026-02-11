"""
Walk-Forward Validation Framework for Chloe 0.6.1
Professional out-of-sample validation with rolling windows
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation"""
    train_window_size: int = 126        # Training period in days (6 months)
    test_window_size: int = 21          # Testing period in days (1 month)
    step_size: int = 7                  # Step size in days (1 week)
    min_train_samples: int = 63         # Minimum training samples required
    n_splits: Optional[int] = None      # Number of splits (auto-calculated if None)

@dataclass
class ValidationResult:
    """Results from a single validation fold"""
    fold_number: int
    train_period: Tuple[datetime, datetime]
    test_period: Tuple[datetime, datetime]
    train_samples: int
    test_samples: int
    metrics: Dict[str, float]
    predictions: Optional[pd.Series] = None
    actuals: Optional[pd.Series] = None

@dataclass
class WalkForwardResults:
    """Complete walk-forward validation results"""
    config: WalkForwardConfig
    validation_results: List[ValidationResult]
    overall_metrics: Dict[str, float]
    stability_analysis: Dict[str, float]

class WalkForwardValidator:
    """Professional walk-forward validation engine"""
    
    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.validation_history = []
        logger.info("Walk-Forward Validator initialized")

    def validate_strategy(self, 
                         data: pd.DataFrame,
                         strategy_func: Callable[[pd.DataFrame], pd.Series],
                         metrics_func: Callable[[pd.Series, pd.Series], Dict[str, float]],
                         target_column: str = 'returns') -> WalkForwardResults:
        """
        Perform walk-forward validation of trading strategy
        
        Args:
            data: DataFrame with datetime index and market data
            strategy_func: Function that takes training data and returns predictions
            metrics_func: Function that calculates performance metrics
            target_column: Column name containing target returns
        """
        try:
            logger.info("🔬 Starting Walk-Forward Validation...")
            logger.info(f"   Data shape: {data.shape}")
            logger.info(f"   Train window: {self.config.train_window_size} days")
            logger.info(f"   Test window: {self.config.test_window_size} days")
            logger.info(f"   Step size: {self.config.step_size} days")
            
            # Sort data by timestamp
            data = data.sort_index()
            
            # Generate walk-forward splits
            splits = self._generate_time_splits(data)
            logger.info(f"   Generated {len(splits)} validation folds")
            
            if len(splits) < 2:
                raise ValueError("Insufficient data for walk-forward validation")
            
            # Execute validation for each fold
            validation_results = []
            
            for fold_num, (train_idx, test_idx) in enumerate(splits, 1):
                logger.info(f"   Processing fold {fold_num}/{len(splits)}")
                
                # Split data
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                
                # Validate minimum sample size
                if len(train_data) < self.config.min_train_samples:
                    logger.warning(f"      Insufficient training samples: {len(train_data)}")
                    continue
                
                try:
                    # Train strategy
                    predictions = strategy_func(train_data)
                    
                    # Get actual returns for test period
                    actual_returns = test_data[target_column]
                    
                    # Align predictions with actual returns
                    aligned_predictions = predictions.reindex(test_data.index).dropna()
                    aligned_actuals = actual_returns.reindex(aligned_predictions.index)
                    
                    if len(aligned_predictions) == 0:
                        logger.warning(f"      No overlapping predictions/actuals")
                        continue
                    
                    # Calculate metrics
                    fold_metrics = metrics_func(aligned_predictions, aligned_actuals)
                    
                    # Store results
                    result = ValidationResult(
                        fold_number=fold_num,
                        train_period=(train_data.index[0], train_data.index[-1]),
                        test_period=(test_data.index[0], test_data.index[-1]),
                        train_samples=len(train_data),
                        test_samples=len(aligned_predictions),
                        metrics=fold_metrics,
                        predictions=aligned_predictions,
                        actuals=aligned_actuals
                    )
                    
                    validation_results.append(result)
                    logger.info(f"      Fold {fold_num}: {len(aligned_predictions)} samples, "
                               f"Sharpe={fold_metrics.get('sharpe_ratio', 0):.3f}")
                    
                except Exception as e:
                    logger.error(f"      Fold {fold_num} failed: {e}")
                    continue
            
            if not validation_results:
                raise RuntimeError("All validation folds failed")
            
            # Calculate overall metrics
            overall_metrics = self._calculate_overall_metrics(validation_results)
            
            # Perform stability analysis
            stability_analysis = self._analyze_stability(validation_results)
            
            # Create results object
            results = WalkForwardResults(
                config=self.config,
                validation_results=validation_results,
                overall_metrics=overall_metrics,
                stability_analysis=stability_analysis
            )
            
            self.validation_history.append(results)
            logger.info("✅ Walk-Forward Validation completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Walk-forward validation failed: {e}")
            raise

    def _generate_time_splits(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate time-based splits for walk-forward validation"""
        try:
            # Use TimeSeriesSplit for proper time series splitting
            if self.config.n_splits:
                n_splits = self.config.n_splits
            else:
                # Calculate automatic number of splits
                total_days = (data.index[-1] - data.index[0]).days
                n_splits = max(2, (total_days - self.config.train_window_size) // self.config.step_size)
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            splits = []
            for train_idx, test_idx in tscv.split(data):
                # Filter test indices to respect test window size
                if len(test_idx) > self.config.test_window_size:
                    test_idx = test_idx[:self.config.test_window_size]
                
                splits.append((train_idx, test_idx))
            
            return splits
            
        except Exception as e:
            logger.error(f"Time split generation failed: {e}")
            raise

    def _calculate_overall_metrics(self, results: List[ValidationResult]) -> Dict[str, float]:
        """Calculate overall performance metrics across all folds"""
        try:
            # Aggregate metrics
            metric_names = results[0].metrics.keys()
            aggregated_metrics = {}
            
            for metric_name in metric_names:
                fold_values = [result.metrics[metric_name] for result in results]
                aggregated_metrics[f'{metric_name}_mean'] = np.mean(fold_values)
                aggregated_metrics[f'{metric_name}_std'] = np.std(fold_values)
                aggregated_metrics[f'{metric_name}_min'] = np.min(fold_values)
                aggregated_metrics[f'{metric_name}_max'] = np.max(fold_values)
            
            # Calculate consistency measures
            sharpe_ratios = [result.metrics.get('sharpe_ratio', 0) for result in results]
            aggregated_metrics['sharpe_consistency'] = self._calculate_consistency(sharpe_ratios)
            
            win_rates = [result.metrics.get('win_rate', 0) for result in results]
            aggregated_metrics['win_rate_consistency'] = self._calculate_consistency(win_rates)
            
            return aggregated_metrics
            
        except Exception as e:
            logger.error(f"Overall metrics calculation failed: {e}")
            return {}

    def _analyze_stability(self, results: List[ValidationResult]) -> Dict[str, float]:
        """Analyze stability of strategy performance"""
        try:
            stability_metrics = {}
            
            # Sharpe ratio stability
            sharpe_values = [result.metrics.get('sharpe_ratio', 0) for result in results]
            stability_metrics['sharpe_stability'] = self._calculate_stability(sharpe_values)
            
            # Return stability
            return_values = [result.metrics.get('annual_return', 0) for result in results]
            stability_metrics['return_stability'] = self._calculate_stability(return_values)
            
            # Drawdown stability
            dd_values = [result.metrics.get('max_drawdown', 0) for result in results]
            stability_metrics['drawdown_stability'] = self._calculate_stability(dd_values, inverse=True)
            
            # Consistency of positive periods
            positive_periods = sum(1 for result in results 
                                 if result.metrics.get('sharpe_ratio', 0) > 0)
            stability_metrics['positive_fold_ratio'] = positive_periods / len(results)
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Stability analysis failed: {e}")
            return {}

    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency measure (lower std/mean ratio = more consistent)"""
        if not values or np.mean(values) == 0:
            return 0.0
        return 1.0 / (1.0 + np.std(values) / abs(np.mean(values)))

    def _calculate_stability(self, values: List[float], inverse: bool = False) -> float:
        """Calculate stability score (0-1, higher = more stable)"""
        if len(values) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0
            
        cv = np.std(values) / abs(mean_val)
        
        # Convert to stability score (0-1)
        stability = 1.0 / (1.0 + cv)
        
        # For inverse metrics (like drawdown), higher values are worse
        if inverse:
            stability = 1.0 - stability
            
        return stability

    def generate_validation_report(self, results: WalkForwardResults) -> str:
        """Generate comprehensive validation report"""
        try:
            report = f"""
WALK-FORWARD VALIDATION REPORT
=============================

CONFIGURATION:
  Train Window: {results.config.train_window_size} days
  Test Window: {results.config.test_window_size} days  
  Step Size: {results.config.step_size} days
  Folds Processed: {len(results.validation_results)}

OVERALL PERFORMANCE:
"""
            
            # Key metrics
            metrics = results.overall_metrics
            report += f"  Mean Sharpe Ratio: {metrics.get('sharpe_ratio_mean', 0):.3f} ± {metrics.get('sharpe_ratio_std', 0):.3f}\n"
            report += f"  Mean Annual Return: {metrics.get('annual_return_mean', 0)*100:.2f}% ± {metrics.get('annual_return_std', 0)*100:.2f}%\n"
            report += f"  Mean Max Drawdown: {metrics.get('max_drawdown_mean', 0)*100:.2f}% ± {metrics.get('max_drawdown_std', 0)*100:.2f}%\n"
            report += f"  Mean Win Rate: {metrics.get('win_rate_mean', 0)*100:.1f}% ± {metrics.get('win_rate_std', 0)*100:.1f}%\n"
            
            # Stability analysis
            stability = results.stability_analysis
            report += f"\nSTABILITY ANALYSIS:\n"
            report += f"  Sharpe Stability: {stability.get('sharpe_stability', 0):.3f}\n"
            report += f"  Return Stability: {stability.get('return_stability', 0):.3f}\n"
            report += f"  Drawdown Stability: {stability.get('drawdown_stability', 0):.3f}\n"
            report += f"  Positive Periods: {stability.get('positive_fold_ratio', 0)*100:.1f}%\n"
            
            # Individual fold results
            report += f"\nFOLD-BY-FOLD RESULTS:\n"
            report += f"{'Fold':<6} {'Period':<20} {'Samples':<8} {'Sharpe':<8} {'Return':<8} {'Drawdown':<8}\n"
            report += "-" * 70 + "\n"
            
            for result in results.validation_results:
                period = f"{result.test_period[0].strftime('%Y-%m')} to {result.test_period[1].strftime('%Y-%m')}"
                sharpe = result.metrics.get('sharpe_ratio', 0)
                ret = result.metrics.get('annual_return', 0) * 100
                dd = result.metrics.get('max_drawdown', 0) * 100
                
                report += f"{result.fold_number:<6} {period:<20} {result.test_samples:<8} {sharpe:<8.3f} {ret:<8.2f}% {dd:<8.2f}%\n"
            
            # Validation verdict
            report += f"\nVALIDATION VERDICT:\n"
            
            mean_sharpe = metrics.get('sharpe_ratio_mean', 0)
            sharpe_stability = stability.get('sharpe_stability', 0)
            positive_ratio = stability.get('positive_fold_ratio', 0)
            
            if mean_sharpe > 1.5 and sharpe_stability > 0.7 and positive_ratio > 0.8:
                verdict = "✅ STRONG VALIDATION - Strategy shows consistent outperformance"
            elif mean_sharpe > 1.0 and sharpe_stability > 0.5 and positive_ratio > 0.6:
                verdict = "⚠️  MODERATE VALIDATION - Strategy shows promise but needs monitoring"
            elif mean_sharpe > 0.5 and sharpe_stability > 0.3 and positive_ratio > 0.5:
                verdict = "🔶 WEAK VALIDATION - Strategy marginally profitable but inconsistent"
            else:
                verdict = "❌ FAILED VALIDATION - Strategy does not demonstrate consistent edge"
            
            report += f"  {verdict}\n"
            
            return report.strip()
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Report generation failed: {e}"

# Standard performance metrics calculator
def calculate_standard_metrics(predictions: pd.Series, actuals: pd.Series) -> Dict[str, float]:
    """Calculate standard performance metrics"""
    try:
        # Align data
        aligned_data = pd.concat([predictions, actuals], axis=1, keys=['pred', 'actual']).dropna()
        if len(aligned_data) == 0:
            return {}
        
        pred_returns = aligned_data['pred']
        actual_returns = aligned_data['actual']
        
        # Calculate portfolio returns (assuming predictions are position sizes)
        portfolio_returns = pred_returns * actual_returns
        
        # Basic metrics
        metrics = {
            'annual_return': portfolio_returns.mean() * 252,
            'annual_volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252)) if portfolio_returns.std() > 0 else 0,
            'win_rate': (portfolio_returns > 0).mean(),
            'max_drawdown': calculate_max_drawdown(portfolio_returns),
            'profit_factor': calculate_profit_factor(portfolio_returns),
            'calmar_ratio': calculate_calmar_ratio(portfolio_returns)
        }
        
        # Handle potential NaN values
        for key, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                metrics[key] = 0.0
        
        return metrics
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        return {}

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return abs(drawdown.min())

def calculate_profit_factor(returns: pd.Series) -> float:
    """Calculate profit factor (gross profits / gross losses)"""
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return profits / losses if losses > 0 else float('inf')

def calculate_calmar_ratio(returns: pd.Series) -> float:
    """Calculate Calmar ratio (annual return / max drawdown)"""
    annual_return = returns.mean() * 252
    max_dd = calculate_max_drawdown(returns)
    return annual_return / max_dd if max_dd > 0 else 0

def main():
    """Example usage"""
    print("Walk-Forward Validation Framework ready")
    print("Professional out-of-sample validation with rolling windows")

if __name__ == "__main__":
    main()