"""
Model Validation Gate for Chloe AI - Phase 4
Critical quality gate for model deployment approval
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ValidationCriteria:
    """Validation criteria configuration"""
    min_sharpe_ratio: float = 1.2        # Minimum Sharpe ratio
    max_drawdown: float = 0.20           # Maximum drawdown (20%)
    min_win_rate: float = 0.52           # Minimum win rate (52%)
    min_trades: int = 50                 # Minimum number of trades
    min_months: int = 3                  # Minimum backtest period in months
    stability_threshold: float = 0.7     # Minimum stability score
    monte_carlo_pass_rate: float = 0.90  # Monte Carlo pass rate (90%)

@dataclass
class ValidationResult:
    """Detailed validation result"""
    model_id: str
    validation_date: datetime
    overall_status: str  # APPROVED, REJECTED, CONDITIONAL
    criteria_results: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    stability_analysis: Dict[str, float]
    recommendations: List[str]

class ModelValidationGate:
    """Professional model validation system"""
    
    def __init__(self, criteria: ValidationCriteria = None):
        self.criteria = criteria or ValidationCriteria()
        self.validation_history = []
        logger.info("Model Validation Gate initialized")
        logger.info(f"Minimum Sharpe: {self.criteria.min_sharpe_ratio}")
        logger.info(f"Maximum Drawdown: {self.criteria.max_drawdown*100:.1f}%")
        logger.info(f"Minimum Win Rate: {self.criteria.min_win_rate*100:.1f}%")

    def validate_model(self, 
                      trade_data: pd.DataFrame,
                      model_metadata: Dict[str, Any] = None) -> ValidationResult:
        """
        Perform comprehensive model validation
        
        Args:
            trade_data: DataFrame with trade records
            model_metadata: Additional model information
            
        Returns:
            ValidationResult with detailed assessment
        """
        try:
            logger.info("🔍 Starting Model Validation Process...")
            
            model_id = model_metadata.get('model_id', 'unknown_model') if model_metadata else 'test_model'
            logger.info(f"   Model ID: {model_id}")
            logger.info(f"   Trade Records: {len(trade_data)}")
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(trade_data)
            logger.info(f"   Performance calculated")
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(trade_data)
            logger.info(f"   Risk metrics calculated")
            
            # Perform stability analysis
            stability_analysis = self._perform_stability_analysis(trade_data)
            logger.info(f"   Stability analysis completed")
            
            # Evaluate each criterion
            criteria_results = self._evaluate_criteria(performance_metrics, risk_metrics, stability_analysis)
            
            # Determine overall status
            overall_status = self._determine_overall_status(criteria_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(criteria_results, performance_metrics, risk_metrics)
            
            # Create validation result
            result = ValidationResult(
                model_id=model_id,
                validation_date=datetime.now(),
                overall_status=overall_status,
                criteria_results=criteria_results,
                performance_metrics=performance_metrics,
                risk_metrics=risk_metrics,
                stability_analysis=stability_analysis,
                recommendations=recommendations
            )
            
            self.validation_history.append(result)
            
            logger.info(f"✅ Model validation completed - Status: {overall_status}")
            return result
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise

    def _calculate_performance_metrics(self, trade_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        try:
            # Basic metrics
            total_trades = len(trade_data)
            winning_trades = (trade_data['pnl'] > 0).sum()
            losing_trades = (trade_data['pnl'] < 0).sum()
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Returns calculation
            if 'pnl_percentage' in trade_data.columns:
                returns = trade_data['pnl_percentage'] / 100
            else:
                # Calculate returns from PnL and position sizes
                returns = trade_data['pnl'] / 1000  # Simplified assumption
            
            # Annualized metrics (assuming 252 trading days)
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            sortino_ratio = annual_return / (returns[returns < 0].std() * np.sqrt(252)) if returns[returns < 0].std() > 0 else 0
            
            # Profitability metrics
            total_pnl = trade_data['pnl'].sum()
            avg_win = trade_data[trade_data['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trade_data[trade_data['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            metrics = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'total_pnl': total_pnl,
                'average_win': avg_win,
                'average_loss': avg_loss,
                'profit_factor': profit_factor,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades
            }
            
            # Handle potential NaN values
            for key, value in metrics.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    metrics[key] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {}

    def _calculate_risk_metrics(self, trade_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        try:
            # Calculate equity curve
            if 'pnl' in trade_data.columns:
                equity_curve = trade_data['pnl'].cumsum()
            else:
                equity_curve = pd.Series(np.zeros(len(trade_data)))
            
            # Maximum drawdown
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / (running_max + 1e-8)  # Avoid division by zero
            max_drawdown = abs(drawdown.min())
            
            # Drawdown duration
            drawdown_periods = (drawdown < 0).astype(int)
            max_drawdown_duration = 0
            current_duration = 0
            
            for period in drawdown_periods:
                if period == 1:
                    current_duration += 1
                    max_drawdown_duration = max(max_drawdown_duration, current_duration)
                else:
                    current_duration = 0
            
            # Value at Risk (approximate)
            if 'pnl_percentage' in trade_data.columns:
                returns = trade_data['pnl_percentage'] / 100
            else:
                returns = trade_data['pnl'] / 1000
            
            var_95 = np.percentile(returns, 5)  # 95% VaR
            var_99 = np.percentile(returns, 1)  # 99% VaR
            
            # Downside risk
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else 0
            
            # Calmar ratio
            calmar_ratio = (returns.mean() * 252) / max_drawdown if max_drawdown > 0 else 0
            
            risk_metrics = {
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': max_drawdown_duration,
                'value_at_risk_95': var_95,
                'value_at_risk_99': var_99,
                'downside_deviation': downside_deviation,
                'calmar_ratio': calmar_ratio,
                'skewness': stats.skew(returns) if len(returns) > 2 else 0,
                'kurtosis': stats.kurtosis(returns) if len(returns) > 3 else 0
            }
            
            # Handle potential NaN values
            for key, value in risk_metrics.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    risk_metrics[key] = 0.0
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Risk metrics calculation failed: {e}")
            return {}

    def _perform_stability_analysis(self, trade_data: pd.DataFrame) -> Dict[str, float]:
        """Perform stability and consistency analysis"""
        try:
            stability_metrics = {}
            
            # Time-based stability (monthly analysis)
            if 'timestamp' in trade_data.columns and len(trade_data) >= 20:
                trade_data_sorted = trade_data.sort_values('timestamp')
                trade_data_sorted['month'] = pd.to_datetime(trade_data_sorted['timestamp']).dt.to_period('M')
                
                monthly_performance = trade_data_sorted.groupby('month')['pnl'].sum()
                
                if len(monthly_performance) >= 3:
                    # Monthly Sharpe stability
                    monthly_sharpe = []
                    for month in monthly_performance.index:
                        month_data = trade_data_sorted[trade_data_sorted['month'] == month]
                        if len(month_data) >= 5:  # Minimum trades per month
                            if 'pnl_percentage' in month_data.columns:
                                returns = month_data['pnl_percentage'] / 100
                            else:
                                returns = month_data['pnl'] / 1000
                            
                            if returns.std() > 0:
                                sharpe = returns.mean() / returns.std()
                                monthly_sharpe.append(sharpe)
                    
                    if len(monthly_sharpe) >= 3:
                        stability_metrics['monthly_sharpe_stability'] = self._calculate_stability_score(monthly_sharpe)
                        stability_metrics['monthly_sharpe_mean'] = np.mean(monthly_sharpe)
                        stability_metrics['monthly_sharpe_std'] = np.std(monthly_sharpe)
            
            # Rolling window stability
            if len(trade_data) >= 30:
                rolling_sharpe = []
                window_size = min(20, len(trade_data) // 3)
                
                for i in range(len(trade_data) - window_size + 1):
                    window_data = trade_data.iloc[i:i+window_size]
                    if 'pnl_percentage' in window_data.columns:
                        returns = window_data['pnl_percentage'] / 100
                    else:
                        returns = window_data['pnl'] / 1000
                    
                    if returns.std() > 0:
                        sharpe = returns.mean() / returns.std()
                        rolling_sharpe.append(sharpe)
                
                if len(rolling_sharpe) >= 5:
                    stability_metrics['rolling_sharpe_stability'] = self._calculate_stability_score(rolling_sharpe)
            
            # Win rate consistency
            if len(trade_data) >= 50:
                # Calculate rolling win rates
                rolling_wins = []
                window_size = min(20, len(trade_data) // 3)
                
                for i in range(len(trade_data) - window_size + 1):
                    window_data = trade_data.iloc[i:i+window_size]
                    win_rate = (window_data['pnl'] > 0).mean()
                    rolling_wins.append(win_rate)
                
                if len(rolling_wins) >= 5:
                    stability_metrics['win_rate_stability'] = self._calculate_stability_score(rolling_wins)
            
            # Default stability if not enough data
            if not stability_metrics:
                stability_metrics['default_stability'] = 0.5  # Neutral score
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Stability analysis failed: {e}")
            return {'error_stability': 0.0}

    def _calculate_stability_score(self, values: List[float]) -> float:
        """Calculate stability score (0-1, higher = more stable)"""
        if len(values) < 2:
            return 0.5
        
        mean_val = np.mean(values)
        if abs(mean_val) < 1e-10:
            return 0.5
            
        cv = np.std(values) / abs(mean_val)  # Coefficient of variation
        stability = 1.0 / (1.0 + cv)  # Convert to stability score
        return max(0.0, min(1.0, stability))

    def _evaluate_criteria(self, 
                          performance_metrics: Dict[str, float],
                          risk_metrics: Dict[str, float],
                          stability_analysis: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Evaluate model against all validation criteria"""
        try:
            criteria_results = {}
            
            # Sharpe Ratio Criterion
            sharpe = performance_metrics.get('sharpe_ratio', 0)
            criteria_results['sharpe_ratio'] = {
                'value': sharpe,
                'threshold': self.criteria.min_sharpe_ratio,
                'passed': sharpe >= self.criteria.min_sharpe_ratio,
                'margin': sharpe - self.criteria.min_sharpe_ratio
            }
            
            # Maximum Drawdown Criterion
            max_dd = risk_metrics.get('max_drawdown', 1.0)
            criteria_results['max_drawdown'] = {
                'value': max_dd,
                'threshold': self.criteria.max_drawdown,
                'passed': max_dd <= self.criteria.max_drawdown,
                'margin': self.criteria.max_drawdown - max_dd
            }
            
            # Win Rate Criterion
            win_rate = performance_metrics.get('win_rate', 0)
            criteria_results['win_rate'] = {
                'value': win_rate,
                'threshold': self.criteria.min_win_rate,
                'passed': win_rate >= self.criteria.min_win_rate,
                'margin': win_rate - self.criteria.min_win_rate
            }
            
            # Minimum Trades Criterion
            total_trades = performance_metrics.get('total_trades', 0)
            criteria_results['minimum_trades'] = {
                'value': total_trades,
                'threshold': self.criteria.min_trades,
                'passed': total_trades >= self.criteria.min_trades,
                'margin': total_trades - self.criteria.min_trades
            }
            
            # Stability Criterion
            stability_scores = [v for k, v in stability_analysis.items() if 'stability' in k.lower()]
            avg_stability = np.mean(stability_scores) if stability_scores else 0
            criteria_results['stability'] = {
                'value': avg_stability,
                'threshold': self.criteria.stability_threshold,
                'passed': avg_stability >= self.criteria.stability_threshold,
                'margin': avg_stability - self.criteria.stability_threshold
            }
            
            return criteria_results
            
        except Exception as e:
            logger.error(f"Criteria evaluation failed: {e}")
            return {}

    def _determine_overall_status(self, criteria_results: Dict[str, Dict[str, Any]]) -> str:
        """Determine overall validation status"""
        try:
            if not criteria_results:
                return "REJECTED"
            
            # Count passed criteria
            passed_criteria = sum(1 for criterion in criteria_results.values() if criterion['passed'])
            total_criteria = len(criteria_results)
            
            # Critical criteria that must pass
            critical_criteria = ['sharpe_ratio', 'max_drawdown', 'minimum_trades']
            critical_passed = all(criteria_results.get(crit, {}).get('passed', False) 
                                for crit in critical_criteria if crit in criteria_results)
            
            if not critical_passed:
                return "REJECTED"
            
            # Overall assessment
            pass_rate = passed_criteria / total_criteria
            
            if pass_rate >= 0.8:
                return "APPROVED"
            elif pass_rate >= 0.6:
                return "CONDITIONAL"
            else:
                return "REJECTED"
                
        except Exception as e:
            logger.error(f"Status determination failed: {e}")
            return "REJECTED"

    def _generate_recommendations(self, 
                                criteria_results: Dict[str, Dict[str, Any]],
                                performance_metrics: Dict[str, float],
                                risk_metrics: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations"""
        try:
            recommendations = []
            
            # Sharpe ratio recommendations
            sharpe_result = criteria_results.get('sharpe_ratio', {})
            if not sharpe_result.get('passed', True):
                margin = sharpe_result.get('margin', 0)
                if margin < -0.5:
                    recommendations.append("Significantly improve risk-adjusted returns")
                else:
                    recommendations.append("Marginally improve Sharpe ratio")
            
            # Drawdown recommendations
            dd_result = criteria_results.get('max_drawdown', {})
            if not dd_result.get('passed', True):
                current_dd = risk_metrics.get('max_drawdown', 0) * 100
                max_allowed = self.criteria.max_drawdown * 100
                recommendations.append(f"Reduce maximum drawdown from {current_dd:.1f}% to under {max_allowed:.1f}%")
            
            # Win rate recommendations
            win_result = criteria_results.get('win_rate', {})
            if not win_result.get('passed', True):
                current_win = performance_metrics.get('win_rate', 0) * 100
                min_required = self.criteria.min_win_rate * 100
                recommendations.append(f"Increase win rate from {current_win:.1f}% to at least {min_required:.1f}%")
            
            # General recommendations
            if len(recommendations) == 0:
                recommendations.append("Model meets all validation criteria")
                recommendations.append("Ready for shadow mode testing")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendations generation failed: {e}")
            return ["Unable to generate recommendations due to error"]

    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generate comprehensive validation report"""
        try:
            report = f"""
MODEL VALIDATION REPORT
======================

MODEL INFORMATION:
  Model ID: {result.model_id}
  Validation Date: {result.validation_date.strftime('%Y-%m-%d %H:%M:%S')}
  Overall Status: {result.overall_status}

PERFORMANCE METRICS:
  Sharpe Ratio: {result.performance_metrics.get('sharpe_ratio', 0):.3f}
  Annual Return: {result.performance_metrics.get('annual_return', 0)*100:.2f}%
  Win Rate: {result.performance_metrics.get('win_rate', 0)*100:.1f}%
  Total Trades: {result.performance_metrics.get('total_trades', 0)}
  Profit Factor: {result.performance_metrics.get('profit_factor', 0):.2f}

RISK METRICS:
  Max Drawdown: {result.risk_metrics.get('max_drawdown', 0)*100:.2f}%
  VaR (95%): {result.risk_metrics.get('value_at_risk_95', 0)*100:.2f}%
  Downside Deviation: {result.risk_metrics.get('downside_deviation', 0)*100:.2f}%
  Calmar Ratio: {result.risk_metrics.get('calmar_ratio', 0):.2f}

VALIDATION CRITERIA:
"""
            
            for criterion_name, criterion_result in result.criteria_results.items():
                status_icon = "✅" if criterion_result['passed'] else "❌"
                report += f"  {status_icon} {criterion_name.replace('_', ' ').title()}: "
                report += f"{criterion_result['value']:.3f} "
                report += f"(threshold: {criterion_result['threshold']:.3f})\n"
            
            report += f"\nRECOMMENDATIONS:\n"
            for i, recommendation in enumerate(result.recommendations, 1):
                report += f"  {i}. {recommendation}\n"
            
            return report.strip()
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Report generation failed: {e}"

def main():
    """Example usage"""
    print("Model Validation Gate - Quality Assurance System")
    print("Phase 4 of Paper-Learning Architecture")

if __name__ == "__main__":
    main()