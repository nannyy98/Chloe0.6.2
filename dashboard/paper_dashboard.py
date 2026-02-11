"""
Paper Dashboard for Chloe AI - Phase 8
Interactive performance monitoring and visualization
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import asyncio
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    refresh_interval_seconds: int = 30
    equity_curve_window_days: int = 90
    rolling_metrics_window: int = 30  # days for rolling calculations
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.alert_thresholds:
            self.alert_thresholds = {
                'max_drawdown': 0.15,      # 15% drawdown alert
                'sharpe_ratio': 1.0,       # Minimum Sharpe ratio
                'win_rate': 0.50,          # Minimum win rate
                'daily_loss_limit': 0.05   # 5% daily loss limit
            }

@dataclass
class PerformanceMetrics:
    """Key performance metrics"""
    timestamp: datetime
    equity: float
    daily_pnl: float
    total_pnl: float
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    calmar_ratio: float
    profit_factor: float
    stability_score: float

@dataclass
class RiskMetrics:
    """Risk-related metrics"""
    timestamp: datetime
    value_at_risk: float
    conditional_var: float
    max_position_size: float
    current_exposure: float
    leverage_ratio: float
    correlation_risk: float
    liquidity_risk: float
    regime_risk: float

@dataclass
class ModelMetrics:
    """Model performance tracking"""
    model_id: str
    timestamp: datetime
    trades_today: int
    accuracy_today: float
    confidence_avg: float
    risk_adjusted_return: float
    consistency_score: float
    active_signals: int

class PaperDashboard:
    """Interactive paper trading dashboard"""
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.performance_history = []
        self.risk_history = []
        self.model_performance = defaultdict(list)
        self.alerts = []
        self.active_models = set()
        
        logger.info("Paper Dashboard initialized")
        logger.info(f"Refresh interval: {self.config.refresh_interval_seconds} seconds")
        logger.info(f"Equity window: {self.config.equity_curve_window_days} days")

    def update_performance_data(self, 
                              equity: float,
                              daily_pnl: float,
                              trade_data: pd.DataFrame = None,
                              model_data: Dict[str, Any] = None) -> PerformanceMetrics:
        """Update performance metrics"""
        try:
            timestamp = datetime.now()
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(equity, daily_pnl, trade_data, timestamp)
            
            # Store in history
            self.performance_history.append(performance)
            
            # Trim old data
            self._trim_history()
            
            # Check for alerts
            self._check_performance_alerts(performance)
            
            # Update model metrics if provided
            if model_data:
                self._update_model_metrics(model_data, timestamp)
            
            logger.debug(f"Performance updated: Equity=${equity:,.2f}, PnL=${daily_pnl:+.2f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Performance update failed: {e}")
            raise

    def update_risk_data(self, risk_data: Dict[str, float]) -> RiskMetrics:
        """Update risk metrics"""
        try:
            timestamp = datetime.now()
            
            risk_metrics = RiskMetrics(
                timestamp=timestamp,
                value_at_risk=risk_data.get('var_95', 0.0),
                conditional_var=risk_data.get('cvar_95', 0.0),
                max_position_size=risk_data.get('max_position_size', 0.0),
                current_exposure=risk_data.get('current_exposure', 0.0),
                leverage_ratio=risk_data.get('leverage_ratio', 0.0),
                correlation_risk=risk_data.get('correlation_risk', 0.0),
                liquidity_risk=risk_data.get('liquidity_risk', 0.0),
                regime_risk=risk_data.get('regime_risk', 0.0)
            )
            
            self.risk_history.append(risk_metrics)
            self._trim_history()
            
            # Check risk alerts
            self._check_risk_alerts(risk_metrics)
            
            logger.debug(f"Risk metrics updated: VaR={risk_metrics.value_at_risk:.2f}")
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Risk update failed: {e}")
            raise

    def _calculate_performance_metrics(self, equity: float, daily_pnl: float, 
                                     trade_data: pd.DataFrame, timestamp: datetime) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            # Basic metrics
            total_pnl = equity - 10000  # Assuming $10k starting capital
            
            # Calculate from trade data if available
            if trade_data is not None and not trade_data.empty:
                # Win rate calculation
                winning_trades = (trade_data['pnl'] > 0).sum()
                total_trades = len(trade_data)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                # Returns calculation
                if 'pnl_percentage' in trade_data.columns:
                    returns = trade_data['pnl_percentage'] / 100
                else:
                    returns = trade_data['pnl'] / 10000  # As percentage of capital
                
                # Risk-adjusted metrics
                if len(returns) > 1:
                    annual_return = returns.mean() * 252
                    annual_volatility = returns.std() * np.sqrt(252)
                    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
                    
                    # Downside deviation for Sortino ratio
                    downside_returns = returns[returns < 0]
                    downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                    sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
                    
                    # Maximum drawdown
                    equity_curve = (1 + returns).cumprod()
                    running_max = equity_curve.expanding().max()
                    drawdown = (equity_curve - running_max) / running_max
                    max_drawdown = abs(drawdown.min())
                    
                    # Profit factor
                    gross_profits = trade_data[trade_data['pnl'] > 0]['pnl'].sum()
                    gross_losses = abs(trade_data[trade_data['pnl'] < 0]['pnl'].sum())
                    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
                    
                    # Calmar ratio
                    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
                    
                    # Stability score (based on return consistency)
                    recent_returns = returns.tail(30)
                    if len(recent_returns) > 1:
                        stability_score = 1.0 / (1.0 + recent_returns.std())
                    else:
                        stability_score = 0.5
                else:
                    # Default values when insufficient data
                    win_rate = 0.0
                    sharpe_ratio = 0.0
                    sortino_ratio = 0.0
                    max_drawdown = 0.0
                    annual_volatility = 0.0
                    profit_factor = 0.0
                    calmar_ratio = 0.0
                    stability_score = 0.5
            else:
                # Default values when no trade data
                win_rate = 0.0
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
                max_drawdown = 0.0
                annual_volatility = 0.0
                profit_factor = 0.0
                calmar_ratio = 0.0
                stability_score = 0.5
            
            return PerformanceMetrics(
                timestamp=timestamp,
                equity=equity,
                daily_pnl=daily_pnl,
                total_pnl=total_pnl,
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                volatility=annual_volatility,
                calmar_ratio=calmar_ratio,
                profit_factor=profit_factor,
                stability_score=stability_score
            )
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            # Return minimal metrics
            return PerformanceMetrics(
                timestamp=timestamp,
                equity=equity,
                daily_pnl=daily_pnl,
                total_pnl=equity - 10000,
                win_rate=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                calmar_ratio=0.0,
                profit_factor=0.0,
                stability_score=0.5
            )

    def _update_model_metrics(self, model_data: Dict[str, Any], timestamp: datetime):
        """Update model-specific metrics"""
        try:
            for model_id, metrics in model_data.items():
                model_metrics = ModelMetrics(
                    model_id=model_id,
                    timestamp=timestamp,
                    trades_today=metrics.get('trades_today', 0),
                    accuracy_today=metrics.get('accuracy_today', 0.0),
                    confidence_avg=metrics.get('confidence_avg', 0.0),
                    risk_adjusted_return=metrics.get('risk_adjusted_return', 0.0),
                    consistency_score=metrics.get('consistency_score', 0.0),
                    active_signals=metrics.get('active_signals', 0)
                )
                
                self.model_performance[model_id].append(model_metrics)
                self.active_models.add(model_id)
                
                # Trim old model data
                if len(self.model_performance[model_id]) > 1000:
                    self.model_performance[model_id] = self.model_performance[model_id][-500:]
            
        except Exception as e:
            logger.error(f"Model metrics update failed: {e}")

    def _check_performance_alerts(self, performance: PerformanceMetrics):
        """Check for performance-related alerts"""
        try:
            alerts_generated = []
            
            # Drawdown alert
            if performance.max_drawdown > self.config.alert_thresholds['max_drawdown']:
                alerts_generated.append({
                    'type': 'MAX_DRAWDOWN',
                    'severity': 'HIGH',
                    'message': f'Maximum drawdown {performance.max_drawdown*100:.1f}% exceeds threshold',
                    'timestamp': performance.timestamp
                })
            
            # Sharpe ratio alert
            if performance.sharpe_ratio < self.config.alert_thresholds['sharpe_ratio']:
                alerts_generated.append({
                    'type': 'LOW_SHARPE',
                    'severity': 'MEDIUM',
                    'message': f'Sharpe ratio {performance.sharpe_ratio:.2f} below minimum threshold',
                    'timestamp': performance.timestamp
                })
            
            # Win rate alert
            if performance.win_rate < self.config.alert_thresholds['win_rate'] and performance.win_rate > 0:
                alerts_generated.append({
                    'type': 'LOW_WIN_RATE',
                    'severity': 'MEDIUM',
                    'message': f'Win rate {performance.win_rate*100:.1f}% below threshold',
                    'timestamp': performance.timestamp
                })
            
            # Daily loss alert
            daily_loss_pct = abs(performance.daily_pnl / performance.equity)
            if daily_loss_pct > self.config.alert_thresholds['daily_loss_limit']:
                alerts_generated.append({
                    'type': 'DAILY_LOSS',
                    'severity': 'HIGH',
                    'message': f'Daily loss {daily_loss_pct*100:.1f}% exceeds limit',
                    'timestamp': performance.timestamp
                })
            
            # Add alerts to history
            self.alerts.extend(alerts_generated)
            
            # Log alerts
            for alert in alerts_generated:
                logger.warning(f"🚨 ALERT: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Performance alert checking failed: {e}")

    def _check_risk_alerts(self, risk_metrics: RiskMetrics):
        """Check for risk-related alerts"""
        try:
            alerts_generated = []
            
            # VaR alert
            if abs(risk_metrics.value_at_risk) > 0.10:  # 10% VaR threshold
                alerts_generated.append({
                    'type': 'HIGH_VAR',
                    'severity': 'HIGH',
                    'message': f'Value at Risk {risk_metrics.value_at_risk*100:.1f}% is elevated',
                    'timestamp': risk_metrics.timestamp
                })
            
            # Leverage alert
            if risk_metrics.leverage_ratio > 3.0:  # 3x leverage threshold
                alerts_generated.append({
                    'type': 'HIGH_LEVERAGE',
                    'severity': 'MEDIUM',
                    'message': f'Leverage ratio {risk_metrics.leverage_ratio:.1f}x is high',
                    'timestamp': risk_metrics.timestamp
                })
            
            # Exposure alert
            exposure_ratio = risk_metrics.current_exposure / risk_metrics.max_position_size
            if exposure_ratio > 0.9:  # Near maximum exposure
                alerts_generated.append({
                    'type': 'HIGH_EXPOSURE',
                    'severity': 'MEDIUM',
                    'message': f'Current exposure {exposure_ratio*100:.1f}% of maximum',
                    'timestamp': risk_metrics.timestamp
                })
            
            # Add risk alerts
            self.alerts.extend(alerts_generated)
            
            # Log risk alerts
            for alert in alerts_generated:
                logger.warning(f"⚠️  RISK ALERT: {alert['message']}")
            
        except Exception as e:
            logger.error(f"Risk alert checking failed: {e}")

    def _trim_history(self):
        """Trim historical data to maintain reasonable size"""
        try:
            # Keep last 365 days of performance data
            cutoff_date = datetime.now() - timedelta(days=365)
            
            self.performance_history = [
                p for p in self.performance_history 
                if p.timestamp >= cutoff_date
            ]
            
            self.risk_history = [
                r for r in self.risk_history 
                if r.timestamp >= cutoff_date
            ]
            
            # Keep alerts from last 30 days
            alert_cutoff = datetime.now() - timedelta(days=30)
            self.alerts = [
                a for a in self.alerts 
                if a['timestamp'] >= alert_cutoff
            ]
            
        except Exception as e:
            logger.error(f"History trimming failed: {e}")

    def get_equity_curve(self, days: int = None) -> pd.DataFrame:
        """Get equity curve data"""
        try:
            if not self.performance_history:
                return pd.DataFrame()
            
            days = days or self.config.equity_curve_window_days
            cutoff_date = datetime.now() - timedelta(days=days)
            
            recent_data = [
                p for p in self.performance_history 
                if p.timestamp >= cutoff_date
            ]
            
            if not recent_data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for perf in recent_data:
                data.append({
                    'timestamp': perf.timestamp,
                    'equity': perf.equity,
                    'daily_pnl': perf.daily_pnl,
                    'total_pnl': perf.total_pnl,
                    'drawdown': perf.max_drawdown
                })
            
            return pd.DataFrame(data).set_index('timestamp').sort_index()
            
        except Exception as e:
            logger.error(f"Equity curve generation failed: {e}")
            return pd.DataFrame()

    def get_rolling_metrics(self, window_days: int = None) -> Dict[str, float]:
        """Get rolling performance metrics"""
        try:
            window_days = window_days or self.config.rolling_metrics_window
            
            if len(self.performance_history) < window_days:
                return {}
            
            # Get recent data
            cutoff_date = datetime.now() - timedelta(days=window_days)
            recent_data = [
                p for p in self.performance_history 
                if p.timestamp >= cutoff_date
            ]
            
            if not recent_data:
                return {}
            
            # Calculate rolling metrics
            equities = [p.equity for p in recent_data]
            daily_pnls = [p.daily_pnl for p in recent_data]
            win_rates = [p.win_rate for p in recent_data if p.win_rate > 0]
            
            rolling_metrics = {
                'avg_equity': np.mean(equities),
                'equity_volatility': np.std(equities),
                'avg_daily_pnl': np.mean(daily_pnls),
                'daily_pnl_volatility': np.std(daily_pnls),
                'avg_win_rate': np.mean(win_rates) if win_rates else 0,
                'current_drawdown': recent_data[-1].max_drawdown if recent_data else 0,
                'period_return': (equities[-1] - equities[0]) / equities[0] if len(equities) > 1 else 0
            }
            
            return rolling_metrics
            
        except Exception as e:
            logger.error(f"Rolling metrics calculation failed: {e}")
            return {}

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all model performances"""
        try:
            summary = {}
            
            for model_id in self.active_models:
                model_data = self.model_performance[model_id]
                if not model_data:
                    continue
                
                # Get recent data (last 30 entries)
                recent_data = model_data[-30:] if len(model_data) > 30 else model_data
                
                summary[model_id] = {
                    'total_trades': sum(m.trades_today for m in recent_data),
                    'avg_accuracy': np.mean([m.accuracy_today for m in recent_data]),
                    'avg_confidence': np.mean([m.confidence_avg for m in recent_data]),
                    'risk_adjusted_return': np.mean([m.risk_adjusted_return for m in recent_data]),
                    'consistency_score': np.mean([m.consistency_score for m in recent_data]),
                    'active_signals': sum(m.active_signals for m in recent_data[-5:]),  # Recent signals
                    'data_points': len(recent_data)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Model performance summary failed: {e}")
            return {}

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of recent alerts"""
        try:
            if not self.alerts:
                return {
                    'total_alerts': 0,
                    'critical_alerts': 0,
                    'warning_alerts': 0,
                    'recent_alerts': []
                }
            
            # Categorize alerts by severity
            critical_alerts = [a for a in self.alerts if a['severity'] == 'HIGH']
            warning_alerts = [a for a in self.alerts if a['severity'] == 'MEDIUM']
            
            # Get recent alerts (last 10)
            recent_alerts = sorted(self.alerts, key=lambda x: x['timestamp'], reverse=True)[:10]
            
            return {
                'total_alerts': len(self.alerts),
                'critical_alerts': len(critical_alerts),
                'warning_alerts': len(warning_alerts),
                'recent_alerts': recent_alerts
            }
            
        except Exception as e:
            logger.error(f"Alert summary failed: {e}")
            return {}

    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        """Get complete dashboard snapshot"""
        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'performance': {},
                'risk': {},
                'models': {},
                'alerts': {},
                'rolling_metrics': {}
            }
            
            # Add latest performance data
            if self.performance_history:
                latest_perf = self.performance_history[-1]
                snapshot['performance'] = {
                    'equity': latest_perf.equity,
                    'daily_pnl': latest_perf.daily_pnl,
                    'total_pnl': latest_perf.total_pnl,
                    'win_rate': latest_perf.win_rate,
                    'sharpe_ratio': latest_perf.sharpe_ratio,
                    'max_drawdown': latest_perf.max_drawdown,
                    'volatility': latest_perf.volatility
                }
            
            # Add latest risk data
            if self.risk_history:
                latest_risk = self.risk_history[-1]
                snapshot['risk'] = {
                    'value_at_risk': latest_risk.value_at_risk,
                    'current_exposure': latest_risk.current_exposure,
                    'leverage_ratio': latest_risk.leverage_ratio,
                    'max_position_size': latest_risk.max_position_size
                }
            
            # Add summaries
            snapshot['models'] = self.get_model_performance_summary()
            snapshot['alerts'] = self.get_alert_summary()
            snapshot['rolling_metrics'] = self.get_rolling_metrics()
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Dashboard snapshot failed: {e}")
            return {}

    async def start_real_time_updates(self, data_source: Callable):
        """Start real-time dashboard updates"""
        logger.info("🚀 Starting real-time dashboard updates")
        logger.info(f"   Refresh interval: {self.config.refresh_interval_seconds} seconds")
        
        try:
            while True:
                try:
                    # Get fresh data from source
                    data = await data_source()
                    
                    # Update dashboard
                    if 'equity' in data and 'daily_pnl' in data:
                        self.update_performance_data(
                            equity=data['equity'],
                            daily_pnl=data['daily_pnl'],
                            trade_data=data.get('trade_data'),
                            model_data=data.get('model_data')
                        )
                    
                    if 'risk_data' in data:
                        self.update_risk_data(data['risk_data'])
                    
                    # Wait for next update
                    await asyncio.sleep(self.config.refresh_interval_seconds)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Real-time updates stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Real-time update error: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retry
                    
        except Exception as e:
            logger.error(f"Real-time updates failed: {e}")

def main():
    """Example usage"""
    print("Paper Dashboard - Interactive Performance Monitoring")
    print("Phase 8 of Paper-Learning Architecture")

if __name__ == "__main__":
    main()