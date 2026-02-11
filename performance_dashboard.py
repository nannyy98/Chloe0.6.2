"""
Performance Monitoring Dashboard for Chloe 0.6
Real-time performance monitoring and visualization dashboard
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import threading
import time
from collections import deque
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MetricCategory(Enum):
    """Categories of performance metrics"""
    PORTFOLIO = "PORTFOLIO"           # Portfolio-level metrics
    RISK = "RISK"                     # Risk management metrics
    TRADING = "TRADING"               # Trading activity metrics
    SYSTEM = "SYSTEM"                 # System performance metrics
    MARKET = "MARKET"                 # Market condition metrics

class AlertLevel(Enum):
    """Alert levels for monitoring"""
    NORMAL = "NORMAL"                 # Normal operating conditions
    WARNING = "WARNING"               # Warning conditions
    CRITICAL = "CRITICAL"             # Critical conditions requiring attention

@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    category: MetricCategory
    value: float
    timestamp: datetime
    unit: str = ""
    description: str = ""
    alert_level: AlertLevel = AlertLevel.NORMAL
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None

@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    update_frequency: float = 1.0           # Update frequency in seconds
    history_length: int = 1000              # Number of historical points to keep
    alert_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)
    enabled_metrics: List[str] = field(default_factory=list)

class PerformanceMonitor:
    """Core performance monitoring engine"""
    
    def __init__(self, config: DashboardConfig = None):
        self.config = config or DashboardConfig()
        self.metrics = {}
        self.metric_history = {}
        self.subscribers = []
        self.monitoring_active = False
        self.monitoring_thread = None
        self.lock = threading.Lock()
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        logger.info("Performance Monitor initialized")

    def _initialize_default_metrics(self):
        """Initialize default performance metrics"""
        default_metrics = {
            # Portfolio metrics
            'portfolio_value': {'category': MetricCategory.PORTFOLIO, 'unit': '$', 'description': 'Total portfolio value'},
            'portfolio_pnl': {'category': MetricCategory.PORTFOLIO, 'unit': '$', 'description': 'Portfolio profit/loss'},
            'portfolio_return': {'category': MetricCategory.PORTFOLIO, 'unit': '%', 'description': 'Portfolio return'},
            'drawdown': {'category': MetricCategory.PORTFOLIO, 'unit': '%', 'description': 'Portfolio drawdown'},
            'sharpe_ratio': {'category': MetricCategory.PORTFOLIO, 'unit': '', 'description': 'Risk-adjusted return'},
            
            # Risk metrics
            'value_at_risk': {'category': MetricCategory.RISK, 'unit': '$', 'description': 'Value at Risk (VaR)'},
            'conditional_var': {'category': MetricCategory.RISK, 'unit': '$', 'description': 'Conditional VaR'},
            'portfolio_volatility': {'category': MetricCategory.RISK, 'unit': '%', 'description': 'Portfolio volatility'},
            'max_position_size': {'category': MetricCategory.RISK, 'unit': '%', 'description': 'Largest position size'},
            'risk_exposure': {'category': MetricCategory.RISK, 'unit': '%', 'description': 'Total risk exposure'},
            
            # Trading metrics
            'daily_trades': {'category': MetricCategory.TRADING, 'unit': '', 'description': 'Number of daily trades'},
            'win_rate': {'category': MetricCategory.TRADING, 'unit': '%', 'description': 'Trading win rate'},
            'average_trade_pnl': {'category': MetricCategory.TRADING, 'unit': '$', 'description': 'Average trade PNL'},
            'largest_winner': {'category': MetricCategory.TRADING, 'unit': '$', 'description': 'Largest winning trade'},
            'largest_loser': {'category': MetricCategory.TRADING, 'unit': '$', 'description': 'Largest losing trade'},
            
            # System metrics
            'cpu_usage': {'category': MetricCategory.SYSTEM, 'unit': '%', 'description': 'CPU utilization'},
            'memory_usage': {'category': MetricCategory.SYSTEM, 'unit': '%', 'description': 'Memory utilization'},
            'disk_usage': {'category': MetricCategory.SYSTEM, 'unit': '%', 'description': 'Disk space usage'},
            'response_time': {'category': MetricCategory.SYSTEM, 'unit': 'ms', 'description': 'System response time'},
            'uptime': {'category': MetricCategory.SYSTEM, 'unit': 'hours', 'description': 'System uptime'},
            
            # Market metrics
            'market_volatility': {'category': MetricCategory.MARKET, 'unit': '%', 'description': 'Market volatility'},
            'correlation_index': {'category': MetricCategory.MARKET, 'unit': '', 'description': 'Market correlation'},
            'regime_confidence': {'category': MetricCategory.MARKET, 'unit': '%', 'description': 'Regime detection confidence'},
            'volume_change': {'category': MetricCategory.MARKET, 'unit': '%', 'description': 'Volume change'},
            'price_impact': {'category': MetricCategory.MARKET, 'unit': 'bps', 'description': 'Market price impact'}
        }
        
        # Initialize metrics with default values
        for name, props in default_metrics.items():
            self.metrics[name] = PerformanceMetric(
                name=name,
                category=props['category'],
                value=0.0,
                timestamp=datetime.now(),
                unit=props['unit'],
                description=props['description']
            )
            self.metric_history[name] = deque(maxlen=self.config.history_length)

    def subscribe_to_updates(self, callback: Callable[[Dict[str, PerformanceMetric]], None]):
        """Subscribe to metric updates"""
        self.subscribers.append(callback)

    def update_metric(self, name: str, value: float, timestamp: datetime = None):
        """Update a performance metric"""
        try:
            with self.lock:
                if name in self.metrics:
                    metric = self.metrics[name]
                    metric.value = value
                    metric.timestamp = timestamp or datetime.now()
                    
                    # Check thresholds and set alert level
                    self._evaluate_alert_level(metric)
                    
                    # Add to history
                    self.metric_history[name].append({
                        'timestamp': metric.timestamp,
                        'value': metric.value
                    })
                    
                    # Notify subscribers
                    self._notify_subscribers()
                    
        except Exception as e:
            logger.error(f"Failed to update metric {name}: {e}")

    def _evaluate_alert_level(self, metric: PerformanceMetric):
        """Evaluate alert level based on thresholds"""
        try:
            thresholds = self.config.alert_thresholds.get(metric.name, {})
            warning_threshold = thresholds.get('warning')
            critical_threshold = thresholds.get('critical')
            
            if critical_threshold is not None:
                if ((critical_threshold > 0 and metric.value >= critical_threshold) or
                    (critical_threshold < 0 and metric.value <= critical_threshold)):
                    metric.alert_level = AlertLevel.CRITICAL
                    return
            
            if warning_threshold is not None:
                if ((warning_threshold > 0 and metric.value >= warning_threshold) or
                    (warning_threshold < 0 and metric.value <= warning_threshold)):
                    metric.alert_level = AlertLevel.WARNING
                    return
            
            metric.alert_level = AlertLevel.NORMAL
            
        except Exception as e:
            logger.error(f"Alert level evaluation failed: {e}")

    def _notify_subscribers(self):
        """Notify all subscribers of metric updates"""
        try:
            for callback in self.subscribers:
                try:
                    callback(self.metrics.copy())
                except Exception as e:
                    logger.error(f"Subscriber notification failed: {e}")
        except Exception as e:
            logger.error(f"Notification system failed: {e}")

    def get_metrics_by_category(self, category: MetricCategory) -> Dict[str, PerformanceMetric]:
        """Get metrics filtered by category"""
        return {name: metric for name, metric in self.metrics.items() 
                if metric.category == category}

    def get_alert_metrics(self) -> Dict[str, PerformanceMetric]:
        """Get metrics with active alerts"""
        return {name: metric for name, metric in self.metrics.items() 
                if metric.alert_level != AlertLevel.NORMAL}

    def get_metric_history(self, metric_name: str, hours_back: int = 24) -> List[Dict]:
        """Get historical data for a metric"""
        try:
            if metric_name not in self.metric_history:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            history = list(self.metric_history[metric_name])
            
            # Filter by time
            filtered_history = [
                point for point in history 
                if point['timestamp'] >= cutoff_time
            ]
            
            return filtered_history
            
        except Exception as e:
            logger.error(f"Failed to get metric history: {e}")
            return []

    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect portfolio metrics (would integrate with real system)
                self._collect_portfolio_metrics()
                
                # Collect risk metrics
                self._collect_risk_metrics()
                
                # Sleep for update frequency
                time.sleep(self.config.update_frequency)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)  # Wait longer on error

    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            # Simulate system metrics collection
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.update_metric('cpu_usage', cpu_percent)
            
            # Memory usage
            memory_percent = psutil.virtual_memory().percent
            self.update_metric('memory_usage', memory_percent)
            
            # Disk usage
            disk_percent = psutil.disk_usage('/').percent
            self.update_metric('disk_usage', disk_percent)
            
        except ImportError:
            # Fallback if psutil not available
            self.update_metric('cpu_usage', np.random.uniform(10, 90))
            self.update_metric('memory_usage', np.random.uniform(20, 80))
            self.update_metric('disk_usage', np.random.uniform(30, 70))
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")

    def _collect_portfolio_metrics(self):
        """Collect portfolio performance metrics"""
        try:
            # Simulate portfolio metrics (would integrate with real portfolio system)
            self.update_metric('portfolio_value', np.random.uniform(95000, 105000))
            self.update_metric('portfolio_pnl', np.random.uniform(-1000, 2000))
            self.update_metric('portfolio_return', np.random.uniform(-0.02, 0.03) * 100)
            self.update_metric('drawdown', abs(np.random.uniform(0, 0.05)) * 100)
            self.update_metric('sharpe_ratio', np.random.uniform(0.5, 2.5))
            
        except Exception as e:
            logger.error(f"Portfolio metrics collection failed: {e}")

    def _collect_risk_metrics(self):
        """Collect risk management metrics"""
        try:
            # Simulate risk metrics (would integrate with real risk system)
            self.update_metric('value_at_risk', np.random.uniform(1000, 3000))
            self.update_metric('conditional_var', np.random.uniform(1500, 4000))
            self.update_metric('portfolio_volatility', np.random.uniform(0.10, 0.25) * 100)
            self.update_metric('max_position_size', np.random.uniform(0.05, 0.20) * 100)
            self.update_metric('risk_exposure', np.random.uniform(0.30, 0.80) * 100)
            
        except Exception as e:
            logger.error(f"Risk metrics collection failed: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            with self.lock:
                # Organize metrics by category
                dashboard_data = {
                    'timestamp': datetime.now().isoformat(),
                    'categories': {},
                    'alerts': {},
                    'summary': {}
                }
                
                # Group metrics by category
                for category in MetricCategory:
                    category_metrics = self.get_metrics_by_category(category)
                    if category_metrics:
                        dashboard_data['categories'][category.value] = {
                            name: {
                                'value': metric.value,
                                'unit': metric.unit,
                                'description': metric.description,
                                'alert_level': metric.alert_level.value,
                                'timestamp': metric.timestamp.isoformat()
                            }
                            for name, metric in category_metrics.items()
                        }
                
                # Add alert information
                alert_metrics = self.get_alert_metrics()
                dashboard_data['alerts'] = {
                    name: {
                        'value': metric.value,
                        'level': metric.alert_level.value,
                        'description': metric.description
                    }
                    for name, metric in alert_metrics.items()
                }
                
                # Add summary statistics
                dashboard_data['summary'] = {
                    'total_metrics': len(self.metrics),
                    'active_alerts': len(alert_metrics),
                    'categories_monitored': len(dashboard_data['categories']),
                    'monitoring_active': self.monitoring_active
                }
                
                return dashboard_data
                
        except Exception as e:
            logger.error(f"Dashboard data collection failed: {e}")
            return {}

    def export_dashboard_data(self, filename: str = None) -> str:
        """Export dashboard data to JSON file"""
        try:
            data = self.get_dashboard_data()
            
            if filename is None:
                filename = f"dashboard_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Dashboard data exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Dashboard export failed: {e}")
            raise

class DashboardRenderer:
    """Dashboard visualization renderer"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        logger.info("Dashboard Renderer initialized")

    def render_console_dashboard(self):
        """Render dashboard to console"""
        try:
            data = self.monitor.get_dashboard_data()
            
            print("\n" + "="*60)
            print("📈 CHLOE 0.6 PERFORMANCE DASHBOARD")
            print("="*60)
            
            # Summary section
            summary = data.get('summary', {})
            print(f"📊 Summary:")
            print(f"   Total Metrics: {summary.get('total_metrics', 0)}")
            print(f"   Active Alerts: {summary.get('active_alerts', 0)}")
            print(f"   Categories: {summary.get('categories_monitored', 0)}")
            print(f"   Status: {'🟢 ACTIVE' if summary.get('monitoring_active', False) else '🔴 INACTIVE'}")
            
            # Alerts section
            alerts = data.get('alerts', {})
            if alerts:
                print(f"\n🚨 ACTIVE ALERTS:")
                for name, alert in alerts.items():
                    level_indicator = "⚠️" if alert['level'] == 'WARNING' else "🔴"
                    print(f"   {level_indicator} {name}: {alert['value']:.2f} - {alert['description']}")
            else:
                print(f"\n✅ NO ACTIVE ALERTS")
            
            # Metrics by category
            categories = data.get('categories', {})
            for category_name, metrics in categories.items():
                print(f"\n📁 {category_name.upper()} METRICS:")
                for name, metric in metrics.items():
                    alert_symbol = ""
                    if metric['alert_level'] == 'WARNING':
                        alert_symbol = " ⚠️"
                    elif metric['alert_level'] == 'CRITICAL':
                        alert_symbol = " 🔴"
                    
                    value_str = f"{metric['value']:.2f}"
                    if metric['unit']:
                        value_str += f" {metric['unit']}"
                    
                    print(f"   {name}: {value_str}{alert_symbol}")
                    if metric['description']:
                        print(f"      {metric['description']}")
            
            print("="*60)
            
        except Exception as e:
            logger.error(f"Console dashboard rendering failed: {e}")

    def render_simple_dashboard(self):
        """Render simple dashboard view"""
        try:
            # Portfolio section
            portfolio_value = self.monitor.metrics.get('portfolio_value', {}).value
            portfolio_pnl = self.monitor.metrics.get('portfolio_pnl', {}).value
            drawdown = self.monitor.metrics.get('drawdown', {}).value
            
            print(f"\n💰 PORTFOLIO OVERVIEW:")
            print(f"   Value: ${portfolio_value:,.2f}")
            print(f"   PNL: ${portfolio_pnl:+,.2f}")
            print(f"   Drawdown: {drawdown:.2f}%")
            
            # Risk section
            var = self.monitor.metrics.get('value_at_risk', {}).value
            volatility = self.monitor.metrics.get('portfolio_volatility', {}).value
            
            print(f"\n🛡️ RISK METRICS:")
            print(f"   VaR: ${var:,.2f}")
            print(f"   Volatility: {volatility:.2f}%")
            
            # System section
            cpu = self.monitor.metrics.get('cpu_usage', {}).value
            memory = self.monitor.metrics.get('memory_usage', {}).value
            
            print(f"\n🖥️ SYSTEM HEALTH:")
            print(f"   CPU: {cpu:.1f}%")
            print(f"   Memory: {memory:.1f}%")
            
        except Exception as e:
            logger.error(f"Simple dashboard rendering failed: {e}")

# Global instances
_performance_monitor = None
_dashboard_renderer = None

def get_performance_monitor(config: DashboardConfig = None) -> PerformanceMonitor:
    """Get singleton performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(config)
    return _performance_monitor

def get_dashboard_renderer() -> DashboardRenderer:
    """Get singleton dashboard renderer instance"""
    global _dashboard_renderer
    if _dashboard_renderer is None:
        monitor = get_performance_monitor()
        _dashboard_renderer = DashboardRenderer(monitor)
    return _dashboard_renderer

def main():
    """Example usage"""
    print("Performance Monitoring Dashboard ready")
    print("Real-time performance monitoring and visualization")

if __name__ == "__main__":
    main()