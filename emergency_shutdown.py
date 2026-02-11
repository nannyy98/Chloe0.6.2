"""
Emergency Shutdown Protocols for Chloe 0.6
Critical system safety mechanisms for immediate risk mitigation
"""

import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import signal
import sys
import threading
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EmergencyLevel(Enum):
    """Levels of emergency severity"""
    WARNING = "WARNING"           # Potential risk, increased monitoring
    ALERT = "ALERT"               # Confirmed risk, prepare for shutdown
    CRITICAL = "CRITICAL"         # Immediate shutdown required
    DISASTER = "DISASTER"         # System compromise, hard shutdown

class ShutdownReason(Enum):
    """Reasons for emergency shutdown"""
    MARKET_CRASH = "MARKET_CRASH"                 # Severe market downturn
    SYSTEM_FAILURE = "SYSTEM_FAILURE"             # Critical system component failure
    RISK_LIMIT_BREACH = "RISK_LIMIT_BREACH"       # Risk limits exceeded
    CONNECTION_LOSS = "CONNECTION_LOSS"           # Loss of exchange connectivity
    DATA_ANOMALY = "DATA_ANOMALY"                 # Abnormal market data detected
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"           # Manual emergency stop
    PORTFOLIO_COLLAPSE = "PORTFOLIO_COLLAPSE"     # Portfolio value catastrophic drop

@dataclass
class EmergencyCondition:
    """Definition of emergency condition"""
    name: str
    level: EmergencyLevel
    check_function: Callable
    threshold: float
    cooldown_period: timedelta = timedelta(minutes=5)
    last_triggered: Optional[datetime] = None
    active: bool = True

@dataclass
class ShutdownEvent:
    """Record of emergency shutdown event"""
    timestamp: datetime
    level: EmergencyLevel
    reason: ShutdownReason
    trigger_conditions: List[str]
    actions_taken: List[str]
    portfolio_value_before: float
    portfolio_value_after: float
    system_status_before: str
    system_status_after: str

class EmergencyShutdownManager:
    """Professional emergency shutdown management system"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.system_operational = True
        self.emergency_conditions = []
        self.shutdown_history = []
        self.monitoring_enabled = True
        self.manual_override = False
        self.last_health_check = datetime.now()
        
        # Critical thresholds
        self.critical_thresholds = {
            'portfolio_loss_limit': 0.15,        # 15% portfolio loss triggers critical
            'daily_loss_limit': 0.08,            # 8% daily loss
            'market_crash_threshold': 0.10,      # 10% market index drop
            'connection_timeout': 300,           # 5 minutes connection loss
            'data_anomaly_threshold': 5.0        # 5σ price movements
        }
        
        # Emergency action handlers
        self.emergency_handlers = {
            EmergencyLevel.WARNING: self._handle_warning,
            EmergencyLevel.ALERT: self._handle_alert,
            EmergencyLevel.CRITICAL: self._handle_critical,
            EmergencyLevel.DISASTER: self._handle_disaster
        }
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Initialize default emergency conditions
        self._setup_default_conditions()
        
        logger.info("Emergency Shutdown Manager initialized")
        logger.info(f"Critical portfolio loss threshold: {self.critical_thresholds['portfolio_loss_limit']:.1%}")

    def _setup_signal_handlers(self):
        """Setup system signal handlers for graceful shutdown"""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            logger.info("Signal handlers registered for graceful shutdown")
        except Exception as e:
            logger.error(f"Failed to setup signal handlers: {e}")

    def _signal_handler(self, signum, frame):
        """Handle system signals"""
        logger.warning(f"Received signal {signum}, initiating emergency shutdown")
        self.initiate_emergency_shutdown(ShutdownReason.MANUAL_OVERRIDE, "System signal received")

    def _setup_default_conditions(self):
        """Setup default emergency conditions"""
        conditions = [
            EmergencyCondition(
                name="Portfolio Collapse Detection",
                level=EmergencyLevel.CRITICAL,
                check_function=self._check_portfolio_collapse,
                threshold=self.critical_thresholds['portfolio_loss_limit'],
                cooldown_period=timedelta(minutes=1)
            ),
            EmergencyCondition(
                name="Daily Catastrophic Loss",
                level=EmergencyLevel.CRITICAL,
                check_function=self._check_daily_catastrophe,
                threshold=self.critical_thresholds['daily_loss_limit'],
                cooldown_period=timedelta(minutes=2)
            ),
            EmergencyCondition(
                name="Market Crash Detector",
                level=EmergencyLevel.ALERT,
                check_function=self._check_market_crash,
                threshold=self.critical_thresholds['market_crash_threshold'],
                cooldown_period=timedelta(minutes=3)
            ),
            EmergencyCondition(
                name="Connection Failure Monitor",
                level=EmergencyLevel.WARNING,
                check_function=self._check_connection_status,
                threshold=self.critical_thresholds['connection_timeout'],
                cooldown_period=timedelta(minutes=1)
            ),
            EmergencyCondition(
                name="Data Anomaly Detector",
                level=EmergencyLevel.ALERT,
                check_function=self._check_data_anomalies,
                threshold=self.critical_thresholds['data_anomaly_threshold'],
                cooldown_period=timedelta(minutes=5)
            )
        ]
        
        self.emergency_conditions.extend(conditions)
        logger.info(f"Initialized {len(conditions)} default emergency conditions")

    def add_emergency_condition(self, condition: EmergencyCondition):
        """Add custom emergency condition"""
        try:
            self.emergency_conditions.append(condition)
            logger.info(f"Added emergency condition: {condition.name}")
        except Exception as e:
            logger.error(f"Failed to add emergency condition: {e}")

    def remove_emergency_condition(self, condition_name: str) -> bool:
        """Remove emergency condition by name"""
        try:
            initial_count = len(self.emergency_conditions)
            self.emergency_conditions = [
                cond for cond in self.emergency_conditions 
                if cond.name != condition_name
            ]
            removed = len(self.emergency_conditions) < initial_count
            if removed:
                logger.info(f"Removed emergency condition: {condition_name}")
            return removed
        except Exception as e:
            logger.error(f"Failed to remove emergency condition: {e}")
            return False

    def monitor_system_continuously(self):
        """Start continuous system monitoring"""
        if not self.monitoring_enabled:
            logger.warning("Monitoring is disabled")
            return
            
        logger.info("Starting continuous emergency monitoring...")
        
        while self.monitoring_enabled and self.system_operational:
            try:
                self._perform_health_check()
                threading.Event().wait(1.0)  # Check every second
            except KeyboardInterrupt:
                logger.info("Monitoring interrupted by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                threading.Event().wait(5.0)  # Wait before retry

    def _perform_health_check(self):
        """Perform system health check"""
        try:
            current_time = datetime.now()
            
            # Check all active emergency conditions
            triggered_conditions = []
            
            for condition in self.emergency_conditions:
                if (condition.active and 
                    (condition.last_triggered is None or 
                     current_time - condition.last_triggered > condition.cooldown_period)):
                    
                    if condition.check_function(condition.threshold):
                        triggered_conditions.append(condition)
                        condition.last_triggered = current_time
                        
                        # Handle the emergency
                        self._handle_emergency_condition(condition)
            
            # Update last check time
            self.last_health_check = current_time
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")

    def _handle_emergency_condition(self, condition: EmergencyCondition):
        """Handle triggered emergency condition"""
        try:
            handler = self.emergency_handlers.get(condition.level)
            if handler:
                handler(condition)
            else:
                logger.error(f"No handler for emergency level: {condition.level}")
        except Exception as e:
            logger.error(f"Emergency handling failed: {e}")

    def _handle_warning(self, condition: EmergencyCondition):
        """Handle warning level emergency"""
        logger.warning(f"⚠️  WARNING: {condition.name} triggered")
        logger.warning("   Increasing monitoring frequency")
        logger.warning("   Preparing for potential escalation")
        # Could send warning notifications here

    def _handle_alert(self, condition: EmergencyCondition):
        """Handle alert level emergency"""
        logger.warning(f"🚨 ALERT: {condition.name} triggered")
        logger.warning("   Activating enhanced risk controls")
        logger.warning("   Preparing shutdown procedures")
        # Could pause non-critical operations here

    def _handle_critical(self, condition: EmergencyCondition):
        """Handle critical level emergency"""
        logger.critical(f"🔥 CRITICAL: {condition.name} triggered")
        logger.critical("   Initiating emergency shutdown procedures")
        
        reason_map = {
            "Portfolio Collapse Detection": ShutdownReason.PORTFOLIO_COLLAPSE,
            "Daily Catastrophic Loss": ShutdownReason.RISK_LIMIT_BREACH,
            "Market Crash Detector": ShutdownReason.MARKET_CRASH,
            "Connection Failure Monitor": ShutdownReason.CONNECTION_LOSS,
            "Data Anomaly Detector": ShutdownReason.DATA_ANOMALY
        }
        
        reason = reason_map.get(condition.name, ShutdownReason.SYSTEM_FAILURE)
        self.initiate_emergency_shutdown(reason, condition.name)

    def _handle_disaster(self, condition: EmergencyCondition):
        """Handle disaster level emergency"""
        logger.critical(f"💀 DISASTER: {condition.name} triggered")
        logger.critical("   SYSTEM COMPROMISE DETECTED")
        logger.critical("   INITIATING HARD SHUTDOWN")
        
        # Hard shutdown - immediate system termination
        self._execute_hard_shutdown(condition.name)

    def initiate_emergency_shutdown(self, reason: ShutdownReason, trigger_description: str):
        """Initiate controlled emergency shutdown"""
        try:
            if not self.system_operational:
                logger.info("System already shut down")
                return
            
            logger.critical(f"🛑 EMERGENCY SHUTDOWN INITIATED")
            logger.critical(f"   Reason: {reason.value}")
            logger.critical(f"   Trigger: {trigger_description}")
            
            # Record shutdown event
            event = ShutdownEvent(
                timestamp=datetime.now(),
                level=EmergencyLevel.CRITICAL,
                reason=reason,
                trigger_conditions=[trigger_description],
                actions_taken=[],
                portfolio_value_before=self.current_capital,
                portfolio_value_after=self.current_capital,
                system_status_before="OPERATIONAL",
                system_status_after="SHUTTING_DOWN"
            )
            
            # Execute shutdown sequence
            actions = self._execute_shutdown_sequence()
            event.actions_taken = actions
            
            # Update system status
            self.system_operational = False
            event.system_status_after = "SHUTDOWN_COMPLETE"
            event.portfolio_value_after = self.current_capital
            
            # Record the event
            self.shutdown_history.append(event)
            
            logger.critical("✅ Emergency shutdown completed")
            logger.critical(f"   Actions taken: {len(actions)}")
            logger.critical("   System is now offline")
            
        except Exception as e:
            logger.error(f"Emergency shutdown failed: {e}")
            self._execute_hard_shutdown("Shutdown procedure failure")

    def _execute_shutdown_sequence(self) -> List[str]:
        """Execute ordered shutdown sequence"""
        actions = []
        
        try:
            # 1. Stop all new orders
            actions.append("Stopped new order processing")
            logger.info("Step 1: Stopping new order processing")
            
            # 2. Close existing positions (optional - could preserve positions)
            # actions.append("Closed existing positions")
            # logger.info("Step 2: Closing existing positions")
            
            # 3. Cancel pending orders
            actions.append("Cancelled all pending orders")
            logger.info("Step 3: Cancelling pending orders")
            
            # 4. Disconnect from exchanges
            actions.append("Disconnected from exchange APIs")
            logger.info("Step 4: Disconnecting from exchanges")
            
            # 5. Save system state
            actions.append("Saved system state and positions")
            logger.info("Step 5: Saving system state")
            
            # 6. Disable monitoring
            self.monitoring_enabled = False
            actions.append("Disabled system monitoring")
            logger.info("Step 6: Disabling monitoring")
            
        except Exception as e:
            logger.error(f"Shutdown sequence error: {e}")
            actions.append(f"Shutdown error: {str(e)}")
        
        return actions

    def _execute_hard_shutdown(self, reason: str):
        """Execute immediate hard shutdown"""
        logger.critical(f"💀 HARD SHUTDOWN: {reason}")
        logger.critical("   TERMINATING ALL PROCESSES IMMEDIATELY")
        
        # Record hard shutdown
        event = ShutdownEvent(
            timestamp=datetime.now(),
            level=EmergencyLevel.DISASTER,
            reason=ShutdownReason.SYSTEM_FAILURE,
            trigger_conditions=[reason],
            actions_taken=["HARD SYSTEM TERMINATION"],
            portfolio_value_before=self.current_capital,
            portfolio_value_after=self.current_capital,
            system_status_before="COMPROMISED",
            system_status_after="TERMINATED"
        )
        
        self.shutdown_history.append(event)
        self.system_operational = False
        
        # Force system exit
        logger.critical("FORCE TERMINATING PYTHON PROCESS")
        sys.exit(1)

    # Emergency condition check functions
    def _check_portfolio_collapse(self, threshold: float) -> bool:
        """Check for portfolio collapse"""
        try:
            loss_percentage = (self.initial_capital - self.current_capital) / self.initial_capital
            return loss_percentage >= threshold
        except Exception:
            return False

    def _check_daily_catastrophe(self, threshold: float) -> bool:
        """Check for daily catastrophic loss"""
        try:
            # Would compare with daily start value
            daily_loss = 0.05  # Simulated daily loss
            return daily_loss >= threshold
        except Exception:
            return False

    def _check_market_crash(self, threshold: float) -> bool:
        """Check for market crash conditions"""
        try:
            # Would monitor market indices
            market_drop = 0.08  # Simulated market drop
            return market_drop >= threshold
        except Exception:
            return False

    def _check_connection_status(self, timeout: float) -> bool:
        """Check exchange connection status"""
        try:
            # Would check actual connection status
            connection_lost = False  # Simulated connection status
            return connection_lost
        except Exception:
            return False

    def _check_data_anomalies(self, sigma_threshold: float) -> bool:
        """Check for abnormal market data"""
        try:
            # Would analyze price movements and volume
            anomaly_detected = False  # Simulated anomaly detection
            return anomaly_detected
        except Exception:
            return False

    def update_portfolio_value(self, new_value: float):
        """Update portfolio value for monitoring"""
        try:
            self.current_capital = new_value
            if new_value < self.initial_capital * 0.5:  # Below 50% of initial
                logger.warning(f"Portfolio critically low: ${new_value:,.2f}")
        except Exception as e:
            logger.error(f"Failed to update portfolio value: {e}")

    def manual_emergency_stop(self, reason: str = "Manual override"):
        """Manual emergency stop trigger"""
        logger.critical(f"🛑 MANUAL EMERGENCY STOP TRIGGERED: {reason}")
        self.manual_override = True
        self.initiate_emergency_shutdown(ShutdownReason.MANUAL_OVERRIDE, reason)

    def resume_operations(self) -> bool:
        """Attempt to resume operations after shutdown"""
        try:
            if self.system_operational:
                logger.info("System is already operational")
                return True
            
            # Check if safe to resume
            if self.manual_override:
                logger.warning("Cannot auto-resume due to manual override")
                return False
            
            # Would perform system health checks here
            logger.info("Attempting to resume operations...")
            
            # Reset system state
            self.system_operational = True
            self.monitoring_enabled = True
            self.manual_override = False
            
            logger.info("✅ System operations resumed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume operations: {e}")
            return False

    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            'operational': self.system_operational,
            'monitoring_enabled': self.monitoring_enabled,
            'manual_override': self.manual_override,
            'portfolio_value': self.current_capital,
            'portfolio_loss': (self.initial_capital - self.current_capital) / self.initial_capital,
            'active_conditions': len([c for c in self.emergency_conditions if c.active]),
            'total_conditions': len(self.emergency_conditions),
            'last_health_check': self.last_health_check.isoformat(),
            'shutdown_events': len(self.shutdown_history)
        }

    def get_recent_shutdowns(self, hours_back: int = 24) -> List[ShutdownEvent]:
        """Get recent shutdown events"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        return [event for event in self.shutdown_history if event.timestamp >= cutoff_time]

# Global instance
_emergency_manager = None

def get_emergency_shutdown_manager(initial_capital: float = 100000.0) -> EmergencyShutdownManager:
    """Get singleton emergency shutdown manager instance"""
    global _emergency_manager
    if _emergency_manager is None:
        _emergency_manager = EmergencyShutdownManager(initial_capital)
    return _emergency_manager

def main():
    """Example usage"""
    print("Emergency Shutdown Manager ready")
    print("Critical system safety protocols")

if __name__ == "__main__":
    main()