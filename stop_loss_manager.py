"""
Systemic Stop-Loss Mechanisms for Chloe 0.6
Professional portfolio and position-level risk protection
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class StopLossType(Enum):
    """Types of stop-loss mechanisms"""
    PORTFOLIO_DRAWDOWN = "PORTFOLIO_DRAWDOWN"      # Portfolio-level maximum drawdown
    POSITION_LOSS = "POSITION_LOSS"                # Individual position loss limits
    DAILY_LOSS = "DAILY_LOSS"                      # Daily maximum loss
    TRAILING_STOP = "TRAILING_STOP"                # Trailing stop based on peaks
    TIME_BASED = "TIME_BASED"                      # Time-based expiration

@dataclass
class StopLossRule:
    """Individual stop-loss rule definition"""
    rule_type: StopLossType
    threshold: float                           # Threshold value (percentage or absolute)
    symbol: Optional[str] = None               # Specific symbol (None = portfolio-wide)
    active: bool = True                        # Whether rule is currently active
    triggered: bool = False                    # Whether rule has been triggered
    trigger_time: Optional[datetime] = None    # When rule was triggered
    description: str = ""                      # Human-readable description

@dataclass
class StopLossEvent:
    """Record of stop-loss trigger event"""
    timestamp: datetime
    rule_type: StopLossType
    symbol: Optional[str]
    trigger_value: float
    threshold: float
    action_taken: str                          # Description of protective action
    portfolio_value_before: float
    portfolio_value_after: float

class StopLossManager:
    """Professional systemic stop-loss management system"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.high_water_mark = initial_capital
        self.daily_start_value = initial_capital
        self.daily_high_water_mark = initial_capital
        self.position_peaks = {}                   # Track peak values for trailing stops
        self.stop_loss_rules = []                  # Active stop-loss rules
        self.trigger_history = []                  # History of triggered events
        self.protection_enabled = True             # Master switch for all protections
        
        # Default protection levels
        self.default_limits = {
            'portfolio_max_drawdown': 0.05,        # 5% portfolio drawdown
            'position_max_loss': 0.03,             # 3% per position loss
            'daily_max_loss': 0.02,                # 2% daily loss
            'trailing_stop_distance': 0.05         # 5% trailing distance
        }
        
        logger.info(f"Stop-Loss Manager initialized with ${initial_capital:,.2f} capital")
        self._setup_default_protections()

    def _setup_default_protections(self):
        """Setup default stop-loss protections"""
        # Portfolio-level drawdown protection
        self.add_stop_loss_rule(
            StopLossType.PORTFOLIO_DRAWDOWN,
            self.default_limits['portfolio_max_drawdown'],
            description=f"Portfolio maximum drawdown protection ({self.default_limits['portfolio_max_drawdown']:.1%})"
        )
        
        # Daily loss protection
        self.add_stop_loss_rule(
            StopLossType.DAILY_LOSS,
            self.default_limits['daily_max_loss'],
            description=f"Daily maximum loss protection ({self.default_limits['daily_max_loss']:.1%})"
        )
        
        logger.info("✅ Default stop-loss protections activated")

    def add_stop_loss_rule(self, rule_type: StopLossType, threshold: float,
                          symbol: Optional[str] = None, description: str = "") -> StopLossRule:
        """Add a new stop-loss rule"""
        try:
            rule = StopLossRule(
                rule_type=rule_type,
                threshold=threshold,
                symbol=symbol,
                description=description or f"{rule_type.value} protection at {threshold:.1%}"
            )
            
            self.stop_loss_rules.append(rule)
            logger.info(f"➕ Added stop-loss rule: {rule.description}")
            return rule
            
        except Exception as e:
            logger.error(f"Failed to add stop-loss rule: {e}")
            raise

    def remove_stop_loss_rule(self, rule_index: int) -> bool:
        """Remove stop-loss rule by index"""
        try:
            if 0 <= rule_index < len(self.stop_loss_rules):
                removed_rule = self.stop_loss_rules.pop(rule_index)
                logger.info(f"➖ Removed stop-loss rule: {removed_rule.description}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove stop-loss rule: {e}")
            return False

    def update_portfolio_value(self, current_value: float, timestamp: datetime = None):
        """Update portfolio value and check protections"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            previous_value = self.current_capital
            self.current_capital = current_value
            
            # Update high water marks
            if current_value > self.high_water_mark:
                self.high_water_mark = current_value
                logger.debug(f"New portfolio high water mark: ${current_value:,.2f}")
            
            # Update daily tracking
            self._update_daily_tracking(timestamp)
            
            # Check all active protections
            triggered_actions = []
            for rule in self.stop_loss_rules:
                if rule.active and not rule.triggered:
                    action = self._check_single_rule(rule, timestamp)
                    if action:
                        triggered_actions.append(action)
            
            return triggered_actions
            
        except Exception as e:
            logger.error(f"Portfolio value update failed: {e}")
            return []

    def _update_daily_tracking(self, timestamp: datetime):
        """Update daily tracking for daily loss protection"""
        # Reset daily tracking if new day
        if hasattr(self, '_last_check_date'):
            if timestamp.date() != self._last_check_date:
                self.daily_start_value = self.current_capital
                self.daily_high_water_mark = self.current_capital
                self._last_check_date = timestamp.date()
        else:
            self._last_check_date = timestamp.date()

    def _check_single_rule(self, rule: StopLossRule, timestamp: datetime) -> Optional[str]:
        """Check individual stop-loss rule"""
        try:
            action = None
            
            if rule.rule_type == StopLossType.PORTFOLIO_DRAWDOWN:
                action = self._check_portfolio_drawdown(rule, timestamp)
            
            elif rule.rule_type == StopLossType.POSITION_LOSS:
                action = self._check_position_loss(rule, timestamp)
            
            elif rule.rule_type == StopLossType.DAILY_LOSS:
                action = self._check_daily_loss(rule, timestamp)
            
            elif rule.rule_type == StopLossType.TRAILING_STOP:
                action = self._check_trailing_stop(rule, timestamp)
            
            elif rule.rule_type == StopLossType.TIME_BASED:
                action = self._check_time_based(rule, timestamp)
            
            # Record trigger event
            if action and not rule.triggered:
                rule.triggered = True
                rule.trigger_time = timestamp
                self._record_trigger_event(rule, timestamp, action)
                logger.warning(f"🚨 STOP-LOSS TRIGGERED: {rule.description}")
                logger.warning(f"   Action taken: {action}")
            
            return action
            
        except Exception as e:
            logger.error(f"Rule checking failed for {rule.rule_type}: {e}")
            return None

    def _check_portfolio_drawdown(self, rule: StopLossRule, timestamp: datetime) -> Optional[str]:
        """Check portfolio drawdown protection"""
        try:
            current_drawdown = (self.high_water_mark - self.current_capital) / self.high_water_mark
            
            if current_drawdown >= rule.threshold:
                return self._execute_portfolio_protection(rule, current_drawdown)
            return None
            
        except Exception as e:
            logger.error(f"Portfolio drawdown check failed: {e}")
            return None

    def _check_position_loss(self, rule: StopLossRule, timestamp: datetime) -> Optional[str]:
        """Check individual position loss limits"""
        try:
            # This would integrate with actual position data
            # For now, simulating position loss checking
            if rule.symbol:
                # Check specific position
                position_loss = self._calculate_position_loss(rule.symbol)
                if position_loss >= rule.threshold:
                    return self._execute_position_protection(rule, position_loss)
            else:
                # Check all positions
                for symbol in self.position_peaks.keys():
                    position_loss = self._calculate_position_loss(symbol)
                    if position_loss >= rule.threshold:
                        temp_rule = StopLossRule(
                            rule_type=rule.rule_type,
                            threshold=rule.threshold,
                            symbol=symbol,
                            description=f"Position loss protection for {symbol}"
                        )
                        return self._execute_position_protection(temp_rule, position_loss)
            
            return None
            
        except Exception as e:
            logger.error(f"Position loss check failed: {e}")
            return None

    def _check_daily_loss(self, rule: StopLossRule, timestamp: datetime) -> Optional[str]:
        """Check daily maximum loss protection"""
        try:
            daily_return = (self.current_capital - self.daily_start_value) / self.daily_start_value
            daily_drawdown = max(0, -daily_return)
            
            if daily_drawdown >= rule.threshold:
                return self._execute_daily_protection(rule, daily_drawdown)
            return None
            
        except Exception as e:
            logger.error(f"Daily loss check failed: {e}")
            return None

    def _check_trailing_stop(self, rule: StopLossType, timestamp: datetime) -> Optional[str]:
        """Check trailing stop protection"""
        try:
            if rule.symbol:
                # Check trailing stop for specific symbol
                trailing_loss = self._calculate_trailing_loss(rule.symbol)
                if trailing_loss >= rule.threshold:
                    return self._execute_trailing_protection(rule, trailing_loss)
            else:
                # Check trailing stops for all positions
                for symbol in self.position_peaks.keys():
                    trailing_loss = self._calculate_trailing_loss(symbol)
                    if trailing_loss >= rule.threshold:
                        temp_rule = StopLossRule(
                            rule_type=rule.rule_type,
                            threshold=rule.threshold,
                            symbol=symbol,
                            description=f"Trailing stop for {symbol}"
                        )
                        return self._execute_trailing_protection(temp_rule, trailing_loss)
            
            return None
            
        except Exception as e:
            logger.error(f"Trailing stop check failed: {e}")
            return None

    def _check_time_based(self, rule: StopLossRule, timestamp: datetime) -> Optional[str]:
        """Check time-based expiration"""
        try:
            # This would check if positions have been held too long
            # Implementation depends on specific time-based rules
            return None
        except Exception as e:
            logger.error(f"Time-based check failed: {e}")
            return None

    def _execute_portfolio_protection(self, rule: StopLossRule, drawdown: float) -> str:
        """Execute portfolio-level protection"""
        try:
            action = f"Reduce portfolio exposure by {drawdown:.1%}, trigger value: {drawdown:.1%}"
            # In real implementation, this would:
            # 1. Reduce overall position sizes
            # 2. Move to safer assets
            # 3. Increase cash holdings
            # 4. Possibly halt new trades
            return action
        except Exception as e:
            logger.error(f"Portfolio protection execution failed: {e}")
            return "Protection execution failed"

    def _execute_position_protection(self, rule: StopLossRule, loss: float) -> str:
        """Execute position-level protection"""
        try:
            action = f"Close position in {rule.symbol} with {loss:.1%} loss"
            # In real implementation, this would:
            # 1. Close the specific position
            # 2. Prevent re-entry for specified period
            # 3. Log the loss for analysis
            return action
        except Exception as e:
            logger.error(f"Position protection execution failed: {e}")
            return "Protection execution failed"

    def _execute_daily_protection(self, rule: StopLossRule, loss: float) -> str:
        """Execute daily loss protection"""
        try:
            action = f"Halt trading for remainder of day, daily loss: {loss:.1%}"
            # In real implementation, this would:
            # 1. Stop all new trades
            # 2. Optionally close existing positions
            # 3. Wait until next trading day
            return action
        except Exception as e:
            logger.error(f"Daily protection execution failed: {e}")
            return "Protection execution failed"

    def _execute_trailing_protection(self, rule: StopLossRule, loss: float) -> str:
        """Execute trailing stop protection"""
        try:
            action = f"Close {rule.symbol} position at trailing stop loss: {loss:.1%}"
            # In real implementation, this would:
            # 1. Close position at current market price
            # 2. Update position records
            # 3. Prevent immediate re-entry
            return action
        except Exception as e:
            logger.error(f"Trailing protection execution failed: {e}")
            return "Protection execution failed"

    def _calculate_position_loss(self, symbol: str) -> float:
        """Calculate current loss for a position"""
        # Simulated calculation - would use actual position data
        return np.random.uniform(0, 0.05)  # 0-5% loss

    def _calculate_trailing_loss(self, symbol: str) -> float:
        """Calculate trailing loss from peak"""
        # Simulated calculation - would track actual peaks
        return np.random.uniform(0, 0.08)  # 0-8% trailing loss

    def _record_trigger_event(self, rule: StopLossRule, timestamp: datetime, action: str):
        """Record stop-loss trigger event"""
        try:
            event = StopLossEvent(
                timestamp=timestamp,
                rule_type=rule.rule_type,
                symbol=rule.symbol,
                trigger_value=getattr(rule, 'trigger_value', rule.threshold),
                threshold=rule.threshold,
                action_taken=action,
                portfolio_value_before=self.current_capital,
                portfolio_value_after=self.current_capital  # Would be updated after action
            )
            self.trigger_history.append(event)
        except Exception as e:
            logger.error(f"Failed to record trigger event: {e}")

    def get_active_protections(self) -> List[StopLossRule]:
        """Get list of currently active protections"""
        return [rule for rule in self.stop_loss_rules if rule.active and not rule.triggered]

    def get_triggered_events(self, hours_back: int = 24) -> List[StopLossEvent]:
        """Get recently triggered events"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        return [event for event in self.trigger_history if event.timestamp >= cutoff_time]

    def reset_protections(self):
        """Reset all triggered protections"""
        try:
            reset_count = 0
            for rule in self.stop_loss_rules:
                if rule.triggered:
                    rule.triggered = False
                    rule.trigger_time = None
                    reset_count += 1
            
            logger.info(f"Reset {reset_count} triggered protections")
            return reset_count
        except Exception as e:
            logger.error(f"Protection reset failed: {e}")
            return 0

    def disable_protection(self, protection_type: StopLossType = None):
        """Temporarily disable protections"""
        if protection_type:
            for rule in self.stop_loss_rules:
                if rule.rule_type == protection_type:
                    rule.active = False
                    logger.info(f"Disabled {protection_type.value} protection")
        else:
            self.protection_enabled = False
            logger.warning("All stop-loss protections disabled")

    def enable_protection(self, protection_type: StopLossType = None):
        """Re-enable protections"""
        if protection_type:
            for rule in self.stop_loss_rules:
                if rule.rule_type == protection_type:
                    rule.active = True
                    logger.info(f"Enabled {protection_type.value} protection")
        else:
            self.protection_enabled = True
            logger.info("All stop-loss protections enabled")

    def get_protection_status(self) -> Dict:
        """Get comprehensive protection status"""
        return {
            'capital': self.current_capital,
            'high_water_mark': self.high_water_mark,
            'current_drawdown': (self.high_water_mark - self.current_capital) / self.high_water_mark,
            'active_rules': len(self.get_active_protections()),
            'triggered_rules': len([r for r in self.stop_loss_rules if r.triggered]),
            'total_rules': len(self.stop_loss_rules),
            'protection_enabled': self.protection_enabled,
            'recent_triggers': len(self.get_triggered_events(24))
        }

# Global instance
_stop_loss_manager = None

def get_stop_loss_manager(initial_capital: float = 100000.0) -> StopLossManager:
    """Get singleton stop-loss manager instance"""
    global _stop_loss_manager
    if _stop_loss_manager is None:
        _stop_loss_manager = StopLossManager(initial_capital)
    return _stop_loss_manager

def main():
    """Example usage"""
    print("Systemic Stop-Loss Manager ready")
    print("Professional portfolio protection system")

if __name__ == "__main__":
    main()