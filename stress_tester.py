"""
Stress Testing on Crisis Data for Chloe 0.6
Professional stress testing using historical crisis scenarios
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class CrisisScenario(Enum):
    """Historical crisis scenarios for stress testing"""
    MAR2020_PANIC = "MAR2020_PANIC"           # March 2020 COVID-19 market crash
    OCT2008_FINANCIAL = "OCT2008_FINANCIAL"   # 2008 Financial crisis
    MAY2010_FLASH = "MAY2010_FLASH"           # 2010 Flash crash
    JUN2013_TAPER = "JUN2013_TAPER"           # 2013 Taper tantrum
    JAN2018_VOL = "JAN2018_VOL"               # 2018 Volatility shock
    NOV2021_LUNA = "NOV2021_LUNA"             # Luna/UST collapse
    NOV2022_FTX = "NOV2022_FTX"               # FTX collapse
    MAR2023_BANKS = "MAR2023_BANKS"           # Banking crisis fears

@dataclass
class CrisisParameters:
    """Parameters defining a crisis scenario"""
    name: str
    duration_days: int
    max_drawdown: float          # Maximum portfolio drawdown expected
    volatility_spike: float      # Volatility multiplier
    correlation_increase: float  # Increase in asset correlations
    liquidity_drop: float        # Liquidity reduction factor
    volume_spike: float          # Trading volume spike
    start_date: datetime
    end_date: datetime

@dataclass
class StressTestResult:
    """Results from stress test scenario"""
    scenario: CrisisScenario
    start_date: datetime
    end_date: datetime
    initial_portfolio_value: float
    final_portfolio_value: float
    max_drawdown: float
    max_daily_loss: float
    volatility_during_crisis: float
    sharpe_ratio_during_crisis: float
    number_of_trades: int
    stop_losses_triggered: int
    emergency_shutdowns: int
    recovery_time_days: int
    performance_vs_benchmark: float  # Relative performance

class CrisisDataGenerator:
    """Generate realistic crisis scenario data"""
    
    def __init__(self):
        self.crisis_scenarios = self._define_crisis_scenarios()
        logger.info("Crisis Data Generator initialized")

    def _define_crisis_scenarios(self) -> Dict[CrisisScenario, CrisisParameters]:
        """Define historical crisis parameters"""
        return {
            CrisisScenario.MAR2020_PANIC: CrisisParameters(
                name="March 2020 COVID-19 Panic",
                duration_days=45,
                max_drawdown=0.35,        # 35% drawdown
                volatility_spike=3.0,     # 3x normal volatility
                correlation_increase=0.4, # 40% increase in correlations
                liquidity_drop=0.6,       # 40% liquidity reduction
                volume_spike=2.5,         # 2.5x normal volume
                start_date=datetime(2020, 2, 20),
                end_date=datetime(2020, 4, 5)
            ),
            CrisisScenario.OCT2008_FINANCIAL: CrisisParameters(
                name="October 2008 Financial Crisis",
                duration_days=90,
                max_drawdown=0.45,
                volatility_spike=4.0,
                correlation_increase=0.6,
                liquidity_drop=0.4,
                volume_spike=1.8,
                start_date=datetime(2008, 9, 15),
                end_date=datetime(2008, 12, 15)
            ),
            CrisisScenario.MAY2010_FLASH: CrisisParameters(
                name="May 2010 Flash Crash",
                duration_days=1,
                max_drawdown=0.10,
                volatility_spike=8.0,
                correlation_increase=0.8,
                liquidity_drop=0.9,
                volume_spike=5.0,
                start_date=datetime(2010, 5, 6),
                end_date=datetime(2010, 5, 6)
            ),
            CrisisScenario.JUN2013_TAPER: CrisisParameters(
                name="June 2013 Taper Tantrum",
                duration_days=60,
                max_drawdown=0.15,
                volatility_spike=2.5,
                correlation_increase=0.3,
                liquidity_drop=0.7,
                volume_spike=2.0,
                start_date=datetime(2013, 5, 1),
                end_date=datetime(2013, 7, 1)
            ),
            CrisisScenario.JAN2018_VOL: CrisisParameters(
                name="January 2018 Volatility Shock",
                duration_days=30,
                max_drawdown=0.12,
                volatility_spike=3.5,
                correlation_increase=0.2,
                liquidity_drop=0.8,
                volume_spike=3.0,
                start_date=datetime(2018, 1, 24),
                end_date=datetime(2018, 2, 24)
            ),
            CrisisScenario.NOV2021_LUNA: CrisisParameters(
                name="November 2021 Luna/UST Collapse",
                duration_days=14,
                max_drawdown=0.25,
                volatility_spike=5.0,
                correlation_increase=0.5,
                liquidity_drop=0.5,
                volume_spike=4.0,
                start_date=datetime(2021, 11, 9),
                end_date=datetime(2021, 11, 23)
            ),
            CrisisScenario.NOV2022_FTX: CrisisParameters(
                name="November 2022 FTX Collapse",
                duration_days=21,
                max_drawdown=0.30,
                volatility_spike=4.5,
                correlation_increase=0.6,
                liquidity_drop=0.4,
                volume_spike=3.5,
                start_date=datetime(2022, 11, 8),
                end_date=datetime(2022, 11, 29)
            ),
            CrisisScenario.MAR2023_BANKS: CrisisParameters(
                name="March 2023 Banking Crisis",
                duration_days=30,
                max_drawdown=0.20,
                volatility_spike=2.8,
                correlation_increase=0.4,
                liquidity_drop=0.6,
                volume_spike=2.2,
                start_date=datetime(2023, 3, 10),
                end_date=datetime(2023, 4, 10)
            )
        }

    def generate_crisis_data(self, scenario: CrisisScenario, 
                           base_data: pd.DataFrame) -> pd.DataFrame:
        """Generate crisis-affected market data"""
        try:
            params = self.crisis_scenarios[scenario]
            
            # Copy base data
            crisis_data = base_data.copy()
            
            # Apply crisis effects
            crisis_multiplier = 1 - params.max_drawdown
            volatility_multiplier = params.volatility_spike
            
            # Modify prices to reflect crisis
            for column in ['open', 'high', 'low', 'close']:
                if column in crisis_data.columns:
                    # Apply drawdown
                    crisis_data[column] = crisis_data[column] * crisis_multiplier
                    
                    # Add crisis volatility
                    noise = np.random.normal(0, 0.02 * volatility_multiplier, len(crisis_data))
                    crisis_data[column] = crisis_data[column] * (1 + noise)
            
            # Modify volume
            if 'volume' in crisis_data.columns:
                crisis_data['volume'] = crisis_data['volume'] * params.volume_spike
            
            # Ensure logical price relationships
            crisis_data['high'] = np.maximum(crisis_data['open'], crisis_data['close']) * (1 + np.abs(noise) * 0.5)
            crisis_data['low'] = np.minimum(crisis_data['open'], crisis_data['close']) * (1 - np.abs(noise) * 0.5)
            
            return crisis_data
            
        except Exception as e:
            logger.error(f"Crisis data generation failed: {e}")
            return base_data

class StressTester:
    """Professional stress testing engine"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.crisis_generator = CrisisDataGenerator()
        self.test_results = []
        self.risk_controls = {
            'max_daily_loss': 0.05,      # 5% maximum daily loss
            'max_portfolio_loss': 0.15,  # 15% maximum portfolio loss
            'stop_loss_enabled': True,
            'emergency_shutdown_enabled': True
        }
        
        logger.info(f"Stress Tester initialized with ${initial_capital:,.2f}")

    def run_stress_test(self, scenario: CrisisScenario, 
                       market_data: Dict[str, pd.DataFrame],
                       strategy_function) -> StressTestResult:
        """Run stress test for specific crisis scenario"""
        try:
            logger.info(f"🧪 Running stress test: {scenario.value}")
            
            params = self.crisis_generator.crisis_scenarios[scenario]
            start_value = self.current_capital
            peak_value = start_value
            max_drawdown = 0.0
            max_daily_loss = 0.0
            trades_executed = 0
            stop_losses_triggered = 0
            emergency_shutdowns = 0
            daily_values = []
            
            # Generate crisis data for each asset
            crisis_market_data = {}
            for symbol, data in market_data.items():
                crisis_data = self.crisis_generator.generate_crisis_data(scenario, data)
                crisis_market_data[symbol] = crisis_data
            
            # Simulate trading during crisis period
            crisis_dates = pd.date_range(params.start_date, params.end_date, freq='D')
            crisis_dates = crisis_dates[crisis_dates.weekday < 5]  # Weekdays only
            
            for current_date in crisis_dates:
                daily_start_value = self.current_capital
                
                # Get market data for current date
                daily_data = {}
                for symbol, df in crisis_market_data.items():
                    if current_date in df.index:
                        daily_data[symbol] = df.loc[current_date]
                
                if not daily_data:
                    continue
                
                # Execute trading strategy
                trade_signals = strategy_function(daily_data)
                daily_pnl = self._execute_trades(trade_signals, daily_data)
                
                # Update portfolio value
                self.current_capital += daily_pnl
                daily_values.append(self.current_capital)
                trades_executed += 1 if daily_pnl != 0 else 0
                
                # Check risk controls
                daily_loss = (daily_start_value - self.current_capital) / daily_start_value
                if daily_loss > max_daily_loss:
                    max_daily_loss = daily_loss
                
                # Check stop-loss triggers
                if daily_loss > self.risk_controls['max_daily_loss']:
                    stop_losses_triggered += 1
                    logger.warning(f"Stop-loss triggered on {current_date}: {daily_loss:.2%} loss")
                
                # Check emergency shutdown
                portfolio_loss = (start_value - self.current_capital) / start_value
                if (portfolio_loss > self.risk_controls['max_portfolio_loss'] and 
                    self.risk_controls['emergency_shutdown_enabled']):
                    emergency_shutdowns += 1
                    logger.critical(f"Emergency shutdown triggered: {portfolio_loss:.2%} portfolio loss")
                    break
                
                # Update peak and drawdown
                if self.current_capital > peak_value:
                    peak_value = self.current_capital
                
                current_drawdown = (peak_value - self.current_capital) / peak_value
                if current_drawdown > max_drawdown:
                    max_drawdown = current_drawdown
            
            # Calculate recovery time
            recovery_time = self._calculate_recovery_time(daily_values, start_value)
            
            # Calculate crisis metrics
            volatility = np.std(np.diff(np.log(daily_values))) * np.sqrt(252) if len(daily_values) > 1 else 0
            sharpe = self._calculate_crisis_sharpe(daily_values)
            
            # Create result
            result = StressTestResult(
                scenario=scenario,
                start_date=params.start_date,
                end_date=params.end_date,
                initial_portfolio_value=start_value,
                final_portfolio_value=self.current_capital,
                max_drawdown=max_drawdown,
                max_daily_loss=max_daily_loss,
                volatility_during_crisis=volatility,
                sharpe_ratio_during_crisis=sharpe,
                number_of_trades=trades_executed,
                stop_losses_triggered=stop_losses_triggered,
                emergency_shutdowns=emergency_shutdowns,
                recovery_time_days=recovery_time,
                performance_vs_benchmark=0.0  # Would compare to buy-and-hold
            )
            
            self.test_results.append(result)
            self.current_capital = self.initial_capital  # Reset for next test
            
            return result
            
        except Exception as e:
            logger.error(f"Stress test failed for {scenario.value}: {e}")
            raise

    def _execute_trades(self, signals: Dict[str, float], 
                       market_data: Dict[str, pd.Series]) -> float:
        """Execute trades based on signals"""
        try:
            total_pnl = 0.0
            
            for symbol, signal in signals.items():
                if signal != 0 and symbol in market_data:
                    price = market_data[symbol]['close']
                    position_size = self.current_capital * 0.01  # 1% position
                    
                    # Simulate execution with crisis slippage
                    if signal > 0:  # Buy
                        execution_price = price * 1.005  # 0.5% buy slippage
                        pnl = -position_size * 0.005  # Slippage cost
                    else:  # Sell
                        execution_price = price * 0.995  # 0.5% sell slippage
                        pnl = -position_size * 0.005  # Slippage cost
                    
                    # Add random market impact
                    market_impact = np.random.normal(0, 0.002) * position_size
                    pnl += market_impact
                    
                    total_pnl += pnl
            
            return total_pnl
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return 0.0

    def _calculate_recovery_time(self, daily_values: List[float], 
                                initial_value: float) -> int:
        """Calculate days to recover to initial value"""
        try:
            if not daily_values:
                return 0
            
            # Find when portfolio recovers to initial value
            for i, value in enumerate(daily_values):
                if value >= initial_value:
                    return i
            
            # If never recovered, return length of test
            return len(daily_values)
            
        except Exception:
            return len(daily_values)

    def _calculate_crisis_sharpe(self, daily_values: List[float]) -> float:
        """Calculate Sharpe ratio during crisis period"""
        try:
            if len(daily_values) < 2:
                return 0.0
            
            # Calculate daily returns
            returns = np.diff(np.log(daily_values))
            
            if len(returns) == 0 or np.std(returns) == 0:
                return 0.0
            
            # Annualized Sharpe (risk-free rate = 0 for crisis period)
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            return sharpe
            
        except Exception:
            return 0.0

    def run_comprehensive_stress_suite(self, 
                                     market_data: Dict[str, pd.DataFrame],
                                     strategy_function) -> List[StressTestResult]:
        """Run stress tests for all crisis scenarios"""
        try:
            logger.info("🧪 Running comprehensive stress test suite...")
            results = []
            
            for scenario in CrisisScenario:
                result = self.run_stress_test(scenario, market_data, strategy_function)
                results.append(result)
                
                logger.info(f"   {scenario.value}:")
                logger.info(f"      Max Drawdown: {result.max_drawdown:.2%}")
                logger.info(f"      Final Value: ${result.final_portfolio_value:,.2f}")
                logger.info(f"      Stop-Losses: {result.stop_losses_triggered}")
                logger.info(f"      Emergency Shutdowns: {result.emergency_shutdowns}")
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive stress test suite failed: {e}")
            raise

    def get_stress_test_summary(self) -> Dict:
        """Get summary statistics of all stress tests"""
        try:
            if not self.test_results:
                return {'total_tests': 0}
            
            drawdowns = [r.max_drawdown for r in self.test_results]
            final_values = [r.final_portfolio_value for r in self.test_results]
            stop_losses = [r.stop_losses_triggered for r in self.test_results]
            shutdowns = [r.emergency_shutdowns for r in self.test_results]
            
            return {
                'total_tests': len(self.test_results),
                'average_max_drawdown': np.mean(drawdowns),
                'worst_drawdown': np.max(drawdowns),
                'best_drawdown': np.min(drawdowns),
                'average_final_value': np.mean(final_values),
                'total_stop_losses': np.sum(stop_losses),
                'total_emergency_shutdowns': np.sum(shutdowns),
                'successful_recoveries': sum(1 for r in self.test_results if r.final_portfolio_value >= r.initial_portfolio_value),
                'failed_scenarios': sum(1 for r in self.test_results if r.emergency_shutdowns > 0)
            }
            
        except Exception as e:
            logger.error(f"Stress test summary failed: {e}")
            return {'error': str(e)}

    def generate_stress_test_report(self) -> str:
        """Generate comprehensive stress test report"""
        try:
            summary = self.get_stress_test_summary()
            
            report = f"""
STRESS TEST REPORT FOR CHLOE 0.6
================================

SUMMARY STATISTICS:
Total Scenarios Tested: {summary.get('total_tests', 0)}
Average Maximum Drawdown: {summary.get('average_max_drawdown', 0):.2%}
Worst Case Drawdown: {summary.get('worst_drawdown', 0):.2%}
Best Case Drawdown: {summary.get('best_drawdown', 0):.2%}
Average Final Portfolio Value: ${summary.get('average_final_value', 0):,.2f}
Total Stop-Losses Triggered: {summary.get('total_stop_losses', 0)}
Total Emergency Shutdowns: {summary.get('total_emergency_shutdowns', 0)}
Successful Recoveries: {summary.get('successful_recoveries', 0)}
Failed Scenarios: {summary.get('failed_scenarios', 0)}

DETAILED RESULTS BY SCENARIO:
"""
            
            for result in self.test_results:
                report += f"""
{result.scenario.value}:
  Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}
  Max Drawdown: {result.max_drawdown:.2%}
  Max Daily Loss: {result.max_daily_loss:.2%}
  Final Value: ${result.final_portfolio_value:,.2f}
  Trades Executed: {result.number_of_trades}
  Stop-Losses: {result.stop_losses_triggered}
  Emergency Shutdowns: {result.emergency_shutdowns}
  Recovery Time: {result.recovery_time_days} days
"""
            
            report += f"""
RISK ASSESSMENT:
{'✅ PASSED' if summary.get('failed_scenarios', 0) == 0 else '❌ FAILED'} - All scenarios survived
{'⚠️  WARNING' if summary.get('total_stop_losses', 0) > 0 else '✅ GOOD'} - Stop-loss system active
{'🔴 CRITICAL' if summary.get('total_emergency_shutdowns', 0) > 0 else '✅ SAFE'} - Emergency protocols engaged
"""
            
            return report.strip()
            
        except Exception as e:
            logger.error(f"Stress test report generation failed: {e}")
            return f"Report generation failed: {e}"

# Global instance
_stress_tester = None

def get_stress_tester(initial_capital: float = 100000.0) -> StressTester:
    """Get singleton stress tester instance"""
    global _stress_tester
    if _stress_tester is None:
        _stress_tester = StressTester(initial_capital)
    return _stress_tester

def main():
    """Example usage"""
    print("Stress Testing Engine ready")
    print("Professional crisis scenario testing")

if __name__ == "__main__":
    main()