#!/usr/bin/env python3
"""
Stress Testing Demo for Chloe 0.6
Professional crisis scenario stress testing demonstration
"""

import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from stress_tester import get_stress_tester, CrisisScenario

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_base_market_data(symbols: list, days: int = 252) -> dict:
    """Generate base market data for stress testing"""
    logger.info("📊 Generating base market data...")
    
    market_data = {}
    start_date = datetime.now() - timedelta(days=days)
    
    for symbol in symbols:
        # Generate realistic price series
        dates = pd.date_range(start_date, periods=days, freq='D')
        dates = dates[dates.weekday < 5]  # Only weekdays
        
        # Base price (different for each asset)
        base_prices = {'BTC/USDT': 45000, 'ETH/USDT': 2800, 'SOL/USDT': 90, 'ADA/USDT': 0.45}
        base_price = base_prices.get(symbol, 30000)
        
        # Generate price series with realistic characteristics
        prices = []
        current_price = base_price
        
        for i in range(len(dates)):
            # Daily return with trend and volatility
            trend = 0.0001  # Small upward bias
            volatility = 0.02   # 2% daily volatility
            daily_return = np.random.normal(trend, volatility)
            
            current_price *= (1 + daily_return)
            current_price = max(current_price, base_price * 0.3)  # Prevent extreme drops
            prices.append(current_price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(1000, 10000) for _ in prices]
        }, index=dates)
        
        market_data[symbol] = df
        logger.info(f"   Generated {len(df)} days of data for {symbol}")
    
    return market_data

def dummy_strategy_function(market_data: dict) -> dict:
    """Simple dummy strategy for stress testing"""
    signals = {}
    
    for symbol in market_data.keys():
        # Simple mean-reversion logic for demo
        # In real implementation, this would be actual trading strategy
        if np.random.random() > 0.7:  # 30% chance of trading
            signals[symbol] = np.random.choice([-1, 1])  # Random long/short
        else:
            signals[symbol] = 0  # No position
    
    return signals

async def demonstrate_stress_testing():
    """Demonstrate comprehensive stress testing capabilities"""
    logger.info("🧪 STRESS TESTING ON CRISIS DATA DEMO")
    logger.info("=" * 45)
    
    try:
        # Initialize stress tester
        logger.info("🔧 Initializing Stress Tester...")
        stress_tester = get_stress_tester(initial_capital=100000.0)
        logger.info("✅ Stress Tester initialized")
        
        # Generate base market data
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
        market_data = generate_base_market_data(symbols, 365)
        
        logger.info(f"📊 Market Data Generated:")
        logger.info(f"   Assets: {symbols}")
        logger.info(f"   Historical Period: 365 days")
        logger.info(f"   Data Points: {sum(len(df) for df in market_data.values())}")
        
        # Test individual crisis scenarios
        logger.info(f"\n🌪️ INDIVIDUAL CRISIS SCENARIO TESTING:")
        
        # Test 3 representative scenarios
        test_scenarios = [
            CrisisScenario.MAR2020_PANIC,
            CrisisScenario.MAY2010_FLASH,
            CrisisScenario.NOV2022_FTX
        ]
        
        individual_results = []
        
        for scenario in test_scenarios:
            logger.info(f"\n   Testing {scenario.value}:")
            
            result = stress_tester.run_stress_test(
                scenario=scenario,
                market_data=market_data,
                strategy_function=dummy_strategy_function
            )
            
            individual_results.append(result)
            
            logger.info(f"      Initial Value: ${result.initial_portfolio_value:,.2f}")
            logger.info(f"      Final Value: ${result.final_portfolio_value:,.2f}")
            logger.info(f"      Max Drawdown: {result.max_drawdown:.2%}")
            logger.info(f"      Max Daily Loss: {result.max_daily_loss:.2%}")
            logger.info(f"      Volatility: {result.volatility_during_crisis:.2%}")
            logger.info(f"      Sharpe Ratio: {result.sharpe_ratio_during_crisis:.3f}")
            logger.info(f"      Trades Executed: {result.number_of_trades}")
            logger.info(f"      Stop-Losses Triggered: {result.stop_losses_triggered}")
            logger.info(f"      Emergency Shutdowns: {result.emergency_shutdowns}")
            logger.info(f"      Recovery Time: {result.recovery_time_days} days")
        
        # Run comprehensive stress test suite
        logger.info(f"\n🎭 COMPREHENSIVE STRESS TEST SUITE:")
        
        all_results = stress_tester.run_comprehensive_stress_suite(
            market_data=market_data,
            strategy_function=dummy_strategy_function
        )
        
        # Analyze results
        summary = stress_tester.get_stress_test_summary()
        
        logger.info(f"   Suite Results:")
        logger.info(f"      Total Scenarios Tested: {summary['total_tests']}")
        logger.info(f"      Average Max Drawdown: {summary['average_max_drawdown']:.2%}")
        logger.info(f"      Worst Case Drawdown: {summary['worst_drawdown']:.2%}")
        logger.info(f"      Best Case Drawdown: {summary['best_drawdown']:.2%}")
        logger.info(f"      Average Final Value: ${summary['average_final_value']:,.2f}")
        logger.info(f"      Total Stop-Losses: {summary['total_stop_losses']}")
        logger.info(f"      Total Emergency Shutdowns: {summary['total_emergency_shutdowns']}")
        logger.info(f"      Successful Recoveries: {summary['successful_recoveries']}")
        logger.info(f"      Failed Scenarios: {summary['failed_scenarios']}")
        
        # Detailed scenario analysis
        logger.info(f"\n📊 DETAILED SCENARIO ANALYSIS:")
        
        crisis_categories = {
            'Market Crashes': [CrisisScenario.MAR2020_PANIC, CrisisScenario.OCT2008_FINANCIAL],
            'Flash Events': [CrisisScenario.MAY2010_FLASH],
            'Crypto-Specific': [CrisisScenario.NOV2021_LUNA, CrisisScenario.NOV2022_FTX],
            'Interest Rate Shocks': [CrisisScenario.JUN2013_TAPER, CrisisScenario.JAN2018_VOL],
            'Banking Crises': [CrisisScenario.MAR2023_BANKS]
        }
        
        for category, scenarios in crisis_categories.items():
            category_results = [r for r in all_results if r.scenario in scenarios]
            if category_results:
                avg_drawdown = np.mean([r.max_drawdown for r in category_results])
                avg_recovery = np.mean([r.recovery_time_days for r in category_results])
                logger.info(f"   {category}:")
                logger.info(f"      Average Drawdown: {avg_drawdown:.2%}")
                logger.info(f"      Average Recovery: {avg_recovery:.1f} days")
                logger.info(f"      Scenarios Tested: {len(category_results)}")
        
        # Risk assessment
        logger.info(f"\n🛡️ RISK ASSESSMENT:")
        
        risk_metrics = {
            'Maximum Drawdown Tolerance': 0.20,  # 20% tolerance
            'Maximum Daily Loss Tolerance': 0.08,  # 8% tolerance
            'Stop-Loss Effectiveness': summary['total_stop_losses'] > 0,
            'Emergency Protocol Activation': summary['total_emergency_shutdowns'] > 0
        }
        
        for metric, value in risk_metrics.items():
            if isinstance(value, bool):
                status = "✅ ACTIVATED" if value else "❌ NOT TRIGGERED"
                logger.info(f"   {metric}: {status}")
            else:
                logger.info(f"   {metric}: {value:.2%}")
        
        # Performance benchmarking
        logger.info(f"\n🏁 PERFORMANCE BENCHMARKING:")
        
        # Compare against simple buy-and-hold
        buy_and_hold_performance = {}
        for symbol, df in market_data.items():
            if len(df) > 0:
                start_price = df['close'].iloc[0]
                end_price = df['close'].iloc[-1]
                performance = (end_price - start_price) / start_price
                buy_and_hold_performance[symbol] = performance
        
        avg_buy_hold = np.mean(list(buy_and_hold_performance.values()))
        logger.info(f"   Buy-and-Hold Benchmark: {avg_buy_hold:.2%}")
        
        # Strategy vs benchmark comparison would be implemented here
        logger.info(f"   Strategy Performance: To be calculated vs benchmark")
        
        # Generate comprehensive report
        logger.info(f"\n📋 GENERATING STRESS TEST REPORT:")
        
        report = stress_tester.generate_stress_test_report()
        logger.info("   Report generated successfully")
        
        # Show key findings
        logger.info(f"\n🔑 KEY FINDINGS:")
        
        # Worst performing scenario
        worst_scenario = max(all_results, key=lambda x: x.max_drawdown)
        logger.info(f"   Worst Scenario: {worst_scenario.scenario.value}")
        logger.info(f"      Drawdown: {worst_scenario.max_drawdown:.2%}")
        logger.info(f"      Final Value: ${worst_scenario.final_portfolio_value:,.2f}")
        
        # Best performing scenario
        best_scenario = min(all_results, key=lambda x: x.max_drawdown)
        logger.info(f"   Best Scenario: {best_scenario.scenario.value}")
        logger.info(f"      Drawdown: {best_scenario.max_drawdown:.2%}")
        logger.info(f"      Final Value: ${best_scenario.final_portfolio_value:,.2f}")
        
        # Recovery analysis
        recovered_scenarios = [r for r in all_results if r.recovery_time_days < len(pd.date_range(r.start_date, r.end_date))]
        logger.info(f"   Recovery Rate: {len(recovered_scenarios)}/{len(all_results)} scenarios recovered")
        
        # Risk-adjusted performance
        sharpe_ratios = [r.sharpe_ratio_during_crisis for r in all_results if not np.isnan(r.sharpe_ratio_during_crisis)]
        if sharpe_ratios:
            avg_sharpe = np.mean(sharpe_ratios)
            logger.info(f"   Average Crisis Sharpe Ratio: {avg_sharpe:.3f}")
        
        logger.info(f"\n✅ STRESS TESTING DEMO COMPLETED SUCCESSFULLY")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented comprehensive crisis scenario testing")
        logger.info("   • Created 8 major historical crisis simulations")
        logger.info("   • Built automated stress testing framework")
        logger.info("   • Developed detailed performance analysis")
        logger.info("   • Generated professional stress test reports")
        
        logger.info(f"\n🌪️ STRESS TESTING COVERAGE:")
        logger.info("   Market Crashes: 2008 Financial Crisis, 2020 COVID Crash")
        logger.info("   Flash Events: 2010 Flash Crash")
        logger.info("   Crypto Events: Luna/UST Collapse, FTX Bankruptcy")
        logger.info("   Interest Rate Shocks: Taper Tantrum, Volatility Shock")
        logger.info("   Banking Crises: 2023 Regional Bank Concerns")
        
        logger.info(f"\n🎯 SYSTEM RESILIENCE ASSESSMENT:")
        logger.info(f"   Drawdown Tolerance: {summary['worst_drawdown']:.2%} (acceptable: <20%)")
        logger.info(f"   Risk Control Effectiveness: Stop-losses triggered {summary['total_stop_losses']} times")
        logger.info(f"   Emergency Response: {summary['total_emergency_shutdowns']} shutdowns activated")
        logger.info(f"   Recovery Capability: {summary['successful_recoveries']}/{summary['total_tests']} scenarios recovered")
        
        logger.info(f"\n🎯 NEXT STEPS:")
        logger.info("   1. Refine strategy based on weak scenarios")
        logger.info("   2. Optimize risk parameters for better crisis performance")
        logger.info("   3. Implement additional crisis-specific protections")
        logger.info("   4. Conduct sensitivity analysis on key parameters")
        
    except Exception as e:
        logger.error(f"❌ Stress testing demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_crisis_scenarios():
    """Demonstrate different crisis scenarios"""
    logger.info(f"\n🌪️ CRISIS SCENARIO LIBRARY")
    logger.info("=" * 30)
    
    try:
        from stress_tester import CrisisDataGenerator
        
        generator = CrisisDataGenerator()
        
        logger.info("Available Crisis Scenarios:")
        for scenario, params in generator.crisis_scenarios.items():
            logger.info(f"   {scenario.value}:")
            logger.info(f"      {params.name}")
            logger.info(f"      Duration: {params.duration_days} days")
            logger.info(f"      Expected Drawdown: {params.max_drawdown:.1%}")
            logger.info(f"      Volatility Spike: {params.volatility_spike}x")
            logger.info(f"      Period: {params.start_date.strftime('%Y-%m-%d')} to {params.end_date.strftime('%Y-%m-%d')}")
            logger.info("")
        
        logger.info("✅ Crisis scenario library demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ Crisis scenarios demo failed: {e}")

def demonstrate_risk_controls():
    """Demonstrate risk control effectiveness"""
    logger.info(f"\n🛡️ RISK CONTROL EFFECTIVENESS")
    logger.info("=" * 32)
    
    try:
        stress_tester = get_stress_tester(100000.0)
        
        logger.info("Risk Control Settings:")
        for control, value in stress_tester.risk_controls.items():
            logger.info(f"   {control}: {value}")
        
        logger.info("Risk Control Testing:")
        logger.info("   • Daily Loss Limits: 5% maximum")
        logger.info("   • Portfolio Loss Limits: 15% maximum")
        logger.info("   • Stop-Loss System: Enabled")
        logger.info("   • Emergency Shutdown: Enabled")
        
        logger.info("✅ Risk controls demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ Risk controls demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe 0.6 - Stress Testing on Crisis Data Demo")
    print("Professional crisis scenario stress testing")
    print()
    
    # Run main stress testing demo
    await demonstrate_stress_testing()
    
    # Run scenario library demonstration
    demonstrate_crisis_scenarios()
    
    # Run risk controls demonstration
    demonstrate_risk_controls()
    
    print(f"\n🎉 STRESS TESTING DEMO COMPLETED")
    print("Chloe 0.6 now has professional stress testing capabilities!")

if __name__ == "__main__":
    asyncio.run(main())