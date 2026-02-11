#!/usr/bin/env python3
"""
Walk-Forward Validation Demo for Chloe 0.6
Professional out-of-sample testing with periodic retraining
"""

import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from walk_forward_validator import get_walk_forward_validator, ValidationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_market_data(start_date: datetime, end_date: datetime, 
                                 symbols: list) -> dict:
    """Generate synthetic market data for walk-forward testing"""
    logger.info("📊 Generating synthetic market data...")
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    date_range = date_range[date_range.weekday < 5]  # Only weekdays
    
    market_data = {}
    
    # Base parameters for different market regimes
    regime_periods = [
        {'start': 0, 'end': len(date_range) // 3, 'volatility': 0.02, 'trend': 0.0005},      # Stable period
        {'start': len(date_range) // 3, 'end': 2 * len(date_range) // 3, 'volatility': 0.04, 'trend': -0.001}, # Volatile period
        {'start': 2 * len(date_range) // 3, 'end': len(date_range), 'volatility': 0.015, 'trend': 0.0015}     # Trending period
    ]
    
    for symbol in symbols:
        # Generate base price series
        prices = []
        current_price = np.random.uniform(30000, 60000)  # BTC-like prices
        
        for i, date in enumerate(date_range):
            # Determine current regime
            current_regime = None
            for regime in regime_periods:
                if regime['start'] <= i < regime['end']:
                    current_regime = regime
                    break
            
            if current_regime:
                # Generate price movement based on regime
                daily_return = np.random.normal(current_regime['trend'], current_regime['volatility'])
                current_price *= (1 + daily_return)
                current_price = max(current_price, current_price * 0.5)  # Prevent extreme drops
            else:
                current_price *= (1 + np.random.normal(0.0001, 0.02))
            
            prices.append(current_price)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(1000, 10000) for _ in prices]
        }, index=date_range)
        
        market_data[symbol] = df
        logger.info(f"   Generated {len(df)} days of data for {symbol}")
    
    return market_data

def dummy_strategy_function(data_dict):
    """Dummy strategy function for demonstration"""
    # This would be replaced with actual Chloe strategy
    return {
        'signals': {},
        'parameters': {'lookback': 20, 'threshold': 0.5}
    }

async def demonstrate_walk_forward_validation():
    """Demonstrate walk-forward validation capabilities"""
    logger.info("🔍 WALK-FORWARD VALIDATION DEMO")
    logger.info("=" * 45)
    
    try:
        # Setup validation configuration
        config = ValidationConfig(
            training_window_months=12,
            testing_window_months=3,
            retrain_frequency_months=6,
            min_training_samples=252,
            risk_free_rate=0.02
        )
        
        logger.info("🔧 Initializing Walk-Forward Validator...")
        validator = get_walk_forward_validator(config)
        logger.info("✅ Validator initialized")
        
        # Generate synthetic market data (2 years of data)
        logger.info("📅 Generating 2 years of market data...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)  # 2 years ago
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        market_data = generate_synthetic_market_data(start_date, end_date, symbols)
        
        logger.info(f"   Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"   Assets: {symbols}")
        logger.info(f"   Total Trading Days: {len(next(iter(market_data.values())).index)}")
        
        # Perform walk-forward validation
        logger.info("\n🚀 Performing Walk-Forward Validation...")
        results = validator.validate_strategy(
            historical_data=market_data,
            strategy_function=dummy_strategy_function,
            initial_capital=100000.0
        )
        
        # Display results
        logger.info(f"\n📊 WALK-FORWARD VALIDATION RESULTS:")
        logger.info(f"   Total Validation Periods: {len(results)}")
        
        for i, result in enumerate(results, 1):
            logger.info(f"   Period {i}: {result.period_start.strftime('%Y-%m')} to {result.period_end.strftime('%Y-%m')}")
            logger.info(f"      Sharpe Ratio: {result.sharpe_ratio:.3f}")
            logger.info(f"      Total Return: {result.total_return:.2%}")
            logger.info(f"      Max Drawdown: {result.max_drawdown:.2%}")
            logger.info(f"      Volatility: {result.volatility:.2%}")
            logger.info(f"      Win Rate: {result.win_rate:.1%}")
            logger.info(f"      Number of Trades: {result.number_of_trades}")
        
        # Overall performance metrics
        metrics = validator.performance_metrics
        logger.info(f"\n🏆 OVERALL PERFORMANCE METRICS:")
        logger.info(f"   Average Sharpe Ratio: {metrics['average_sharpe']:.3f} ± {metrics['sharpe_std']:.3f}")
        logger.info(f"   Average Annual Return: {metrics['average_return']*252/len(results):.2%}")
        logger.info(f"   Average Maximum Drawdown: {metrics['average_drawdown']:.2%}")
        logger.info(f"   Average Volatility: {metrics['average_volatility']:.2%}")
        logger.info(f"   Average Win Rate: {metrics['average_win_rate']:.1%}")
        logger.info(f"   Strategy Consistency: {metrics['consistency']:.1%}")
        logger.info(f"   Performance Degradation: {metrics['degradation']:.3f}")
        
        # Viability assessment
        logger.info(f"\n🎯 STRATEGY VIABILITY ASSESSMENT:")
        is_viable = validator.is_strategy_viable(
            minimum_sharpe=0.5,
            maximum_drawdown=0.2,
            minimum_consistency=0.6
        )
        logger.info(f"   Overall Status: {'✅ VIABLE' if is_viable else '❌ NOT VIABLE'}")
        
        # Performance stability analysis
        logger.info(f"\n📈 PERFORMANCE STABILITY ANALYSIS:")
        
        sharpe_ratios = [r.sharpe_ratio for r in results]
        returns = [r.total_return for r in results]
        
        logger.info(f"   Sharpe Ratio Stability: {np.std(sharpe_ratios):.3f} (lower = more stable)")
        logger.info(f"   Return Stability: {np.std(returns):.3f} (lower = more stable)")
        logger.info(f"   Best Period Sharpe: {max(sharpe_ratios):.3f}")
        logger.info(f"   Worst Period Sharpe: {min(sharpe_ratios):.3f}")
        logger.info(f"   Positive Periods: {sum(1 for sr in sharpe_ratios if sr > 0)}/{len(sharpe_ratios)}")
        
        # Regime performance comparison
        logger.info(f"\n📊 REGIME PERFORMANCE COMPARISON:")
        
        # Split results by time periods to simulate different market regimes
        if len(results) >= 3:
            third = len(results) // 3
            stable_results = results[:third]
            volatile_results = results[third:2*third]
            trending_results = results[2*third:]
            
            logger.info(f"   Stable Market Periods ({len(stable_results)}):")
            logger.info(f"      Avg Sharpe: {np.mean([r.sharpe_ratio for r in stable_results]):.3f}")
            logger.info(f"      Avg Return: {np.mean([r.total_return for r in stable_results]):.2%}")
            
            logger.info(f"   Volatile Market Periods ({len(volatile_results)}):")
            logger.info(f"      Avg Sharpe: {np.mean([r.sharpe_ratio for r in volatile_results]):.3f}")
            logger.info(f"      Avg Return: {np.mean([r.total_return for r in volatile_results]):.2%}")
            
            logger.info(f"   Trending Market Periods ({len(trending_results)}):")
            logger.info(f"      Avg Sharpe: {np.mean([r.sharpe_ratio for r in trending_results]):.3f}")
            logger.info(f"      Avg Return: {np.mean([r.total_return for r in trending_results]):.2%}")
        
        # Generate validation report
        logger.info(f"\n📋 GENERATING VALIDATION REPORT:")
        report = validator.get_validation_report()
        
        logger.info(f"   Report Generated with {report['total_periods']} periods")
        logger.info(f"   Configuration: {report['config']['training_window_months']}M training, "
                   f"{report['config']['testing_window_months']}M testing")
        
        logger.info(f"\n✅ WALK-FORWARD VALIDATION DEMO COMPLETED SUCCESSFULLY")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented professional walk-forward validation framework")
        logger.info("   • Tested strategy performance across multiple market regimes")
        logger.info("   • Calculated comprehensive performance metrics")
        logger.info("   • Assessed strategy viability and stability")
        logger.info("   • Generated detailed validation report")
        
        logger.info(f"\n📊 FINAL VALIDATION RESULTS:")
        logger.info(f"   Strategy Viability: {'PASS' if is_viable else 'FAIL'}")
        logger.info(f"   Performance Consistency: {metrics['consistency']:.1%}")
        logger.info(f"   Risk-Adjusted Returns: {metrics['average_sharpe']:.3f}")
        logger.info(f"   Maximum Drawdown Control: {metrics['max_drawdown']:.2%}")
        
        logger.info(f"\n🎯 NEXT STEPS:")
        logger.info("   1. Refine strategy based on weak periods")
        logger.info("   2. Optimize parameters for better consistency")
        logger.info("   3. Implement additional risk controls")
        logger.info("   4. Proceed to systemic stop-loss implementation")
        
    except Exception as e:
        logger.error(f"❌ Walk-forward validation demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_validation_configurations():
    """Demonstrate different validation configurations"""
    logger.info(f"\n⚙️ VALIDATION CONFIGURATION TESTING")
    logger.info("=" * 40)
    
    try:
        # Test different configurations
        configs = [
            ValidationConfig(training_window_months=6, testing_window_months=3, retrain_frequency_months=3),
            ValidationConfig(training_window_months=12, testing_window_months=3, retrain_frequency_months=6),
            ValidationConfig(training_window_months=24, testing_window_months=6, retrain_frequency_months=12)
        ]
        
        config_names = ["Aggressive (6M/3M)", "Standard (12M/3M)", "Conservative (24M/6M)"]
        
        logger.info("Testing different validation configurations:")
        
        for name, config in zip(config_names, configs):
            validator = get_walk_forward_validator(config)
            logger.info(f"   {name} Configuration:")
            logger.info(f"      Training Window: {config.training_window_months} months")
            logger.info(f"      Testing Window: {config.testing_window_months} months")
            logger.info(f"      Retrain Frequency: {config.retrain_frequency_months} months")
        
        logger.info("✅ Configuration testing completed")
        
    except Exception as e:
        logger.error(f"❌ Configuration testing failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe 0.6 - Walk-Forward Validation Demo")
    print("Professional out-of-sample strategy testing")
    print()
    
    # Run main validation demo
    await demonstrate_walk_forward_validation()
    
    # Run configuration demonstration
    demonstrate_validation_configurations()
    
    print(f"\n🎉 WALK-FORWARD VALIDATION DEMO COMPLETED")
    print("Chloe 0.6 now has professional validation capabilities!")

if __name__ == "__main__":
    asyncio.run(main())