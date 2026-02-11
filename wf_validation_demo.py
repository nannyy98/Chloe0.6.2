#!/usr/bin/env python3
"""
Walk-Forward Validation Demo for Chloe 0.6.1
Demonstrating professional out-of-sample validation
"""

import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from walk_forward_validator import WalkForwardValidator, WalkForwardConfig, calculate_standard_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_strategy_data(days: int = 756) -> pd.DataFrame:
    """Create sample market data for strategy testing"""
    logger.info(f"📊 Creating {days}-day sample market data...")
    
    # Generate realistic market data
    dates = pd.date_range(start='2022-01-01', periods=days, freq='D')
    
    # Simulate market returns with some realistic characteristics
    np.random.seed(42)  # For reproducible results
    
    # Base trend + volatility clustering + regime changes
    base_trend = 0.0002  # Small positive drift
    volatility = 0.02    # 2% daily volatility
    
    returns = []
    current_vol = volatility
    regime_switch_prob = 0.005  # 0.5% chance of regime switch per day
    
    for i in range(days):
        # Occasional regime switches
        if np.random.random() < regime_switch_prob:
            # Switch between normal and stressed regimes
            if current_vol == volatility:
                current_vol = volatility * 2.5  # High volatility regime
            else:
                current_vol = volatility  # Normal regime
        
        # Generate return with current regime characteristics
        daily_return = np.random.normal(base_trend, current_vol)
        returns.append(daily_return)
    
    # Create DataFrame
    data = pd.DataFrame({
        'returns': returns,
        'price': 100 * np.exp(np.cumsum(returns)),  # Price series
        'volume': np.random.lognormal(10, 1, days),  # Volume
        'rsi': 50 + np.random.normal(0, 15, days),   # RSI indicator
        'macd': np.random.normal(0, 0.5, days)       # MACD indicator
    }, index=dates)
    
    # Ensure RSI stays in valid range
    data['rsi'] = np.clip(data['rsi'], 0, 100)
    
    logger.info(f"   Generated {len(data)} days of market data")
    logger.info(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
    logger.info(f"   Average daily return: {data['returns'].mean()*100:.3f}%")
    logger.info(f"   Volatility: {data['returns'].std()*100:.2f}%")
    
    return data

def simple_momentum_strategy(train_data: pd.DataFrame) -> pd.Series:
    """Simple momentum strategy for demonstration"""
    try:
        # Simple moving average crossover strategy
        prices = train_data['price']
        
        # Calculate short and long moving averages
        short_ma = prices.rolling(window=10).mean()
        long_ma = prices.rolling(window=30).mean()
        
        # Generate position signals
        positions = pd.Series(0.0, index=train_data.index)
        
        # Long when short MA > long MA, short when short MA < long MA
        positions[short_ma > long_ma] = 1.0   # Long positions
        positions[short_ma < long_ma] = -1.0  # Short positions
        
        # Forward fill positions
        positions = positions.ffill().fillna(0.0)
        
        return positions
        
    except Exception as e:
        logger.error(f"Strategy execution failed: {e}")
        return pd.Series(0.0, index=train_data.index)

async def demonstrate_walk_forward_validation():
    """Demonstrate walk-forward validation capabilities"""
    logger.info("🔬 WALK-FORWARD VALIDATION DEMO")
    logger.info("=" * 35)
    
    try:
        # Create sample data
        logger.info("🔧 Setting up validation environment...")
        market_data = create_sample_strategy_data(days=756)  # ~2.5 years of data
        
        # Configure walk-forward validation
        config = WalkForwardConfig(
            train_window_size=126,    # 6 months training
            test_window_size=21,      # 1 month testing  
            step_size=7,              # 1 week steps
            min_train_samples=63      # 3 months minimum
        )
        
        validator = WalkForwardValidator(config)
        logger.info("✅ Walk-Forward Validator initialized")
        logger.info(f"   Configuration: {config.train_window_size}d train, {config.test_window_size}d test, {config.step_size}d steps")
        
        # Perform walk-forward validation
        logger.info("🚀 Executing Walk-Forward Validation...")
        results = validator.validate_strategy(
            data=market_data,
            strategy_func=simple_momentum_strategy,
            metrics_func=calculate_standard_metrics,
            target_column='returns'
        )
        
        # Display results
        logger.info(f"   Completed {len(results.validation_results)} validation folds")
        logger.info(f"   Overall Sharpe Ratio: {results.overall_metrics.get('sharpe_ratio_mean', 0):.3f}")
        logger.info(f"   Sharpe Stability: {results.stability_analysis.get('sharpe_stability', 0):.3f}")
        logger.info(f"   Positive Periods: {results.stability_analysis.get('positive_fold_ratio', 0)*100:.1f}%")
        
        # Generate detailed report
        logger.info(f"\n📋 DETAILED VALIDATION REPORT:")
        report = validator.generate_validation_report(results)
        print(report)
        
        # Analyze validation quality
        logger.info(f"\n🔍 VALIDATION QUALITY ANALYSIS:")
        
        mean_sharpe = results.overall_metrics.get('sharpe_ratio_mean', 0)
        sharpe_stability = results.stability_analysis.get('sharpe_stability', 0)
        positive_ratio = results.stability_analysis.get('positive_fold_ratio', 0)
        
        logger.info(f"   Mean Sharpe Ratio: {mean_sharpe:.3f}")
        logger.info(f"   Sharpe Stability: {sharpe_stability:.3f}")
        logger.info(f"   Positive Periods: {positive_ratio*100:.1f}%")
        
        # Validation quality assessment
        if mean_sharpe > 1.5 and sharpe_stability > 0.7 and positive_ratio > 0.8:
            quality = "🥇 EXCELLENT - Strong, consistent performance"
        elif mean_sharpe > 1.0 and sharpe_stability > 0.5 and positive_ratio > 0.6:
            quality = "🥈 GOOD - Promising but requires monitoring"
        elif mean_sharpe > 0.5 and sharpe_stability > 0.3 and positive_ratio > 0.5:
            quality = "🥉 FAIR - Marginal edge, needs improvement"
        else:
            quality = "❌ POOR - No demonstrable edge"
        
        logger.info(f"   Quality Assessment: {quality}")
        
        # Compare with traditional backtest
        logger.info(f"\n⚖️ BACKTEST vs WALK-FORWARD COMPARISON:")
        
        # Traditional backtest (full period)
        full_positions = simple_momentum_strategy(market_data)
        full_metrics = calculate_standard_metrics(full_positions, market_data['returns'])
        
        logger.info("Traditional Backtest (Full Period):")
        logger.info(f"   Sharpe Ratio: {full_metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"   Annual Return: {full_metrics.get('annual_return', 0)*100:.2f}%")
        logger.info(f"   Max Drawdown: {full_metrics.get('max_drawdown', 0)*100:.2f}%")
        
        logger.info("Walk-Forward Validation:")
        logger.info(f"   Mean Sharpe Ratio: {mean_sharpe:.3f}")
        logger.info(f"   Sharpe Std Dev: {results.overall_metrics.get('sharpe_ratio_std', 0):.3f}")
        logger.info(f"   Best Sharpe: {results.overall_metrics.get('sharpe_ratio_max', 0):.3f}")
        logger.info(f"   Worst Sharpe: {results.overall_metrics.get('sharpe_ratio_min', 0):.3f}")
        
        overfitting_risk = "HIGH" if abs(full_metrics.get('sharpe_ratio', 0) - mean_sharpe) > 0.5 else "MODERATE" if abs(full_metrics.get('sharpe_ratio', 0) - mean_sharpe) > 0.2 else "LOW"
        logger.info(f"   Overfitting Risk: {overfitting_risk}")
        
        # Show validation timeline
        logger.info(f"\n📅 VALIDATION TIMELINE:")
        for result in results.validation_results[:5]:  # Show first 5 folds
            period = f"{result.test_period[0].strftime('%Y-%m')} to {result.test_period[1].strftime('%Y-%m')}"
            sharpe = result.metrics.get('sharpe_ratio', 0)
            logger.info(f"   Fold {result.fold_number}: {period} - Sharpe: {sharpe:.3f}")
        
        if len(results.validation_results) > 5:
            logger.info(f"   ... and {len(results.validation_results) - 5} more folds")
        
        logger.info(f"\n✅ WALK-FORWARD VALIDATION DEMO COMPLETED")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented professional walk-forward validation framework")
        logger.info("   • Demonstrated out-of-sample strategy testing")
        logger.info("   • Provided stability and consistency analysis")
        logger.info("   • Enabled comparison with traditional backtesting")
        logger.info("   • Established foundation for robust strategy validation")
        
        logger.info(f"\n🎯 VALIDATION BENEFITS:")
        logger.info("   Prevents overfitting to historical data")
        logger.info("   Tests strategy adaptability to changing markets")
        logger.info("   Provides realistic performance expectations")
        logger.info("   Identifies strategy degradation over time")
        logger.info("   Enables proper risk management parameter setting")
        
        logger.info(f"\n⏭️ NEXT STEPS:")
        logger.info("   1. Implement Live Risk Governor (independent risk daemon)")
        logger.info("   2. Create extended paper trading infrastructure")
        logger.info("   3. Establish micro-capital testing protocol ($50-100)")
        logger.info("   4. Develop comprehensive performance monitoring")
        
    except Exception as e:
        logger.error(f"❌ Walk-forward validation demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_validation_concepts():
    """Demonstrate key validation concepts"""
    logger.info(f"\n🎯 VALIDATION CONCEPTS DEMONSTRATION")
    logger.info("=" * 37)
    
    try:
        concepts = {
            "Walk-Forward Validation": [
                "Tests strategy on multiple out-of-sample periods",
                "Prevents curve fitting and overoptimization", 
                "Measures strategy adaptability over time",
                "Provides realistic performance expectations"
            ],
            
            "Overfitting Detection": [
                "Compare in-sample vs out-of-sample performance",
                "Large differences indicate overfitting",
                "Walk-forward helps identify unstable strategies",
                "Stability metrics quantify consistency"
            ],
            
            "Risk Management Benefits": [
                "Proper position sizing based on validated performance",
                "Realistic drawdown expectations",
                "Appropriate stop-loss levels",
                "Confidence intervals for returns"
            ]
        }
        
        logger.info("Key Validation Concepts:")
        for concept, explanations in concepts.items():
            logger.info(f"\n{concept}:")
            for explanation in explanations:
                logger.info(f"   • {explanation}")
        
        logger.info("✅ Validation concepts demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ Validation concepts demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe 0.6.1 - Walk-Forward Validation Demo")
    print("Professional out-of-sample strategy validation")
    print()
    
    # Run main validation demo
    await demonstrate_walk_forward_validation()
    
    # Run concepts demonstration
    demonstrate_validation_concepts()
    
    print(f"\n🎉 WALK-FORWARD VALIDATION DEMO COMPLETED")
    print("Chloe 0.6.1 now has professional validation capabilities!")

if __name__ == "__main__":
    asyncio.run(main())