#!/usr/bin/env python3
"""
Shadow Mode Demo for Chloe AI
Demonstrating parallel model comparison and performance monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from learning.shadow_mode import ShadowModeManager, ShadowModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_shadow_mode():
    """Demonstrate shadow mode capabilities"""
    logger.info("🎭 SHADOW MODE DEMO")
    logger.info("=" * 21)
    
    try:
        # Initialize shadow mode manager
        logger.info("🔧 Initializing Shadow Mode Manager...")
        shadow_manager = ShadowModeManager(comparison_window_hours=6)
        logger.info("✅ Shadow Mode Manager initialized")
        logger.info(f"   Comparison window: 6 hours")
        
        # Create sample models
        logger.info(f"\n🤖 CREATING SAMPLE MODELS:")
        
        # Model 1: Conservative model (high accuracy, lower returns)
        conservative_model = create_sample_model("conservative", accuracy_bias=0.7, risk_level=0.3)
        conservative_registered = shadow_manager.register_model(
            "conservative_v1.0", 
            conservative_model, 
            ["rsi", "macd", "volatility"],
            "1.0"
        )
        logger.info(f"   Conservative Model: {'✅' if conservative_registered else '❌'}")
        
        # Model 2: Aggressive model (higher returns, higher risk)
        aggressive_model = create_sample_model("aggressive", accuracy_bias=0.55, risk_level=0.8)
        aggressive_registered = shadow_manager.register_model(
            "aggressive_v1.0",
            aggressive_model,
            ["rsi", "volume", "momentum", "trend"],
            "1.0"
        )
        logger.info(f"   Aggressive Model: {'✅' if aggressive_registered else '❌'}")
        
        # Model 3: Balanced model (middle ground)
        balanced_model = create_sample_model("balanced", accuracy_bias=0.62, risk_level=0.5)
        balanced_registered = shadow_manager.register_model(
            "balanced_v1.0",
            balanced_model,
            ["rsi", "macd", "bollinger", "atr"],
            "1.0"
        )
        logger.info(f"   Balanced Model: {'✅' if balanced_registered else '❌'}")
        
        # Simulate market data and shadow trading
        logger.info(f"\n📊 SIMULATING MARKET DATA AND SHADOW TRADING:")
        
        # Generate sample market features for multiple time periods
        market_scenarios = [
            ("Stable Market", generate_stable_market_data(20)),
            ("Volatile Market", generate_volatile_market_data(20)),
            ("Trending Market", generate_trending_market_data(20))
        ]
        
        all_comparison_results = []
        
        for scenario_name, market_features in market_scenarios:
            logger.info(f"\n   📈 {scenario_name.upper()}:")
            
            # Update market data
            for _, row in market_features.iterrows():
                price_data = {
                    'close': row['close'],
                    'high': row['high'],
                    'low': row['low'],
                    'volume': row['volume']
                }
                shadow_manager.update_market_data(row['symbol'], price_data)
            
            # Process shadow trades
            comparison_result = await shadow_manager.process_shadow_trades(market_features)
            all_comparison_results.append((scenario_name, comparison_result))
            
            # Show immediate results
            logger.info(f"      Active Models: {len(comparison_result.active_models)}")
            logger.info(f"      Best Performer: {comparison_result.best_performing_model}")
            logger.info(f"      Consensus Signals: {len(comparison_result.consensus_signals)} symbols")
            
            # Show model rankings
            logger.info(f"      Performance Rankings:")
            for rank, (model_id, score) in enumerate(comparison_result.performance_ranking[:3], 1):
                logger.info(f"         {rank}. {model_id}: {score:.3f}")
        
        # Detailed analysis of final comparison
        logger.info(f"\n📊 DETAILED PERFORMANCE ANALYSIS:")
        final_comparison = all_comparison_results[-1][1]  # Last scenario results
        
        logger.info(f"Model Performance Summary:")
        logger.info(f"{'Model':<15} {'Trades':<8} {'Win Rate':<10} {'Sharpe':<8} {'Accuracy':<10} {'Score':<8}")
        logger.info("-" * 65)
        
        for model_id, metrics in final_comparison.model_metrics.items():
            logger.info(f"{model_id:<15} {metrics.total_trades:<8} "
                       f"{metrics.win_rate*100:<10.1f}% {metrics.sharpe_ratio:<8.2f} "
                       f"{metrics.accuracy*100:<10.1f}% {final_comparison.performance_ranking[0][1]:<8.3f}"
                       if model_id == final_comparison.performance_ranking[0][0] else "")
        
        # Show consensus signals
        logger.info(f"\n🤝 CONSENSUS SIGNALS:")
        if final_comparison.consensus_signals:
            for symbol, signal in list(final_comparison.consensus_signals.items())[:5]:
                signal_desc = {1: "BUY", -1: "SELL", 0: "HOLD"}.get(signal, "UNKNOWN")
                logger.info(f"   {symbol}: {signal_desc} (strength: {abs(signal)})")
        else:
            logger.info("   No strong consensus signals generated")
        
        # Show divergence alerts
        logger.info(f"\n⚠️  DIVERGENCE ALERTS:")
        if final_comparison.divergence_alerts:
            for alert in final_comparison.divergence_alerts:
                logger.info(f"   [{alert['severity']}] {alert['message']}")
        else:
            logger.info("   No significant divergences detected")
        
        # Test model management
        logger.info(f"\n🔧 MODEL MANAGEMENT:")
        
        # Pause a model
        shadow_manager.models["aggressive_v1.0"].status = "PAUSED"
        logger.info("   Paused aggressive model")
        
        # Process with one model paused
        market_features = generate_stable_market_data(10)
        comparison_result = await shadow_manager.process_shadow_trades(market_features)
        logger.info(f"   Active models after pause: {len(comparison_result.active_models)}")
        
        # Resume model
        shadow_manager.models["aggressive_v1.0"].status = "ACTIVE"
        logger.info("   Resumed aggressive model")
        
        # Show performance evolution
        logger.info(f"\n📈 PERFORMANCE EVOLUTION:")
        logger.info(f"Scenario Progression:")
        for scenario_name, comparison_result in all_comparison_results:
            best_model = comparison_result.best_performing_model
            best_score = next((score for mid, score in comparison_result.performance_ranking if mid == best_model), 0)
            logger.info(f"   {scenario_name}: {best_model} ({best_score:.3f})")
        
        # Generate detailed reports
        logger.info(f"\n📋 DETAILED MODEL REPORTS:")
        
        for model_id in shadow_manager.models.keys():
            if shadow_manager.models[model_id].status == "ACTIVE":
                report = shadow_manager.get_model_performance_report(model_id)
                if report:
                    logger.info(f"\n{model_id.upper()} REPORT:")
                    metrics = report['metrics']
                    logger.info(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
                    logger.info(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                    logger.info(f"   Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
                    logger.info(f"   Accuracy: {metrics['accuracy']*100:.1f}%")
                    logger.info(f"   Recent Performance: {metrics['recent_performance']:+.4f}")
        
        # Show shadow mode summary
        logger.info(f"\n🎭 SHADOW MODE SUMMARY:")
        summary = shadow_manager.get_shadow_summary()
        logger.info(f"   Total Models: {summary['total_models']}")
        logger.info(f"   Active Models: {summary['active_models']}")
        logger.info(f"   Total Shadow Trades: {summary['total_shadow_trades']}")
        logger.info(f"   Symbols Tracked: {summary['symbols_tracked']}")
        logger.info(f"   Recent Comparisons: {summary['recent_comparisons']}")
        
        logger.info(f"\n✅ SHADOW MODE DEMO COMPLETED")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented parallel model comparison system")
        logger.info("   • Created real-time performance monitoring")
        logger.info("   • Built consensus signal generation")
        logger.info("   • Added divergence detection and alerts")
        logger.info("   • Developed comprehensive reporting system")
        
        logger.info(f"\n🎯 SHADOW MODE FEATURES:")
        logger.info("   Multi-model parallel evaluation")
        logger.info("   Real-time performance comparison")
        logger.info("   Consensus signal aggregation")
        logger.info("   Divergence detection and alerts")
        logger.info("   Historical performance tracking")
        logger.info("   Dynamic model management")
        
        logger.info(f"\n⏭️ NEXT STEPS:")
        logger.info("   1. Implement Controlled Self-Learning Loop")
        logger.info("   2. Build Risk Sandbox for stress testing")
        logger.info("   3. Create Paper Performance Dashboard")
        logger.info("   4. Add Live Trading Integration")
        
    except Exception as e:
        logger.error(f"❌ Shadow mode demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_sample_model(model_type: str, accuracy_bias: float, risk_level: float):
    """Create a sample model for demonstration"""
    try:
        # Create a simple logistic regression model with specific characteristics
        model = LogisticRegression(random_state=42)
        
        # Generate training data that reflects the model's bias
        n_samples = 1000
        np.random.seed(sum(ord(c) for c in model_type))  # Different seed per model type
        
        # Features that influence the model's behavior
        rsi = np.random.uniform(30, 70, n_samples)
        macd = np.random.normal(0, 1, n_samples)
        volatility = np.random.uniform(0.01, 0.05, n_samples)
        
        # Create target based on model characteristics
        signal_strength = accuracy_bias * risk_level
        noise_level = 1 - accuracy_bias
        
        # Generate target with bias
        target_prob = (
            0.3 * (rsi > 50).astype(int) +
            0.4 * (macd > 0).astype(int) +
            0.2 * (volatility < 0.03).astype(int) +
            np.random.normal(0, noise_level, n_samples)
        )
        
        # Convert to binary target
        y = (target_prob > 0.5).astype(int)
        
        # Features matrix
        X = np.column_stack([rsi, macd, volatility])
        
        # Train model
        model.fit(X, y)
        
        return model
        
    except Exception as e:
        logger.error(f"Sample model creation failed: {e}")
        # Return dummy model
        class DummyModel:
            def predict_proba(self, X):
                return [[0.5, 0.5]] * len(X)
            def predict(self, X):
                return [0] * len(X)
        return DummyModel()

def generate_stable_market_data(n_periods: int) -> pd.DataFrame:
    """Generate stable market conditions data"""
    try:
        np.random.seed(42)
        timestamps = [datetime.now() - timedelta(minutes=i*15) for i in range(n_periods-1, -1, -1)]
        
        data = []
        base_price = 50000
        
        for i, timestamp in enumerate(timestamps):
            # Stable price movement
            price_change = np.random.normal(0, 50)  # Small, stable changes
            current_price = base_price + price_change + i * 10  # Slight upward trend
            
            data.append({
                'timestamp': timestamp,
                'symbol': 'BTC/USDT',
                'open': current_price - np.random.normal(0, 20),
                'high': current_price + abs(np.random.normal(0, 30)),
                'low': current_price - abs(np.random.normal(0, 30)),
                'close': current_price,
                'volume': np.random.uniform(1000, 2000),
                'rsi': np.random.uniform(45, 55),  # Centered around 50
                'macd': np.random.normal(0, 0.5),
                'volatility': np.random.uniform(0.01, 0.02)
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Stable market data generation failed: {e}")
        return pd.DataFrame()

def generate_volatile_market_data(n_periods: int) -> pd.DataFrame:
    """Generate volatile market conditions data"""
    try:
        np.random.seed(123)
        timestamps = [datetime.now() - timedelta(minutes=i*15) for i in range(n_periods-1, -1, -1)]
        
        data = []
        base_price = 48000
        
        for i, timestamp in enumerate(timestamps):
            # Volatile price movement
            price_change = np.random.normal(0, 300)  # Large price swings
            current_price = base_price + price_change
            
            data.append({
                'timestamp': timestamp,
                'symbol': 'BTC/USDT',
                'open': current_price - np.random.normal(0, 150),
                'high': current_price + abs(np.random.normal(0, 200)),
                'low': current_price - abs(np.random.normal(0, 200)),
                'close': current_price,
                'volume': np.random.uniform(1500, 3000),
                'rsi': np.random.uniform(30, 70),  # Wide RSI range
                'macd': np.random.normal(0, 2.0),  # Higher MACD volatility
                'volatility': np.random.uniform(0.03, 0.08)
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Volatile market data generation failed: {e}")
        return pd.DataFrame()

def generate_trending_market_data(n_periods: int) -> pd.DataFrame:
    """Generate trending market conditions data"""
    try:
        np.random.seed(456)
        timestamps = [datetime.now() - timedelta(minutes=i*15) for i in range(n_periods-1, -1, -1)]
        
        data = []
        base_price = 45000
        
        for i, timestamp in enumerate(timestamps):
            # Strong upward trend with some noise
            trend_component = i * 200  # Strong upward trend
            noise = np.random.normal(0, 100)
            current_price = base_price + trend_component + noise
            
            data.append({
                'timestamp': timestamp,
                'symbol': 'BTC/USDT',
                'open': current_price - np.random.normal(0, 80),
                'high': current_price + abs(np.random.normal(0, 120)),
                'low': current_price - abs(np.random.normal(0, 120)),
                'close': current_price,
                'volume': np.random.uniform(2000, 4000),
                'rsi': np.random.uniform(55, 75),  # RSI biased upward
                'macd': np.random.normal(1.0, 0.8),  # Positive MACD bias
                'volatility': np.random.uniform(0.02, 0.04)
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Trending market data generation failed: {e}")
        return pd.DataFrame()

def demonstrate_shadow_concepts():
    """Demonstrate key shadow mode concepts"""
    logger.info(f"\n🧠 SHADOW MODE CONCEPTS")
    logger.info("=" * 23)
    
    try:
        concepts = {
            "Parallel Evaluation": [
                "Multiple models trade simultaneously on same data",
                "Real-time performance comparison",
                "No single point of failure",
                "Continuous model improvement identification"
            ],
            
            "Consensus Building": [
                "Aggregates signals from multiple models",
                "Reduces individual model bias",
                "Provides more robust trading signals",
                "Weighted voting based on model confidence"
            ],
            
            "Risk Mitigation": [
                "Identifies model divergence early",
                "Prevents catastrophic model failures",
                "Provides backup when primary model fails",
                "Maintains system uptime during model issues"
            ],
            
            "Performance Monitoring": [
                "Continuous performance tracking",
                "Automated model ranking and selection",
                "Historical performance analysis",
                "Adaptive model management"
            ]
        }
        
        logger.info("Key Shadow Mode Concepts:")
        for concept, explanations in concepts.items():
            logger.info(f"\n{concept}:")
            for explanation in explanations:
                logger.info(f"   • {explanation}")
        
        logger.info("✅ Shadow concepts demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ Shadow concepts demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe AI - Shadow Mode Demo")
    print("Parallel model comparison and performance monitoring")
    print()
    
    # Run main shadow mode demo
    await demonstrate_shadow_mode()
    
    # Run concepts demonstration
    demonstrate_shadow_concepts()
    
    print(f"\n🎉 SHADOW MODE DEMO COMPLETED")
    print("Chloe AI now has professional parallel model comparison capabilities!")

if __name__ == "__main__":
    asyncio.run(main())