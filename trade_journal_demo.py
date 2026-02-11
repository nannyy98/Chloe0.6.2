#!/usr/bin/env python3
"""
Trade Journal Demo for Chloe AI
Demonstrating comprehensive trade logging for machine learning datasets
"""

import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
from learning.trade_logger import TradeJournal, TradeRecord

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_trade_journal():
    """Demonstrate trade journal capabilities"""
    logger.info("📊 TRADE JOURNAL DEMO")
    logger.info("=" * 25)
    
    try:
        # Initialize trade journal
        logger.info("🔧 Initializing Trade Journal...")
        journal = TradeJournal(storage_path="./data/demo_trade_logs")
        logger.info("✅ Trade Journal initialized")
        logger.info(f"   Session ID: {journal.session_id}")
        
        # Generate sample trade records
        logger.info(f"\n📝 GENERATING SAMPLE TRADE RECORDS:")
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
        regimes = ['STABLE', 'TRENDING', 'VOLATILE', 'CRISIS']
        strategies = ['momentum_long', 'mean_reversion', 'breakout']
        
        sample_trades = []
        
        # Create 50 sample trades
        for i in range(50):
            # Random trade parameters
            symbol = np.random.choice(symbols)
            side = np.random.choice(['BUY', 'SELL'])
            quantity = np.random.uniform(0.1, 5.0)
            
            # Simulate realistic prices
            base_prices = {'BTC/USDT': 50000, 'ETH/USDT': 2950, 'SOL/USDT': 95, 'ADA/USDT': 0.48}
            entry_price = base_prices[symbol] * np.random.uniform(0.98, 1.02)
            exit_price = entry_price * np.random.uniform(0.95, 1.05)  # 5% swing
            
            # Timing
            entry_time = datetime.now() - timedelta(hours=np.random.uniform(1, 48))
            holding_hours = np.random.uniform(0.5, 24)
            exit_time = entry_time + timedelta(hours=holding_hours)
            
            # Market conditions
            regime = np.random.choice(regimes)
            volatility = np.random.uniform(0.01, 0.05)
            volume = np.random.uniform(1000000, 10000000)
            spread = entry_price * np.random.uniform(0.0001, 0.0005)
            
            # Model signals
            confidence = np.random.uniform(0.6, 0.95)
            predicted_return = np.random.uniform(-0.02, 0.03)
            model_version = f"model_v{np.random.randint(1, 5)}"
            
            # Features
            features = {
                'rsi': np.random.uniform(30, 70),
                'macd': np.random.uniform(-2, 2),
                'volume_change': np.random.uniform(-0.3, 0.5),
                'atr': np.random.uniform(0.01, 0.03),
                'ema_distance': np.random.uniform(-0.02, 0.02)
            }
            
            # Performance calculation
            if side == 'BUY':
                pnl = (exit_price - entry_price) * quantity
            else:  # SELL
                pnl = (entry_price - exit_price) * quantity
            
            commission = abs(pnl) * 0.001  # 0.1% commission
            slippage = abs(pnl) * np.random.uniform(0.0002, 0.001)
            net_pnl = pnl - commission - slippage
            pnl_percentage = (net_pnl / (entry_price * quantity)) * 100
            
            # Risk metrics
            position_size = entry_price * quantity
            risk_per_trade = position_size * 0.02  # 2% risk
            stop_loss = entry_price * (0.98 if side == 'BUY' else 1.02)
            take_profit = entry_price * (1.03 if side == 'BUY' else 0.97)
            
            # Create trade record
            trade_record = TradeRecord(
                trade_id=f"TRADE_{datetime.now().strftime('%Y%m%d')}_{i:04d}",
                timestamp=exit_time,
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=entry_time,
                exit_time=exit_time,
                signal_confidence=confidence,
                predicted_return=predicted_return,
                model_version=model_version,
                features_used=features,
                market_regime=regime,
                volatility=volatility,
                volume=volume,
                spread=spread,
                pnl=net_pnl,
                pnl_percentage=pnl_percentage,
                holding_period=holding_hours,
                slippage=slippage,
                commission=commission,
                position_size=position_size,
                risk_per_trade=risk_per_trade,
                stop_loss_level=stop_loss,
                take_profit_level=take_profit,
                strategy_name=np.random.choice(strategies)
            )
            
            sample_trades.append(trade_record)
            
            if i < 5:  # Show first 5 trades
                logger.info(f"   Trade {i+1}: {side} {quantity:.2f} {symbol} "
                           f"@ ${entry_price:.2f} → ${exit_price:.2f} "
                           f"PnL: ${net_pnl:+.2f} ({pnl_percentage:+.2f}%)")
        
        logger.info(f"   ... and {len(sample_trades) - 5} more trades")
        
        # Log all trades
        logger.info(f"\n📝 LOGGING TRADES TO JOURNAL:")
        successful_logs = 0
        for trade in sample_trades:
            if journal.log_trade(trade):
                successful_logs += 1
        
        logger.info(f"   Successfully logged: {successful_logs}/{len(sample_trades)} trades")
        
        # Show performance metrics
        logger.info(f"\n📊 PERFORMANCE METRICS:")
        metrics = journal.get_performance_metrics()
        
        if metrics:
            logger.info(f"   Total Trades: {metrics['total_trades']}")
            logger.info(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
            logger.info(f"   Total PnL: ${metrics['total_pnl']:+.2f}")
            logger.info(f"   Average PnL: ${metrics['average_pnl']:+.2f}")
            logger.info(f"   Best Trade: ${metrics['best_trade']:+.2f}")
            logger.info(f"   Worst Trade: ${metrics['worst_trade']:+.2f}")
            logger.info(f"   Profit Factor: {metrics['profit_factor']:.2f}")
            logger.info(f"   Average Holding: {metrics['average_holding_hours']:.1f} hours")
        
        # Create dataset
        logger.info(f"\n📊 CREATING LEARNING DATASET:")
        dataset_path = journal.create_dataset()
        
        if dataset_path:
            # Export metadata
            metadata_path = journal.export_metadata(dataset_path)
            logger.info(f"   Dataset created: {dataset_path}")
            logger.info(f"   Metadata exported: {metadata_path}")
            
            # Load and examine dataset
            logger.info(f"\n📂 EXAMINING CREATED DATASET:")
            df = journal.load_dataset(dataset_path)
            
            logger.info(f"   Dataset shape: {df.shape}")
            logger.info(f"   Columns: {len(df.columns)}")
            logger.info(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            # Show sample data
            logger.info(f"\n📋 SAMPLE DATASET ROWS:")
            print(df[['symbol', 'side', 'entry_price', 'exit_price', 'pnl', 'pnl_percentage', 
                     'market_regime', 'signal_confidence']].head(3).to_string())
            
            # Show feature columns
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            logger.info(f"\n🎯 FEATURE COLUMNS ({len(feature_cols)}):")
            for i, col in enumerate(feature_cols[:5]):
                logger.info(f"   {col}")
            if len(feature_cols) > 5:
                logger.info(f"   ... and {len(feature_cols) - 5} more features")
        
        # Test error handling
        logger.info(f"\n🧪 ERROR HANDLING TEST:")
        
        # Try invalid trade record
        invalid_trade = TradeRecord(
            trade_id="INVALID_001",
            timestamp=datetime.now(),
            symbol="TEST/USDT",
            side="BUY",
            quantity=-1.0,  # Invalid negative quantity
            entry_price=100.0,
            exit_price=105.0,
            entry_time=datetime.now(),
            exit_time=datetime.now() - timedelta(hours=1)  # Invalid timing
        )
        
        invalid_result = journal.log_trade(invalid_trade)
        logger.info(f"   Invalid trade logging: {'SUCCESS' if invalid_result else 'REJECTED'}")
        
        # Reset session
        logger.info(f"\n🔄 SESSION MANAGEMENT:")
        old_session = journal.session_id
        journal.reset_session()
        logger.info(f"   Session reset: {old_session} → {journal.session_id}")
        
        logger.info(f"\n✅ TRADE JOURNAL DEMO COMPLETED SUCCESSFULLY")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented comprehensive trade logging system")
        logger.info("   • Created structured dataset for machine learning")
        logger.info("   • Built performance metrics and analytics")
        logger.info("   • Added metadata export and dataset management")
        logger.info("   • Included robust error handling and validation")
        
        logger.info(f"\n🎯 TRADE JOURNAL FEATURES:")
        logger.info("   Rich trade record structure with 25+ fields")
        logger.info("   Parquet dataset creation for efficient storage")
        logger.info("   Automatic feature column generation")
        logger.info("   Performance metrics calculation")
        logger.info("   Metadata export for dataset documentation")
        logger.info("   Session management for experiment tracking")
        
        logger.info(f"\n⏭️ NEXT STEPS:")
        logger.info("   1. Implement Learning Pipeline (Model training)")
        logger.info("   2. Build Model Validation Gate")
        logger.info("   3. Add Shadow Mode capabilities")
        logger.info("   4. Create Controlled Self-Learning Loop")
        
    except Exception as e:
        logger.error(f"❌ Trade journal demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_learning_concepts():
    """Demonstrate key machine learning concepts for trading"""
    logger.info(f"\n🧠 MACHINE LEARNING CONCEPTS")
    logger.info("=" * 32)
    
    try:
        concepts = {
            "Dataset Structure": [
                "Features: RSI, MACD, Volume, Volatility indicators",
                "Targets: Profitability classification, return prediction",
                "Metadata: Market regime, strategy, confidence scores",
                "Time series: Entry/exit timestamps, holding periods"
            ],
            
            "Learning Approach": [
                "Supervised learning: Predict trade profitability",
                "Reinforcement learning: Optimize strategy parameters",
                "Online learning: Continuous model updates",
                "Ensemble methods: Combine multiple model predictions"
            ],
            
            "Validation Methods": [
                "Walk-forward validation for time series",
                "Cross-validation with temporal splits",
                "Out-of-sample testing on recent data",
                "Monte Carlo simulations for robustness"
            ],
            
            "Risk Integration": [
                "Position sizing based on model confidence",
                "Risk-adjusted return optimization",
                "Regime-aware model selection",
                "Dynamic stop-loss/take-profit levels"
            ]
        }
        
        logger.info("Key Machine Learning Concepts for Trading:")
        for concept, explanations in concepts.items():
            logger.info(f"\n{concept}:")
            for explanation in explanations:
                logger.info(f"   • {explanation}")
        
        logger.info("✅ Learning concepts demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ Learning concepts demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe AI - Trade Journal Demo")
    print("Comprehensive trade logging for machine learning datasets")
    print()
    
    # Run main trade journal demo
    await demonstrate_trade_journal()
    
    # Run learning concepts demonstration
    demonstrate_learning_concepts()
    
    print(f"\n🎉 TRADE JOURNAL DEMO COMPLETED")
    print("Chloe AI now has professional trade logging capabilities!")

if __name__ == "__main__":
    asyncio.run(main())