#!/usr/bin/env python3
"""
Controlled Self-Learning Loop Demo for Chloe AI
Demonstrating automated daily model improvement with safety controls
"""

import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
from pathlib import Path
from learning.self_learner import SelfLearningController, LearningSchedule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_self_learning():
    """Demonstrate controlled self-learning capabilities"""
    logger.info("🎓 CONTROLLED SELF-LEARNING DEMO")
    logger.info("=" * 35)
    
    try:
        # Initialize self-learning controller
        logger.info("🔧 Initializing Self-Learning Controller...")
        
        schedule = LearningSchedule(
            daily_learning_time="02:00",
            retrain_frequency_days=1,
            min_new_data_points=20,  # Lower for demo
            max_training_time_minutes=5,  # Shorter for demo
            performance_improvement_threshold=0.02  # Lower for demo
        )
        
        learner = SelfLearningController(schedule)
        logger.info("✅ Self-Learning Controller initialized")
        logger.info(f"   Schedule: Daily at {schedule.daily_learning_time}")
        logger.info(f"   Min data points: {schedule.min_new_data_points}")
        logger.info(f"   Improvement threshold: {schedule.performance_improvement_threshold*100:.1f}%")
        
        # Create sample trade data for initial training
        logger.info(f"\n📊 CREATING INITIAL TRAINING DATA:")
        
        # Create directory for trade data
        data_dir = Path("./data/trade_journal")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate initial training dataset
        initial_data = create_sample_trade_dataset(100, "initial_training")
        initial_file = data_dir / "initial_trades.csv"
        initial_data.to_csv(initial_file, index=False)
        logger.info(f"   Initial data: {len(initial_data)} trades saved to {initial_file}")
        
        # Set initial performance baseline
        learner.performance_baseline = 0.55  # 55% baseline accuracy
        logger.info(f"   Performance baseline set to: {learner.performance_baseline:.3f}")
        
        # Demonstrate single learning cycle
        logger.info(f"\n🎓 DEMONSTRATING SINGLE LEARNING CYCLE:")
        
        # Force learning for demo (ignore schedule)
        session1 = await learner.execute_learning_cycle(force_learning=True)
        
        logger.info(f"   Session ID: {session1.session_id}")
        logger.info(f"   Outcome: {session1.learning_outcome}")
        logger.info(f"   Training samples: {session1.training_samples}")
        logger.info(f"   Data quality: {session1.data_quality_score:.3f}")
        logger.info(f"   Performance: {session1.performance_before:.3f} → {session1.performance_after:.3f}")
        logger.info(f"   Improvement: {session1.improvement:+.4f}")
        
        # Create improved data for second cycle
        logger.info(f"\n📊 CREATING IMPROVED TRAINING DATA:")
        improved_data = create_sample_trade_dataset(80, "improved_data", quality_factor=1.2)
        improved_file = data_dir / "improved_trades.csv"
        improved_data.to_csv(improved_file, index=False)
        logger.info(f"   Improved data: {len(improved_data)} trades saved to {improved_file}")
        
        # Second learning cycle
        logger.info(f"\n🎓 DEMONSTRATING IMPROVEMENT LEARNING CYCLE:")
        session2 = await learner.execute_learning_cycle(force_learning=True)
        
        logger.info(f"   Session ID: {session2.session_id}")
        logger.info(f"   Outcome: {session2.learning_outcome}")
        logger.info(f"   Performance: {session2.performance_before:.3f} → {session2.performance_after:.3f}")
        logger.info(f"   Improvement: {session2.improvement:+.4f}")
        
        # Demonstrate insufficient data scenario
        logger.info(f"\n📊 DEMONSTRATING INSUFFICIENT DATA SCENARIO:")
        
        # Create minimal dataset
        minimal_data = create_sample_trade_dataset(10, "minimal_data")  # Below threshold
        minimal_file = data_dir / "minimal_trades.csv"
        minimal_data.to_csv(minimal_file, index=False)
        logger.info(f"   Minimal data: {len(minimal_data)} trades (< {schedule.min_new_data_points} threshold)")
        
        session3 = await learner.execute_learning_cycle(force_learning=True)
        logger.info(f"   Session outcome: {session3.learning_outcome}")
        logger.info(f"   Reason: {session3.validation_result}")
        
        # Show learning history
        logger.info(f"\n📚 LEARNING HISTORY:")
        summary = learner.get_learning_summary()
        
        logger.info(f"   Total sessions: {summary['total_sessions']}")
        logger.info(f"   Successful sessions: {summary['successful_sessions']}")
        logger.info(f"   Success rate: {summary['success_rate']*100:.1f}%")
        logger.info(f"   Current model version: {summary['current_model_version']}")
        logger.info(f"   Performance baseline: {summary['performance_baseline']:.3f}")
        
        if 'recent_sessions' in summary:
            logger.info(f"   Recent sessions:")
            for session in summary['recent_sessions']:
                logger.info(f"      {session['session_id']}: {session['outcome']} "
                           f"({session['samples']} samples, {session['improvement']:+.4f})")
        
        # Test learning schedule logic
        logger.info(f"\n⏰ LEARNING SCHEDULE DEMO:")
        
        # Test schedule checking
        should_learn = learner._should_learn_now()
        logger.info(f"   Should learn now: {should_learn}")
        
        next_run = learner._calculate_next_run_time()
        time_until_next = (next_run - datetime.now()).total_seconds() / 3600
        logger.info(f"   Next scheduled run: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   Hours until next run: {time_until_next:.1f}")
        
        # Demonstrate version management
        logger.info(f"\n🔢 VERSION MANAGEMENT:")
        logger.info(f"   Starting version: {learner.current_model_version}")
        
        # Simulate multiple version increments
        for i in range(3):
            new_version = learner._increment_version()
            logger.info(f"   Increment {i+1}: {new_version}")
            learner.current_model_version = new_version
        
        # Export learning log
        logger.info(f"\n📝 EXPORTING LEARNING LOG:")
        log_file = learner.export_learning_log()
        logger.info(f"   Log exported to: {log_file}")
        
        # Show data quality assessment
        logger.info(f"\n📈 DATA QUALITY ASSESSMENT:")
        
        quality_tests = [
            ("High Quality Data", create_sample_trade_dataset(50, "hq", quality_factor=1.0)),
            ("Low Quality Data", create_sample_trade_dataset(50, "lq", quality_factor=0.3)),
            ("Outlier Data", create_sample_trade_dataset(50, "outliers", outlier_factor=2.0))
        ]
        
        for test_name, test_data in quality_tests:
            quality_score = learner._assess_data_quality(test_data)
            logger.info(f"   {test_name}: {quality_score:.3f}")
        
        logger.info(f"\n✅ CONTROLLED SELF-LEARNING DEMO COMPLETED")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented automated daily learning system")
        logger.info("   • Created safety controls and validation gates")
        logger.info("   • Built performance improvement tracking")
        logger.info("   • Added data quality assessment")
        logger.info("   • Developed version management system")
        logger.info("   • Provided comprehensive logging and reporting")
        
        logger.info(f"\n🎯 SELF-LEARNING FEATURES:")
        logger.info("   Scheduled daily learning cycles")
        logger.info("   Automatic data quality assessment")
        logger.info("   Performance improvement validation")
        logger.info("   Model version tracking")
        logger.info("   Comprehensive session logging")
        logger.info("   Configurable safety thresholds")
        
        logger.info(f"\n⏭️ NEXT STEPS:")
        logger.info("   1. Implement Risk Sandbox for stress testing")
        logger.info("   2. Create Paper Performance Dashboard")
        logger.info("   3. Add Live Trading Integration")
        logger.info("   4. Implement Production Deployment Pipeline")
        
    except Exception as e:
        logger.error(f"❌ Self-learning demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_sample_trade_dataset(n_trades: int, dataset_name: str, 
                              quality_factor: float = 1.0, 
                              outlier_factor: float = 1.0) -> pd.DataFrame:
    """Create sample trade dataset for demonstration"""
    try:
        np.random.seed(sum(ord(c) for c in dataset_name))
        
        # Base parameters
        base_win_rate = 0.55 * quality_factor
        base_avg_win = 120 * quality_factor
        base_avg_loss = -100 * quality_factor
        volatility = 0.03 / quality_factor
        
        # Generate trades
        n_wins = int(n_trades * base_win_rate)
        n_losses = n_trades - n_wins
        
        # Winning trades
        wins = np.random.normal(base_avg_win, abs(base_avg_win * volatility), n_wins)
        wins = np.abs(wins) * outlier_factor
        
        # Losing trades
        losses = np.random.normal(base_avg_loss, abs(base_avg_loss * volatility), n_losses)
        losses = -np.abs(losses) * outlier_factor
        
        # Combine and shuffle
        all_pnl = np.concatenate([wins, losses])
        np.random.shuffle(all_pnl)
        
        # Generate timestamps
        start_date = datetime.now() - timedelta(days=7)
        timestamps = [start_date + timedelta(hours=i*2) for i in range(n_trades)]
        
        # Calculate returns
        base_position_size = 1000
        returns_pct = (all_pnl / base_position_size) * 100
        
        # Generate features
        rsi_values = np.random.uniform(30, 70, n_trades)
        macd_values = np.random.normal(0, 1, n_trades)
        volatility_values = np.random.uniform(0.01, 0.05, n_trades)
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'symbol': np.random.choice(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'], n_trades),
            'side': np.random.choice(['BUY', 'SELL'], n_trades),
            'pnl': all_pnl,
            'pnl_percentage': returns_pct,
            'is_profitable': all_pnl > 0,
            'feature_rsi': rsi_values,
            'feature_macd': macd_values,
            'feature_volatility': volatility_values,
            'volume': np.random.uniform(1000, 5000, n_trades),
            'price_entry': np.random.uniform(40000, 60000, n_trades),
            'price_exit': np.random.uniform(40000, 60000, n_trades)
        })
        
        return data
        
    except Exception as e:
        logger.error(f"Sample dataset creation failed: {e}")
        return pd.DataFrame()

def demonstrate_learning_concepts():
    """Demonstrate key self-learning concepts"""
    logger.info(f"\n🧠 CONTROLLED SELF-LEARNING CONCEPTS")
    logger.info("=" * 38)
    
    try:
        concepts = {
            "Safety Controls": [
                "Scheduled learning prevents continuous resource consumption",
                "Minimum data thresholds ensure statistical significance",
                "Validation gates prevent deployment of poor models",
                "Performance improvement requirements maintain quality standards"
            ],
            
            "Data Management": [
                "Automatic data quality assessment filters poor data",
                "Version tracking maintains model lineage",
                "Historical session logging enables audit trails",
                "Baseline performance tracking measures true improvement"
            ],
            
            "Process Automation": [
                "Daily learning cycles without manual intervention",
                "Automatic success/failure handling",
                "Self-healing from temporary failures",
                "Progressive model versioning"
            ],
            
            "Risk Mitigation": [
                "Controlled learning frequency prevents overfitting",
                "Data quality gates protect against garbage-in-garbage-out",
                "Performance validation ensures models meet standards",
                "Rollback capability to previous working versions"
            ]
        }
        
        logger.info("Key Controlled Self-Learning Concepts:")
        for concept, explanations in concepts.items():
            logger.info(f"\n{concept}:")
            for explanation in explanations:
                logger.info(f"   • {explanation}")
        
        logger.info("✅ Learning concepts demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ Learning concepts demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe AI - Controlled Self-Learning Loop Demo")
    print("Automated daily model improvement with safety controls")
    print()
    
    # Run main self-learning demo
    await demonstrate_self_learning()
    
    # Run concepts demonstration
    demonstrate_learning_concepts()
    
    print(f"\n🎉 CONTROLLED SELF-LEARNING DEMO COMPLETED")
    print("Chloe AI now has professional automated learning capabilities!")

if __name__ == "__main__":
    asyncio.run(main())