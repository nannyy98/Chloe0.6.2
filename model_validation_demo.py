#!/usr/bin/env python3
"""
Model Validation Gate Demo for Chloe AI
Demonstrating professional model quality assurance
"""

import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from learning.validator import ModelValidationGate, ValidationCriteria

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_model_validation():
    """Demonstrate model validation gate capabilities"""
    logger.info("🔍 MODEL VALIDATION GATE DEMO")
    logger.info("=" * 32)
    
    try:
        # Initialize validation gate
        logger.info("🔧 Initializing Model Validation Gate...")
        
        criteria = ValidationCriteria(
            min_sharpe_ratio=1.2,
            max_drawdown=0.20,
            min_win_rate=0.52,
            min_trades=50,
            min_months=3,
            stability_threshold=0.7
        )
        
        validator = ModelValidationGate(criteria)
        logger.info("✅ Model Validation Gate initialized")
        logger.info(f"   Sharpe threshold: {criteria.min_sharpe_ratio}")
        logger.info(f"   Max drawdown: {criteria.max_drawdown*100:.1f}%")
        logger.info(f"   Min win rate: {criteria.min_win_rate*100:.1f}%")
        logger.info(f"   Min trades: {criteria.min_trades}")
        
        # Test Case 1: Excellent Model
        logger.info(f"\n🧪 TEST CASE 1: EXCELLENT MODEL")
        excellent_data = create_sample_trade_data(
            n_trades=120,
            win_rate=0.65,
            avg_win=150,
            avg_loss=-100,
            volatility=0.02
        )
        
        excellent_result = validator.validate_model(
            excellent_data,
            {'model_id': 'excellent_model_v1.0'}
        )
        
        logger.info(f"   Status: {excellent_result.overall_status}")
        logger.info(f"   Sharpe: {excellent_result.performance_metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"   Drawdown: {excellent_result.risk_metrics.get('max_drawdown', 0)*100:.1f}%")
        logger.info(f"   Win Rate: {excellent_result.performance_metrics.get('win_rate', 0)*100:.1f}%")
        
        # Test Case 2: Marginal Model
        logger.info(f"\n🧪 TEST CASE 2: MARGINAL MODEL")
        marginal_data = create_sample_trade_data(
            n_trades=75,
            win_rate=0.53,
            avg_win=120,
            avg_loss=-110,
            volatility=0.03
        )
        
        marginal_result = validator.validate_model(
            marginal_data,
            {'model_id': 'marginal_model_v1.0'}
        )
        
        logger.info(f"   Status: {marginal_result.overall_status}")
        logger.info(f"   Sharpe: {marginal_result.performance_metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"   Drawdown: {marginal_result.risk_metrics.get('max_drawdown', 0)*100:.1f}%")
        logger.info(f"   Win Rate: {marginal_result.performance_metrics.get('win_rate', 0)*100:.1f}%")
        
        # Test Case 3: Poor Model
        logger.info(f"\n🧪 TEST CASE 3: POOR MODEL")
        poor_data = create_sample_trade_data(
            n_trades=40,
            win_rate=0.45,
            avg_win=80,
            avg_loss=-120,
            volatility=0.04
        )
        
        poor_result = validator.validate_model(
            poor_data,
            {'model_id': 'poor_model_v1.0'}
        )
        
        logger.info(f"   Status: {poor_result.overall_status}")
        logger.info(f"   Sharpe: {poor_result.performance_metrics.get('sharpe_ratio', 0):.3f}")
        logger.info(f"   Drawdown: {poor_result.risk_metrics.get('max_drawdown', 0)*100:.1f}%")
        logger.info(f"   Win Rate: {poor_result.performance_metrics.get('win_rate', 0)*100:.1f}%")
        
        # Show detailed validation report
        logger.info(f"\n📋 DETAILED VALIDATION REPORT (Excellent Model):")
        excellent_report = validator.generate_validation_report(excellent_result)
        print(excellent_report)
        
        # Compare all models
        logger.info(f"\n📊 MODEL COMPARISON:")
        models_comparison = [
            ("Excellent Model", excellent_result),
            ("Marginal Model", marginal_result),
            ("Poor Model", poor_result)
        ]
        
        logger.info(f"{'Model':<15} {'Status':<12} {'Sharpe':<8} {'Drawdown':<10} {'Win Rate':<10} {'Trades':<8}")
        logger.info("-" * 65)
        
        for model_name, result in models_comparison:
            status = result.overall_status
            sharpe = result.performance_metrics.get('sharpe_ratio', 0)
            drawdown = result.risk_metrics.get('max_drawdown', 0) * 100
            win_rate = result.performance_metrics.get('win_rate', 0) * 100
            trades = result.performance_metrics.get('total_trades', 0)
            
            logger.info(f"{model_name:<15} {status:<12} {sharpe:<8.3f} {drawdown:<10.1f}% {win_rate:<10.1f}% {trades:<8}")
        
        # Show validation criteria details
        logger.info(f"\n🎯 VALIDATION CRITERIA BREAKDOWN:")
        
        criteria_names = {
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown': 'Max Drawdown',
            'win_rate': 'Win Rate',
            'minimum_trades': 'Minimum Trades',
            'stability': 'Stability'
        }
        
        for criterion_key, criterion_name in criteria_names.items():
            excellent_criterion = excellent_result.criteria_results.get(criterion_key, {})
            marginal_criterion = marginal_result.criteria_results.get(criterion_key, {})
            poor_criterion = poor_result.criteria_results.get(criterion_key, {})
            
            logger.info(f"\n{criterion_name}:")
            logger.info(f"  Excellent: {'✅' if excellent_criterion.get('passed', False) else '❌'} "
                       f"({excellent_criterion.get('value', 0):.3f})")
            logger.info(f"  Marginal:  {'✅' if marginal_criterion.get('passed', False) else '❌'} "
                       f"({marginal_criterion.get('value', 0):.3f})")
            logger.info(f"  Poor:      {'✅' if poor_criterion.get('passed', False) else '❌'} "
                       f"({poor_criterion.get('value', 0):.3f})")
        
        # Show recommendations
        logger.info(f"\n💡 RECOMMENDATIONS BY MODEL:")
        
        for model_name, result in models_comparison:
            logger.info(f"\n{model_name}:")
            for i, rec in enumerate(result.recommendations[:3], 1):  # Show first 3 recommendations
                logger.info(f"  {i}. {rec}")
        
        # Test edge cases
        logger.info(f"\n🧪 EDGE CASE TESTING:")
        
        # Very short dataset
        short_data = create_sample_trade_data(n_trades=20, win_rate=0.7, avg_win=200, avg_loss=-50)
        short_result = validator.validate_model(short_data, {'model_id': 'short_test'})
        logger.info(f"   Short dataset (20 trades): {short_result.overall_status}")
        
        # High volatility dataset
        volatile_data = create_sample_trade_data(n_trades=100, win_rate=0.6, avg_win=300, avg_loss=-200, volatility=0.06)
        volatile_result = validator.validate_model(volatile_data, {'model_id': 'volatile_test'})
        logger.info(f"   High volatility: {volatile_result.overall_status} "
                   f"(Drawdown: {volatile_result.risk_metrics.get('max_drawdown', 0)*100:.1f}%)")
        
        # Validation history
        logger.info(f"\n📚 VALIDATION HISTORY:")
        logger.info(f"   Total validations performed: {len(validator.validation_history)}")
        
        status_counts = {}
        for result in validator.validation_history:
            status = result.overall_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in status_counts.items():
            logger.info(f"   {status}: {count}")
        
        logger.info(f"\n✅ MODEL VALIDATION GATE DEMO COMPLETED")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented professional model validation system")
        logger.info("   • Created comprehensive quality criteria")
        logger.info("   • Built detailed performance and risk analysis")
        logger.info("   • Added stability and consistency evaluation")
        logger.info("   • Provided actionable improvement recommendations")
        
        logger.info(f"\n🎯 VALIDATION GATE FEATURES:")
        logger.info("   Multi-criteria evaluation (Sharpe, Drawdown, Win Rate, etc.)")
        logger.info("   Stability analysis across time periods")
        logger.info("   Risk-adjusted performance assessment")
        logger.info("   Detailed reporting and recommendations")
        logger.info("   Historical validation tracking")
        logger.info("   Configurable quality thresholds")
        
        logger.info(f"\n⏭️ NEXT STEPS:")
        logger.info("   1. Implement Shadow Mode (parallel model comparison)")
        logger.info("   2. Create Controlled Self-Learning Loop")
        logger.info("   3. Build Risk Sandbox for stress testing")
        logger.info("   4. Develop Paper Performance Dashboard")
        
    except Exception as e:
        logger.error(f"❌ Model validation demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_sample_trade_data(n_trades: int, win_rate: float, avg_win: float, 
                           avg_loss: float, volatility: float = 0.02) -> pd.DataFrame:
    """Create realistic sample trade data"""
    try:
        # Generate winning and losing trades
        n_wins = int(n_trades * win_rate)
        n_losses = n_trades - n_wins
        
        # Winning trades
        wins = np.random.normal(avg_win, abs(avg_win * volatility), n_wins)
        wins = np.abs(wins)  # Ensure positive
        
        # Losing trades  
        losses = np.random.normal(avg_loss, abs(avg_loss * volatility), n_losses)
        losses = -np.abs(losses)  # Ensure negative
        
        # Combine and shuffle
        all_pnl = np.concatenate([wins, losses])
        np.random.shuffle(all_pnl)
        
        # Generate timestamps
        start_date = datetime(2023, 1, 1)
        timestamps = [start_date + timedelta(hours=i*4) for i in range(n_trades)]
        
        # Calculate returns
        base_position_size = 1000
        returns_pct = (all_pnl / base_position_size) * 100
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'pnl': all_pnl,
            'pnl_percentage': returns_pct,
            'is_profitable': all_pnl > 0
        })
        
        return data
        
    except Exception as e:
        logger.error(f"Sample data creation failed: {e}")
        # Return minimal dataframe
        return pd.DataFrame({
            'timestamp': [datetime.now()],
            'pnl': [0],
            'pnl_percentage': [0],
            'is_profitable': [False]
        })

def demonstrate_validation_concepts():
    """Demonstrate key validation concepts"""
    logger.info(f"\n🧠 MODEL VALIDATION CONCEPTS")
    logger.info("=" * 28)
    
    try:
        concepts = {
            "Quality Gates": [
                "Models must pass minimum performance thresholds",
                "Risk metrics are as important as return metrics",
                "Stability across time periods is crucial",
                "No single metric determines model quality"
            ],
            
            "Risk Management": [
                "Maximum drawdown limits protect capital",
                "VaR measures tail risk exposure",
                "Downside deviation focuses on losses",
                "Calmar ratio balances return vs risk"
            ],
            
            "Performance Metrics": [
                "Sharpe ratio: Risk-adjusted returns",
                "Win rate: Consistency of profitable trades",
                "Profit factor: Gross profits vs losses",
                "Stability score: Consistent performance over time"
            ],
            
            "Validation Process": [
                "Multi-dimensional evaluation approach",
                "Clear pass/fail criteria",
                "Actionable improvement recommendations",
                "Historical tracking of model performance"
            ]
        }
        
        logger.info("Key Model Validation Concepts:")
        for concept, explanations in concepts.items():
            logger.info(f"\n{concept}:")
            for explanation in explanations:
                logger.info(f"   • {explanation}")
        
        logger.info("✅ Validation concepts demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ Validation concepts demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe AI - Model Validation Gate Demo")
    print("Professional model quality assurance system")
    print()
    
    # Run main validation demo
    await demonstrate_model_validation()
    
    # Run concepts demonstration
    demonstrate_validation_concepts()
    
    print(f"\n🎉 MODEL VALIDATION GATE DEMO COMPLETED")
    print("Chloe AI now has professional model validation capabilities!")

if __name__ == "__main__":
    asyncio.run(main())