#!/usr/bin/env python3
"""
Learning Pipeline Demo for Chloe AI
Demonstrating machine learning training on trade data
"""

import asyncio
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from learning.trainer import LearningPipeline, ModelConfig, XGB_AVAILABLE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_learning_pipeline():
    """Demonstrate learning pipeline capabilities"""
    logger.info("🤖 LEARNING PIPELINE DEMO")
    logger.info("=" * 27)
    
    try:
        # Initialize learning pipeline
        logger.info("🔧 Initializing Learning Pipeline...")
        
        config = ModelConfig(
            model_type="xgboost",
            target_variable="is_profitable",
            cross_validation_folds=3,
            validation_metric="accuracy"
        )
        
        pipeline = LearningPipeline(config)
        logger.info("✅ Learning Pipeline initialized")
        logger.info(f"   Model type: {config.model_type}")
        logger.info(f"   Target variable: {config.target_variable}")
        logger.info(f"   CV folds: {config.cross_validation_folds}")
        
        # Create sample training data (since we don't have real trade data yet)
        logger.info(f"\n📊 CREATING SAMPLE TRAINING DATA:")
        
        # Generate realistic trade-like data
        np.random.seed(42)
        n_samples = 200
        
        # Create feature data
        data = {
            'feature_rsi': np.random.uniform(30, 70, n_samples),
            'feature_macd': np.random.normal(0, 2, n_samples),
            'feature_volume_change': np.random.uniform(-0.5, 0.8, n_samples),
            'feature_atr': np.random.uniform(0.01, 0.05, n_samples),
            'feature_ema_distance': np.random.uniform(-0.03, 0.03, n_samples),
            'feature_bollinger_width': np.random.uniform(1.5, 3.5, n_samples),
            'feature_stochastic_k': np.random.uniform(20, 80, n_samples),
            'volatility': np.random.uniform(0.01, 0.08, n_samples),
            'market_regime_STABLE': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'market_regime_TRENDING': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'market_regime_VOLATILE': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        }
        
        # Create synthetic target based on features (with some noise)
        # Profitable trades when RSI is in middle range and MACD is positive
        rsi_signal = ((data['feature_rsi'] > 40) & (data['feature_rsi'] < 60)).astype(int)
        macd_signal = (data['feature_macd'] > 0).astype(int)
        vol_signal = (data['volatility'] < 0.04).astype(int)
        
        # Combine signals with weights
        profit_probability = (
            0.3 * rsi_signal + 
            0.4 * macd_signal + 
            0.2 * vol_signal + 
            0.1 * np.random.uniform(0, 1, n_samples)  # Noise
        )
        
        # Generate binary target
        data['is_profitable'] = (np.random.random(n_samples) < profit_probability).astype(int)
        data['pnl_percentage'] = np.random.normal(0, 2, n_samples) + (data['is_profitable'] * 1.5)
        
        # Add timestamps
        start_date = datetime(2023, 1, 1)
        dates = [start_date + pd.Timedelta(hours=i) for i in range(n_samples)]
        data['timestamp'] = dates
        data['entry_time'] = dates
        data['exit_time'] = [d + pd.Timedelta(hours=4) for d in dates]
        
        # Add metadata
        data['symbol'] = np.random.choice(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'], n_samples)
        data['side'] = np.random.choice(['BUY', 'SELL'], n_samples)
        data['pnl'] = data['pnl_percentage'] * 100  # Scale for realism
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        logger.info(f"   Generated {len(df)} sample trades")
        logger.info(f"   Feature columns: {len([col for col in df.columns if col.startswith('feature_')])}")
        logger.info(f"   Profitable trades: {df['is_profitable'].sum()} ({df['is_profitable'].mean()*100:.1f}%)")
        logger.info(f"   Average PnL: {df['pnl_percentage'].mean():+.2f}%")
        
        # Save sample data
        sample_data_path = "./data/sample_trade_data.csv"
        df.to_csv(sample_data_path, index=False)
        logger.info(f"   Sample data saved to: {sample_data_path}")
        
        # Load and prepare data
        logger.info(f"\n📊 LOADING AND PREPARING DATA:")
        loaded_df = pipeline.load_training_data(sample_data_path)
        
        X, y = pipeline.prepare_features_and_target(loaded_df)
        
        logger.info(f"   Features shape: {X.shape}")
        logger.info(f"   Target shape: {y.shape}")
        logger.info(f"   Feature columns: {list(X.columns)}")
        
        # Train model
        logger.info(f"\n🤖 TRAINING MACHINE LEARNING MODEL:")
        training_result = pipeline.train_model(X, y)
        
        # Show training results
        logger.info(f"   Model: {training_result.model_name} v{training_result.model_version}")
        logger.info(f"   Training samples: {training_result.training_samples}")
        logger.info(f"   Validation samples: {training_result.validation_samples}")
        logger.info(f"   Training score: {training_result.train_score:.4f}")
        logger.info(f"   Validation score: {training_result.validation_score:.4f}")
        logger.info(f"   Cross-validation: {np.mean(training_result.cross_validation_scores):.4f} "
                   f"± {np.std(training_result.cross_validation_scores):.4f}")
        
        # Evaluate model performance
        logger.info(f"\n📊 MODEL PERFORMANCE EVALUATION:")
        evaluation = pipeline.evaluate_model_performance(training_result)
        
        perf_metrics = evaluation['performance_metrics']
        logger.info(f"   Performance Metrics:")
        logger.info(f"      Train Accuracy: {perf_metrics['train_score']:.4f}")
        logger.info(f"      Validation Accuracy: {perf_metrics['validation_score']:.4f}")
        logger.info(f"      CV Mean: {perf_metrics['cv_mean']:.4f}")
        logger.info(f"      CV Std: {perf_metrics['cv_std']:.4f}")
        
        stability_metrics = evaluation['stability_metrics']
        logger.info(f"   Stability Metrics:")
        logger.info(f"      CV Stability: {stability_metrics['cv_stability']:.4f}")
        logger.info(f"      Overfitting Risk: {stability_metrics['overfitting_risk']:.4f}")
        
        # Show feature importance
        logger.info(f"\n🎯 FEATURE IMPORTANCE:")
        feature_importance = evaluation['feature_importance']
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        for feature, importance in sorted_features[:5]:
            logger.info(f"   {feature}: {importance:.4f}")
        
        if len(sorted_features) > 5:
            logger.info(f"   ... and {len(sorted_features) - 5} more features")
        
        # Test model predictions
        logger.info(f"\n🔮 MODEL PREDICTIONS DEMO:")
        
        # Make predictions on sample data
        if pipeline.current_model and hasattr(pipeline.current_model, 'predict'):
            sample_predictions = pipeline.current_model.predict(X.head(10))
            sample_actual = y.head(10)
            
            logger.info("   Sample predictions vs actual:")
            for i, (pred, actual) in enumerate(zip(sample_predictions, sample_actual)):
                pred_display = f"{pred:.3f}" if isinstance(pred, float) else str(pred)
                actual_display = f"{actual:.3f}" if isinstance(actual, float) else str(actual)
                logger.info(f"      Sample {i+1}: Predicted={pred_display}, Actual={actual_display}")
        
        # Save model
        logger.info(f"\n💾 MODEL PERSISTENCE:")
        model_path = pipeline.save_model()
        logger.info(f"   Model saved to: {model_path}")
        
        # Test model loading
        pipeline_new = LearningPipeline()
        pipeline_new.load_model(model_path)
        logger.info(f"   Model successfully loaded from: {model_path}")
        
        # Test different model types
        logger.info(f"\n🔍 MODEL TYPE COMPARISON:")
        
        model_types = ["random_forest", "basic"]  # Skip xgboost if not available
        if not XGB_AVAILABLE:
            model_types.insert(0, "xgboost")
        
        comparison_results = []
        
        for model_type in model_types:
            logger.info(f"   Testing {model_type}...")
            
            test_config = ModelConfig(
                model_type=model_type,
                target_variable="is_profitable"
            )
            
            test_pipeline = LearningPipeline(test_config)
            test_result = test_pipeline.train_model(X, y)
            comparison_results.append((model_type, test_result.validation_score))
            
            logger.info(f"      Validation score: {test_result.validation_score:.4f}")
        
        # Show comparison
        logger.info(f"\n📊 MODEL COMPARISON SUMMARY:")
        for model_type, score in comparison_results:
            logger.info(f"   {model_type}: {score:.4f}")
        
        best_model = max(comparison_results, key=lambda x: x[1])
        logger.info(f"   Best model: {best_model[0]} ({best_model[1]:.4f})")
        
        # Show training history
        logger.info(f"\n📚 TRAINING HISTORY:")
        logger.info(f"   Total models trained: {len(pipeline.training_history)}")
        
        for i, result in enumerate(pipeline.training_history, 1):
            logger.info(f"   Model {i}: {result.model_name} v{result.model_version} - "
                       f"Val Score: {result.validation_score:.4f}")
        
        logger.info(f"\n✅ LEARNING PIPELINE DEMO COMPLETED SUCCESSFULLY")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented comprehensive ML pipeline for trading data")
        logger.info("   • Created automatic feature detection and preprocessing")
        logger.info("   • Built model training with cross-validation")
        logger.info("   • Added performance evaluation and feature importance")
        logger.info("   • Included model persistence and version management")
        
        logger.info(f"\n🎯 LEARNING PIPELINE FEATURES:")
        logger.info("   Multiple model support (XGBoost, Random Forest, basic models)")
        logger.info("   Automatic feature column detection")
        logger.info("   Time series cross-validation")
        logger.info("   Comprehensive performance metrics")
        logger.info("   Feature importance analysis")
        logger.info("   Model versioning and persistence")
        
        logger.info(f"\n⏭️ NEXT STEPS:")
        logger.info("   1. Implement Model Validation Gate (Sharpe > 1.2, DD < 20%)")
        logger.info("   2. Add Shadow Mode capabilities")
        logger.info("   3. Create Controlled Self-Learning Loop")
        logger.info("   4. Build Risk Sandbox for stress testing")
        
    except Exception as e:
        logger.error(f"❌ Learning pipeline demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_ml_concepts():
    """Demonstrate key machine learning concepts for trading"""
    logger.info(f"\n🧠 MACHINE LEARNING CONCEPTS FOR TRADING")
    logger.info("=" * 42)
    
    try:
        concepts = {
            "Feature Engineering": [
                "Technical indicators (RSI, MACD, ATR, Bollinger Bands)",
                "Market regime features (volatility, trend strength)",
                "Volume and liquidity indicators",
                "Price pattern recognition features"
            ],
            
            "Target Variables": [
                "Binary classification: Profitable vs Non-profitable",
                "Regression: Expected return percentage",
                "Multi-class: Strong Buy/Hold/Strong Sell",
                "Risk-adjusted targets: Sharpe ratio optimization"
            ],
            
            "Model Selection": [
                "Tree-based models: XGBoost, Random Forest (good for tabular data)",
                "Linear models: Logistic Regression (interpretable)",
                "Neural networks: LSTM, Transformers (sequential patterns)",
                "Ensemble methods: Combining multiple model predictions"
            ],
            
            "Validation Strategy": [
                "Time series cross-validation (avoid lookahead bias)",
                "Walk-forward analysis for sequential data",
                "Out-of-sample testing on recent data",
                "Monte Carlo simulations for robustness"
            ],
            
            "Risk Integration": [
                "Position sizing based on prediction confidence",
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
        
        logger.info("✅ ML concepts demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ ML concepts demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe AI - Learning Pipeline Demo")
    print("Machine learning training on trade data")
    print()
    
    # Run main learning pipeline demo
    await demonstrate_learning_pipeline()
    
    # Run ML concepts demonstration
    demonstrate_ml_concepts()
    
    print(f"\n🎉 LEARNING PIPELINE DEMO COMPLETED")
    print("Chloe AI now has professional machine learning capabilities!")

if __name__ == "__main__":
    asyncio.run(main())