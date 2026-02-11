#!/usr/bin/env python3
"""
Simple Learning Pipeline Demo
Working version for Chloe AI
"""

import logging
from datetime import datetime
import numpy as np
import pandas as pd
from learning.trainer import LearningPipeline, ModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_learning_demo():
    """Simple working demonstration"""
    print("🤖 SIMPLE LEARNING PIPELINE DEMO")
    print("=" * 35)
    
    try:
        # Initialize pipeline
        print("🔧 Initializing Learning Pipeline...")
        config = ModelConfig(model_type="basic", target_variable="is_profitable")
        pipeline = LearningPipeline(config)
        print("✅ Pipeline initialized")
        
        # Create simple sample data
        print("\n📊 Creating sample data...")
        np.random.seed(42)
        n_samples = 100
        
        # Simple features
        data = {
            'feature_rsi': np.random.uniform(30, 70, n_samples),
            'feature_macd': np.random.normal(0, 2, n_samples),
            'feature_volatility': np.random.uniform(0.01, 0.05, n_samples),
            'is_profitable': np.random.choice([0, 1], n_samples, p=[0.55, 0.45])
        }
        
        df = pd.DataFrame(data)
        print(f"   Created {len(df)} sample records")
        print(f"   Profitable trades: {df['is_profitable'].sum()} ({df['is_profitable'].mean()*100:.1f}%)")
        
        # Save data
        data_path = "./data/simple_sample.csv"
        df.to_csv(data_path, index=False)
        print(f"   Data saved to: {data_path}")
        
        # Load and prepare data
        print("\n📊 Preparing data for training...")
        loaded_df = pipeline.load_training_data(data_path)
        X, y = pipeline.prepare_features_and_target(loaded_df)
        print(f"   Features: {X.shape}")
        print(f"   Target: {y.shape}")
        
        # Train model
        print("\n🤖 Training model...")
        result = pipeline.train_model(X, y)
        print(f"   Training completed!")
        print(f"   Model: {result.model_name}")
        print(f"   Validation score: {result.validation_score:.4f}")
        print(f"   Training samples: {result.training_samples}")
        
        # Show feature importance
        print("\n🎯 Feature Importance:")
        for feature, importance in list(result.feature_importance.items())[:3]:
            print(f"   {feature}: {importance:.4f}")
        
        # Save model
        print("\n💾 Saving model...")
        model_path = pipeline.save_model()
        print(f"   Model saved to: {model_path}")
        
        print("\n✅ SIMPLE LEARNING PIPELINE DEMO COMPLETED!")
        print("🚀 Key achievements:")
        print("   • Created working ML pipeline")
        print("   • Trained model on sample data")
        print("   • Generated feature importance")
        print("   • Saved model for future use")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_learning_demo()