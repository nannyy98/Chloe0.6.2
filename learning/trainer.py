"""
Learning Pipeline for Chloe AI - Phase 3
Machine learning pipeline for strategy improvement using trade data
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import machine learning libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logging.warning("XGBoost not available, using basic ML")

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available")

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for machine learning models"""
    model_type: str = "xgboost"  # xgboost, random_forest, logistic_regression
    target_variable: str = "is_profitable"  # or "pnl_percentage"
    feature_columns: List[str] = field(default_factory=list)
    train_test_split: float = 0.8
    cross_validation_folds: int = 5
    validation_metric: str = "accuracy"  # accuracy, precision, f1, sharpe

@dataclass
class TrainingResult:
    """Results from model training"""
    model_name: str
    model_version: str
    training_date: datetime
    train_score: float
    validation_score: float
    cross_validation_scores: List[float]
    feature_importance: Dict[str, float]
    model_parameters: Dict[str, Any]
    training_samples: int
    validation_samples: int

class LearningPipeline:
    """Machine learning pipeline for trading strategy improvement"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.training_history = []
        self.current_model = None
        self.current_version = "1.0.0"
        
        logger.info("Learning Pipeline initialized")
        logger.info(f"Model type: {self.config.model_type}")
        logger.info(f"Target variable: {self.config.target_variable}")

    def load_training_data(self, dataset_path: str) -> pd.DataFrame:
        """Load training data from trade dataset"""
        try:
            logger.info(f"📂 Loading training data from: {dataset_path}")
            
            # Load dataset
            df = pd.read_csv(dataset_path)
            
            # Convert datetime columns
            datetime_cols = ['timestamp', 'entry_time', 'exit_time']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            logger.info(f"   Loaded {len(df)} records with {len(df.columns)} columns")
            
            # Auto-detect feature columns if not specified
            if not self.config.feature_columns:
                self.config.feature_columns = self._detect_feature_columns(df)
                logger.info(f"   Auto-detected {len(self.config.feature_columns)} feature columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise

    def _detect_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Auto-detect feature columns from dataset"""
        # Common feature prefixes
        feature_prefixes = ['feature_', 'rsi', 'macd', 'volatility', 'volume', 'atr', 'ema']
        
        feature_columns = []
        for column in df.columns:
            # Check if column name matches feature patterns
            if (column.startswith(tuple(feature_prefixes)) or 
                any(prefix in column.lower() for prefix in ['rsi', 'macd', 'atr', 'ema']) or
                (column.startswith(('feature_', 'indicator_')))):
                feature_columns.append(column)
        
        # Remove target and metadata columns
        exclude_columns = [
            'trade_id', 'timestamp', 'symbol', 'side', 'entry_time', 'exit_time',
            'pnl', 'pnl_percentage', 'is_profitable', 'win_rate', 'strategy_name',
            'session_id', 'paper_trading', 'data_quality_score'
        ]
        
        feature_columns = [col for col in feature_columns if col not in exclude_columns]
        
        return feature_columns

    def prepare_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target variables"""
        try:
            logger.info("🔧 Preparing features and target variables...")
            
            # Auto-detect feature columns if not specified
            if not self.config.feature_columns:
                self.config.feature_columns = self._detect_feature_columns(df)
                logger.info(f"   Auto-detected {len(self.config.feature_columns)} feature columns")
            
            # Check if all feature columns exist
            missing_features = [col for col in self.config.feature_columns if col not in df.columns]
            if missing_features:
                logger.warning(f"Missing feature columns: {missing_features}")
                # Remove missing columns
                self.config.feature_columns = [col for col in self.config.feature_columns if col not in missing_features]
            
            if not self.config.feature_columns:
                raise ValueError("No valid feature columns found")
            
            X = df[self.config.feature_columns].copy()
            
            # Handle target variable
            if self.config.target_variable == "is_profitable":
                if 'pnl' in df.columns:
                    y = (df['pnl'] > 0).astype(int)
                elif 'is_profitable' in df.columns:
                    y = df['is_profitable'].astype(int)
                else:
                    raise ValueError("No profit/loss data available for target variable")
            elif self.config.target_variable == "pnl_percentage":
                if 'pnl_percentage' in df.columns:
                    y = df['pnl_percentage']
                else:
                    raise ValueError("pnl_percentage column not found")
            else:
                if self.config.target_variable in df.columns:
                    y = df[self.config.target_variable]
                else:
                    raise ValueError(f"Target column '{self.config.target_variable}' not found")
            
            # Handle missing values
            X = X.fillna(X.median())
            
            logger.info(f"   Features shape: {X.shape}")
            logger.info(f"   Target shape: {y.shape}")
            logger.info(f"   Target distribution: {y.value_counts().to_dict() if y.dtype == 'int' else 'continuous'}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            raise

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> TrainingResult:
        """Train machine learning model"""
        try:
            logger.info(f"🤖 Training {self.config.model_type} model...")
            
            # Split data
            split_idx = int(len(X) * self.config.train_test_split)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            logger.info(f"   Training samples: {len(X_train)}")
            logger.info(f"   Validation samples: {len(X_val)}")
            
            # Train model based on type
            if self.config.model_type == "xgboost" and XGB_AVAILABLE:
                model = self._train_xgboost(X_train, y_train)
                model_name = "XGBoost"
            elif self.config.model_type == "random_forest" and SKLEARN_AVAILABLE:
                model = self._train_random_forest(X_train, y_train)
                model_name = "Random Forest"
            else:
                model = self._train_basic_model(X_train, y_train)
                model_name = "Basic Model"
            
            # Make predictions
            if hasattr(model, 'predict'):
                y_train_pred = model.predict(X_train)
                y_val_pred = model.predict(X_val)
                
                # Calculate scores
                if y.dtype == 'int':  # Classification
                    train_score = accuracy_score(y_train, y_train_pred)
                    val_score = accuracy_score(y_val, y_val_pred)
                else:  # Regression
                    train_score = 1 - (abs(y_train - y_train_pred).mean() / abs(y_train).mean())
                    val_score = 1 - (abs(y_val - y_val_pred).mean() / abs(y_val).mean())
            else:
                train_score = 0.0
                val_score = 0.0
            
            # Cross-validation
            cv_scores = self._perform_cross_validation(model, X, y)
            
            # Feature importance
            feature_importance = self._get_feature_importance(model, X.columns)
            
            # Create training result
            result = TrainingResult(
                model_name=model_name,
                model_version=self.current_version,
                training_date=datetime.now(),
                train_score=train_score,
                validation_score=val_score,
                cross_validation_scores=cv_scores,
                feature_importance=feature_importance,
                model_parameters=self._get_model_parameters(model),
                training_samples=len(X_train),
                validation_samples=len(X_val)
            )
            
            # Store model
            self.models[self.current_version] = model
            self.current_model = model
            self.training_history.append(result)
            
            logger.info(f"✅ Model training completed")
            logger.info(f"   Training score: {train_score:.4f}")
            logger.info(f"   Validation score: {val_score:.4f}")
            logger.info(f"   CV mean score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise

    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series):
        """Train XGBoost model"""
        try:
            params = {
                'objective': 'binary:logistic' if y.dtype == 'int' else 'reg:squarederror',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params) if y.dtype == 'int' else xgb.XGBRegressor(**params)
            model.fit(X, y)
            return model
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return self._train_basic_model(X, y)

    def _train_random_forest(self, X: pd.DataFrame, y: pd.Series):
        """Train Random Forest model"""
        try:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
            
            model = RandomForestClassifier(**params) if y.dtype == 'int' else RandomForestRegressor(**params)
            model.fit(X, y)
            return model
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            return self._train_basic_model(X, y)

    def _train_basic_model(self, X: pd.DataFrame, y: pd.Series):
        """Train basic linear model as fallback"""
        try:
            from sklearn.linear_model import LogisticRegression, LinearRegression
            
            if y.dtype == 'int':
                model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                model = LinearRegression()
            
            model.fit(X, y)
            return model
            
        except Exception as e:
            logger.error(f"Basic model training failed: {e}")
            # Return dummy model
            class DummyModel:
                def predict(self, X):
                    return np.zeros(len(X))
            return DummyModel()

    def _perform_cross_validation(self, model, X: pd.DataFrame, y: pd.Series) -> List[float]:
        """Perform time series cross-validation"""
        try:
            if len(X) < 10:  # Too few samples
                return [0.0]
            
            tscv = TimeSeriesSplit(n_splits=min(self.config.cross_validation_folds, 5))
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train on fold
                model_copy = type(model)(**self._get_model_parameters(model))
                model_copy.fit(X_train_fold, y_train_fold)
                
                # Evaluate
                y_pred = model_copy.predict(X_val_fold)
                if y.dtype == 'int':
                    score = accuracy_score(y_val_fold, y_pred)
                else:
                    score = 1 - (abs(y_val_fold - y_pred).mean() / (abs(y_val_fold).mean() + 1e-8))
                
                scores.append(score)
            
            return scores
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return [0.0]

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = abs(model.coef_[0]) if len(model.coef_.shape) > 1 else abs(model.coef_)
            else:
                # Equal importance if not available
                importance = np.ones(len(feature_names)) / len(feature_names)
            
            # Normalize
            if np.sum(importance) > 0:
                importance = importance / np.sum(importance)
            
            return dict(zip(feature_names, importance))
            
        except Exception as e:
            logger.error(f"Feature importance extraction failed: {e}")
            return {name: 1.0/len(feature_names) for name in feature_names}

    def _get_model_parameters(self, model) -> Dict[str, Any]:
        """Extract model parameters"""
        try:
            if hasattr(model, 'get_params'):
                return model.get_params()
            else:
                return {'model_type': str(type(model).__name__)}
        except Exception:
            return {}

    def evaluate_model_performance(self, result: TrainingResult) -> Dict[str, Any]:
        """Comprehensive model performance evaluation"""
        try:
            evaluation = {
                'model_name': result.model_name,
                'model_version': result.model_version,
                'training_date': result.training_date.isoformat(),
                'performance_metrics': {
                    'train_score': result.train_score,
                    'validation_score': result.validation_score,
                    'cv_mean': np.mean(result.cross_validation_scores),
                    'cv_std': np.std(result.cross_validation_scores),
                    'cv_min': np.min(result.cross_validation_scores),
                    'cv_max': np.max(result.cross_validation_scores)
                },
                'stability_metrics': {
                    'cv_stability': 1.0 / (1.0 + np.std(result.cross_validation_scores)),  # Higher = more stable
                    'overfitting_risk': abs(result.train_score - result.validation_score)
                },
                'feature_importance': result.feature_importance,
                'training_info': {
                    'training_samples': result.training_samples,
                    'validation_samples': result.validation_samples,
                    'total_features': len(result.feature_importance)
                }
            }
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}

    def save_model(self, model_path: str = None) -> str:
        """Save trained model"""
        try:
            if not self.current_model:
                raise ValueError("No trained model to save")
            
            if model_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f"./models/chloe_model_{timestamp}.pkl"
            
            # Save model (simplified approach)
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.current_model,
                    'config': self.config,
                    'version': self.current_version,
                    'training_date': datetime.now()
                }, f)
            
            logger.info(f"💾 Model saved to: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"Model saving failed: {e}")
            raise

    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.current_model = model_data['model']
            self.config = model_data['config']
            self.current_version = model_data['version']
            
            logger.info(f"📂 Model loaded from: {model_path}")
            logger.info(f"   Version: {self.current_version}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def get_latest_training_result(self) -> Optional[TrainingResult]:
        """Get the most recent training result"""
        return self.training_history[-1] if self.training_history else None

    def get_model_versions(self) -> List[str]:
        """Get list of trained model versions"""
        return list(self.models.keys())

def main():
    """Example usage"""
    print("Learning Pipeline - Machine Learning for Trading Strategies")
    print("Phase 3 of Paper-Learning Architecture")

if __name__ == "__main__":
    main()