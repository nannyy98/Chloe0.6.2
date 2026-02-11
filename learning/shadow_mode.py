"""
Shadow Mode for Chloe AI - Phase 5
Parallel model comparison and performance monitoring
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import asyncio
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ShadowModel:
    """Representation of a model in shadow mode"""
    model_id: str
    model_version: str
    model_object: Any  # Trained model object
    features_used: List[str]
    creation_date: datetime
    status: str = "ACTIVE"  # ACTIVE, PAUSED, RETIRED
    
@dataclass
class ShadowTrade:
    """Individual trade record in shadow mode"""
    timestamp: datetime
    symbol: str
    model_id: str
    predicted_direction: int  # 1 for buy, -1 for sell, 0 for hold
    predicted_confidence: float
    actual_price: float
    predicted_price: float
    actual_pnl: float = 0.0
    predicted_pnl: float = 0.0
    trade_duration: int = 0  # minutes
    
@dataclass
class ModelComparisonMetrics:
    """Metrics for comparing shadow models"""
    model_id: str
    total_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    accuracy: float
    avg_return: float
    consistency_score: float
    recent_performance: float  # Last 30 trades
    
@dataclass
class ShadowComparisonResult:
    """Result of shadow mode comparison"""
    comparison_timestamp: datetime
    active_models: List[str]
    best_performing_model: str
    performance_ranking: List[Tuple[str, float]]  # (model_id, score)
    consensus_signals: Dict[str, int]  # symbol -> consensus direction
    divergence_alerts: List[Dict[str, Any]]  # Models with conflicting signals
    model_metrics: Dict[str, ModelComparisonMetrics]

class ShadowModeManager:
    """Manages parallel model comparison in shadow mode"""
    
    def __init__(self, comparison_window_hours: int = 24):
        self.comparison_window = timedelta(hours=comparison_window_hours)
        self.models = {}  # model_id -> ShadowModel
        self.shadow_trades = []  # List of all shadow trades
        self.live_market_data = {}  # symbol -> current price data
        self.comparison_history = []
        self.alert_threshold = 0.3  # Threshold for divergence alerts
        
        logger.info("Shadow Mode Manager initialized")
        logger.info(f"Comparison window: {comparison_window_hours} hours")

    def register_model(self, model_id: str, model_object: Any, 
                      features: List[str], version: str = "1.0") -> bool:
        """Register a model for shadow mode comparison"""
        try:
            if model_id in self.models:
                logger.warning(f"Model {model_id} already registered, updating...")
            
            shadow_model = ShadowModel(
                model_id=model_id,
                model_version=version,
                model_object=model_object,
                features_used=features,
                creation_date=datetime.now(),
                status="ACTIVE"
            )
            
            self.models[model_id] = shadow_model
            logger.info(f"✅ Registered model: {model_id} (v{version})")
            logger.info(f"   Features: {len(features)} used")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            return False

    def unregister_model(self, model_id: str) -> bool:
        """Unregister a model from shadow mode"""
        try:
            if model_id in self.models:
                self.models[model_id].status = "RETIRED"
                logger.info(f"📤 Unregistered model: {model_id}")
                return True
            else:
                logger.warning(f"Model {model_id} not found for unregistration")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unregister model {model_id}: {e}")
            return False

    def update_market_data(self, symbol: str, price_data: Dict[str, float]):
        """Update current market data for shadow trading"""
        try:
            self.live_market_data[symbol] = price_data
            logger.debug(f"Updated market data for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to update market data for {symbol}: {e}")

    async def process_shadow_trades(self, market_features: pd.DataFrame) -> ShadowComparisonResult:
        """
        Process all registered models and generate shadow trades
        
        Args:
            market_features: DataFrame with current market features
            
        Returns:
            ShadowComparisonResult with comparison analysis
        """
        try:
            logger.info("🔄 Processing shadow trades...")
            current_time = datetime.now()
            
            # Generate trades for each active model
            model_trades = {}
            
            for model_id, shadow_model in self.models.items():
                if shadow_model.status == "ACTIVE":
                    try:
                        trades = await self._generate_model_trades(shadow_model, market_features, current_time)
                        model_trades[model_id] = trades
                        logger.debug(f"   {model_id}: Generated {len(trades)} shadow trades")
                        
                    except Exception as e:
                        logger.error(f"Failed to generate trades for {model_id}: {e}")
                        model_trades[model_id] = []
            
            # Store all trades
            for trades in model_trades.values():
                self.shadow_trades.extend(trades)
            
            # Clean old trades
            self._cleanup_old_trades()
            
            # Analyze performance
            comparison_result = self._analyze_model_performance(model_trades, current_time)
            
            # Generate alerts
            self._check_for_divergence_alerts(comparison_result)
            
            # Store comparison result
            self.comparison_history.append(comparison_result)
            
            logger.info(f"✅ Shadow processing completed")
            logger.info(f"   Active models: {len(comparison_result.active_models)}")
            logger.info(f"   Best performer: {comparison_result.best_performing_model}")
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"Shadow trade processing failed: {e}")
            raise

    async def _generate_model_trades(self, shadow_model: ShadowModel, 
                                   market_features: pd.DataFrame, 
                                   timestamp: datetime) -> List[ShadowTrade]:
        """Generate shadow trades for a specific model"""
        try:
            trades = []
            
            # For each symbol in market features
            for symbol in market_features['symbol'].unique():
                symbol_data = market_features[market_features['symbol'] == symbol]
                
                if symbol_data.empty:
                    continue
                
                # Get latest features for this symbol
                latest_features = symbol_data.iloc[-1]
                
                # Extract features used by this model
                model_features = latest_features[shadow_model.features_used]
                
                # Handle missing features
                model_features = model_features.fillna(0)
                
                # Make prediction
                try:
                    if hasattr(shadow_model.model_object, 'predict_proba'):
                        # Classification model
                        prediction_proba = shadow_model.model_object.predict_proba([model_features])[0]
                        predicted_direction = 1 if prediction_proba[1] > 0.5 else -1
                        confidence = max(prediction_proba)
                    elif hasattr(shadow_model.model_object, 'predict'):
                        # Regression or other model
                        prediction = shadow_model.model_object.predict([model_features])[0]
                        predicted_direction = 1 if prediction > 0 else -1
                        confidence = abs(prediction) / (abs(prediction) + 1)  # Normalize confidence
                    else:
                        # Fallback prediction
                        predicted_direction = 0
                        confidence = 0.5
                
                except Exception as e:
                    logger.warning(f"Prediction failed for {shadow_model.model_id} on {symbol}: {e}")
                    predicted_direction = 0
                    confidence = 0.0
                
                # Create shadow trade
                if predicted_direction != 0 and confidence > 0.1:  # Only significant signals
                    current_price = latest_features.get('close', 0)
                    
                    shadow_trade = ShadowTrade(
                        timestamp=timestamp,
                        symbol=symbol,
                        model_id=shadow_model.model_id,
                        predicted_direction=predicted_direction,
                        predicted_confidence=confidence,
                        actual_price=current_price,
                        predicted_price=current_price * (1 + predicted_direction * 0.01 * confidence),
                        trade_duration=60  # Default 1 hour
                    )
                    
                    trades.append(shadow_trade)
            
            return trades
            
        except Exception as e:
            logger.error(f"Failed to generate trades for model {shadow_model.model_id}: {e}")
            return []

    def _analyze_model_performance(self, model_trades: Dict[str, List[ShadowTrade]], 
                                 current_time: datetime) -> ShadowComparisonResult:
        """Analyze performance of all models in shadow mode"""
        try:
            # Calculate metrics for each model
            model_metrics = {}
            active_models = []
            
            for model_id, trades in model_trades.items():
                if trades:  # Only analyze models with trades
                    metrics = self._calculate_model_metrics(trades, current_time)
                    model_metrics[model_id] = metrics
                    active_models.append(model_id)
            
            # Rank models by performance
            performance_ranking = self._rank_models(model_metrics)
            
            # Determine best performing model
            best_model = performance_ranking[0][0] if performance_ranking else "NONE"
            
            # Generate consensus signals
            consensus_signals = self._generate_consensus_signals(model_trades)
            
            # Create comparison result
            result = ShadowComparisonResult(
                comparison_timestamp=current_time,
                active_models=active_models,
                best_performing_model=best_model,
                performance_ranking=performance_ranking,
                consensus_signals=consensus_signals,
                divergence_alerts=[],  # Will be filled by alert system
                model_metrics=model_metrics
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return ShadowComparisonResult(
                comparison_timestamp=current_time,
                active_models=[],
                best_performing_model="NONE",
                performance_ranking=[],
                consensus_signals={},
                divergence_alerts=[],
                model_metrics={}
            )

    def _calculate_model_metrics(self, trades: List[ShadowTrade], 
                               current_time: datetime) -> ModelComparisonMetrics:
        """Calculate performance metrics for a model"""
        try:
            if not trades:
                return ModelComparisonMetrics(
                    model_id="unknown",
                    total_trades=0,
                    win_rate=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0,
                    accuracy=0.0,
                    avg_return=0.0,
                    consistency_score=0.0,
                    recent_performance=0.0
                )
            
            model_id = trades[0].model_id
            
            # Calculate basic metrics
            total_trades = len(trades)
            winning_trades = sum(1 for trade in trades if trade.actual_pnl > 0)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate returns
            returns = [trade.actual_pnl for trade in trades]
            avg_return = np.mean(returns) if returns else 0
            
            # Sharpe ratio (simplified)
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = avg_return / np.std(returns)
            else:
                sharpe_ratio = 0
            
            # Maximum drawdown
            equity_curve = np.cumsum(returns)
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / (running_max + 1e-8)
            max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
            
            # Accuracy (direction prediction accuracy)
            correct_directions = sum(1 for trade in trades 
                                   if trade.predicted_direction * trade.actual_pnl > 0)
            accuracy = correct_directions / total_trades if total_trades > 0 else 0
            
            # Consistency score (based on recent performance stability)
            recent_returns = returns[-10:] if len(returns) >= 10 else returns
            if len(recent_returns) > 1:
                consistency_score = 1.0 / (1.0 + np.std(recent_returns))
            else:
                consistency_score = 0.5
            
            # Recent performance (last 30 trades or all if less)
            recent_trades_count = min(30, len(trades))
            recent_returns = returns[-recent_trades_count:] if recent_trades_count > 0 else [0]
            recent_performance = np.mean(recent_returns) if recent_returns else 0
            
            return ModelComparisonMetrics(
                model_id=model_id,
                total_trades=total_trades,
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                accuracy=accuracy,
                avg_return=avg_return,
                consistency_score=consistency_score,
                recent_performance=recent_performance
            )
            
        except Exception as e:
            logger.error(f"Metrics calculation failed: {e}")
            return ModelComparisonMetrics(
                model_id="error",
                total_trades=0,
                win_rate=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                accuracy=0.0,
                avg_return=0.0,
                consistency_score=0.0,
                recent_performance=0.0
            )

    def _rank_models(self, model_metrics: Dict[str, ModelComparisonMetrics]) -> List[Tuple[str, float]]:
        """Rank models by composite performance score"""
        try:
            rankings = []
            
            for model_id, metrics in model_metrics.items():
                # Composite score calculation
                # Weighted combination of key metrics
                score = (
                    0.3 * min(metrics.win_rate / 0.6, 1.0) +  # Win rate weight (normalized to 60% target)
                    0.25 * min(metrics.sharpe_ratio / 2.0, 1.0) +  # Sharpe ratio weight
                    0.2 * (1.0 - min(metrics.max_drawdown / 0.2, 1.0)) +  # Drawdown penalty
                    0.15 * min(metrics.accuracy / 0.6, 1.0) +  # Accuracy weight
                    0.1 * min(metrics.consistency_score, 1.0)  # Consistency weight
                )
                
                rankings.append((model_id, score))
            
            # Sort by score descending
            rankings.sort(key=lambda x: x[1], reverse=True)
            return rankings
            
        except Exception as e:
            logger.error(f"Model ranking failed: {e}")
            return []

    def _generate_consensus_signals(self, model_trades: Dict[str, List[ShadowTrade]]) -> Dict[str, int]:
        """Generate consensus signals across all models"""
        try:
            symbol_signals = defaultdict(list)
            
            # Collect all signals by symbol
            for model_trades_list in model_trades.values():
                for trade in model_trades_list:
                    symbol_signals[trade.symbol].append({
                        'direction': trade.predicted_direction,
                        'confidence': trade.predicted_confidence,
                        'model_id': trade.model_id
                    })
            
            # Calculate consensus for each symbol
            consensus_signals = {}
            
            for symbol, signals in symbol_signals.items():
                if not signals:
                    continue
                
                # Weighted average of directions
                total_weight = sum(signal['confidence'] for signal in signals)
                if total_weight > 0:
                    weighted_direction = sum(signal['direction'] * signal['confidence'] 
                                           for signal in signals) / total_weight
                    
                    # Convert to discrete signal
                    if weighted_direction > 0.3:
                        consensus_signals[symbol] = 1  # Strong buy
                    elif weighted_direction < -0.3:
                        consensus_signals[symbol] = -1  # Strong sell
                    elif abs(weighted_direction) > 0.1:
                        consensus_signals[symbol] = int(weighted_direction)  # Weak signal
                    else:
                        consensus_signals[symbol] = 0  # No consensus
            
            return consensus_signals
            
        except Exception as e:
            logger.error(f"Consensus signal generation failed: {e}")
            return {}

    def _check_for_divergence_alerts(self, comparison_result: ShadowComparisonResult):
        """Check for significant divergence between models"""
        try:
            alerts = []
            
            # Check for models with conflicting strong signals
            symbol_model_signals = defaultdict(lambda: defaultdict(list))
            
            # Reorganize data by symbol and model
            for model_id, metrics in comparison_result.model_metrics.items():
                # This is simplified - in practice, you'd track actual trade signals
                pass
            
            # Generate alerts for significant performance divergence
            if len(comparison_result.performance_ranking) >= 2:
                top_model_score = comparison_result.performance_ranking[0][1]
                bottom_model_score = comparison_result.performance_ranking[-1][1]
                
                if top_model_score - bottom_model_score > self.alert_threshold:
                    alerts.append({
                        'type': 'PERFORMANCE_DIVERGENCE',
                        'severity': 'HIGH',
                        'message': f'Significant performance gap between top and bottom models: {top_model_score:.3f} vs {bottom_model_score:.3f}',
                        'timestamp': comparison_result.comparison_timestamp
                    })
            
            comparison_result.divergence_alerts = alerts
            
        except Exception as e:
            logger.error(f"Divergence checking failed: {e}")

    def _cleanup_old_trades(self):
        """Remove old trades outside comparison window"""
        try:
            cutoff_time = datetime.now() - self.comparison_window
            
            self.shadow_trades = [
                trade for trade in self.shadow_trades 
                if trade.timestamp >= cutoff_time
            ]
            
            logger.debug(f"Cleaned up old trades. Remaining: {len(self.shadow_trades)}")
            
        except Exception as e:
            logger.error(f"Trade cleanup failed: {e}")

    def get_model_performance_report(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed performance report for a specific model"""
        try:
            if model_id not in self.models:
                return None
            
            # Get recent trades for this model
            recent_trades = [
                trade for trade in self.shadow_trades 
                if trade.model_id == model_id 
                and trade.timestamp >= datetime.now() - timedelta(hours=24)
            ]
            
            if not recent_trades:
                return None
            
            metrics = self._calculate_model_metrics(recent_trades, datetime.now())
            
            report = {
                'model_id': model_id,
                'model_version': self.models[model_id].model_version,
                'report_timestamp': datetime.now().isoformat(),
                'metrics': {
                    'total_trades': metrics.total_trades,
                    'win_rate': metrics.win_rate,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'accuracy': metrics.accuracy,
                    'avg_return': metrics.avg_return,
                    'consistency_score': metrics.consistency_score,
                    'recent_performance': metrics.recent_performance
                },
                'status': self.models[model_id].status,
                'features_used': self.models[model_id].features_used
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Performance report generation failed: {e}")
            return None

    def get_shadow_summary(self) -> Dict[str, Any]:
        """Get overall shadow mode summary"""
        try:
            active_models = [mid for mid, model in self.models.items() if model.status == "ACTIVE"]
            retired_models = [mid for mid, model in self.models.items() if model.status == "RETIRED"]
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_models': len(self.models),
                'active_models': len(active_models),
                'retired_models': len(retired_models),
                'total_shadow_trades': len(self.shadow_trades),
                'symbols_tracked': len(set(trade.symbol for trade in self.shadow_trades)),
                'active_model_list': active_models,
                'recent_comparisons': len(self.comparison_history[-10:]) if self.comparison_history else 0
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Shadow summary generation failed: {e}")
            return {}

def main():
    """Example usage"""
    print("Shadow Mode Manager - Parallel Model Comparison")
    print("Phase 5 of Paper-Learning Architecture")

if __name__ == "__main__":
    main()