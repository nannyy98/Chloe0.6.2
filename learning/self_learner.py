"""
Controlled Self-Learning Loop for Chloe AI - Phase 6
Daily batch learning with safety controls and validation
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import asyncio
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our existing components
from learning.trainer import LearningPipeline, ModelConfig
from learning.validator import ModelValidationGate, ValidationCriteria

logger = logging.getLogger(__name__)

@dataclass
class LearningSchedule:
    """Self-learning schedule configuration"""
    daily_learning_time: str = "02:00"  # UTC time for daily learning
    retrain_frequency_days: int = 1      # Retrain every N days
    min_new_data_points: int = 50        # Minimum new trades for retraining
    max_training_time_minutes: int = 30  # Maximum training time allowed
    performance_improvement_threshold: float = 0.05  # 5% minimum improvement

@dataclass
class LearningSession:
    """Record of a learning session"""
    session_id: str
    start_time: datetime
    end_time: datetime
    model_version: str
    training_samples: int
    validation_result: str  # APPROVED, REJECTED, CONDITIONAL
    performance_before: float
    performance_after: float
    improvement: float
    data_quality_score: float
    learning_outcome: str  # SUCCESS, FAILED_VALIDATION, TIMEOUT, INSUFFICIENT_DATA

class SelfLearningController:
    """Controls the automated self-learning process"""
    
    def __init__(self, schedule: LearningSchedule = None, 
                 data_directory: str = "./data/trade_journal"):
        self.schedule = schedule or LearningSchedule()
        self.data_directory = Path(data_directory)
        self.learning_history = []
        self.current_model_version = "1.0.0"
        self.last_training_date = None
        self.performance_baseline = 0.0
        
        # Initialize components
        self.learning_pipeline = LearningPipeline()
        self.validation_gate = ModelValidationGate()
        
        # Create data directory if it doesn't exist
        self.data_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info("Controlled Self-Learning Controller initialized")
        logger.info(f"Learning schedule: Daily at {self.schedule.daily_learning_time} UTC")
        logger.info(f"Retrain frequency: Every {self.schedule.retrain_frequency_days} days")
        logger.info(f"Min new data points: {self.schedule.min_new_data_points}")

    async def execute_learning_cycle(self, force_learning: bool = False) -> LearningSession:
        """
        Execute one complete learning cycle
        
        Args:
            force_learning: Force learning regardless of schedule
            
        Returns:
            LearningSession with results
        """
        try:
            session_id = f"learn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            start_time = datetime.now()
            
            logger.info(f"🎓 STARTING LEARNING CYCLE: {session_id}")
            logger.info(f"   Start time: {start_time}")
            
            # Check if learning should proceed
            if not force_learning and not self._should_learn_now():
                logger.info("   ⏰ Not scheduled for learning - skipping")
                return self._create_skipped_session(session_id, start_time)
            
            # Load and validate new data
            logger.info("   📊 Loading new training data...")
            new_data = self._load_recent_trade_data()
            
            if len(new_data) < self.schedule.min_new_data_points:
                logger.info(f"   📉 Insufficient new data ({len(new_data)} < {self.schedule.min_new_data_points})")
                return self._create_insufficient_data_session(session_id, start_time, len(new_data))
            
            logger.info(f"   ✅ Loaded {len(new_data)} new trade records")
            
            # Assess data quality
            data_quality_score = self._assess_data_quality(new_data)
            logger.info(f"   📈 Data quality score: {data_quality_score:.3f}")
            
            if data_quality_score < 0.7:
                logger.warning("   ⚠️  Low data quality - proceeding with caution")
            
            # Prepare data for training
            logger.info("   🔧 Preparing data for training...")
            X, y = self.learning_pipeline.prepare_features_and_target(new_data)
            logger.info(f"   ✅ Prepared {X.shape[0]} samples with {X.shape[1]} features")
            
            # Store baseline performance
            baseline_performance = self.performance_baseline
            
            # Train new model
            logger.info("   🤖 Training new model...")
            training_start = datetime.now()
            
            try:
                training_result = self.learning_pipeline.train_model(X, y)
                training_time = (datetime.now() - training_start).total_seconds() / 60
                
                logger.info(f"   ✅ Training completed in {training_time:.1f} minutes")
                logger.info(f"   🎯 Validation score: {training_result.validation_score:.4f}")
                
                # Validate new model
                logger.info("   🔍 Validating new model...")
                validation_result = self.validation_gate.validate_model(
                    new_data, 
                    {'model_id': f'model_v{self._increment_version()}'}
                )
                
                logger.info(f"   ✅ Validation status: {validation_result.overall_status}")
                
                # Calculate performance improvement
                new_performance = training_result.validation_score
                improvement = new_performance - baseline_performance
                
                logger.info(f"   📊 Performance: {baseline_performance:.4f} → {new_performance:.4f} "
                           f"({improvement:+.4f})")
                
                # Determine learning outcome
                if validation_result.overall_status == "APPROVED":
                    if improvement >= self.schedule.performance_improvement_threshold:
                        outcome = "SUCCESS"
                        logger.info("   🎉 Learning cycle successful - approved model with improvement")
                    else:
                        outcome = "FAILED_VALIDATION"
                        logger.info("   ⚠️  Model approved but insufficient improvement")
                else:
                    outcome = "FAILED_VALIDATION"
                    logger.info("   ❌ Model rejected by validation gate")
                
                # Update baseline if successful
                if outcome == "SUCCESS":
                    self.performance_baseline = new_performance
                    self.current_model_version = self._increment_version()
                    self.last_training_date = datetime.now()
                    logger.info(f"   🆕 Updated model version to {self.current_model_version}")
                
            except Exception as e:
                logger.error(f"   ❌ Training failed: {e}")
                outcome = "FAILED_VALIDATION"
                training_time = (datetime.now() - training_start).total_seconds() / 60
                improvement = 0.0
                new_performance = 0.0
                validation_result = None
            
            # Create learning session record
            end_time = datetime.now()
            session = LearningSession(
                session_id=session_id,
                start_time=start_time,
                end_time=end_time,
                model_version=self.current_model_version,
                training_samples=len(new_data),
                validation_result=validation_result.overall_status if validation_result else "FAILED",
                performance_before=baseline_performance,
                performance_after=new_performance,
                improvement=improvement,
                data_quality_score=data_quality_score,
                learning_outcome=outcome
            )
            
            # Store session
            self.learning_history.append(session)
            
            # Log completion
            duration = (end_time - start_time).total_seconds() / 60
            logger.info(f"✅ LEARNING CYCLE COMPLETED: {session_id}")
            logger.info(f"   Duration: {duration:.1f} minutes")
            logger.info(f"   Outcome: {outcome}")
            logger.info(f"   Samples: {len(new_data)}")
            
            return session
            
        except Exception as e:
            logger.error(f"Learning cycle failed: {e}")
            raise

    def _should_learn_now(self) -> bool:
        """Determine if learning should execute now"""
        try:
            now = datetime.now()
            
            # Check if enough time has passed since last training
            if self.last_training_date:
                days_since_training = (now - self.last_training_date).days
                if days_since_training < self.schedule.retrain_frequency_days:
                    logger.debug(f"Too soon since last training ({days_since_training} days)")
                    return False
            
            # Check if it's the scheduled learning time
            current_time_str = now.strftime("%H:%M")
            scheduled_time = self.schedule.daily_learning_time
            
            # Allow 30-minute window around scheduled time
            scheduled_hour, scheduled_minute = map(int, scheduled_time.split(':'))
            current_hour, current_minute = map(int, current_time_str.split(':'))
            
            time_diff = abs((current_hour * 60 + current_minute) - (scheduled_hour * 60 + scheduled_minute))
            
            if time_diff > 30:  # 30 minutes window
                logger.debug(f"Not scheduled learning time ({current_time_str} vs {scheduled_time})")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Learning schedule check failed: {e}")
            return False

    def _load_recent_trade_data(self) -> pd.DataFrame:
        """Load recent trade data for training"""
        try:
            # Look for trade journal files
            journal_files = list(self.data_directory.glob("*.csv"))
            
            if not journal_files:
                logger.warning("No trade journal files found")
                return pd.DataFrame()
            
            # Load most recent data
            all_data = []
            
            for file_path in journal_files:
                try:
                    df = pd.read_csv(file_path)
                    if not df.empty:
                        # Convert timestamp if present
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            # Filter for recent data (last 30 days)
                            cutoff_date = datetime.now() - timedelta(days=30)
                            df = df[df['timestamp'] >= cutoff_date]
                        
                        all_data.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                # Remove duplicates
                combined_data = combined_data.drop_duplicates()
                return combined_data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load trade data: {e}")
            return pd.DataFrame()

    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess the quality of training data"""
        try:
            if len(data) == 0:
                return 0.0
            
            quality_score = 1.0
            
            # Check for missing values
            missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            quality_score -= missing_pct * 0.3  # Up to 30% penalty for missing data
            
            # Check for extreme outliers
            if 'pnl' in data.columns:
                pnl_data = data['pnl']
                z_scores = np.abs((pnl_data - pnl_data.mean()) / pnl_data.std())
                outlier_pct = (z_scores > 3).mean()
                quality_score -= outlier_pct * 0.2  # Up to 20% penalty for outliers
            
            # Check data distribution
            if 'is_profitable' in data.columns:
                win_rate = data['is_profitable'].mean()
                # Penalize extreme win rates (too good or too bad)
                if win_rate < 0.3 or win_rate > 0.8:
                    quality_score -= 0.1
            
            # Ensure minimum quality
            quality_score = max(0.0, min(1.0, quality_score))
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            return 0.0

    def _increment_version(self) -> str:
        """Increment model version number"""
        try:
            version_parts = self.current_model_version.split('.')
            major, minor, patch = map(int, version_parts)
            patch += 1
            return f"{major}.{minor}.{patch}"
        except Exception:
            # Fallback version increment
            return "1.0.1"

    def _create_skipped_session(self, session_id: str, start_time: datetime) -> LearningSession:
        """Create session record for skipped learning"""
        return LearningSession(
            session_id=session_id,
            start_time=start_time,
            end_time=start_time,
            model_version=self.current_model_version,
            training_samples=0,
            validation_result="SKIPPED",
            performance_before=self.performance_baseline,
            performance_after=self.performance_baseline,
            improvement=0.0,
            data_quality_score=0.0,
            learning_outcome="SKIPPED"
        )

    def _create_insufficient_data_session(self, session_id: str, start_time: datetime, 
                                        data_count: int) -> LearningSession:
        """Create session record for insufficient data"""
        return LearningSession(
            session_id=session_id,
            start_time=start_time,
            end_time=start_time,
            model_version=self.current_model_version,
            training_samples=data_count,
            validation_result="INSUFFICIENT_DATA",
            performance_before=self.performance_baseline,
            performance_after=self.performance_baseline,
            improvement=0.0,
            data_quality_score=0.0,
            learning_outcome="INSUFFICIENT_DATA"
        )

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of learning activities"""
        try:
            if not self.learning_history:
                return {
                    'total_sessions': 0,
                    'successful_sessions': 0,
                    'current_model_version': self.current_model_version,
                    'last_training_date': self.last_training_date.isoformat() if self.last_training_date else None,
                    'performance_baseline': self.performance_baseline
                }
            
            successful_sessions = sum(1 for session in self.learning_history 
                                    if session.learning_outcome == "SUCCESS")
            
            recent_sessions = self.learning_history[-10:]  # Last 10 sessions
            
            summary = {
                'total_sessions': len(self.learning_history),
                'successful_sessions': successful_sessions,
                'success_rate': successful_sessions / len(self.learning_history),
                'current_model_version': self.current_model_version,
                'last_training_date': self.last_training_date.isoformat() if self.last_training_date else None,
                'performance_baseline': self.performance_baseline,
                'recent_sessions': [
                    {
                        'session_id': session.session_id,
                        'date': session.start_time.isoformat(),
                        'outcome': session.learning_outcome,
                        'samples': session.training_samples,
                        'improvement': session.improvement,
                        'data_quality': session.data_quality_score
                    }
                    for session in recent_sessions
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Learning summary generation failed: {e}")
            return {}

    def export_learning_log(self, filepath: str = None) -> str:
        """Export learning history to file"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"./logs/learning_log_{timestamp}.json"
            
            # Create logs directory if needed
            log_dir = Path(filepath).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for export
            summary = self.get_learning_summary()
            summary['export_timestamp'] = datetime.now().isoformat()
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"📝 Learning log exported to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Learning log export failed: {e}")
            raise

    async def start_autonomous_learning(self):
        """Start continuous autonomous learning loop"""
        logger.info("🚀 STARTING AUTONOMOUS LEARNING LOOP")
        logger.info("   Press Ctrl+C to stop")
        
        try:
            while True:
                try:
                    # Execute learning cycle
                    session = await self.execute_learning_cycle()
                    
                    # Wait until next scheduled time
                    next_run = self._calculate_next_run_time()
                    wait_time = (next_run - datetime.now()).total_seconds()
                    
                    if wait_time > 0:
                        logger.info(f"⏳ Waiting {wait_time/3600:.1f} hours until next learning cycle")
                        await asyncio.sleep(min(wait_time, 3600))  # Max 1 hour sleep to stay responsive
                    else:
                        await asyncio.sleep(60)  # Check every minute
                        
                except KeyboardInterrupt:
                    logger.info("🛑 Autonomous learning stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Autonomous learning error: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retry
                    
        except Exception as e:
            logger.error(f"Autonomous learning loop failed: {e}")

    def _calculate_next_run_time(self) -> datetime:
        """Calculate next scheduled run time"""
        try:
            now = datetime.now()
            scheduled_time = self.schedule.daily_learning_time
            
            hour, minute = map(int, scheduled_time.split(':'))
            
            # Calculate today's scheduled time
            today_scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # If today's time has passed, schedule for tomorrow
            if today_scheduled <= now:
                next_run = today_scheduled + timedelta(days=1)
            else:
                next_run = today_scheduled
                
            return next_run
            
        except Exception as e:
            logger.error(f"Next run time calculation failed: {e}")
            return datetime.now() + timedelta(hours=24)

def main():
    """Example usage"""
    print("Controlled Self-Learning Controller - Automated Model Improvement")
    print("Phase 6 of Paper-Learning Architecture")

if __name__ == "__main__":
    main()