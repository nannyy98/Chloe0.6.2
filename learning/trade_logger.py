"""
Trade Journal for Chloe AI - Phase 2
Comprehensive trade logging system for machine learning dataset creation
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
import json
import hashlib
import os

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Comprehensive trade record for learning dataset"""
    # Basic trade information
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    
    # Model signals
    signal_confidence: float
    predicted_return: float
    model_version: str
    features_used: Dict[str, float]
    
    # Market conditions
    market_regime: str
    volatility: float
    volume: float
    spread: float
    
    # Performance metrics
    pnl: float
    pnl_percentage: float
    holding_period: float  # in hours
    slippage: float
    commission: float
    
    # Risk metrics
    position_size: float
    risk_per_trade: float
    stop_loss_level: Optional[float]
    take_profit_level: Optional[float]
    
    # Metadata
    strategy_name: str
    paper_trading: bool = True
    data_quality_score: float = 1.0

class TradeJournal:
    """Trade journal system for collecting learning data"""
    
    def __init__(self, storage_path: str = "./data/trade_logs"):
        """
        Initialize trade journal
        
        Args:
            storage_path: Directory for storing csv files
        """
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        self.current_session_records: List[TradeRecord] = []
        self.session_id = self._generate_session_id()
        
        logger.info(f"Trade Journal initialized with session ID: {self.session_id}")
        logger.info(f"Storage path: {self.storage_path}")

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}_{hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]}"

    def log_trade(self, trade_record: TradeRecord) -> bool:
        """
        Log a completed trade
        
        Args:
            trade_record: Complete trade record
            
        Returns:
            bool: Success status
        """
        try:
            # Validate trade record
            if not self._validate_trade_record(trade_record):
                logger.warning(f"Invalid trade record: {trade_record.trade_id}")
                return False
            
            # Add to current session
            self.current_session_records.append(trade_record)
            
            logger.info(f"📝 Trade logged: {trade_record.trade_id} "
                       f"{trade_record.side} {trade_record.quantity} {trade_record.symbol} "
                       f"PnL: ${trade_record.pnl:+.2f} ({trade_record.pnl_percentage:+.2f}%)")
            
            return True
            
        except Exception as e:
            logger.error(f"Trade logging failed: {e}")
            return False

    def _validate_trade_record(self, record: TradeRecord) -> bool:
        """Validate trade record completeness"""
        try:
            # Check required fields
            required_fields = [
                record.trade_id, record.symbol, record.side,
                record.entry_price, record.exit_price,
                record.entry_time, record.exit_time
            ]
            
            if any(field is None for field in required_fields):
                return False
            
            # Check logical consistency
            if record.entry_time >= record.exit_time:
                logger.warning("Entry time must be before exit time")
                return False
            
            if record.quantity <= 0:
                logger.warning("Quantity must be positive")
                return False
            
            # Check price validity
            if record.entry_price <= 0 or record.exit_price <= 0:
                logger.warning("Prices must be positive")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def create_dataset(self, 
                      filename: Optional[str] = None) -> str:
        """
        Create CSV dataset from current session records
        
        Args:
            filename: Custom filename (optional)
            
        Returns:
            str: Path to created dataset
        """
        try:
            if not self.current_session_records:
                logger.warning("No trades to create dataset from")
                return ""
            
            # Convert to DataFrame
            df = self._records_to_dataframe(self.current_session_records)
            
            # Generate filename
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trade_dataset_{timestamp}.csv"
            
            filepath = os.path.join(self.storage_path, filename)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            
            logger.info(f"📊 Dataset created: {filepath}")
            logger.info(f"   Records: {len(df)}")
            logger.info(f"   Columns: {len(df.columns)}")
            logger.info(f"   Size: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")
            
            # Print dataset summary
            self._print_dataset_summary(df)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Dataset creation failed: {e}")
            raise

    def _records_to_dataframe(self, records: List[TradeRecord]) -> pd.DataFrame:
        """Convert trade records to DataFrame"""
        try:
            # Extract basic data
            data = []
            for record in records:
                row = {
                    # Basic info
                    'trade_id': record.trade_id,
                    'timestamp': record.timestamp,
                    'symbol': record.symbol,
                    'side': record.side,
                    'quantity': record.quantity,
                    'entry_price': record.entry_price,
                    'exit_price': record.exit_price,
                    'entry_time': record.entry_time,
                    'exit_time': record.exit_time,
                    
                    # Model signals
                    'signal_confidence': record.signal_confidence,
                    'predicted_return': record.predicted_return,
                    'model_version': record.model_version,
                    
                    # Market conditions
                    'market_regime': record.market_regime,
                    'volatility': record.volatility,
                    'volume': record.volume,
                    'spread': record.spread,
                    
                    # Performance
                    'pnl': record.pnl,
                    'pnl_percentage': record.pnl_percentage,
                    'holding_period_hours': record.holding_period,
                    'slippage': record.slippage,
                    'commission': record.commission,
                    
                    # Risk metrics
                    'position_size': record.position_size,
                    'risk_per_trade': record.risk_per_trade,
                    'stop_loss_level': record.stop_loss_level,
                    'take_profit_level': record.take_profit_level,
                    
                    # Metadata
                    'strategy_name': record.strategy_name,
                    'paper_trading': record.paper_trading,
                    'data_quality_score': record.data_quality_score,
                    'session_id': self.session_id
                }
                
                # Add features as separate columns
                for feature_name, feature_value in record.features_used.items():
                    row[f"feature_{feature_name}"] = feature_value
                
                data.append(row)
            
            df = pd.DataFrame(data)
            
            # Add derived columns
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_profitable'] = df['pnl'] > 0
            df['win_rate'] = df['is_profitable'].rolling(window=min(20, len(df)), min_periods=1).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"DataFrame conversion failed: {e}")
            raise

    def _print_dataset_summary(self, df: pd.DataFrame):
        """Print dataset summary statistics"""
        try:
            logger.info("📊 DATASET SUMMARY:")
            
            # Basic statistics
            logger.info(f"   Total Trades: {len(df)}")
            logger.info(f"   Symbols: {df['symbol'].nunique()}")
            logger.info(f"   Strategies: {df['strategy_name'].nunique()}")
            logger.info(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
            
            # Performance metrics
            total_pnl = df['pnl'].sum()
            win_rate = (df['pnl'] > 0).mean() * 100
            avg_pnl = df['pnl'].mean()
            max_win = df['pnl'].max()
            max_loss = df['pnl'].min()
            
            logger.info(f"   Total PnL: ${total_pnl:+.2f}")
            logger.info(f"   Win Rate: {win_rate:.1f}%")
            logger.info(f"   Average PnL: ${avg_pnl:+.2f}")
            logger.info(f"   Best Trade: ${max_win:+.2f}")
            logger.info(f"   Worst Trade: ${max_loss:+.2f}")
            
            # Risk metrics
            avg_volatility = df['volatility'].mean()
            avg_holding = df['holding_period_hours'].mean()
            logger.info(f"   Avg Volatility: {avg_volatility:.2f}")
            logger.info(f"   Avg Holding Period: {avg_holding:.1f} hours")
            
        except Exception as e:
            logger.error(f"Summary printing failed: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        try:
            if not self.current_session_records:
                return {}
            
            df = self._records_to_dataframe(self.current_session_records)
            
            metrics = {
                # Basic metrics
                'total_trades': len(df),
                'winning_trades': (df['pnl'] > 0).sum(),
                'losing_trades': (df['pnl'] < 0).sum(),
                'win_rate': (df['pnl'] > 0).mean(),
                
                # PnL metrics
                'total_pnl': df['pnl'].sum(),
                'average_pnl': df['pnl'].mean(),
                'pnl_std': df['pnl'].std(),
                'best_trade': df['pnl'].max(),
                'worst_trade': df['pnl'].min(),
                
                # Risk metrics
                'profit_factor': df[df['pnl'] > 0]['pnl'].sum() / abs(df[df['pnl'] < 0]['pnl'].sum()) if (df['pnl'] < 0).any() else float('inf'),
                'average_win': df[df['pnl'] > 0]['pnl'].mean() if (df['pnl'] > 0).any() else 0,
                'average_loss': df[df['pnl'] < 0]['pnl'].mean() if (df['pnl'] < 0).any() else 0,
                
                # Time metrics
                'average_holding_hours': df['holding_period_hours'].mean(),
                'total_holding_days': df['holding_period_hours'].sum() / 24,
                
                # Signal quality
                'average_confidence': df['signal_confidence'].mean(),
                'confidence_std': df['signal_confidence'].std(),
                
                # Market conditions
                'average_volatility': df['volatility'].mean(),
                'volatility_std': df['volatility'].std()
            }
            
            # Handle potential NaN values
            for key, value in metrics.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    metrics[key] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {}

    def export_metadata(self, dataset_path: str) -> str:
        """Export dataset metadata"""
        try:
            metadata = {
                'session_id': self.session_id,
                'created_at': datetime.now().isoformat(),
                'total_records': len(self.current_session_records),
                'symbols': list(set(record.symbol for record in self.current_session_records)),
                'strategies': list(set(record.strategy_name for record in self.current_session_records)),
                'performance_metrics': self.get_performance_metrics(),
                'columns': [
                    'trade_id', 'timestamp', 'symbol', 'side', 'quantity',
                    'entry_price', 'exit_price', 'entry_time', 'exit_time',
                    'signal_confidence', 'predicted_return', 'model_version',
                    'market_regime', 'volatility', 'volume', 'spread',
                    'pnl', 'pnl_percentage', 'holding_period_hours', 'slippage', 'commission',
                    'position_size', 'risk_per_trade', 'stop_loss_level', 'take_profit_level',
                    'strategy_name', 'paper_trading', 'data_quality_score'
                ]
            }
            
            # Create metadata path
            base_path = os.path.splitext(dataset_path)[0]
            metadata_path = f"{base_path}.metadata.json"
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"📄 Metadata exported: {metadata_path}")
            return str(metadata_path)
            
        except Exception as e:
            logger.error(f"Metadata export failed: {e}")
            raise

    def reset_session(self):
        """Reset current session"""
        self.current_session_records = []
        self.session_id = self._generate_session_id()
        logger.info(f"🔄 Session reset. New session ID: {self.session_id}")

    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Load existing dataset"""
        try:
            df = pd.read_csv(filepath)
            # Convert datetime columns
            datetime_cols = ['timestamp', 'entry_time', 'exit_time']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            logger.info(f"📂 Dataset loaded: {filepath}")
            logger.info(f"   Records: {len(df)}")
            logger.info(f"   Columns: {len(df.columns)}")
            return df
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
            raise

def main():
    """Example usage"""
    print("Trade Journal - Learning Dataset Creation System")
    print("Phase 2 of Paper-Learning Architecture")

if __name__ == "__main__":
    main()