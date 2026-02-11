"""
Paper Trading Environment for Chloe 0.6
Professional simulated trading environment with real market data
"""

import logging
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import asyncio
import threading
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PaperTradeStatus(Enum):
    """Status of paper trades"""
    PENDING = "PENDING"          # Order placed but not filled
    FILLED = "FILLED"            # Order completely filled
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Partially filled
    CANCELLED = "CANCELLED"      # Order cancelled
    REJECTED = "REJECTED"        # Order rejected

class OrderType(Enum):
    """Order types for paper trading"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

@dataclass
class PaperOrder:
    """Paper trading order"""
    order_id: str
    symbol: str
    order_type: OrderType
    side: str  # BUY or SELL
    quantity: float
    price: Optional[float]  # None for market orders
    status: PaperTradeStatus = PaperTradeStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    fill_timestamp: Optional[datetime] = None

@dataclass
class PaperPosition:
    """Paper trading position"""
    symbol: str
    quantity: float
    average_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

@dataclass
class PaperAccount:
    """Paper trading account"""
    initial_balance: float
    current_balance: float
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    orders: List[PaperOrder] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    total_fees: float = 0.0

class MarketDataProvider:
    """Provider of real market data for paper trading"""
    
    def __init__(self):
        self.current_data = {}
        self.data_subscribers = []
        logger.info("Market Data Provider initialized")

    def subscribe_to_data(self, callback: Callable):
        """Subscribe to market data updates"""
        self.data_subscribers.append(callback)

    def update_market_data(self, symbol: str, data: Dict[str, float]):
        """Update market data for a symbol"""
        self.current_data[symbol] = data
        # Notify subscribers
        for callback in self.data_subscribers:
            try:
                callback(symbol, data)
            except Exception as e:
                logger.error(f"Data callback error: {e}")

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        if symbol in self.current_data:
            return self.current_data[symbol].get('close')
        return None

    def get_market_snapshot(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get complete market snapshot"""
        return self.current_data.get(symbol)

class PaperTradingEngine:
    """Professional paper trading engine"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.account = PaperAccount(
            initial_balance=initial_balance,
            current_balance=initial_balance
        )
        self.market_provider = MarketDataProvider()
        self.trading_enabled = True
        self.order_counter = 0
        self.simulation_speed = 1.0  # 1.0 = real-time, 2.0 = 2x speed
        self.trading_lock = threading.Lock()
        
        # Trading fees configuration
        self.fee_structure = {
            'maker_fee': 0.001,  # 0.1% maker fee
            'taker_fee': 0.002,  # 0.2% taker fee
            'min_fee': 0.1       # Minimum $0.10 fee
        }
        
        # Subscribe to market data updates
        self.market_provider.subscribe_to_data(self._process_market_update)
        
        logger.info(f"Paper Trading Engine initialized with ${initial_balance:,.2f}")

    def place_order(self, symbol: str, order_type: OrderType, side: str,
                   quantity: float, price: Optional[float] = None) -> str:
        """Place a paper trading order"""
        try:
            with self.trading_lock:
                if not self.trading_enabled:
                    raise Exception("Trading is currently disabled")
                
                self.order_counter += 1
                order_id = f"PT_{datetime.now().strftime('%Y%m%d')}_{self.order_counter:06d}"
                
                order = PaperOrder(
                    order_id=order_id,
                    symbol=symbol,
                    order_type=order_type,
                    side=side.upper(),
                    quantity=quantity,
                    price=price
                )
                
                self.account.orders.append(order)
                
                # Process market orders immediately
                if order_type == OrderType.MARKET:
                    self._execute_market_order(order)
                
                logger.info(f"📝 Order placed: {order_id} {side} {quantity} {symbol} @{price or 'MARKET'}")
                return order_id
                
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            raise

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        try:
            with self.trading_lock:
                for order in self.account.orders:
                    if order.order_id == order_id and order.status == PaperTradeStatus.PENDING:
                        order.status = PaperTradeStatus.CANCELLED
                        logger.info(f"❌ Order cancelled: {order_id}")
                        return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def _execute_market_order(self, order: PaperOrder):
        """Execute market order immediately"""
        try:
            current_price = self.market_provider.get_current_price(order.symbol)
            if current_price is None:
                order.status = PaperTradeStatus.REJECTED
                logger.warning(f"❌ Order rejected - no market data for {order.symbol}")
                return
            
            # Apply slippage for market orders
            slippage = 0.001  # 0.1% slippage
            if order.side == 'BUY':
                execution_price = current_price * (1 + slippage)
            else:
                execution_price = current_price * (1 - slippage)
            
            self._fill_order(order, execution_price, order.quantity)
            
        except Exception as e:
            logger.error(f"Market order execution failed: {e}")
            order.status = PaperTradeStatus.REJECTED

    def _process_market_update(self, symbol: str, market_data: Dict[str, float]):
        """Process market data update"""
        try:
            if not self.trading_enabled:
                return
            
            current_price = market_data.get('close')
            if current_price is None:
                return
            
            # Check limit orders
            with self.trading_lock:
                for order in self.account.orders:
                    if (order.status == PaperTradeStatus.PENDING and 
                        order.symbol == symbol and 
                        order.order_type == OrderType.LIMIT):
                        
                        if ((order.side == 'BUY' and current_price <= order.price) or
                            (order.side == 'SELL' and current_price >= order.price)):
                            self._fill_order(order, order.price, order.quantity)
            
            # Update position values
            self._update_position_values(symbol, current_price)
            
        except Exception as e:
            logger.error(f"Market update processing failed: {e}")

    def _fill_order(self, order: PaperOrder, fill_price: float, fill_quantity: float):
        """Fill an order"""
        try:
            # Calculate fees
            order_value = fill_quantity * fill_price
            fee_rate = self.fee_structure['taker_fee']  # Assume taker for simplicity
            fees = max(order_value * fee_rate, self.fee_structure['min_fee'])
            
            # Update account balance
            if order.side == 'BUY':
                cost = order_value + fees
                if self.account.current_balance >= cost:
                    self.account.current_balance -= cost
                    self._update_position(order.symbol, fill_quantity, fill_price, 'BUY')
                else:
                    order.status = PaperTradeStatus.REJECTED
                    logger.warning(f"❌ Order rejected - insufficient funds")
                    return
            else:  # SELL
                proceeds = order_value - fees
                self.account.current_balance += proceeds
                self._update_position(order.symbol, fill_quantity, fill_price, 'SELL')
            
            # Update order status
            order.status = PaperTradeStatus.FILLED
            order.filled_quantity = fill_quantity
            order.average_fill_price = fill_price
            order.fill_timestamp = datetime.now()
            
            # Record trade
            self.account.trade_history.append({
                'timestamp': order.fill_timestamp,
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side,
                'quantity': fill_quantity,
                'price': fill_price,
                'fees': fees,
                'value': order_value
            })
            
            self.account.total_fees += fees
            
            logger.info(f"✅ Order filled: {order.order_id} {order.side} {fill_quantity} {order.symbol} @ ${fill_price:.2f}")
            
        except Exception as e:
            logger.error(f"Order filling failed: {e}")
            order.status = PaperTradeStatus.REJECTED

    def _update_position(self, symbol: str, quantity: float, price: float, side: str):
        """Update position after order fill"""
        try:
            if symbol not in self.account.positions:
                # New position
                self.account.positions[symbol] = PaperPosition(
                    symbol=symbol,
                    quantity=quantity if side == 'BUY' else -quantity,
                    average_price=price
                )
            else:
                # Existing position
                position = self.account.positions[symbol]
                old_quantity = position.quantity
                old_average_price = position.average_price
                
                if side == 'BUY':
                    # Adding to position
                    new_quantity = old_quantity + quantity
                    if new_quantity > 0:
                        # Average down/up
                        position.average_price = (
                            (old_quantity * old_average_price) + (quantity * price)
                        ) / new_quantity
                    position.quantity = new_quantity
                else:
                    # Reducing position
                    new_quantity = old_quantity - quantity
                    if new_quantity < 0:
                        # Position reversed
                        position.average_price = price
                    position.quantity = new_quantity
                    
                    # Calculate realized PNL
                    if (old_quantity > 0 and new_quantity < old_quantity) or \
                       (old_quantity < 0 and new_quantity > old_quantity):
                        # Closing part of position
                        closed_quantity = min(quantity, abs(old_quantity))
                        if old_quantity > 0:  # Was long
                            pnl = closed_quantity * (price - old_average_price)
                        else:  # Was short
                            pnl = closed_quantity * (old_average_price - price)
                        position.realized_pnl += pnl
                
                # Remove position if flat
                if position.quantity == 0:
                    del self.account.positions[symbol]
                    
        except Exception as e:
            logger.error(f"Position update failed: {e}")

    def _update_position_values(self, symbol: str, current_price: float):
        """Update unrealized PNL for positions"""
        try:
            if symbol in self.account.positions:
                position = self.account.positions[symbol]
                if position.quantity > 0:
                    position.unrealized_pnl = position.quantity * (current_price - position.average_price)
                else:
                    position.unrealized_pnl = abs(position.quantity) * (position.average_price - current_price)
        except Exception as e:
            logger.error(f"Position value update failed: {e}")

    def get_account_summary(self) -> Dict:
        """Get comprehensive account summary"""
        try:
            total_positions_value = 0
            total_unrealized_pnl = 0
            
            # Calculate position values
            for symbol, position in self.account.positions.items():
                current_price = self.market_provider.get_current_price(symbol)
                if current_price:
                    position_value = abs(position.quantity) * current_price
                    total_positions_value += position_value
                    total_unrealized_pnl += position.unrealized_pnl
            
            total_portfolio_value = self.account.current_balance + total_positions_value
            total_pnl = total_unrealized_pnl + sum(pos.realized_pnl for pos in self.account.positions.values())
            
            return {
                'cash_balance': self.account.current_balance,
                'positions_value': total_positions_value,
                'total_portfolio_value': total_portfolio_value,
                'unrealized_pnl': total_unrealized_pnl,
                'realized_pnl': sum(pos.realized_pnl for pos in self.account.positions.values()),
                'total_pnl': total_pnl,
                'pnl_percentage': (total_pnl / self.account.initial_balance) * 100 if self.account.initial_balance > 0 else 0,
                'number_of_positions': len(self.account.positions),
                'number_of_open_orders': len([o for o in self.account.orders if o.status == PaperTradeStatus.PENDING]),
                'total_fees_paid': self.account.total_fees,
                'total_trades': len(self.account.trade_history)
            }
            
        except Exception as e:
            logger.error(f"Account summary failed: {e}")
            return {}

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        positions = []
        for symbol, position in self.account.positions.items():
            current_price = self.market_provider.get_current_price(symbol)
            market_value = abs(position.quantity) * (current_price or 0)
            
            positions.append({
                'symbol': position.symbol,
                'quantity': position.quantity,
                'average_price': position.average_price,
                'current_price': current_price,
                'market_value': market_value,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'pnl_percentage': ((current_price - position.average_price) / position.average_price * 100) 
                                if position.average_price > 0 and current_price else 0
            })
        return positions

    def get_order_history(self, limit: int = 50) -> List[Dict]:
        """Get order history"""
        recent_orders = sorted(self.account.orders, key=lambda x: x.timestamp, reverse=True)[:limit]
        return [
            {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'type': order.order_type.value,
                'side': order.side,
                'quantity': order.quantity,
                'price': order.price,
                'status': order.status.value,
                'timestamp': order.timestamp.isoformat(),
                'filled_quantity': order.filled_quantity,
                'average_fill_price': order.average_fill_price
            }
            for order in recent_orders
        ]

    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get trade history"""
        recent_trades = sorted(self.account.trade_history, key=lambda x: x['timestamp'], reverse=True)[:limit]
        return recent_trades

    def enable_trading(self):
        """Enable trading"""
        self.trading_enabled = True
        logger.info("🟢 Paper trading enabled")

    def disable_trading(self):
        """Disable trading"""
        self.trading_enabled = False
        logger.info("🔴 Paper trading disabled")

    def reset_account(self, new_balance: Optional[float] = None):
        """Reset account to initial state"""
        try:
            new_balance = new_balance or self.account.initial_balance
            self.account = PaperAccount(
                initial_balance=new_balance,
                current_balance=new_balance
            )
            self.order_counter = 0
            logger.info(f"🔄 Account reset with ${new_balance:,.2f}")
        except Exception as e:
            logger.error(f"Account reset failed: {e}")

    def load_simulation_data(self, data_file: str):
        """Load historical data for backtesting simulation"""
        try:
            # This would load historical market data for simulation
            logger.info(f"Loading simulation data from {data_file}")
            # Implementation would depend on data format
        except Exception as e:
            logger.error(f"Failed to load simulation data: {e}")

# Global instance
_paper_trading_engine = None

def get_paper_trading_engine(initial_balance: float = 100000.0) -> PaperTradingEngine:
    """Get singleton paper trading engine instance"""
    global _paper_trading_engine
    if _paper_trading_engine is None:
        _paper_trading_engine = PaperTradingEngine(initial_balance)
    return _paper_trading_engine

def main():
    """Example usage"""
    print("Paper Trading Environment ready")
    print("Professional simulated trading with real market data")

if __name__ == "__main__":
    main()