"""
Paper Broker for Chloe AI - Phase 1
Safe execution simulator that mimics real trading without actual orders
"""

import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import uuid

logger = logging.getLogger(__name__)

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"

class OrderStatus(Enum):
    """Order status enumeration"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class OrderEvent:
    """Order event for paper trading"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # None for market orders
    order_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.NEW

@dataclass
class FillEvent:
    """Fill event from paper broker"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    fill_price: float
    fill_quantity: float
    commission: float
    slippage: float
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: float = 0.0  # Latency simulation

@dataclass
class Position:
    """Current position tracking"""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0

class PaperBroker:
    """Safe paper trading execution simulator"""
    
    def __init__(self, 
                 initial_balance: float = 100000.0,
                 commission_rate: float = 0.001,  # 0.1% commission
                 slippage_range: tuple = (0.0002, 0.001),  # 0.02% to 0.1% slippage
                 latency_range: tuple = (0.01, 0.1)):     # 10ms to 100ms latency
        """
        Initialize paper broker with realistic trading conditions
        
        Args:
            initial_balance: Starting cash balance
            commission_rate: Commission per trade (0.001 = 0.1%)
            slippage_range: Slippage range as (min, max) tuple
            latency_range: Latency simulation range in seconds
        """
        self.cash_balance = initial_balance
        self.commission_rate = commission_rate
        self.slippage_range = slippage_range
        self.latency_range = latency_range
        
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, OrderEvent] = {}
        self.trade_history: List[FillEvent] = []
        self.order_counter = 0
        
        logger.info(f"Paper Broker initialized with ${initial_balance:,.2f}")
        logger.info(f"Commission rate: {commission_rate*100:.2f}%")
        logger.info(f"Slippage range: {slippage_range[0]*100:.2f}% to {slippage_range[1]*100:.2f}%")
        logger.info(f"Latency range: {latency_range[0]*1000:.1f}ms to {latency_range[1]*1000:.1f}ms")

    def submit_order(self, order_event: OrderEvent) -> str:
        """Submit order to paper broker (no real execution)"""
        try:
            # Generate unique order ID
            self.order_counter += 1
            order_event.order_id = f"PAPER_{datetime.now().strftime('%Y%m%d')}_{self.order_counter:06d}"
            
            # Validate order
            if not self._validate_order(order_event):
                order_event.status = OrderStatus.REJECTED
                logger.warning(f"❌ Order rejected: {order_event.order_id}")
                return order_event.order_id
            
            # Add to open orders
            self.open_orders[order_event.order_id] = order_event
            order_event.status = OrderStatus.NEW
            
            logger.info(f"📝 Order submitted: {order_event.order_id} "
                       f"{order_event.side.value} {order_event.quantity} "
                       f"{order_event.symbol} @{order_event.price or 'MARKET'}")
            
            # For market orders, execute immediately
            if order_event.order_type == OrderType.MARKET:
                self._execute_market_order(order_event)
            
            return order_event.order_id
            
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            raise

    def _validate_order(self, order_event: OrderEvent) -> bool:
        """Validate order parameters"""
        try:
            # Check quantity
            if order_event.quantity <= 0:
                logger.error("Invalid quantity")
                return False
            
            # Check price for limit/stop orders
            if order_event.order_type in [OrderType.LIMIT, OrderType.STOP]:
                if order_event.price is None or order_event.price <= 0:
                    logger.error("Invalid price for limit/stop order")
                    return False
            
            # Check available balance for buy orders
            if order_event.side == OrderSide.BUY:
                required_funds = order_event.quantity * (order_event.price or 0) * (1 + self.commission_rate)
                if required_funds > self.cash_balance:
                    logger.error(f"Insufficient funds: need ${required_funds:,.2f}, have ${self.cash_balance:,.2f}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Order validation failed: {e}")
            return False

    def _execute_market_order(self, order_event: OrderEvent):
        """Execute market order with realistic conditions"""
        try:
            # Simulate latency
            execution_latency = np.random.uniform(*self.latency_range)
            
            # Get current market price (would come from market data feed)
            current_price = self._get_current_price(order_event.symbol)
            
            if current_price is None:
                order_event.status = OrderStatus.REJECTED
                logger.warning(f"No market price for {order_event.symbol}")
                return
            
            # Apply slippage
            slippage_pct = np.random.uniform(*self.slippage_range)
            if order_event.side == OrderSide.BUY:
                execution_price = current_price * (1 + slippage_pct)
            else:  # SELL
                execution_price = current_price * (1 - slippage_pct)
            
            # Calculate commission
            commission = order_event.quantity * execution_price * self.commission_rate
            
            # Create fill event
            fill_event = FillEvent(
                order_id=order_event.order_id,
                symbol=order_event.symbol,
                side=order_event.side,
                quantity=order_event.quantity,
                fill_price=execution_price,
                fill_quantity=order_event.quantity,
                commission=commission,
                slippage=slippage_pct,
                execution_time=execution_latency
            )
            
            # Process the fill
            self._process_fill(fill_event)
            
            # Update order status
            order_event.status = OrderStatus.FILLED
            
            # Remove from open orders
            if order_event.order_id in self.open_orders:
                del self.open_orders[order_event.order_id]
            
            logger.info(f"✅ Order filled: {order_event.order_id} "
                       f"{order_event.side.value} {order_event.quantity} "
                       f"{order_event.symbol} @ ${execution_price:.2f} "
                       f"(slippage: {slippage_pct*100:.3f}%, commission: ${commission:.2f})")
            
        except Exception as e:
            logger.error(f"Market order execution failed: {e}")
            order_event.status = OrderStatus.REJECTED

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol"""
        # In real implementation, this would connect to market data feed
        # For demo purposes, we'll simulate prices
        if hasattr(self, '_market_prices'):
            return self._market_prices.get(symbol)
        
        # Default price simulation
        base_prices = {
            'BTC/USDT': 50000.0,
            'ETH/USDT': 2950.0,
            'SOL/USDT': 95.5,
            'ADA/USDT': 0.48
        }
        return base_prices.get(symbol, 100.0)

    def update_market_prices(self, prices: Dict[str, float]):
        """Update current market prices"""
        self._market_prices = prices

    def _process_fill(self, fill_event: FillEvent):
        """Process fill event and update account state"""
        try:
            # Calculate cash flow
            if fill_event.side == OrderSide.BUY:
                cash_outflow = (fill_event.fill_quantity * fill_event.fill_price) + fill_event.commission
                self.cash_balance -= cash_outflow
                self._update_position(fill_event.symbol, fill_event.fill_quantity, fill_event.fill_price, 'BUY')
            else:  # SELL
                cash_inflow = (fill_event.fill_quantity * fill_event.fill_price) - fill_event.commission
                self.cash_balance += cash_inflow
                self._update_position(fill_event.symbol, fill_event.fill_quantity, fill_event.fill_price, 'SELL')
            
            # Add to trade history
            self.trade_history.append(fill_event)
            
        except Exception as e:
            logger.error(f"Fill processing failed: {e}")

    def _update_position(self, symbol: str, quantity: float, price: float, side: str):
        """Update position after fill"""
        try:
            if symbol not in self.positions:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=quantity if side == 'BUY' else -quantity,
                    avg_price=price
                )
            else:
                # Existing position
                position = self.positions[symbol]
                old_quantity = position.quantity
                old_avg_price = position.avg_price
                
                if side == 'BUY':
                    # Adding to position
                    new_quantity = old_quantity + quantity
                    if new_quantity > 0:
                        # Average down/up
                        position.avg_price = (
                            (old_quantity * old_avg_price) + (quantity * price)
                        ) / new_quantity
                    position.quantity = new_quantity
                else:
                    # Reducing position
                    new_quantity = old_quantity - quantity
                    position.quantity = new_quantity
                    
                    # Remove position if flat
                    if position.quantity == 0:
                        del self.positions[symbol]
                        
        except Exception as e:
            logger.error(f"Position update failed: {e}")

    def get_account_state(self) -> Dict:
        """Get current account state"""
        try:
            # Calculate positions value
            positions_value = 0
            for symbol, position in self.positions.items():
                current_price = self._get_current_price(symbol)
                if current_price:
                    positions_value += abs(position.quantity) * current_price
            
            total_value = self.cash_balance + positions_value
            
            return {
                'cash_balance': self.cash_balance,
                'positions_value': positions_value,
                'total_portfolio_value': total_value,
                'positions': {
                    symbol: {
                        'quantity': pos.quantity,
                        'avg_price': pos.avg_price,
                        'market_value': abs(pos.quantity) * (self._get_current_price(symbol) or 0)
                    }
                    for symbol, pos in self.positions.items()
                },
                'open_orders': len(self.open_orders),
                'total_trades': len(self.trade_history)
            }
            
        except Exception as e:
            logger.error(f"Account state retrieval failed: {e}")
            return {}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        try:
            if order_id in self.open_orders:
                order = self.open_orders[order_id]
                order.status = OrderStatus.CANCELLED
                del self.open_orders[order_id]
                logger.info(f"❌ Order cancelled: {order_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False

    def get_open_orders(self) -> List[OrderEvent]:
        """Get list of open orders"""
        return list(self.open_orders.values())

    def get_trade_history(self, limit: int = 50) -> List[FillEvent]:
        """Get recent trade history"""
        return self.trade_history[-limit:] if self.trade_history else []

    def reset(self, new_balance: Optional[float] = None):
        """Reset broker state"""
        new_balance = new_balance or 100000.0
        self.__init__(initial_balance=new_balance,
                     commission_rate=self.commission_rate,
                     slippage_range=self.slippage_range,
                     latency_range=self.latency_range)
        logger.info(f"🔄 Broker reset with ${new_balance:,.2f}")

def main():
    """Example usage"""
    print("Paper Broker - Safe Execution Simulator")
    print("Phase 1 of Paper-Learning Architecture")

if __name__ == "__main__":
    main()