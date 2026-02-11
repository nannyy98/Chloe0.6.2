"""
Transaction Cost Modeling for Chloe 0.6
Professional transaction cost analysis including spreads, fees, and market impact
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types for cost modeling"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class LiquidityImpact(Enum):
    """Liquidity impact levels"""
    LOW = "LOW"         # Minimal market impact
    MODERATE = "MODERATE"  # Noticeable but manageable impact
    HIGH = "HIGH"       # Significant market impact
    EXTREME = "EXTREME" # Severe market disruption

@dataclass
class MarketSnapshot:
    """Market data snapshot for cost calculation"""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_volume: float
    ask_volume: float
    last_price: float
    volume_24h: float
    volatility: float

@dataclass
class TransactionCosts:
    """Detailed transaction cost breakdown"""
    symbol: str
    order_size: float
    order_type: OrderType
    execution_price: float
    spread_cost: float           # Bid-ask spread cost
    commission_cost: float       # Exchange fees
    slippage_cost: float         # Market impact/slippage
    total_cost: float            # Sum of all costs
    cost_percentage: float       # Cost as percentage of order value
    liquidity_impact: LiquidityImpact
    market_conditions: str       # Market regime description

@dataclass
class CostModelConfig:
    """Configuration for transaction cost modeling"""
    # Commission rates (as percentages)
    maker_fee: float = 0.001     # 0.1% maker fee
    taker_fee: float = 0.002     # 0.2% taker fee
    
    # Spread modeling parameters
    base_spread_multiplier: float = 0.5  # Base spread relative to price
    volatility_spread_factor: float = 0.3  # Volatility impact on spread
    
    # Slippage modeling parameters
    slippage_alpha: float = 0.0001  # Linear slippage coefficient
    slippage_beta: float = 0.000001  # Quadratic slippage coefficient
    slippage_gamma: float = 0.1     # Market depth impact factor
    
    # Market conditions multipliers
    stable_market_multiplier: float = 1.0
    volatile_market_multiplier: float = 1.5
    trending_market_multiplier: float = 1.2
    crisis_market_multiplier: float = 2.0

class TransactionCostModel:
    """Professional transaction cost modeling engine"""
    
    def __init__(self, config: CostModelConfig = None):
        self.config = config or CostModelConfig()
        self.market_data_cache = {}  # Cache for market snapshots
        self.cost_history = []       # Historical cost records
        self.exchange_fees = {}      # Exchange-specific fee structures
        
        # Initialize exchange fee structures
        self._setup_exchange_fees()
        
        logger.info("Transaction Cost Model initialized")
        logger.info(f"Base maker fee: {self.config.maker_fee:.3%}")
        logger.info(f"Base taker fee: {self.config.taker_fee:.3%}")

    def _setup_exchange_fees(self):
        """Setup exchange-specific fee structures"""
        self.exchange_fees = {
            'binance': {'maker': 0.001, 'taker': 0.001},
            'coinbase': {'maker': 0.005, 'taker': 0.005},
            'kraken': {'maker': 0.0016, 'taker': 0.0026},
            'bitfinex': {'maker': 0.001, 'taker': 0.002}
        }

    def estimate_transaction_costs(self, 
                                 symbol: str,
                                 order_size: float,
                                 order_type: OrderType,
                                 market_snapshot: MarketSnapshot,
                                 exchange: str = 'binance',
                                 market_regime: str = 'STABLE') -> TransactionCosts:
        """Estimate total transaction costs for an order"""
        try:
            # Calculate individual cost components
            spread_cost = self._calculate_spread_cost(market_snapshot, order_size, order_type)
            commission_cost = self._calculate_commission_cost(order_size, exchange, order_type)
            slippage_cost = self._calculate_slippage_cost(market_snapshot, order_size, market_regime)
            
            # Determine execution price
            execution_price = self._calculate_execution_price(
                market_snapshot, order_size, order_type, slippage_cost
            )
            
            # Calculate total costs
            total_cost = spread_cost + commission_cost + slippage_cost
            order_value = order_size * execution_price
            cost_percentage = (total_cost / order_value) if order_value > 0 else 0
            
            # Determine liquidity impact
            liquidity_impact = self._assess_liquidity_impact(order_size, market_snapshot)
            
            # Create cost breakdown
            costs = TransactionCosts(
                symbol=symbol,
                order_size=order_size,
                order_type=order_type,
                execution_price=execution_price,
                spread_cost=spread_cost,
                commission_cost=commission_cost,
                slippage_cost=slippage_cost,
                total_cost=total_cost,
                cost_percentage=cost_percentage,
                liquidity_impact=liquidity_impact,
                market_conditions=market_regime
            )
            
            # Record in history
            self.cost_history.append(costs)
            
            return costs
            
        except Exception as e:
            logger.error(f"Transaction cost estimation failed: {e}")
            return self._get_default_costs(symbol, order_size, order_type)

    def _calculate_spread_cost(self, snapshot: MarketSnapshot, 
                              order_size: float, order_type: OrderType) -> float:
        """Calculate bid-ask spread cost"""
        try:
            spread = snapshot.ask_price - snapshot.bid_price
            
            # Base spread cost
            base_spread_cost = spread * order_size
            
            # Adjust for order type
            if order_type == OrderType.LIMIT:
                # Limit orders typically execute at midpoint or better
                spread_cost = base_spread_cost * 0.5
            elif order_type == OrderType.MARKET:
                # Market orders typically execute at worst price
                spread_cost = base_spread_cost
            else:
                # Stop and stop-limit orders
                spread_cost = base_spread_cost * 0.75
            
            # Adjust for market volatility
            volatility_factor = 1 + (snapshot.volatility * self.config.volatility_spread_factor)
            spread_cost *= volatility_factor
            
            return max(0, spread_cost)
            
        except Exception as e:
            logger.error(f"Spread cost calculation failed: {e}")
            return 0.0

    def _calculate_commission_cost(self, order_size: float, 
                                  exchange: str, order_type: OrderType) -> float:
        """Calculate exchange commission costs"""
        try:
            # Get exchange fees
            fees = self.exchange_fees.get(exchange.lower(), 
                                        {'maker': self.config.maker_fee, 
                                         'taker': self.config.taker_fee})
            
            # Determine applicable fee rate
            if order_type == OrderType.LIMIT:
                fee_rate = fees['maker']
            else:
                fee_rate = fees['taker']
            
            # Calculate commission (assuming price of $50,000 for crypto)
            estimated_price = 50000.0
            order_value = order_size * estimated_price
            commission_cost = order_value * fee_rate
            
            return max(0, commission_cost)
            
        except Exception as e:
            logger.error(f"Commission cost calculation failed: {e}")
            return order_size * 50000.0 * self.config.taker_fee

    def _calculate_slippage_cost(self, snapshot: MarketSnapshot, 
                                order_size: float, market_regime: str) -> float:
        """Calculate market impact and slippage costs"""
        try:
            # Get regime multiplier
            regime_multipliers = {
                'STABLE': self.config.stable_market_multiplier,
                'TRENDING': self.config.trending_market_multiplier,
                'VOLATILE': self.config.volatile_market_multiplier,
                'CRISIS': self.config.crisis_market_multiplier
            }
            regime_mult = regime_multipliers.get(market_regime.upper(), 1.0)
            
            # Calculate market depth ratio
            avg_depth = (snapshot.bid_volume + snapshot.ask_volume) / 2
            depth_ratio = order_size / (avg_depth + 1e-8)
            
            # Linear slippage component
            linear_slippage = self.config.slippage_alpha * order_size * regime_mult
            
            # Quadratic slippage component (market impact)
            quadratic_slippage = self.config.slippage_beta * (order_size ** 2) * regime_mult
            
            # Depth-based slippage
            depth_slippage = self.config.slippage_gamma * depth_ratio * snapshot.last_price * order_size
            
            # Total slippage cost
            total_slippage = linear_slippage + quadratic_slippage + depth_slippage
            
            return max(0, total_slippage)
            
        except Exception as e:
            logger.error(f"Slippage cost calculation failed: {e}")
            return order_size * 50000.0 * 0.001  # Default 0.1% slippage

    def _calculate_execution_price(self, snapshot: MarketSnapshot,
                                  order_size: float, order_type: OrderType,
                                  slippage_cost: float) -> float:
        """Calculate expected execution price"""
        try:
            base_price = snapshot.last_price
            
            if order_type == OrderType.MARKET:
                # Market orders: execution at average of bid/ask plus slippage
                midpoint = (snapshot.bid_price + snapshot.ask_price) / 2
                slippage_impact = slippage_cost / order_size if order_size > 0 else 0
                execution_price = midpoint + slippage_impact
            elif order_type == OrderType.LIMIT:
                # Limit orders: execution at specified limit price
                # Assuming limit order is placed at midpoint
                execution_price = (snapshot.bid_price + snapshot.ask_price) / 2
            else:
                # Stop orders: similar to market orders but with delay
                midpoint = (snapshot.bid_price + snapshot.ask_price) / 2
                slippage_impact = slippage_cost / order_size * 1.2 if order_size > 0 else 0
                execution_price = midpoint + slippage_impact
            
            return max(0, execution_price)
            
        except Exception as e:
            logger.error(f"Execution price calculation failed: {e}")
            return snapshot.last_price

    def _assess_liquidity_impact(self, order_size: float, 
                                snapshot: MarketSnapshot) -> LiquidityImpact:
        """Assess the liquidity impact level of an order"""
        try:
            # Calculate order size relative to market depth
            avg_depth = (snapshot.bid_volume + snapshot.ask_volume) / 2
            depth_ratio = order_size / (avg_depth + 1e-8)
            
            # Calculate order size relative to 24h volume
            volume_ratio = order_size / (snapshot.volume_24h + 1e-8)
            
            # Combined liquidity pressure metric
            liquidity_pressure = max(depth_ratio, volume_ratio)
            
            # Classify impact level
            if liquidity_pressure < 0.01:
                return LiquidityImpact.LOW
            elif liquidity_pressure < 0.05:
                return LiquidityImpact.MODERATE
            elif liquidity_pressure < 0.2:
                return LiquidityImpact.HIGH
            else:
                return LiquidityImpact.EXTREME
                
        except Exception:
            return LiquidityImpact.MODERATE

    def _get_default_costs(self, symbol: str, order_size: float, 
                          order_type: OrderType) -> TransactionCosts:
        """Get default cost estimates when calculations fail"""
        return TransactionCosts(
            symbol=symbol,
            order_size=order_size,
            order_type=order_type,
            execution_price=50000.0,
            spread_cost=order_size * 50000.0 * 0.0005,  # 0.05% spread
            commission_cost=order_size * 50000.0 * 0.001,  # 0.1% commission
            slippage_cost=order_size * 50000.0 * 0.001,   # 0.1% slippage
            total_cost=order_size * 50000.0 * 0.0025,
            cost_percentage=0.0025,
            liquidity_impact=LiquidityImpact.MODERATE,
            market_conditions="UNKNOWN"
        )

    def optimize_order_size(self, symbol: str, target_value: float,
                           max_cost_percentage: float = 0.005,
                           market_snapshot: MarketSnapshot = None,
                           market_regime: str = 'STABLE') -> Tuple[float, TransactionCosts]:
        """Optimize order size to stay within cost constraints"""
        try:
            if market_snapshot is None:
                # Create default market snapshot
                market_snapshot = MarketSnapshot(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    bid_price=49950.0,
                    ask_price=50050.0,
                    bid_volume=100.0,
                    ask_volume=100.0,
                    last_price=50000.0,
                    volume_24h=10000.0,
                    volatility=0.02
                )
            
            # Binary search for optimal order size
            min_size = 0.0
            max_size = target_value / market_snapshot.last_price
            optimal_size = max_size
            
            for _ in range(20):  # 20 iterations of binary search
                test_size = (min_size + max_size) / 2
                costs = self.estimate_transaction_costs(
                    symbol, test_size, OrderType.MARKET, 
                    market_snapshot, market_regime=market_regime
                )
                
                if costs.cost_percentage <= max_cost_percentage:
                    optimal_size = test_size
                    min_size = test_size
                else:
                    max_size = test_size
            
            # Calculate final costs for optimal size
            final_costs = self.estimate_transaction_costs(
                symbol, optimal_size, OrderType.MARKET,
                market_snapshot, market_regime=market_regime
            )
            
            return optimal_size, final_costs
            
        except Exception as e:
            logger.error(f"Order size optimization failed: {e}")
            default_size = target_value / 50000.0
            default_costs = self._get_default_costs(symbol, default_size, OrderType.MARKET)
            return default_size, default_costs

    def batch_cost_analysis(self, orders: List[Dict]) -> List[TransactionCosts]:
        """Analyze transaction costs for multiple orders"""
        try:
            results = []
            for order in orders:
                costs = self.estimate_transaction_costs(**order)
                results.append(costs)
            return results
        except Exception as e:
            logger.error(f"Batch cost analysis failed: {e}")
            return []

    def get_cost_statistics(self) -> Dict:
        """Get statistics on transaction costs"""
        try:
            if not self.cost_history:
                return {'total_orders': 0}
            
            costs = [c.total_cost for c in self.cost_history]
            percentages = [c.cost_percentage for c in self.cost_history]
            
            return {
                'total_orders': len(self.cost_history),
                'average_total_cost': np.mean(costs),
                'median_total_cost': np.median(costs),
                'average_cost_percentage': np.mean(percentages),
                'median_cost_percentage': np.median(percentages),
                'max_cost': np.max(costs),
                'min_cost': np.min(costs),
                'cost_std': np.std(costs)
            }
        except Exception as e:
            logger.error(f"Cost statistics calculation failed: {e}")
            return {'error': str(e)}

    def update_market_data(self, symbol: str, snapshot: MarketSnapshot):
        """Update cached market data"""
        try:
            self.market_data_cache[symbol] = snapshot
            logger.debug(f"Updated market data for {symbol}")
        except Exception as e:
            logger.error(f"Market data update failed: {e}")

# Global instance
_transaction_cost_model = None

def get_transaction_cost_model(config: CostModelConfig = None) -> TransactionCostModel:
    """Get singleton transaction cost model instance"""
    global _transaction_cost_model
    if _transaction_cost_model is None:
        _transaction_cost_model = TransactionCostModel(config)
    return _transaction_cost_model

def main():
    """Example usage"""
    print("Transaction Cost Model ready")
    print("Professional transaction cost analysis system")

if __name__ == "__main__":
    main()