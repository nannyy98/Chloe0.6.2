#!/usr/bin/env python3
"""
Transaction Cost Modeling Demo for Chloe 0.6
Professional transaction cost analysis demonstration
"""

import asyncio
import logging
from datetime import datetime
import numpy as np
from transaction_cost_model import get_transaction_cost_model, MarketSnapshot, OrderType, CostModelConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_market_snapshots(symbols: list) -> dict:
    """Generate realistic market snapshots for cost modeling"""
    snapshots = {}
    
    # Market data for different assets
    market_data = {
        'BTC/USDT': {
            'bid_price': 49950.0,
            'ask_price': 50050.0,
            'bid_volume': 150.0,
            'ask_volume': 145.0,
            'last_price': 50000.0,
            'volume_24h': 25000.0,
            'volatility': 0.025
        },
        'ETH/USDT': {
            'bid_price': 2950.0,
            'ask_price': 2960.0,
            'bid_volume': 500.0,
            'ask_volume': 480.0,
            'last_price': 2955.0,
            'volume_24h': 150000.0,
            'volatility': 0.030
        },
        'SOL/USDT': {
            'bid_price': 95.0,
            'ask_price': 96.0,
            'bid_volume': 2000.0,
            'ask_volume': 1900.0,
            'last_price': 95.5,
            'volume_24h': 500000.0,
            'volatility': 0.045
        },
        'ADA/USDT': {
            'bid_price': 0.480,
            'ask_price': 0.482,
            'bid_volume': 50000.0,
            'ask_volume': 45000.0,
            'last_price': 0.481,
            'volume_24h': 1000000.0,
            'volatility': 0.020
        }
    }
    
    for symbol in symbols:
        if symbol in market_data:
            data = market_data[symbol]
            snapshot = MarketSnapshot(
                symbol=symbol,
                timestamp=datetime.now(),
                bid_price=data['bid_price'],
                ask_price=data['ask_price'],
                bid_volume=data['bid_volume'],
                ask_volume=data['ask_volume'],
                last_price=data['last_price'],
                volume_24h=data['volume_24h'],
                volatility=data['volatility']
            )
            snapshots[symbol] = snapshot
    
    return snapshots

async def demonstrate_transaction_cost_modeling():
    """Demonstrate transaction cost modeling capabilities"""
    logger.info("💰 TRANSACTION COST MODELING DEMO")
    logger.info("=" * 40)
    
    try:
        # Initialize cost model
        logger.info("🔧 Initializing Transaction Cost Model...")
        config = CostModelConfig(
            maker_fee=0.001,
            taker_fee=0.002,
            base_spread_multiplier=0.5,
            volatility_spread_factor=0.3,
            slippage_alpha=0.0001,
            slippage_beta=0.000001,
            slippage_gamma=0.1
        )
        cost_model = get_transaction_cost_model(config)
        logger.info("✅ Transaction Cost Model initialized")
        
        # Generate market data
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT']
        market_snapshots = generate_market_snapshots(symbols)
        
        logger.info(f"📊 MARKET DATA GENERATED:")
        logger.info(f"   Assets: {list(market_snapshots.keys())}")
        logger.info(f"   Price Range: ${min(s.last_price for s in market_snapshots.values()):,.2f} - ${max(s.last_price for s in market_snapshots.values()):,.2f}")
        logger.info(f"   Volatility Range: {min(s.volatility for s in market_snapshots.values()):.2%} - {max(s.volatility for s in market_snapshots.values()):.2%}")
        
        # Test different order types and sizes
        logger.info(f"\n💱 ORDER TYPE COST COMPARISON:")
        
        test_orders = [
            {'symbol': 'BTC/USDT', 'order_size': 0.1, 'order_type': OrderType.MARKET},
            {'symbol': 'BTC/USDT', 'order_size': 0.1, 'order_type': OrderType.LIMIT},
            {'symbol': 'ETH/USDT', 'order_size': 1.0, 'order_type': OrderType.MARKET},
            {'symbol': 'ETH/USDT', 'order_size': 1.0, 'order_type': OrderType.LIMIT}
        ]
        
        for order in test_orders:
            snapshot = market_snapshots[order['symbol']]
            costs = cost_model.estimate_transaction_costs(
                symbol=order['symbol'],
                order_size=order['order_size'],
                order_type=order['order_type'],
                market_snapshot=snapshot,
                market_regime='STABLE'
            )
            
            order_value = order['order_size'] * snapshot.last_price
            logger.info(f"   {order['symbol']} {order['order_size']} @ {order['order_type'].value}:")
            logger.info(f"      Order Value: ${order_value:,.2f}")
            logger.info(f"      Spread Cost: ${costs.spread_cost:.4f} ({costs.spread_cost/order_value*100:.3f}%)")
            logger.info(f"      Commission: ${costs.commission_cost:.4f} ({costs.commission_cost/order_value*100:.3f}%)")
            logger.info(f"      Slippage: ${costs.slippage_cost:.4f} ({costs.slippage_cost/order_value*100:.3f}%)")
            logger.info(f"      TOTAL COST: ${costs.total_cost:.4f} ({costs.cost_percentage:.3f}%)")
            logger.info(f"      Liquidity Impact: {costs.liquidity_impact.value}")
        
        # Test different market regimes
        logger.info(f"\n🌍 MARKET REGIME IMPACT ANALYSIS:")
        
        regimes = ['STABLE', 'TRENDING', 'VOLATILE', 'CRISIS']
        test_asset = 'BTC/USDT'
        test_order_size = 0.5
        snapshot = market_snapshots[test_asset]
        
        for regime in regimes:
            costs = cost_model.estimate_transaction_costs(
                symbol=test_asset,
                order_size=test_order_size,
                order_type=OrderType.MARKET,
                market_snapshot=snapshot,
                market_regime=regime
            )
            
            order_value = test_order_size * snapshot.last_price
            logger.info(f"   {regime} Regime:")
            logger.info(f"      Total Cost: ${costs.total_cost:.4f} ({costs.cost_percentage:.3f}%)")
            logger.info(f"      Slippage Component: ${costs.slippage_cost:.4f}")
            logger.info(f"      Liquidity Impact: {costs.liquidity_impact.value}")
        
        # Test different order sizes
        logger.info(f"\n📏 ORDER SIZE COST ANALYSIS:")
        
        order_sizes = [0.01, 0.1, 0.5, 1.0, 2.0]
        test_symbol = 'ETH/USDT'
        snapshot = market_snapshots[test_symbol]
        
        for size in order_sizes:
            costs = cost_model.estimate_transaction_costs(
                symbol=test_symbol,
                order_size=size,
                order_type=OrderType.MARKET,
                market_snapshot=snapshot,
                market_regime='STABLE'
            )
            
            order_value = size * snapshot.last_price
            logger.info(f"   Order Size {size}:")
            logger.info(f"      Value: ${order_value:,.2f}")
            logger.info(f"      Cost Percentage: {costs.cost_percentage:.3f}%")
            logger.info(f"      Per Unit Cost: ${costs.total_cost/size:.4f}")
            logger.info(f"      Liquidity Impact: {costs.liquidity_impact.value}")
        
        # Test cost optimization
        logger.info(f"\n🎯 COST OPTIMIZATION DEMO:")
        
        target_values = [1000, 5000, 10000, 25000]
        max_cost_percentages = [0.001, 0.002, 0.003, 0.005]  # 0.1%, 0.2%, 0.3%, 0.5%
        
        for target_value, max_cost in zip(target_values, max_cost_percentages):
            optimal_size, costs = cost_model.optimize_order_size(
                symbol='BTC/USDT',
                target_value=target_value,
                max_cost_percentage=max_cost,
                market_snapshot=market_snapshots['BTC/USDT'],
                market_regime='STABLE'
            )
            
            actual_value = optimal_size * market_snapshots['BTC/USDT'].last_price
            logger.info(f"   Target: ${target_value:,} (max {max_cost:.1%} cost):")
            logger.info(f"      Optimal Size: {optimal_size:.4f} BTC")
            logger.info(f"      Actual Value: ${actual_value:,.2f}")
            logger.info(f"      Achieved Cost: {costs.cost_percentage:.3f}%")
            logger.info(f"      Total Cost: ${costs.total_cost:.4f}")
        
        # Batch analysis
        logger.info(f"\n📊 BATCH COST ANALYSIS:")
        
        batch_orders = [
            {'symbol': 'BTC/USDT', 'order_size': 0.05, 'order_type': OrderType.MARKET, 
             'market_snapshot': market_snapshots['BTC/USDT'], 'market_regime': 'STABLE'},
            {'symbol': 'ETH/USDT', 'order_size': 0.5, 'order_type': OrderType.MARKET,
             'market_snapshot': market_snapshots['ETH/USDT'], 'market_regime': 'VOLATILE'},
            {'symbol': 'SOL/USDT', 'order_size': 10.0, 'order_type': OrderType.LIMIT,
             'market_snapshot': market_snapshots['SOL/USDT'], 'market_regime': 'STABLE'},
            {'symbol': 'ADA/USDT', 'order_size': 1000.0, 'order_type': OrderType.MARKET,
             'market_snapshot': market_snapshots['ADA/USDT'], 'market_regime': 'CRISIS'}
        ]
        
        batch_results = cost_model.batch_cost_analysis(batch_orders)
        
        logger.info(f"   Analyzed {len(batch_results)} orders:")
        for i, result in enumerate(batch_results):
            order_value = batch_orders[i]['order_size'] * batch_orders[i]['market_snapshot'].last_price
            logger.info(f"      {result.symbol}: ${order_value:,.2f} → Cost: {result.cost_percentage:.3f}% ({result.liquidity_impact.value})")
        
        # Cost statistics
        logger.info(f"\n📈 COST STATISTICS ANALYSIS:")
        
        stats = cost_model.get_cost_statistics()
        logger.info(f"   Total Orders Analyzed: {stats.get('total_orders', 0)}")
        logger.info(f"   Average Cost Percentage: {stats.get('average_cost_percentage', 0):.3f}%")
        logger.info(f"   Median Cost Percentage: {stats.get('median_cost_percentage', 0):.3f}%")
        logger.info(f"   Cost Range: {stats.get('min_cost', 0):.4f} - {stats.get('max_cost', 0):.4f}")
        logger.info(f"   Cost Standard Deviation: {stats.get('cost_std', 0):.4f}")
        
        # Exchange comparison
        logger.info(f"\n💱 EXCHANGE FEE COMPARISON:")
        
        exchanges = ['binance', 'coinbase', 'kraken', 'bitfinex']
        test_order = {'symbol': 'BTC/USDT', 'order_size': 0.1, 'order_type': OrderType.MARKET}
        snapshot = market_snapshots['BTC/USDT']
        
        for exchange in exchanges:
            costs = cost_model.estimate_transaction_costs(
                **test_order,
                market_snapshot=snapshot,
                exchange=exchange,
                market_regime='STABLE'
            )
            
            order_value = test_order['order_size'] * snapshot.last_price
            logger.info(f"   {exchange.upper()}:")
            logger.info(f"      Commission: ${costs.commission_cost:.4f} ({costs.commission_cost/order_value*100:.3f}%)")
            logger.info(f"      Total Cost: ${costs.total_cost:.4f} ({costs.cost_percentage:.3f}%)")
        
        logger.info(f"\n✅ TRANSACTION COST MODELING DEMO COMPLETED SUCCESSFULLY")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented professional transaction cost modeling")
        logger.info("   • Created detailed cost breakdown (spread, commission, slippage)")
        logger.info("   • Built market regime-aware cost calculations")
        logger.info("   • Developed order size optimization capabilities")
        logger.info("   • Tested batch analysis and statistics")
        
        logger.info(f"\n💰 COST MODELING INSIGHTS:")
        logger.info("   Spread Costs: Typically 0.05-0.15% of order value")
        logger.info("   Commission Costs: 0.1-0.26% depending on exchange")
        logger.info("   Slippage Costs: Variable based on size and market conditions")
        logger.info("   Total Costs: Usually 0.2-0.8% for typical orders")
        logger.info("   Large Orders: Can reach 1-3% due to market impact")
        
        logger.info(f"\n🎯 NEXT STEPS:")
        logger.info("   1. Integrate with real market data feeds")
        logger.info("   2. Connect to actual exchange APIs for live pricing")
        logger.info("   3. Implement dynamic fee structure updates")
        logger.info("   4. Add advanced slippage modeling algorithms")
        
    except Exception as e:
        logger.error(f"❌ Transaction cost modeling demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_cost_components():
    """Demonstrate different cost components"""
    logger.info(f"\n🧮 COST COMPONENT BREAKDOWN")
    logger.info("=" * 30)
    
    try:
        cost_model = get_transaction_cost_model()
        snapshot = MarketSnapshot(
            symbol='BTC/USDT',
            timestamp=datetime.now(),
            bid_price=49950.0,
            ask_price=50050.0,
            bid_volume=100.0,
            ask_volume=100.0,
            last_price=50000.0,
            volume_24h=20000.0,
            volatility=0.02
        )
        
        # Different order sizes
        sizes = [0.01, 0.1, 1.0]
        
        for size in sizes:
            costs = cost_model.estimate_transaction_costs(
                'BTC/USDT', size, OrderType.MARKET, snapshot
            )
            order_value = size * 50000.0
            
            logger.info(f"Order Size: {size} BTC (${order_value:,.2f}):")
            logger.info(f"   Spread Cost: ${costs.spread_cost:.4f} ({costs.spread_cost/order_value*100:.3f}%)")
            logger.info(f"   Commission: ${costs.commission_cost:.4f} ({costs.commission_cost/order_value*100:.3f}%)")
            logger.info(f"   Slippage: ${costs.slippage_cost:.4f} ({costs.slippage_cost/order_value*100:.3f}%)")
            logger.info(f"   Total: ${costs.total_cost:.4f} ({costs.cost_percentage:.3f}%)")
            logger.info("")
        
        logger.info("✅ Cost components demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ Cost components demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe 0.6 - Transaction Cost Modeling Demo")
    print("Professional transaction cost analysis system")
    print()
    
    # Run main cost modeling demo
    await demonstrate_transaction_cost_modeling()
    
    # Run components demonstration
    demonstrate_cost_components()
    
    print(f"\n🎉 TRANSACTION COST MODELING DEMO COMPLETED")
    print("Chloe 0.6 now has professional transaction cost capabilities!")

if __name__ == "__main__":
    asyncio.run(main())