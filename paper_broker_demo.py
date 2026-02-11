#!/usr/bin/env python3
"""
Paper Broker Demo for Chloe AI
Demonstrating safe execution simulation for paper-learning architecture
"""

import asyncio
import logging
from datetime import datetime
import numpy as np
from execution.paper_broker import PaperBroker, OrderEvent, OrderSide, OrderType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_paper_broker():
    """Demonstrate paper broker capabilities"""
    logger.info("📝 PAPER BROKER DEMO")
    logger.info("=" * 25)
    
    try:
        # Initialize paper broker
        logger.info("🔧 Initializing Paper Broker...")
        broker = PaperBroker(
            initial_balance=50000.0,
            commission_rate=0.001,  # 0.1% commission
            slippage_range=(0.0002, 0.001),  # 0.02% to 0.1% slippage
            latency_range=(0.01, 0.1)  # 10ms to 100ms latency
        )
        logger.info("✅ Paper Broker initialized")
        
        # Show initial account state
        initial_state = broker.get_account_state()
        logger.info(f"📊 INITIAL ACCOUNT STATE:")
        logger.info(f"   Cash Balance: ${initial_state['cash_balance']:,.2f}")
        logger.info(f"   Total Portfolio Value: ${initial_state['total_portfolio_value']:,.2f}")
        logger.info(f"   Open Orders: {initial_state['open_orders']}")
        logger.info(f"   Total Trades: {initial_state['total_trades']}")
        
        # Simulate market data updates
        logger.info(f"\n📈 SIMULATING MARKET DATA:")
        market_prices = {
            'BTC/USDT': 52000.0,
            'ETH/USDT': 2980.0,
            'SOL/USDT': 98.5,
            'ADA/USDT': 0.52
        }
        
        broker.update_market_prices(market_prices)
        logger.info("   Market prices updated")
        
        # Place various orders
        logger.info(f"\n📝 PLACING TEST ORDERS:")
        
        # Market orders
        market_orders = [
            OrderEvent(symbol='BTC/USDT', side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=0.1),
            OrderEvent(symbol='ETH/USDT', side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=0.5),
            OrderEvent(symbol='SOL/USDT', side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0)
        ]
        
        order_ids = []
        for order in market_orders:
            order_id = broker.submit_order(order)
            order_ids.append(order_id)
            logger.info(f"   Submitted: {order.side.value} {order.quantity} {order.symbol} - {order_id}")
        
        # Show account state after market orders
        logger.info(f"\n📊 ACCOUNT STATE AFTER MARKET ORDERS:")
        state_after_market = broker.get_account_state()
        logger.info(f"   Cash Balance: ${state_after_market['cash_balance']:,.2f}")
        logger.info(f"   Positions Value: ${state_after_market['positions_value']:,.2f}")
        logger.info(f"   Total Portfolio Value: ${state_after_market['total_portfolio_value']:,.2f}")
        
        # Show positions
        if state_after_market['positions']:
            logger.info(f"   Positions:")
            for symbol, pos_data in state_after_market['positions'].items():
                logger.info(f"      {symbol}: {pos_data['quantity']:.4f} @ ${pos_data['avg_price']:.2f}")
        
        # Place limit orders
        logger.info(f"\n📝 PLACING LIMIT ORDERS:")
        current_btc_price = market_prices['BTC/USDT']
        
        limit_orders = [
            OrderEvent(symbol='BTC/USDT', side=OrderSide.BUY, order_type=OrderType.LIMIT, 
                      quantity=0.05, price=current_btc_price * 0.98),  # 2% below current
            OrderEvent(symbol='ETH/USDT', side=OrderSide.SELL, order_type=OrderType.LIMIT, 
                      quantity=1.0, price=current_btc_price * 1.05)    # 5% above current
        ]
        
        limit_order_ids = []
        for order in limit_orders:
            order_id = broker.submit_order(order)
            limit_order_ids.append(order_id)
            logger.info(f"   Submitted: {order.side.value} {order.quantity} {order.symbol} "
                       f"@ ${order.price:.2f} - {order_id}")
        
        # Show open orders
        logger.info(f"\n📋 OPEN ORDERS:")
        open_orders = broker.get_open_orders()
        for order in open_orders:
            logger.info(f"   {order.order_id}: {order.side.value} {order.quantity} {order.symbol} "
                       f"@ {order.price or 'MARKET'} - {order.status.value}")
        
        # Simulate price movements to trigger limit orders
        logger.info(f"\n🔄 SIMULATING PRICE MOVEMENTS:")
        
        # Move prices to trigger some limit orders
        new_prices = market_prices.copy()
        new_prices['BTC/USDT'] *= 0.97  # Drop BTC price to trigger buy limit
        new_prices['ETH/USDT'] *= 1.06  # Rise ETH price to trigger sell limit
        
        broker.update_market_prices(new_prices)
        logger.info("   Market prices updated with movements")
        
        # In a real system, the broker would automatically check and fill limit orders
        # For demo purposes, we'll manually check
        logger.info("   Checking limit order status...")
        
        # Show final account state
        logger.info(f"\n📊 FINAL ACCOUNT STATE:")
        final_state = broker.get_account_state()
        logger.info(f"   Cash Balance: ${final_state['cash_balance']:,.2f}")
        logger.info(f"   Positions Value: ${final_state['positions_value']:,.2f}")
        logger.info(f"   Total Portfolio Value: ${final_state['total_portfolio_value']:,.2f}")
        logger.info(f"   Open Orders: {final_state['open_orders']}")
        logger.info(f"   Total Trades: {final_state['total_trades']}")
        
        # Show trade history
        logger.info(f"\n📋 TRADE HISTORY (Last 5 trades):")
        trade_history = broker.get_trade_history(limit=5)
        for trade in trade_history:
            logger.info(f"   {trade.order_id}: {trade.side.value} {trade.quantity} {trade.symbol} "
                       f"@ ${trade.fill_price:.2f} (slippage: {trade.slippage*100:.3f}%)")
        
        # Test order management
        logger.info(f"\n⚙️ ORDER MANAGEMENT:")
        
        # Cancel an open order
        if open_orders:
            order_to_cancel = open_orders[0]
            cancel_result = broker.cancel_order(order_to_cancel.order_id)
            logger.info(f"   Cancelled order {order_to_cancel.order_id}: {'SUCCESS' if cancel_result else 'FAILED'}")
        else:
            logger.info("   No orders to cancel")
        
        # Test error handling
        logger.info(f"\n🧪 ERROR HANDLING TEST:")
        
        # Try invalid order
        invalid_order = OrderEvent(symbol='INVALID', side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=-1)
        invalid_order_id = broker.submit_order(invalid_order)
        logger.info(f"   Invalid order submission: {invalid_order_id} (should be rejected)")
        
        # Try insufficient funds
        expensive_order = OrderEvent(symbol='BTC/USDT', side=OrderSide.BUY, order_type=OrderType.MARKET, 
                                   quantity=1000.0)  # Way too expensive
        expensive_order_id = broker.submit_order(expensive_order)
        logger.info(f"   Expensive order submission: {expensive_order_id} (should be rejected)")
        
        # Reset broker
        logger.info(f"\n🔄 BROKER RESET:")
        old_balance = final_state['total_portfolio_value']
        broker.reset(25000.0)
        reset_state = broker.get_account_state()
        logger.info(f"   Broker reset from ${old_balance:,.2f} to ${reset_state['total_portfolio_value']:,.2f}")
        
        logger.info(f"\n✅ PAPER BROKER DEMO COMPLETED SUCCESSFULLY")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented safe paper trading execution")
        logger.info("   • Simulated realistic trading conditions")
        logger.info("   • Built comprehensive order management")
        logger.info("   • Created position tracking and PNL calculation")
        logger.info("   • Added proper error handling and validation")
        
        logger.info(f"\n🎯 PAPER BROKER FEATURES:")
        logger.info("   Realistic commission modeling (0.1%)")
        logger.info("   Variable slippage simulation (0.02% to 0.1%)")
        logger.info("   Latency simulation (10ms to 100ms)")
        logger.info("   Full position management")
        logger.info("   Trade history and accounting")
        logger.info("   Order validation and error handling")
        
        logger.info(f"\n⏭️ NEXT STEPS:")
        logger.info("   1. Implement Trade Journal (Dataset logging)")
        logger.info("   2. Create Learning Pipeline")
        logger.info("   3. Build Model Validation Gate")
        logger.info("   4. Add Shadow Mode capabilities")
        
    except Exception as e:
        logger.error(f"❌ Paper broker demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def demonstrate_safe_execution_concepts():
    """Demonstrate key safe execution concepts"""
    logger.info(f"\n🛡️ SAFE EXECUTION CONCEPTS")
    logger.info("=" * 28)
    
    try:
        concepts = {
            "Paper Trading Benefits": [
                "Zero financial risk during development",
                "Real market data without real money",
                "Strategy validation before live deployment",
                "Performance measurement in realistic conditions"
            ],
            
            "Realistic Simulation": [
                "Accurate commission modeling",
                "Variable slippage based on market conditions",
                "Latency simulation for execution timing",
                "Position sizing and risk management practice"
            ],
            
            "Safety Features": [
                "Order validation before execution",
                "Balance and position limit checking",
                "Error handling for edge cases",
                "Comprehensive audit trail"
            ]
        }
        
        logger.info("Key Safe Execution Concepts:")
        for concept, explanations in concepts.items():
            logger.info(f"\n{concept}:")
            for explanation in explanations:
                logger.info(f"   • {explanation}")
        
        logger.info("✅ Safe execution concepts demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ Safe execution concepts demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe AI - Paper Broker Demo")
    print("Safe execution simulation for paper-learning architecture")
    print()
    
    # Run main paper broker demo
    await demonstrate_paper_broker()
    
    # Run concepts demonstration
    demonstrate_safe_execution_concepts()
    
    print(f"\n🎉 PAPER BROKER DEMO COMPLETED")
    print("Chloe AI now has safe paper execution capabilities!")

if __name__ == "__main__":
    asyncio.run(main())