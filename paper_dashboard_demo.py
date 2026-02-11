#!/usr/bin/env python3
"""
Paper Dashboard Demo for Chloe AI
Demonstrating interactive performance monitoring and visualization
"""

import asyncio
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dashboard.paper_dashboard import PaperDashboard, DashboardConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demonstrate_paper_dashboard():
    """Demonstrate paper dashboard capabilities"""
    logger.info("📊 PAPER DASHBOARD DEMO")
    logger.info("=" * 25)
    
    try:
        # Initialize paper dashboard
        logger.info("🔧 Initializing Paper Dashboard...")
        
        config = DashboardConfig(
            refresh_interval_seconds=10,  # Fast refresh for demo
            equity_curve_window_days=30,
            rolling_metrics_window=15
        )
        
        dashboard = PaperDashboard(config)
        logger.info("✅ Paper Dashboard initialized")
        logger.info(f"   Refresh interval: {config.refresh_interval_seconds} seconds")
        logger.info(f"   Equity window: {config.equity_curve_window_days} days")
        
        # Simulate initial portfolio setup
        logger.info(f"\n💰 INITIAL PORTFOLIO SETUP:")
        initial_equity = 10000.0  # $10,000 starting capital
        logger.info(f"   Starting Capital: ${initial_equity:,.2f}")
        logger.info(f"   Commission Structure: 0.1% per trade")
        logger.info(f"   Slippage Model: 0.05% market impact")
        
        # Generate sample trade data
        logger.info(f"\n📊 GENERATING SAMPLE TRADE DATA:")
        
        # Create realistic trade history
        trade_data = create_sample_trade_history(100)
        logger.info(f"   Generated {len(trade_data)} sample trades")
        logger.info(f"   Win Rate: {trade_data['is_profitable'].mean()*100:.1f}%")
        logger.info(f"   Average PnL: ${trade_data['pnl'].mean():+.2f}")
        logger.info(f"   Total PnL: ${trade_data['pnl'].sum():+.2f}")
        
        # Simulate real-time performance updates
        logger.info(f"\n📈 SIMULATING REAL-TIME PERFORMANCE UPDATES:")
        
        # Update dashboard with initial data
        initial_performance = dashboard.update_performance_data(
            equity=initial_equity + 1500,  # $1,500 profit
            daily_pnl=250.0,
            trade_data=trade_data
        )
        
        logger.info(f"   Initial Update:")
        logger.info(f"      Equity: ${initial_performance.equity:,.2f}")
        logger.info(f"      Daily PnL: ${initial_performance.daily_pnl:+.2f}")
        logger.info(f"      Total PnL: ${initial_performance.total_pnl:+.2f}")
        logger.info(f"      Win Rate: {initial_performance.win_rate*100:.1f}%")
        logger.info(f"      Sharpe Ratio: {initial_performance.sharpe_ratio:.2f}")
        logger.info(f"      Max Drawdown: {initial_performance.max_drawdown*100:.1f}%")
        
        # Simulate model performance data
        logger.info(f"\n🤖 MODEL PERFORMANCE TRACKING:")
        
        model_data = {
            'conservative_model': {
                'trades_today': 12,
                'accuracy_today': 0.65,
                'confidence_avg': 0.72,
                'risk_adjusted_return': 0.025,
                'consistency_score': 0.85,
                'active_signals': 3
            },
            'aggressive_model': {
                'trades_today': 8,
                'accuracy_today': 0.58,
                'confidence_avg': 0.81,
                'risk_adjusted_return': 0.032,
                'consistency_score': 0.72,
                'active_signals': 2
            },
            'balanced_model': {
                'trades_today': 15,
                'accuracy_today': 0.62,
                'confidence_avg': 0.76,
                'risk_adjusted_return': 0.028,
                'consistency_score': 0.78,
                'active_signals': 4
            }
        }
        
        # Update with model data
        dashboard.update_performance_data(
            equity=initial_equity + 1750,
            daily_pnl=300.0,
            model_data=model_data
        )
        
        logger.info(f"   Registered {len(dashboard.active_models)} models")
        for model_id in dashboard.active_models:
            logger.info(f"      {model_id}: Tracking performance metrics")
        
        # Simulate risk data updates
        logger.info(f"\n🛡️  RISK METRICS MONITORING:")
        
        risk_updates = [
            {'var_95': -0.025, 'current_exposure': 8500, 'leverage_ratio': 1.2, 'max_position_size': 10000},
            {'var_95': -0.032, 'current_exposure': 9200, 'leverage_ratio': 1.4, 'max_position_size': 10000},
            {'var_95': -0.018, 'current_exposure': 7800, 'leverage_ratio': 1.1, 'max_position_size': 10000}
        ]
        
        for i, risk_data in enumerate(risk_updates, 1):
            risk_metrics = dashboard.update_risk_data(risk_data)
            logger.info(f"   Risk Update {i}:")
            logger.info(f"      VaR (95%): {risk_metrics.value_at_risk*100:.2f}%")
            logger.info(f"      Current Exposure: ${risk_metrics.current_exposure:,.0f}")
            logger.info(f"      Leverage Ratio: {risk_metrics.leverage_ratio:.1f}x")
        
        # Generate equity curve
        logger.info(f"\n📈 EQUITY CURVE ANALYSIS:")
        
        equity_curve = dashboard.get_equity_curve(days=30)
        if not equity_curve.empty:
            logger.info(f"   Equity Curve Points: {len(equity_curve)}")
            logger.info(f"   Period Return: {(equity_curve['equity'].iloc[-1] - equity_curve['equity'].iloc[0]) / equity_curve['equity'].iloc[0] * 100:+.2f}%")
            logger.info(f"   Maximum Equity: ${equity_curve['equity'].max():,.2f}")
            logger.info(f"   Minimum Equity: ${equity_curve['equity'].min():,.2f}")
            logger.info(f"   Current Drawdown: {equity_curve['drawdown'].iloc[-1]*100:.2f}%")
        
        # Show rolling metrics
        logger.info(f"\n📊 ROLLING METRICS:")
        
        rolling_metrics = dashboard.get_rolling_metrics(window_days=15)
        if rolling_metrics:
            logger.info(f"   15-Day Rolling Metrics:")
            logger.info(f"      Average Equity: ${rolling_metrics['avg_equity']:,.2f}")
            logger.info(f"      Avg Daily PnL: ${rolling_metrics['avg_daily_pnl']:+.2f}")
            logger.info(f"      Equity Volatility: ${rolling_metrics['equity_volatility']:,.2f}")
            logger.info(f"      Period Return: {rolling_metrics['period_return']*100:+.2f}%")
            logger.info(f"      Current Drawdown: {rolling_metrics['current_drawdown']*100:.2f}%")
        
        # Show model performance summary
        logger.info(f"\n🤖 MODEL PERFORMANCE SUMMARY:")
        
        model_summary = dashboard.get_model_performance_summary()
        if model_summary:
            logger.info(f"   {'Model':<20} {'Trades':<8} {'Accuracy':<10} {'Confidence':<12} {'Return':<8}")
            logger.info("-" * 70)
            
            for model_id, metrics in model_summary.items():
                logger.info(f"   {model_id:<20} {metrics['total_trades']:<8} "
                           f"{metrics['avg_accuracy']*100:<10.1f}% {metrics['avg_confidence']*100:<12.1f}% "
                           f"{metrics['risk_adjusted_return']*100:<8.2f}%")
        else:
            logger.info("   No model performance data available")
        
        # Test alert system
        logger.info(f"\n🚨 ALERT SYSTEM DEMO:")
        
        # Generate some alerts by creating problematic scenarios
        logger.info("   Testing alert generation...")
        
        # High drawdown scenario
        high_dd_performance = dashboard.update_performance_data(
            equity=8500,  # 15% drawdown
            daily_pnl=-2000,
            trade_data=None
        )
        
        # Low Sharpe scenario
        low_sharpe_performance = dashboard.update_performance_data(
            equity=8600,
            daily_pnl=-500,
            trade_data=pd.DataFrame({'pnl': [-100] * 20})  # Poor performance
        )
        
        # High VaR scenario
        high_var_risk = dashboard.update_risk_data({
            'var_95': -0.15,  # 15% VaR
            'current_exposure': 9500,
            'leverage_ratio': 3.5,
            'max_position_size': 10000
        })
        
        # Show alerts
        alert_summary = dashboard.get_alert_summary()
        logger.info(f"   Alert Summary:")
        logger.info(f"      Total Alerts: {alert_summary['total_alerts']}")
        logger.info(f"      Critical Alerts: {alert_summary['critical_alerts']}")
        logger.info(f"      Warning Alerts: {alert_summary['warning_alerts']}")
        
        if alert_summary['recent_alerts']:
            logger.info(f"      Recent Alerts:")
            for alert in alert_summary['recent_alerts'][:3]:
                logger.info(f"         [{alert['severity']}] {alert['type']}: {alert['message']}")
        
        # Show complete dashboard snapshot
        logger.info(f"\n📋 DASHBOARD SNAPSHOT:")
        
        snapshot = dashboard.get_dashboard_snapshot()
        logger.info(f"   Snapshot Timestamp: {snapshot['timestamp']}")
        
        if snapshot['performance']:
            perf = snapshot['performance']
            logger.info(f"   Performance Summary:")
            logger.info(f"      Equity: ${perf['equity']:,.2f}")
            logger.info(f"      Daily PnL: ${perf['daily_pnl']:+.2f}")
            logger.info(f"      Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
            logger.info(f"      Max Drawdown: {perf['max_drawdown']*100:.1f}%")
        
        if snapshot['risk']:
            risk = snapshot['risk']
            logger.info(f"   Risk Summary:")
            logger.info(f"      VaR (95%): {risk['value_at_risk']*100:.2f}%")
            logger.info(f"      Current Exposure: ${risk['current_exposure']:,.0f}")
            logger.info(f"      Leverage: {risk['leverage_ratio']:.1f}x")
        
        logger.info(f"   Active Models: {len(snapshot['models'])}")
        logger.info(f"   Recent Alerts: {alert_summary['total_alerts']}")
        
        # Test real-time simulation
        logger.info(f"\n⚡ REAL-TIME SIMULATION:")
        
        async def mock_data_source():
            """Mock data source for real-time simulation"""
            # Simulate changing market conditions
            equity_change = np.random.normal(0, 500)
            daily_pnl = np.random.normal(100, 200)
            
            return {
                'equity': 10000 + equity_change,
                'daily_pnl': daily_pnl,
                'trade_data': None,
                'model_data': None,
                'risk_data': {
                    'var_95': np.random.uniform(-0.05, -0.01),
                    'current_exposure': np.random.uniform(7000, 9000),
                    'leverage_ratio': np.random.uniform(1.0, 2.0),
                    'max_position_size': 10000
                }
            }
        
        # Run brief real-time simulation
        logger.info("   Running 30-second real-time simulation...")
        simulation_task = asyncio.create_task(
            dashboard.start_real_time_updates(mock_data_source)
        )
        
        # Let it run for a bit
        await asyncio.sleep(30)
        simulation_task.cancel()
        
        try:
            await simulation_task
        except asyncio.CancelledError:
            logger.info("   ✅ Real-time simulation completed")
        
        # Final dashboard status
        logger.info(f"\n📊 FINAL DASHBOARD STATUS:")
        final_snapshot = dashboard.get_dashboard_snapshot()
        logger.info(f"   Total Performance Updates: {len(dashboard.performance_history)}")
        logger.info(f"   Total Risk Updates: {len(dashboard.risk_history)}")
        logger.info(f"   Active Models Tracked: {len(dashboard.active_models)}")
        logger.info(f"   Alerts Generated: {len(dashboard.alerts)}")
        
        logger.info(f"\n✅ PAPER DASHBOARD DEMO COMPLETED")
        logger.info("🚀 Key Achievements:")
        logger.info("   • Implemented comprehensive performance monitoring")
        logger.info("   • Created real-time equity curve tracking")
        logger.info("   • Built model performance analytics")
        logger.info("   • Added intelligent alert system")
        logger.info("   • Developed rolling metrics calculation")
        logger.info("   • Provided complete dashboard snapshots")
        
        logger.info(f"\n🎯 DASHBOARD FEATURES:")
        logger.info("   Real-time performance tracking")
        logger.info("   Interactive equity curve visualization")
        logger.info("   Multi-model performance comparison")
        logger.info("   Risk metrics monitoring")
        logger.info("   Automated alert generation")
        logger.info("   Rolling statistics calculation")
        logger.info("   Historical data retention")
        
        logger.info(f"\n🏁 PAPER-LEARNING ARCHITECTURE COMPLETE!")
        logger.info("All 8 phases successfully implemented:")
        logger.info("   1. ✅ Paper Execution Layer")
        logger.info("   2. ✅ Trade Journal")
        logger.info("   3. ✅ Learning Pipeline")
        logger.info("   4. ✅ Model Validation Gate")
        logger.info("   5. ✅ Shadow Mode")
        logger.info("   6. ✅ Controlled Self-Learning")
        logger.info("   7. ✅ Risk Sandbox")
        logger.info("   8. ✅ Paper Dashboard")
        
    except Exception as e:
        logger.error(f"❌ Paper dashboard demo failed: {e}")
        import traceback
        traceback.print_exc()
        raise

def create_sample_trade_history(n_trades: int) -> pd.DataFrame:
    """Create realistic sample trade history"""
    try:
        np.random.seed(42)
        
        # Generate timestamps
        start_date = datetime.now() - timedelta(days=30)
        timestamps = [start_date + timedelta(hours=i*6) for i in range(n_trades)]
        
        # Generate realistic trade outcomes
        # Mix of winning and losing trades with some correlation
        base_win_rate = 0.58
        winning_streak_prob = 0.3  # Probability of continuing a streak
        
        trades = []
        is_previous_winner = np.random.random() > 0.5
        
        for i in range(n_trades):
            # Adjust win probability based on previous result
            win_prob = base_win_rate
            if is_previous_winner:
                win_prob += winning_streak_prob * (1 - base_win_rate)
            else:
                win_prob -= winning_streak_prob * base_win_rate
            
            # Determine if this trade wins
            is_winner = np.random.random() < win_prob
            is_previous_winner = is_winner
            
            # Generate PnL amounts
            if is_winner:
                # Winning trade: log-normal distribution
                pnl_amount = np.random.lognormal(mean=2.5, sigma=0.8)
            else:
                # Losing trade: exponential distribution
                pnl_amount = -np.random.exponential(scale=150)
            
            # Add some noise and realistic constraints
            pnl_amount = max(-1000, min(2000, pnl_amount))  # Cap extremes
            pnl_amount += np.random.normal(0, 20)  # Small noise
            
            trades.append({
                'timestamp': timestamps[i],
                'symbol': np.random.choice(['BTC/USDT', 'ETH/USDT', 'SOL/USDT'], p=[0.5, 0.3, 0.2]),
                'side': 'BUY' if np.random.random() > 0.5 else 'SELL',
                'pnl': pnl_amount,
                'pnl_percentage': (pnl_amount / 10000) * 100,  # As percentage of capital
                'is_profitable': pnl_amount > 0,
                'holding_period': np.random.exponential(scale=4),  # Hours
                'commission': abs(pnl_amount) * 0.001  # 0.1% commission
            })
        
        return pd.DataFrame(trades)
        
    except Exception as e:
        logger.error(f"Sample trade history creation failed: {e}")
        return pd.DataFrame()

def demonstrate_dashboard_concepts():
    """Demonstrate key dashboard concepts"""
    logger.info(f"\n🧠 PAPER DASHBOARD CONCEPTS")
    logger.info("=" * 28)
    
    try:
        concepts = {
            "Performance Monitoring": [
                "Real-time equity tracking with historical context",
                "Daily PnL and cumulative performance metrics",
                "Risk-adjusted return calculations (Sharpe, Sortino)",
                "Drawdown analysis and maximum drawdown tracking"
            ],
            
            "Risk Analytics": [
                "Value at Risk (VaR) and Conditional VaR monitoring",
                "Position sizing and exposure tracking",
                "Leverage ratio oversight",
                "Correlation and liquidity risk assessment"
            ],
            
            "Model Intelligence": [
                "Multi-model performance comparison",
                "Accuracy and confidence tracking",
                "Risk-adjusted return per model",
                "Consistency scoring and signal analysis"
            ],
            
            "Alert System": [
                "Automated threshold-based alerts",
                "Critical vs warning severity levels",
                "Real-time risk monitoring",
                "Performance degradation detection"
            ]
        }
        
        logger.info("Key Paper Dashboard Concepts:")
        for concept, explanations in concepts.items():
            logger.info(f"\n{concept}:")
            for explanation in explanations:
                logger.info(f"   • {explanation}")
        
        logger.info("✅ Dashboard concepts demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ Dashboard concepts demo failed: {e}")

async def main():
    """Main demo function"""
    print("Chloe AI - Paper Dashboard Demo")
    print("Interactive performance monitoring and visualization")
    print()
    
    # Run main dashboard demo
    await demonstrate_paper_dashboard()
    
    # Run concepts demonstration
    demonstrate_dashboard_concepts()
    
    print(f"\n🎉 PAPER DASHBOARD DEMO COMPLETED")
    print("🎉 ALL 8 PHASES OF PAPER-LEARNING ARCHITECTURE COMPLETED!")
    print("Chloe AI now has complete professional trading system capabilities!")

if __name__ == "__main__":
    asyncio.run(main())