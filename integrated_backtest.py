"""
完整的AI驱动回测引擎
基于真实数据、AI模型预测和完整的策略执行
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass
import json
from pathlib import Path

# 导入AI策略管道
from ai_strategy_simple import SimpleAIStrategy, fetch_real_market_data, SignalEvent, SignalType

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """交易记录"""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: Decimal
    price: Decimal
    commission: Decimal
    pnl: Optional[Decimal] = None
    signal_source: str = ""
    

@dataclass 
class Portfolio:
    """投资组合状态"""
    timestamp: datetime
    cash: Decimal
    position: Decimal  # BTC数量
    market_value: Decimal
    total_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal


class AIBacktestEngine:
    """AI驱动的回测引擎"""
    
    def __init__(self, initial_capital: Decimal = Decimal('100000')):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        
        # 投资组合状态
        self.cash = initial_capital
        self.btc_position = Decimal('0')
        
        # 交易记录
        self.trades: List[Trade] = []
        self.portfolio_history: List[Portfolio] = []
        
        # AI策略
        self.ai_strategy = SimpleAIStrategy()
        
        # 交易参数
        self.commission_rate = Decimal('0.001')  # 0.1% 手续费
        self.slippage_rate = Decimal('0.0005')   # 0.05% 滑点
        self.position_size_pct = Decimal('0.95')  # 95%资金利用率
        
        logger.info(f"AI Backtest Engine initialized with ${initial_capital:,}")
    
    def run_backtest(self, market_data: pd.DataFrame, 
                     window_size: int = 200,
                     rebalance_freq: int = 1) -> Dict[str, Any]:
        """
        运行AI驱动的回测
        
        Args:
            market_data: 市场数据
            window_size: 用于AI预测的数据窗口大小  
            rebalance_freq: 重新平衡频率(小时)
            
        Returns:
            回测结果字典
        """
        logger.info("="*60)
        logger.info("STARTING AI-DRIVEN BACKTEST")
        logger.info("="*60)
        
        logger.info(f"Data period: {market_data.index[0]} to {market_data.index[-1]}")
        logger.info(f"Total data points: {len(market_data):,}")
        logger.info(f"Window size: {window_size}")
        logger.info(f"Initial capital: ${self.initial_capital:,}")
        
        signals_generated = 0
        trades_executed = 0
        
        # 遍历数据进行回测
        for i in range(window_size, len(market_data), rebalance_freq):
            current_timestamp = market_data.index[i]
            current_data = market_data.iloc[i-window_size:i+1]
            current_price = market_data.iloc[i]['close']
            
            # 每100个数据点记录一次进度
            if i % 100 == 0:
                progress = (i - window_size) / (len(market_data) - window_size) * 100
                logger.info(f"Progress: {progress:.1f}% | "
                          f"Price: ${current_price:,.2f} | "
                          f"Portfolio: ${self._get_portfolio_value(current_price):,.2f}")
            
            try:
                # 1. 使用AI策略生成信号
                signal = self.ai_strategy.process_market_data(current_data)
                
                if signal:
                    signals_generated += 1
                    logger.info(f"Signal generated: {signal.signal_type.value} "
                              f"(strength: {signal.strength:.3f}, confidence: {signal.confidence:.3f})")
                    
                    # 2. 执行交易
                    trade_executed = self._execute_signal(signal, current_price, current_timestamp)
                    if trade_executed:
                        trades_executed += 1
                
                # 3. 记录投资组合状态
                self._record_portfolio_state(current_timestamp, current_price)
                
            except Exception as e:
                logger.error(f"Error at timestamp {current_timestamp}: {e}")
                continue
        
        # 最终平仓
        self._close_all_positions(market_data.iloc[-1]['close'], market_data.index[-1])
        
        # 计算和返回结果
        results = self._calculate_backtest_results()
        
        logger.info("="*60)
        logger.info("BACKTEST COMPLETED")
        logger.info("="*60)
        logger.info(f"Signals generated: {signals_generated}")
        logger.info(f"Trades executed: {trades_executed}")
        logger.info(f"Final portfolio value: ${results['final_value']:,.2f}")
        logger.info(f"Total return: {results['total_return']:.2%}")
        logger.info(f"Annualized return: {results['annualized_return']:.2%}")
        logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max drawdown: {results['max_drawdown']:.2%}")
        
        return results
    
    def _execute_signal(self, signal: SignalEvent, current_price: float, 
                       timestamp: datetime) -> bool:
        """执行交易信号"""
        try:
            current_price_decimal = Decimal(str(current_price))
            
            if signal.signal_type == SignalType.LONG:
                # 买入信号
                if self.cash > 0:
                    return self._execute_buy(current_price_decimal, timestamp, signal)
                    
            elif signal.signal_type == SignalType.SHORT:
                # 卖出信号  
                if self.btc_position > 0:
                    return self._execute_sell(current_price_decimal, timestamp, signal)
                    
            elif signal.signal_type == SignalType.EXIT:
                # 平仓信号
                if self.btc_position > 0:
                    return self._execute_sell(current_price_decimal, timestamp, signal)
                    
            return False
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
    
    def _execute_buy(self, price: Decimal, timestamp: datetime, 
                    signal: SignalEvent) -> bool:
        """执行买入"""
        # 计算买入数量（基于信号强度调整仓位大小）
        position_multiplier = min(float(signal.strength) * 2, 1.0)  # 最大100%
        trade_amount = self.cash * self.position_size_pct * Decimal(str(position_multiplier))
        
        # 计算滑点后价格
        slipped_price = price * (1 + self.slippage_rate)
        
        # 计算手续费
        commission = trade_amount * self.commission_rate
        
        # 实际可买入的BTC数量
        net_amount = trade_amount - commission
        btc_quantity = net_amount / slipped_price
        
        if btc_quantity <= 0:
            return False
        
        # 执行交易
        self.cash -= trade_amount
        self.btc_position += btc_quantity
        
        # 记录交易
        trade = Trade(
            timestamp=timestamp,
            symbol="BTCUSDT",
            side="buy",
            quantity=btc_quantity,
            price=slipped_price,
            commission=commission,
            signal_source=f"AI_{signal.strategy_name}"
        )
        self.trades.append(trade)
        
        logger.info(f"BUY: {btc_quantity:.6f} BTC @ ${slipped_price:,.2f} "
                   f"(Amount: ${trade_amount:,.2f}, Commission: ${commission:.2f})")
        
        return True
    
    def _execute_sell(self, price: Decimal, timestamp: datetime,
                     signal: SignalEvent) -> bool:
        """执行卖出"""
        if self.btc_position <= 0:
            return False
        
        # 计算卖出数量（基于信号强度，部分或全部卖出）
        sell_ratio = min(float(signal.strength) * 2, 1.0)  # 最大100%
        sell_quantity = self.btc_position * Decimal(str(sell_ratio))
        
        # 计算滑点后价格
        slipped_price = price * (1 - self.slippage_rate)
        
        # 计算交易金额和手续费
        gross_amount = sell_quantity * slipped_price
        commission = gross_amount * self.commission_rate
        net_amount = gross_amount - commission
        
        # 计算PnL（需要找到对应的买入成本）
        pnl = self._calculate_pnl(sell_quantity, slipped_price)
        
        # 执行交易
        self.cash += net_amount
        self.btc_position -= sell_quantity
        
        # 记录交易
        trade = Trade(
            timestamp=timestamp,
            symbol="BTCUSDT",
            side="sell",
            quantity=sell_quantity,
            price=slipped_price,
            commission=commission,
            pnl=pnl,
            signal_source=f"AI_{signal.strategy_name}"
        )
        self.trades.append(trade)
        
        logger.info(f"SELL: {sell_quantity:.6f} BTC @ ${slipped_price:,.2f} "
                   f"(Amount: ${net_amount:,.2f}, PnL: ${pnl:.2f})")
        
        return True
    
    def _calculate_pnl(self, sell_quantity: Decimal, sell_price: Decimal) -> Decimal:
        """计算PnL（简化FIFO计算）"""
        # 简化计算：使用平均成本
        total_cost = Decimal('0')
        total_quantity = Decimal('0')
        
        for trade in self.trades:
            if trade.side == 'buy':
                total_cost += trade.quantity * trade.price
                total_quantity += trade.quantity
        
        if total_quantity > 0:
            avg_cost = total_cost / total_quantity
            pnl = sell_quantity * (sell_price - avg_cost)
        else:
            pnl = Decimal('0')
        
        return pnl
    
    def _get_portfolio_value(self, current_price: float) -> Decimal:
        """计算投资组合总价值"""
        current_price_decimal = Decimal(str(current_price))
        btc_value = self.btc_position * current_price_decimal
        return self.cash + btc_value
    
    def _record_portfolio_state(self, timestamp: datetime, current_price: float):
        """记录投资组合状态"""
        current_price_decimal = Decimal(str(current_price))
        btc_value = self.btc_position * current_price_decimal
        total_value = self.cash + btc_value
        
        # 计算未实现PnL
        unrealized_pnl = total_value - self.initial_capital
        
        # 计算已实现PnL
        realized_pnl = sum(trade.pnl or Decimal('0') for trade in self.trades)
        
        portfolio = Portfolio(
            timestamp=timestamp,
            cash=self.cash,
            position=self.btc_position,
            market_value=btc_value,
            total_value=total_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl
        )
        
        self.portfolio_history.append(portfolio)
    
    def _close_all_positions(self, final_price: float, timestamp: datetime):
        """最终平仓"""
        if self.btc_position > 0:
            final_price_decimal = Decimal(str(final_price))
            
            # 创建平仓信号
            class ExitSignal:
                signal_type = SignalType.EXIT
                strength = Decimal('1.0')
                strategy_name = "FINAL_EXIT"
            
            self._execute_sell(final_price_decimal, timestamp, ExitSignal())
            logger.info(f"Final position closed at ${final_price:,.2f}")
    
    def _calculate_backtest_results(self) -> Dict[str, Any]:
        """计算回测结果"""
        if not self.portfolio_history:
            return {}
        
        # 提取净值曲线
        portfolio_df = pd.DataFrame([
            {
                'timestamp': p.timestamp,
                'total_value': float(p.total_value),
                'cash': float(p.cash),
                'btc_value': float(p.market_value),
                'unrealized_pnl': float(p.unrealized_pnl)
            }
            for p in self.portfolio_history
        ])
        
        portfolio_df.set_index('timestamp', inplace=True)
        
        # 基本指标
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value - float(self.initial_capital)) / float(self.initial_capital)
        
        # 计算收益率序列
        returns = portfolio_df['total_value'].pct_change().dropna()
        
        # 年化收益率
        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0
        
        # 风险指标
        volatility = returns.std() * np.sqrt(24 * 365)  # 年化波动率
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        
        # 最大回撤
        cumulative_max = portfolio_df['total_value'].expanding().max()
        drawdown = (portfolio_df['total_value'] - cumulative_max) / cumulative_max
        max_drawdown = abs(drawdown.min())
        
        # 交易统计
        trade_df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'side': t.side,
                'quantity': float(t.quantity),
                'price': float(t.price),
                'commission': float(t.commission),
                'pnl': float(t.pnl) if t.pnl else 0
            }
            for t in self.trades
        ])
        
        winning_trades = len(trade_df[trade_df['pnl'] > 0]) if not trade_df.empty else 0
        total_trades = len(trade_df)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 买入持有策略对比
        if not portfolio_df.empty:
            first_price = portfolio_df.index[0]  # 需要价格数据来计算
            last_price = portfolio_df.index[-1]
            # 这里简化计算，实际需要价格数据
            buy_hold_return = 0  # 暂时设为0，需要价格数据
        
        results = {
            'initial_capital': float(self.initial_capital),
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'portfolio_history': portfolio_df.to_dict('records'),
            'trades': trade_df.to_dict('records') if not trade_df.empty else [],
            'ai_signals_used': len([t for t in self.trades if 'AI_' in t.signal_source])
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """保存回测结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_backtest_results_{timestamp}.json"
        
        results_dir = Path("./backtest_results")
        results_dir.mkdir(exist_ok=True)
        
        filepath = results_dir / filename
        
        # 转换Decimal为float以便JSON序列化
        def decimal_to_float(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=decimal_to_float)
        
        logger.info(f"Results saved to {filepath}")


def run_ai_backtest_demo():
    """运行AI回测演示"""
    logger.info("🚀 Starting AI Backtest Demo")
    
    # 1. 获取历史数据
    logger.info("Fetching market data...")
    market_data = fetch_real_market_data("BTCUSDT", days=14)  # 2周数据用于演示
    
    if market_data.empty:
        logger.error("No market data available")
        return
    
    # 2. 初始化回测引擎
    engine = AIBacktestEngine(initial_capital=Decimal('50000'))
    
    # 3. 运行回测
    results = engine.run_backtest(
        market_data=market_data,
        window_size=100,  # 使用100个数据点的窗口
        rebalance_freq=6   # 每6小时重新评估
    )
    
    # 4. 保存结果
    engine.save_results(results)
    
    # 5. 显示摘要
    if results:
        logger.info("\n" + "="*50)
        logger.info("🎯 AI BACKTEST SUMMARY")
        logger.info("="*50)
        logger.info(f"💰 Initial Capital: ${results['initial_capital']:,.2f}")
        logger.info(f"💰 Final Value: ${results['final_value']:,.2f}")
        logger.info(f"📈 Total Return: {results['total_return']:.2%}")
        logger.info(f"📈 Annualized Return: {results['annualized_return']:.2%}")
        logger.info(f"⚡ Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"📉 Max Drawdown: {results['max_drawdown']:.2%}")
        logger.info(f"🎯 Win Rate: {results['win_rate']:.2%}")
        logger.info(f"📊 Total Trades: {results['total_trades']}")
        logger.info(f"🤖 AI Signals Used: {results['ai_signals_used']}")
        
        return True
    
    return False


if __name__ == "__main__":
    success = run_ai_backtest_demo()
    if success:
        print("\n✅ AI Backtest Demo completed successfully!")
    else:
        print("\n❌ AI Backtest Demo failed!")