"""
简单移动平均策略
用于测试回测引擎的基本功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from backtesting.backtest_engine import BaseStrategy, MarketData

logger = logging.getLogger(__name__)


class SimpleMovingAverageStrategy(BaseStrategy):
    """简单移动平均交叉策略"""
    
    def __init__(self, 
                 fast_period: int = 20,
                 slow_period: int = 50,
                 position_size: float = 0.1):  # 10%的资金用于每次交易
        super().__init__()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.position_size = position_size
        
        # 价格历史缓存
        self.price_history: List[float] = []
        self.max_history = max(fast_period, slow_period) + 10
        
        # 移动平均线
        self.fast_ma: float = 0.0
        self.slow_ma: float = 0.0
        self.prev_fast_ma: float = 0.0
        self.prev_slow_ma: float = 0.0
        
        # 持仓状态
        self.current_position: float = 0.0  # 当前持仓数量
        
        # 统计信息
        self.signal_count = 0
        self.trade_signals: List[Dict] = []
    
    def initialize(self, engine):
        """初始化策略"""
        super().initialize(engine)
        logger.info(f"Initialized SMA strategy: fast={self.fast_period}, slow={self.slow_period}")
    
    def on_data(self, market_data: MarketData):
        """处理新的市场数据"""
        # 更新价格历史
        self.price_history.append(market_data.close)
        
        # 保持历史数据长度
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
        
        # 需要足够的数据计算慢移动平均线
        if len(self.price_history) < self.slow_period:
            return
        
        # 保存上一个移动平均线值
        self.prev_fast_ma = self.fast_ma
        self.prev_slow_ma = self.slow_ma
        
        # 计算移动平均线
        self.fast_ma = np.mean(self.price_history[-self.fast_period:])
        self.slow_ma = np.mean(self.price_history[-self.slow_period:])
        
        # 检查交叉信号
        self._check_signals(market_data)
    
    def _check_signals(self, market_data: MarketData):
        """检查交易信号"""
        # 需要至少两个数据点来检测交叉
        if self.prev_fast_ma == 0 or self.prev_slow_ma == 0:
            return
        
        current_price = market_data.close
        portfolio_value = self.engine.portfolio.get_portfolio_value({market_data.symbol: current_price})
        cash = self.engine.portfolio.cash
        
        # 更新当前持仓
        if market_data.symbol in self.engine.portfolio.positions:
            self.current_position = float(self.engine.portfolio.positions[market_data.symbol].quantity)
        else:
            self.current_position = 0.0
        
        # 金叉信号 (快线上穿慢线) - 买入信号
        if (self.prev_fast_ma <= self.prev_slow_ma and 
            self.fast_ma > self.slow_ma and 
            self.current_position <= 0):
            
            # 计算买入数量
            available_cash = float(cash) * self.position_size
            quantity_to_buy = available_cash / current_price
            
            if quantity_to_buy > 0.001:  # 最小交易量
                logger.info(f"Golden Cross detected at {market_data.timestamp}: "
                          f"Fast MA {self.fast_ma:.2f} > Slow MA {self.slow_ma:.2f}, "
                          f"Buying {quantity_to_buy:.6f} {market_data.symbol}")
                
                self.buy(market_data.symbol, quantity_to_buy)
                
                # 记录信号
                self.trade_signals.append({
                    'timestamp': market_data.timestamp,
                    'signal': 'BUY',
                    'price': current_price,
                    'fast_ma': self.fast_ma,
                    'slow_ma': self.slow_ma,
                    'quantity': quantity_to_buy,
                    'portfolio_value': float(portfolio_value)
                })
                
                self.signal_count += 1
        
        # 死叉信号 (快线下穿慢线) - 卖出信号
        elif (self.prev_fast_ma >= self.prev_slow_ma and 
              self.fast_ma < self.slow_ma and 
              self.current_position > 0):
            
            # 卖出所有持仓
            quantity_to_sell = self.current_position
            
            if quantity_to_sell > 0.001:  # 最小交易量
                logger.info(f"Death Cross detected at {market_data.timestamp}: "
                          f"Fast MA {self.fast_ma:.2f} < Slow MA {self.slow_ma:.2f}, "
                          f"Selling {quantity_to_sell:.6f} {market_data.symbol}")
                
                self.sell(market_data.symbol, quantity_to_sell)
                
                # 记录信号
                self.trade_signals.append({
                    'timestamp': market_data.timestamp,
                    'signal': 'SELL',
                    'price': current_price,
                    'fast_ma': self.fast_ma,
                    'slow_ma': self.slow_ma,
                    'quantity': quantity_to_sell,
                    'portfolio_value': float(portfolio_value)
                })
                
                self.signal_count += 1
    
    def get_strategy_stats(self) -> Dict:
        """获取策略统计信息"""
        return {
            'strategy_name': 'Simple Moving Average',
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'position_size': self.position_size,
            'total_signals': self.signal_count,
            'current_fast_ma': self.fast_ma,
            'current_slow_ma': self.slow_ma,
            'current_position': self.current_position,
            'trade_signals': self.trade_signals
        }


class AdaptiveMovingAverageStrategy(BaseStrategy):
    """自适应移动平均策略 - 更高级的版本"""
    
    def __init__(self, 
                 base_fast_period: int = 10,
                 base_slow_period: int = 30,
                 volatility_lookback: int = 20,
                 position_size: float = 0.2):
        super().__init__()
        self.base_fast_period = base_fast_period
        self.base_slow_period = base_slow_period
        self.volatility_lookback = volatility_lookback
        self.position_size = position_size
        
        # 价格和收益率历史
        self.price_history: List[float] = []
        self.returns_history: List[float] = []
        self.max_history = max(base_slow_period, volatility_lookback) * 2
        
        # 自适应参数
        self.adaptive_fast_period: int = base_fast_period
        self.adaptive_slow_period: int = base_slow_period
        
        # 技术指标
        self.fast_ma: float = 0.0
        self.slow_ma: float = 0.0
        self.volatility: float = 0.0
        
        # 上一次的值
        self.prev_fast_ma: float = 0.0
        self.prev_slow_ma: float = 0.0
        
        # 持仓信息
        self.current_position: float = 0.0
        self.last_trade_price: float = 0.0
        
        # 信号记录
        self.signal_count = 0
        self.trade_signals: List[Dict] = []
    
    def initialize(self, engine):
        """初始化策略"""
        super().initialize(engine)
        logger.info(f"Initialized Adaptive MA strategy: "
                   f"base_fast={self.base_fast_period}, base_slow={self.base_slow_period}")
    
    def on_data(self, market_data: MarketData):
        """处理新的市场数据"""
        current_price = market_data.close
        
        # 更新价格历史
        if self.price_history:
            returns = (current_price - self.price_history[-1]) / self.price_history[-1]
            self.returns_history.append(returns)
        
        self.price_history.append(current_price)
        
        # 保持历史数据长度
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)
        if len(self.returns_history) > self.max_history:
            self.returns_history.pop(0)
        
        # 需要足够的数据
        if len(self.price_history) < self.base_slow_period:
            return
        
        # 计算波动率并调整周期
        self._calculate_adaptive_periods()
        
        # 保存上一次的移动平均线
        self.prev_fast_ma = self.fast_ma
        self.prev_slow_ma = self.slow_ma
        
        # 计算自适应移动平均线
        self.fast_ma = np.mean(self.price_history[-self.adaptive_fast_period:])
        self.slow_ma = np.mean(self.price_history[-self.adaptive_slow_period:])
        
        # 检查信号
        self._check_adaptive_signals(market_data)
    
    def _calculate_adaptive_periods(self):
        """根据市场波动率调整移动平均线周期"""
        if len(self.returns_history) < self.volatility_lookback:
            return
        
        # 计算近期波动率
        recent_returns = self.returns_history[-self.volatility_lookback:]
        self.volatility = np.std(recent_returns) * np.sqrt(24 * 365)  # 年化波动率
        
        # 根据波动率调整周期：高波动率用更短周期，低波动率用更长周期
        volatility_multiplier = max(0.5, min(2.0, 1.0 / (self.volatility * 5 + 0.1)))
        
        self.adaptive_fast_period = int(self.base_fast_period * volatility_multiplier)
        self.adaptive_slow_period = int(self.base_slow_period * volatility_multiplier)
        
        # 确保周期合理
        self.adaptive_fast_period = max(5, min(50, self.adaptive_fast_period))
        self.adaptive_slow_period = max(10, min(100, self.adaptive_slow_period))
        
        # 确保快线周期小于慢线周期
        if self.adaptive_fast_period >= self.adaptive_slow_period:
            self.adaptive_slow_period = self.adaptive_fast_period + 5
    
    def _check_adaptive_signals(self, market_data: MarketData):
        """检查自适应交易信号"""
        if self.prev_fast_ma == 0 or self.prev_slow_ma == 0:
            return
        
        current_price = market_data.close
        portfolio_value = self.engine.portfolio.get_portfolio_value({market_data.symbol: current_price})
        
        # 更新当前持仓
        if market_data.symbol in self.engine.portfolio.positions:
            self.current_position = float(self.engine.portfolio.positions[market_data.symbol].quantity)
        else:
            self.current_position = 0.0
        
        # 增强的买入信号：金叉 + 价格动量
        if (self.prev_fast_ma <= self.prev_slow_ma and 
            self.fast_ma > self.slow_ma and 
            self.current_position <= 0):
            
            # 计算价格动量
            price_momentum = (current_price - self.price_history[-10]) / self.price_history[-10] if len(self.price_history) >= 10 else 0
            
            # 只在正动量时买入
            if price_momentum > 0:
                available_cash = float(self.engine.portfolio.cash) * self.position_size
                quantity_to_buy = available_cash / current_price
                
                if quantity_to_buy > 0.001:
                    logger.info(f"Adaptive Buy signal at {market_data.timestamp}: "
                              f"Fast MA {self.fast_ma:.2f} > Slow MA {self.slow_ma:.2f}, "
                              f"Volatility: {self.volatility:.3f}, "
                              f"Momentum: {price_momentum:.3f}")
                    
                    self.buy(market_data.symbol, quantity_to_buy)
                    self.last_trade_price = current_price
                    
                    self.trade_signals.append({
                        'timestamp': market_data.timestamp,
                        'signal': 'BUY',
                        'price': current_price,
                        'fast_ma': self.fast_ma,
                        'slow_ma': self.slow_ma,
                        'volatility': self.volatility,
                        'momentum': price_momentum,
                        'adaptive_fast_period': self.adaptive_fast_period,
                        'adaptive_slow_period': self.adaptive_slow_period
                    })
                    
                    self.signal_count += 1
        
        # 增强的卖出信号：死叉或止损
        elif self.current_position > 0:
            # 死叉信号
            death_cross = (self.prev_fast_ma >= self.prev_slow_ma and self.fast_ma < self.slow_ma)
            
            # 止损信号 (5%止损)
            stop_loss = (self.last_trade_price > 0 and 
                        (current_price - self.last_trade_price) / self.last_trade_price < -0.05)
            
            if death_cross or stop_loss:
                signal_type = "Death Cross" if death_cross else "Stop Loss"
                
                logger.info(f"Adaptive Sell signal ({signal_type}) at {market_data.timestamp}: "
                          f"Fast MA {self.fast_ma:.2f}, Slow MA {self.slow_ma:.2f}")
                
                self.sell(market_data.symbol, self.current_position)
                
                self.trade_signals.append({
                    'timestamp': market_data.timestamp,
                    'signal': 'SELL',
                    'signal_type': signal_type,
                    'price': current_price,
                    'fast_ma': self.fast_ma,
                    'slow_ma': self.slow_ma,
                    'volatility': self.volatility,
                    'entry_price': self.last_trade_price,
                    'pnl_pct': (current_price - self.last_trade_price) / self.last_trade_price if self.last_trade_price > 0 else 0
                })
                
                self.signal_count += 1
    
    def get_strategy_stats(self) -> Dict:
        """获取策略统计信息"""
        return {
            'strategy_name': 'Adaptive Moving Average',
            'base_fast_period': self.base_fast_period,
            'base_slow_period': self.base_slow_period,
            'current_adaptive_fast_period': self.adaptive_fast_period,
            'current_adaptive_slow_period': self.adaptive_slow_period,
            'current_volatility': self.volatility,
            'total_signals': self.signal_count,
            'current_position': self.current_position,
            'trade_signals': self.trade_signals
        }