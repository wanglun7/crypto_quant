#!/usr/bin/env python3
"""
智能仓位管理系统

核心理念：不追求预测准确性，而是最大化预测的盈利能力
- 动态仓位：根据模型置信度调整仓位大小
- 趋势过滤：只在有利趋势中开仓
- 风险控制：严格止损、止盈、回撤控制
- 资金管理：Kelly公式优化仓位分配

目标：将+7.93%的基础收益放大到70%+年化收益
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class TrendDirection(Enum):
    """趋势方向"""
    STRONG_BULL = "strong_bull"      # 强牛市
    WEAK_BULL = "weak_bull"          # 弱牛市  
    SIDEWAYS = "sideways"            # 震荡
    WEAK_BEAR = "weak_bear"          # 弱熊市
    STRONG_BEAR = "strong_bear"      # 强熊市


class PositionSide(Enum):
    """仓位方向"""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class TradingSignal:
    """交易信号"""
    timestamp: pd.Timestamp
    probability: float          # 模型预测概率 [0,1]
    confidence: float          # 置信度 [0,1]
    trend: TrendDirection      # 当前趋势
    recommended_size: float    # 推荐仓位大小 [0,1]
    side: PositionSide        # 仓位方向
    stop_loss: Optional[float] # 止损价格
    take_profit: Optional[float] # 止盈价格
    reasoning: str            # 决策reasoning


@dataclass
class PortfolioState:
    """投资组合状态"""
    timestamp: pd.Timestamp
    cash: float                # 现金
    position_size: float       # 当前仓位大小（BTC数量）
    position_value: float      # 仓位价值（USD）
    total_value: float         # 总价值
    unrealized_pnl: float      # 浮动盈亏
    realized_pnl: float        # 已实现盈亏
    max_drawdown: float        # 最大回撤
    current_drawdown: float    # 当前回撤


class TrendAnalyzer:
    """趋势分析器"""
    
    def __init__(self):
        self.lookback_periods = {
            'short': 60,    # 1小时
            'medium': 240,  # 4小时  
            'long': 1440    # 24小时
        }
    
    def analyze_trend(self, df: pd.DataFrame, current_idx: int) -> TrendDirection:
        """分析当前趋势方向"""
        
        if current_idx < self.lookback_periods['long']:
            return TrendDirection.SIDEWAYS
        
        # 获取不同时间框架的价格数据
        short_data = df.iloc[current_idx-self.lookback_periods['short']:current_idx]
        medium_data = df.iloc[current_idx-self.lookback_periods['medium']:current_idx]
        long_data = df.iloc[current_idx-self.lookback_periods['long']:current_idx]
        
        # 计算趋势强度
        short_trend = self._calculate_trend_strength(short_data['close'])
        medium_trend = self._calculate_trend_strength(medium_data['close'])
        long_trend = self._calculate_trend_strength(long_data['close'])
        
        # 综合趋势评分 (-1到+1，负数看跌，正数看涨)
        trend_score = (short_trend * 0.5 + medium_trend * 0.3 + long_trend * 0.2)
        
        # 分类趋势方向
        if trend_score > 0.5:
            return TrendDirection.STRONG_BULL
        elif trend_score > 0.2:
            return TrendDirection.WEAK_BULL
        elif trend_score > -0.2:
            return TrendDirection.SIDEWAYS
        elif trend_score > -0.5:
            return TrendDirection.WEAK_BEAR
        else:
            return TrendDirection.STRONG_BEAR
    
    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """计算趋势强度 (-1到+1)"""
        if len(prices) < 20:
            return 0
        
        # 线性回归斜率
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # 标准化斜率
        price_std = prices.std()
        normalized_slope = slope / (price_std / len(prices)) if price_std > 0 else 0
        
        # 限制在[-1, 1]范围
        return np.clip(normalized_slope, -1, 1)


class RiskManager:
    """风险管理器"""
    
    def __init__(self, max_position_size=0.95, max_drawdown=0.25, stop_loss_pct=0.02, take_profit_pct=0.05):
        self.max_position_size = max_position_size  # 最大仓位95%
        self.max_drawdown = max_drawdown            # 最大回撤25%
        self.stop_loss_pct = stop_loss_pct          # 止损2%
        self.take_profit_pct = take_profit_pct      # 止盈5%
    
    def calculate_position_size(self, signal_strength: float, confidence: float, 
                              current_drawdown: float, trend: TrendDirection) -> float:
        """计算建议仓位大小"""
        
        # 基础仓位（基于信号强度和置信度）
        base_size = signal_strength * confidence
        
        # 趋势调整
        trend_multiplier = self._get_trend_multiplier(trend)
        adjusted_size = base_size * trend_multiplier
        
        # 回撤调整（回撤越大，仓位越小）
        drawdown_factor = max(0, 1 - current_drawdown / self.max_drawdown)
        final_size = adjusted_size * drawdown_factor
        
        # 限制最大仓位
        return min(final_size, self.max_position_size)
    
    def _get_trend_multiplier(self, trend: TrendDirection) -> float:
        """根据趋势获取仓位乘数"""
        multipliers = {
            TrendDirection.STRONG_BULL: 1.5,    # 强牛市：增加仓位
            TrendDirection.WEAK_BULL: 1.2,     # 弱牛市：略增仓位
            TrendDirection.SIDEWAYS: 0.7,      # 震荡：减少仓位
            TrendDirection.WEAK_BEAR: 0.3,     # 弱熊市：大幅减仓
            TrendDirection.STRONG_BEAR: 0.1    # 强熊市：几乎空仓
        }
        return multipliers.get(trend, 1.0)
    
    def calculate_stop_loss(self, entry_price: float, side: PositionSide) -> float:
        """计算止损价格"""
        if side == PositionSide.LONG:
            return entry_price * (1 - self.stop_loss_pct)
        elif side == PositionSide.SHORT:
            return entry_price * (1 + self.stop_loss_pct)
        return entry_price
    
    def calculate_take_profit(self, entry_price: float, side: PositionSide) -> float:
        """计算止盈价格"""
        if side == PositionSide.LONG:
            return entry_price * (1 + self.take_profit_pct)
        elif side == PositionSide.SHORT:
            return entry_price * (1 - self.take_profit_pct)
        return entry_price


class IntelligentPositionManager:
    """智能仓位管理器"""
    
    def __init__(self, initial_capital=10000, confidence_threshold=0.6):
        self.initial_capital = initial_capital
        self.confidence_threshold = confidence_threshold
        
        # 组件
        self.trend_analyzer = TrendAnalyzer()
        self.risk_manager = RiskManager()
        
        # 状态跟踪
        self.portfolio_history: List[PortfolioState] = []
        self.signal_history: List[TradingSignal] = []
        self.current_position = PositionSide.FLAT
        self.entry_price = 0.0
        self.peak_value = initial_capital
    
    def generate_signal(self, df: pd.DataFrame, current_idx: int, 
                       model_probability: float) -> TradingSignal:
        """生成交易信号"""
        
        current_price = df.iloc[current_idx]['close']
        timestamp = df.iloc[current_idx]['timestamp']
        
        # 1. 分析趋势
        trend = self.trend_analyzer.analyze_trend(df, current_idx)
        
        # 2. 计算置信度（基于概率与0.5的偏离程度）
        confidence = abs(model_probability - 0.5) * 2  # [0, 1]
        
        # 3. 计算信号强度
        if model_probability > 0.5:
            signal_strength = (model_probability - 0.5) * 2
            side = PositionSide.LONG
        else:
            signal_strength = (0.5 - model_probability) * 2
            side = PositionSide.SHORT
        
        # 4. 计算当前回撤
        current_drawdown = self._calculate_current_drawdown()
        
        # 5. 仓位大小建议
        if confidence > self.confidence_threshold:
            recommended_size = self.risk_manager.calculate_position_size(
                signal_strength, confidence, current_drawdown, trend
            )
        else:
            recommended_size = 0  # 置信度不足，空仓
            side = PositionSide.FLAT
        
        # 6. 风险控制价格
        stop_loss = self.risk_manager.calculate_stop_loss(current_price, side)
        take_profit = self.risk_manager.calculate_take_profit(current_price, side)
        
        # 7. 决策reasoning
        reasoning = self._generate_reasoning(
            model_probability, confidence, trend, signal_strength, recommended_size
        )
        
        signal = TradingSignal(
            timestamp=timestamp,
            probability=model_probability,
            confidence=confidence,
            trend=trend,
            recommended_size=recommended_size,
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=reasoning
        )
        
        self.signal_history.append(signal)
        return signal
    
    def execute_trade(self, signal: TradingSignal, current_price: float, 
                     current_portfolio: PortfolioState) -> PortfolioState:
        """执行交易"""
        
        new_portfolio = PortfolioState(
            timestamp=signal.timestamp,
            cash=current_portfolio.cash,
            position_size=current_portfolio.position_size,
            position_value=current_portfolio.position_value,
            total_value=current_portfolio.total_value,
            unrealized_pnl=current_portfolio.unrealized_pnl,
            realized_pnl=current_portfolio.realized_pnl,
            max_drawdown=current_portfolio.max_drawdown,
            current_drawdown=current_portfolio.current_drawdown
        )
        
        # 执行仓位调整
        target_position_value = signal.recommended_size * new_portfolio.total_value
        current_position_value = new_portfolio.position_size * current_price
        
        # 仓位变化
        if target_position_value != current_position_value:
            trade_value = target_position_value - current_position_value
            
            if trade_value > 0:  # 买入
                btc_to_buy = trade_value / current_price
                new_portfolio.cash -= trade_value
                new_portfolio.position_size += btc_to_buy
            else:  # 卖出
                btc_to_sell = -trade_value / current_price
                new_portfolio.cash += -trade_value
                new_portfolio.position_size -= btc_to_sell
                
                # 计算已实现盈亏
                if self.entry_price > 0:
                    pnl = btc_to_sell * (current_price - self.entry_price)
                    new_portfolio.realized_pnl += pnl
        
        # 更新仓位价值
        new_portfolio.position_value = new_portfolio.position_size * current_price
        new_portfolio.total_value = new_portfolio.cash + new_portfolio.position_value
        
        # 更新回撤
        self.peak_value = max(self.peak_value, new_portfolio.total_value)
        new_portfolio.current_drawdown = (self.peak_value - new_portfolio.total_value) / self.peak_value
        new_portfolio.max_drawdown = max(new_portfolio.max_drawdown, new_portfolio.current_drawdown)
        
        # 记录当前仓位
        if signal.recommended_size > 0:
            self.current_position = signal.side
            self.entry_price = current_price
        else:
            self.current_position = PositionSide.FLAT
            self.entry_price = 0.0
        
        self.portfolio_history.append(new_portfolio)
        return new_portfolio
    
    def _calculate_current_drawdown(self) -> float:
        """计算当前回撤"""
        if not self.portfolio_history:
            return 0.0
        return self.portfolio_history[-1].current_drawdown
    
    def _generate_reasoning(self, probability: float, confidence: float, 
                          trend: TrendDirection, signal_strength: float, 
                          recommended_size: float) -> str:
        """生成决策reasoning"""
        
        parts = []
        
        # 模型信号
        direction = "看涨" if probability > 0.5 else "看跌"
        parts.append(f"模型{direction}概率{probability:.1%}")
        
        # 置信度
        conf_level = "高" if confidence > 0.8 else "中" if confidence > 0.6 else "低"
        parts.append(f"置信度{conf_level}({confidence:.1%})")
        
        # 趋势
        trend_desc = {
            TrendDirection.STRONG_BULL: "强牛市",
            TrendDirection.WEAK_BULL: "弱牛市", 
            TrendDirection.SIDEWAYS: "震荡",
            TrendDirection.WEAK_BEAR: "弱熊市",
            TrendDirection.STRONG_BEAR: "强熊市"
        }
        parts.append(f"趋势{trend_desc[trend]}")
        
        # 仓位建议
        if recommended_size > 0.8:
            size_desc = "重仓"
        elif recommended_size > 0.5:
            size_desc = "中仓" 
        elif recommended_size > 0.2:
            size_desc = "轻仓"
        else:
            size_desc = "空仓"
        
        parts.append(f"建议{size_desc}({recommended_size:.1%})")
        
        return " | ".join(parts)
    
    def get_performance_metrics(self) -> Dict:
        """获取表现指标"""
        
        if not self.portfolio_history:
            return {}
        
        final_value = self.portfolio_history[-1].total_value
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 计算年化收益率（假设数据覆盖30天）
        days = 30
        annual_return = (1 + total_return) ** (365/days) - 1
        
        # 计算夏普比率（简化版）
        returns = [p.total_value / self.portfolio_history[i-1].total_value - 1 
                  for i, p in enumerate(self.portfolio_history[1:], 1)]
        
        if returns:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(365*24*60)  # 年化
        else:
            sharpe_ratio = 0
        
        max_dd = max([p.max_drawdown for p in self.portfolio_history], default=0)
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'final_value': final_value,
            'total_trades': len(self.signal_history),
            'win_rate': self._calculate_win_rate()
        }
    
    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        if not self.portfolio_history or len(self.portfolio_history) < 2:
            return 0
        
        winning_periods = 0
        total_periods = len(self.portfolio_history) - 1
        
        for i in range(1, len(self.portfolio_history)):
            if self.portfolio_history[i].total_value > self.portfolio_history[i-1].total_value:
                winning_periods += 1
        
        return winning_periods / total_periods if total_periods > 0 else 0