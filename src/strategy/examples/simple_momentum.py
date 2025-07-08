"""
简单动量策略示例 - 基于移动平均线交叉的策略

这是一个实际可运行的策略示例，展示如何使用框架构建策略
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ..signal_generator import SignalGenerator, SignalConfig, FactorSignal, MarketState, MarketRegime
from ...backtesting.events import EventHandler, Event, MarketDataEvent, SignalEvent, SignalType
from ...backtesting.metrics import PerformanceMetrics

logger = logging.getLogger(__name__)


class SimpleMomentumStrategy(EventHandler):
    """
    简单动量策略
    
    策略逻辑：
    1. 当快线（MA5）上穿慢线（MA20）时，生成买入信号
    2. 当快线（MA5）下穿慢线（MA20）时，生成卖出信号
    3. 结合RSI过滤信号（避免超买超卖区域）
    4. 使用成交量确认信号强度
    """
    
    def __init__(self, 
                 fast_period: int = 5,
                 slow_period: int = 20,
                 rsi_period: int = 14,
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 volume_threshold: float = 1.5,
                 min_signal_strength: float = 0.6):
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.volume_threshold = volume_threshold
        self.min_signal_strength = min_signal_strength
        
        # 历史数据存储
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        
        # 技术指标缓存
        self.ma_fast: Optional[float] = None
        self.ma_slow: Optional[float] = None
        self.rsi: Optional[float] = None
        self.last_signal_time: Optional[datetime] = None
        
        # 状态追踪
        self.position_side: Optional[str] = None  # 'long', 'short', None
        self.signals_generated = 0
        
        logger.info(f"SimpleMomentumStrategy initialized: MA({fast_period},{slow_period}), RSI({rsi_period})")
    
    def handle_event(self, event: Event) -> Optional[List[Event]]:
        """处理事件"""
        if isinstance(event, MarketDataEvent):
            return self._handle_market_data(event)
        return None
    
    def _handle_market_data(self, event: MarketDataEvent) -> List[Event]:
        """处理市场数据"""
        # 更新历史数据
        self._update_history(event)
        
        # 计算技术指标
        self._calculate_indicators()
        
        # 生成信号
        signal = self._generate_signal(event)
        
        if signal:
            self.signals_generated += 1
            self.last_signal_time = event.timestamp
            return [signal]
        
        return []
    
    def _update_history(self, event: MarketDataEvent):
        """更新历史数据"""
        self.price_history.append(float(event.close))
        self.volume_history.append(float(event.volume))
        
        # 保持历史数据长度
        max_history = max(50, self.slow_period * 2)
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
    
    def _calculate_indicators(self):
        """计算技术指标"""
        if len(self.price_history) < self.slow_period:
            return
        
        # 计算移动平均线
        self.ma_fast = np.mean(self.price_history[-self.fast_period:])
        self.ma_slow = np.mean(self.price_history[-self.slow_period:])
        
        # 计算RSI
        if len(self.price_history) >= self.rsi_period + 1:
            self.rsi = self._calculate_rsi()
    
    def _calculate_rsi(self) -> float:
        """计算RSI指标"""
        prices = np.array(self.price_history[-(self.rsi_period + 1):])
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _generate_signal(self, event: MarketDataEvent) -> Optional[SignalEvent]:
        """生成交易信号"""
        # 检查是否有足够的数据
        if (self.ma_fast is None or self.ma_slow is None or 
            len(self.price_history) < self.slow_period + 1):
            return None
        
        # 获取前一个周期的移动平均线
        if len(self.price_history) < self.slow_period + 1:
            return None
        
        prev_ma_fast = np.mean(self.price_history[-(self.fast_period + 1):-1])
        prev_ma_slow = np.mean(self.price_history[-(self.slow_period + 1):-1])
        
        # 检测金叉和死叉
        golden_cross = (prev_ma_fast <= prev_ma_slow and self.ma_fast > self.ma_slow)
        death_cross = (prev_ma_fast >= prev_ma_slow and self.ma_fast < self.ma_slow)
        
        if not (golden_cross or death_cross):
            return None
        
        # 确定信号类型
        if golden_cross:
            signal_type = SignalType.LONG
            signal_strength = self._calculate_long_signal_strength(event)
        else:  # death_cross
            signal_type = SignalType.SHORT
            signal_strength = self._calculate_short_signal_strength(event)
        
        # 检查信号强度是否满足要求
        if signal_strength < self.min_signal_strength:
            logger.debug(f"Signal strength {signal_strength:.2f} below threshold {self.min_signal_strength}")
            return None
        
        # 应用RSI过滤
        if not self._rsi_filter(signal_type):
            logger.debug(f"Signal filtered by RSI: {self.rsi:.1f}")
            return None
        
        # 应用成交量过滤
        if not self._volume_filter():
            logger.debug("Signal filtered by volume")
            return None
        
        # 避免重复信号
        if self._is_duplicate_signal(signal_type):
            logger.debug("Duplicate signal filtered")
            return None
        
        # 创建信号事件
        confidence = self._calculate_confidence(signal_type, signal_strength)
        
        signal = SignalEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            signal_type=signal_type,
            strength=Decimal(str(signal_strength)),
            confidence=Decimal(str(confidence)),
            strategy_name="SimpleMomentumStrategy",
            features={
                'ma_fast': self.ma_fast,
                'ma_slow': self.ma_slow,
                'rsi': self.rsi,
                'volume_ratio': self.volume_history[-1] / np.mean(self.volume_history[-10:]) if len(self.volume_history) >= 10 else 1.0,
                'signal_type': 'golden_cross' if golden_cross else 'death_cross'
            }
        )
        
        # 更新状态
        self.position_side = 'long' if signal_type == SignalType.LONG else 'short'
        
        logger.info(f"Signal generated: {signal_type.value} (strength: {signal_strength:.2f}, confidence: {confidence:.2f})")
        
        return signal
    
    def _calculate_long_signal_strength(self, event: MarketDataEvent) -> float:
        """计算多头信号强度"""
        strength = 0.5  # 基础强度
        
        # MA距离加权
        if self.ma_slow > 0:
            ma_diff = (self.ma_fast - self.ma_slow) / self.ma_slow
            strength += min(ma_diff * 10, 0.3)  # 最多增加0.3
        
        # 价格相对位置
        current_price = float(event.close)
        if current_price > self.ma_fast:
            strength += 0.1
        
        # 趋势加权
        if len(self.price_history) >= 3:
            recent_trend = (self.price_history[-1] - self.price_history[-3]) / self.price_history[-3]
            if recent_trend > 0:
                strength += min(recent_trend * 5, 0.2)
        
        return min(strength, 1.0)
    
    def _calculate_short_signal_strength(self, event: MarketDataEvent) -> float:
        """计算空头信号强度"""
        strength = 0.5  # 基础强度
        
        # MA距离加权
        if self.ma_slow > 0:
            ma_diff = (self.ma_slow - self.ma_fast) / self.ma_slow
            strength += min(ma_diff * 10, 0.3)  # 最多增加0.3
        
        # 价格相对位置
        current_price = float(event.close)
        if current_price < self.ma_fast:
            strength += 0.1
        
        # 趋势加权
        if len(self.price_history) >= 3:
            recent_trend = (self.price_history[-3] - self.price_history[-1]) / self.price_history[-3]
            if recent_trend > 0:
                strength += min(recent_trend * 5, 0.2)
        
        return min(strength, 1.0)
    
    def _rsi_filter(self, signal_type: SignalType) -> bool:
        """RSI过滤器"""
        if self.rsi is None:
            return True  # 如果没有RSI数据，不过滤
        
        if signal_type == SignalType.LONG:
            # 多头信号：避免超买区域
            return self.rsi < self.rsi_overbought
        else:
            # 空头信号：避免超卖区域
            return self.rsi > self.rsi_oversold
    
    def _volume_filter(self) -> bool:
        """成交量过滤器"""
        if len(self.volume_history) < 10:
            return True  # 数据不足，不过滤
        
        current_volume = self.volume_history[-1]
        avg_volume = np.mean(self.volume_history[-10:])
        
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        return volume_ratio >= self.volume_threshold
    
    def _is_duplicate_signal(self, signal_type: SignalType) -> bool:
        """检查是否为重复信号"""
        if self.position_side is None:
            return False
        
        # 如果当前持仓方向与信号方向相同，则为重复信号
        if signal_type == SignalType.LONG and self.position_side == 'long':
            return True
        if signal_type == SignalType.SHORT and self.position_side == 'short':
            return True
        
        return False
    
    def _calculate_confidence(self, signal_type: SignalType, signal_strength: float) -> float:
        """计算信号置信度"""
        confidence = signal_strength
        
        # RSI确认
        if self.rsi is not None:
            if signal_type == SignalType.LONG and self.rsi < 50:
                confidence += 0.1
            elif signal_type == SignalType.SHORT and self.rsi > 50:
                confidence += 0.1
        
        # 成交量确认
        if len(self.volume_history) >= 10:
            volume_ratio = self.volume_history[-1] / np.mean(self.volume_history[-10:])
            if volume_ratio > self.volume_threshold:
                confidence += min((volume_ratio - self.volume_threshold) * 0.2, 0.2)
        
        return min(confidence, 1.0)
    
    def get_strategy_stats(self) -> Dict:
        """获取策略统计信息"""
        return {
            'strategy_name': 'SimpleMomentumStrategy',
            'parameters': {
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'rsi_period': self.rsi_period,
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought,
                'volume_threshold': self.volume_threshold,
                'min_signal_strength': self.min_signal_strength
            },
            'current_state': {
                'ma_fast': self.ma_fast,
                'ma_slow': self.ma_slow,
                'rsi': self.rsi,
                'position_side': self.position_side,
                'signals_generated': self.signals_generated,
                'data_points': len(self.price_history)
            }
        }


def create_simple_momentum_strategy(**kwargs) -> SimpleMomentumStrategy:
    """工厂函数：创建简单动量策略"""
    return SimpleMomentumStrategy(**kwargs)