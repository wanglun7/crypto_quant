"""
信号生成器 - 多因子信号生成和融合

该模块实现了专业的交易信号生成系统，支持：
- 多因子信号生成（技术、基本面、情绪等）
- 机器学习预测信号
- 信号融合和加权
- 自适应阈值调整
- 信号强度量化
- 市场状态识别
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import warnings

from ..backtesting.events import EventHandler, Event, MarketDataEvent, SignalEvent, SignalType

# 设置Decimal精度
getcontext().prec = 28

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市场状态枚举"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class SignalConfidence(Enum):
    """信号置信度等级"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 1.0


@dataclass
class FactorSignal:
    """单因子信号"""
    factor_name: str
    signal_type: SignalType
    strength: float  # [-1, 1]
    confidence: float  # [0, 1]
    timestamp: datetime
    raw_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 规范化信号强度和置信度
        self.strength = max(-1.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class MarketState:
    """市场状态"""
    timestamp: datetime
    regime: MarketRegime
    volatility: float
    trend_strength: float
    momentum: float
    volume_profile: Dict[str, float]
    
    # 技术指标状态
    rsi_level: float = 0.5
    bb_position: float = 0.5  # Bollinger Bands位置
    macd_signal: float = 0.0
    
    # 市场微观结构
    bid_ask_spread: float = 0.0
    order_imbalance: float = 0.0
    price_impact: float = 0.0
    
    # 资金流向
    funding_rate: float = 0.0
    open_interest_change: float = 0.0
    
    # 情绪指标
    fear_greed_index: float = 0.5
    social_sentiment: float = 0.5


@dataclass
class SignalConfig:
    """信号生成配置"""
    # 因子权重
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        'technical': 0.3,
        'momentum': 0.2,
        'mean_reversion': 0.2,
        'volume': 0.1,
        'microstructure': 0.1,
        'ml_prediction': 0.1
    })
    
    # 信号阈值
    signal_threshold: float = 0.6
    confidence_threshold: float = 0.5
    
    # 市场状态调整
    enable_regime_adjustment: bool = True
    regime_multipliers: Dict[MarketRegime, float] = field(default_factory=lambda: {
        MarketRegime.TRENDING_UP: 1.2,
        MarketRegime.TRENDING_DOWN: 1.2,
        MarketRegime.SIDEWAYS: 0.8,
        MarketRegime.HIGH_VOLATILITY: 0.9,
        MarketRegime.LOW_VOLATILITY: 1.1,
        MarketRegime.BREAKOUT: 1.3,
        MarketRegime.REVERSAL: 1.1
    })
    
    # 时间衰减
    enable_time_decay: bool = True
    signal_half_life: timedelta = timedelta(minutes=30)
    
    # 信号过滤
    min_signal_gap: timedelta = timedelta(minutes=5)
    max_signals_per_hour: int = 10
    
    # 自适应参数
    adaptive_threshold: bool = True
    lookback_window: int = 1000
    
    # 风险控制
    max_position_signals: int = 3
    enable_position_sizing: bool = True


class SignalGenerator(EventHandler):
    """信号生成器基类"""
    
    def __init__(self, config: SignalConfig, strategy_name: str = "signal_generator"):
        self.config = config
        self.strategy_name = strategy_name
        self.logger = logging.getLogger(f"{__name__}.{strategy_name}")
        
        # 历史数据
        self.price_history = []
        self.volume_history = []
        self.factor_signals_history = []
        self.market_states_history = []
        
        # 当前状态
        self.current_market_state: Optional[MarketState] = None
        self.current_signals: Dict[str, FactorSignal] = {}
        self.last_signal_time: Optional[datetime] = None
        self.recent_signals: List[SignalEvent] = []
        
        # 自适应参数
        self.adaptive_threshold = config.signal_threshold
        self.performance_tracker = []
        
        logger.info(f"Signal generator '{strategy_name}' initialized")
    
    def handle_event(self, event: Event) -> Optional[List[Event]]:
        """处理事件"""
        if isinstance(event, MarketDataEvent):
            return self._handle_market_data(event)
        return None
    
    def _handle_market_data(self, event: MarketDataEvent) -> List[Event]:
        """处理市场数据事件"""
        # 更新历史数据
        self._update_history(event)
        
        # 分析市场状态
        market_state = self._analyze_market_state(event)
        self.current_market_state = market_state
        
        # 生成因子信号
        factor_signals = self._generate_factor_signals(event, market_state)
        self.current_signals = {fs.factor_name: fs for fs in factor_signals}
        
        # 融合信号
        combined_signal = self._combine_signals(factor_signals, market_state, event)
        
        if combined_signal:
            # 过滤信号
            if self._should_generate_signal(combined_signal, event):
                self.recent_signals.append(combined_signal)
                self.last_signal_time = event.timestamp
                
                # 限制信号数量
                if len(self.recent_signals) > 100:
                    self.recent_signals = self.recent_signals[-100:]
                
                return [combined_signal]
        
        return []
    
    def _update_history(self, event: MarketDataEvent):
        """更新历史数据"""
        self.price_history.append({
            'timestamp': event.timestamp,
            'open': float(event.open),
            'high': float(event.high),
            'low': float(event.low),
            'close': float(event.close),
            'volume': float(event.volume)
        })
        
        # 限制历史数据长度
        max_history = self.config.lookback_window * 2
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
    
    def _analyze_market_state(self, event: MarketDataEvent) -> MarketState:
        """分析市场状态"""
        if len(self.price_history) < 50:
            return MarketState(
                timestamp=event.timestamp,
                regime=MarketRegime.SIDEWAYS,
                volatility=0.0,
                trend_strength=0.0,
                momentum=0.0,
                volume_profile={}
            )
        
        df = pd.DataFrame(self.price_history[-100:])
        
        # 计算趋势强度
        trend_strength = self._calculate_trend_strength(df)
        
        # 计算波动率
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(1440)  # 日化波动率
        
        # 计算动量
        momentum = self._calculate_momentum(df)
        
        # 识别市场状态
        regime = self._identify_market_regime(df, trend_strength, volatility, momentum)
        
        # 计算技术指标
        rsi_level = self._calculate_rsi(df['close'])
        bb_position = self._calculate_bb_position(df['close'])
        macd_signal = self._calculate_macd_signal(df['close'])
        
        return MarketState(
            timestamp=event.timestamp,
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            momentum=momentum,
            volume_profile=self._analyze_volume_profile(df),
            rsi_level=rsi_level,
            bb_position=bb_position,
            macd_signal=macd_signal,
            bid_ask_spread=float(event.spread) if event.spread else 0.0,
            funding_rate=float(event.funding_rate) if event.funding_rate else 0.0
        )
    
    def _generate_factor_signals(self, event: MarketDataEvent, market_state: MarketState) -> List[FactorSignal]:
        """生成各因子信号"""
        signals = []
        
        if len(self.price_history) < 20:
            return signals
        
        df = pd.DataFrame(self.price_history[-100:])
        current_price = float(event.close)
        
        # 技术指标信号
        technical_signal = self._generate_technical_signal(df, current_price, market_state)
        if technical_signal:
            signals.append(technical_signal)
        
        # 动量信号
        momentum_signal = self._generate_momentum_signal(df, current_price, market_state)
        if momentum_signal:
            signals.append(momentum_signal)
        
        # 均值回归信号
        mean_reversion_signal = self._generate_mean_reversion_signal(df, current_price, market_state)
        if mean_reversion_signal:
            signals.append(mean_reversion_signal)
        
        # 成交量信号
        volume_signal = self._generate_volume_signal(df, current_price, market_state)
        if volume_signal:
            signals.append(volume_signal)
        
        # 市场微观结构信号
        microstructure_signal = self._generate_microstructure_signal(event, market_state)
        if microstructure_signal:
            signals.append(microstructure_signal)
        
        return signals
    
    def _generate_technical_signal(self, df: pd.DataFrame, current_price: float, market_state: MarketState) -> Optional[FactorSignal]:
        """生成技术指标信号"""
        if len(df) < 20:
            return None
        
        # RSI信号
        rsi = market_state.rsi_level
        rsi_signal = 0.0
        if rsi < 0.3:
            rsi_signal = 0.8  # 超卖买入
        elif rsi > 0.7:
            rsi_signal = -0.8  # 超买卖出
        
        # MACD信号
        macd_signal = market_state.macd_signal
        
        # Bollinger Bands信号
        bb_signal = 0.0
        if market_state.bb_position < 0.2:
            bb_signal = 0.6  # 接近下轨买入
        elif market_state.bb_position > 0.8:
            bb_signal = -0.6  # 接近上轨卖出
        
        # 移动平均信号
        ma_20 = df['close'].rolling(20).mean().iloc[-1]
        ma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else ma_20
        
        ma_signal = 0.0
        if current_price > ma_20 > ma_50:
            ma_signal = 0.5
        elif current_price < ma_20 < ma_50:
            ma_signal = -0.5
        
        # 综合技术信号
        combined_signal = (rsi_signal * 0.3 + macd_signal * 0.3 + bb_signal * 0.2 + ma_signal * 0.2)
        
        # 转换为信号类型
        if combined_signal > 0.2:
            signal_type = SignalType.LONG
        elif combined_signal < -0.2:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.NEUTRAL
        
        confidence = min(abs(combined_signal), 1.0)
        
        return FactorSignal(
            factor_name='technical',
            signal_type=signal_type,
            strength=combined_signal,
            confidence=confidence,
            timestamp=market_state.timestamp,
            raw_value=combined_signal,
            metadata={
                'rsi': rsi,
                'macd': macd_signal,
                'bb_position': market_state.bb_position,
                'ma_signal': ma_signal
            }
        )
    
    def _generate_momentum_signal(self, df: pd.DataFrame, current_price: float, market_state: MarketState) -> Optional[FactorSignal]:
        """生成动量信号"""
        if len(df) < 10:
            return None
        
        # 价格动量
        price_change_5 = (current_price - df['close'].iloc[-6]) / df['close'].iloc[-6]
        price_change_20 = (current_price - df['close'].iloc[-21]) / df['close'].iloc[-21] if len(df) >= 21 else price_change_5
        
        # 成交量动量
        vol_ma_10 = df['volume'].rolling(10).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_momentum = (current_volume - vol_ma_10) / vol_ma_10 if vol_ma_10 > 0 else 0
        
        # 综合动量信号
        momentum_signal = (price_change_5 * 0.6 + price_change_20 * 0.3 + volume_momentum * 0.1)
        momentum_signal = np.tanh(momentum_signal * 10)  # 标准化到[-1, 1]
        
        # 转换为信号类型
        if momentum_signal > 0.3:
            signal_type = SignalType.LONG
        elif momentum_signal < -0.3:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.NEUTRAL
        
        confidence = min(abs(momentum_signal), 1.0)
        
        return FactorSignal(
            factor_name='momentum',
            signal_type=signal_type,
            strength=momentum_signal,
            confidence=confidence,
            timestamp=market_state.timestamp,
            raw_value=momentum_signal,
            metadata={
                'price_change_5': price_change_5,
                'price_change_20': price_change_20,
                'volume_momentum': volume_momentum
            }
        )
    
    def _generate_mean_reversion_signal(self, df: pd.DataFrame, current_price: float, market_state: MarketState) -> Optional[FactorSignal]:
        """生成均值回归信号"""
        if len(df) < 20:
            return None
        
        # 偏离均值程度
        ma_20 = df['close'].rolling(20).mean().iloc[-1]
        deviation = (current_price - ma_20) / ma_20
        
        # 波动率调整
        vol_adj_deviation = deviation / (market_state.volatility + 0.01)
        
        # 均值回归信号（偏离越大，回归信号越强）
        reversion_signal = -np.tanh(vol_adj_deviation * 5)
        
        # 转换为信号类型
        if reversion_signal > 0.4:
            signal_type = SignalType.LONG
        elif reversion_signal < -0.4:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.NEUTRAL
        
        confidence = min(abs(reversion_signal), 1.0)
        
        return FactorSignal(
            factor_name='mean_reversion',
            signal_type=signal_type,
            strength=reversion_signal,
            confidence=confidence,
            timestamp=market_state.timestamp,
            raw_value=reversion_signal,
            metadata={
                'deviation': deviation,
                'vol_adj_deviation': vol_adj_deviation,
                'ma_20': ma_20
            }
        )
    
    def _generate_volume_signal(self, df: pd.DataFrame, current_price: float, market_state: MarketState) -> Optional[FactorSignal]:
        """生成成交量信号"""
        if len(df) < 20:
            return None
        
        # 成交量异常检测
        vol_ma_20 = df['volume'].rolling(20).mean().iloc[-1]
        vol_std_20 = df['volume'].rolling(20).std().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        
        vol_zscore = (current_volume - vol_ma_20) / (vol_std_20 + 1e-8)
        
        # 价量配合
        price_change = df['close'].pct_change().iloc[-1]
        volume_price_signal = 0.0
        
        if vol_zscore > 2:  # 异常高成交量
            if price_change > 0:
                volume_price_signal = 0.7  # 放量上涨
            else:
                volume_price_signal = -0.7  # 放量下跌
        elif vol_zscore < -1:  # 异常低成交量
            volume_price_signal = -abs(price_change) * 0.5  # 缩量不利
        
        # 转换为信号类型
        if volume_price_signal > 0.3:
            signal_type = SignalType.LONG
        elif volume_price_signal < -0.3:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.NEUTRAL
        
        confidence = min(abs(volume_price_signal), 1.0)
        
        return FactorSignal(
            factor_name='volume',
            signal_type=signal_type,
            strength=volume_price_signal,
            confidence=confidence,
            timestamp=market_state.timestamp,
            raw_value=volume_price_signal,
            metadata={
                'vol_zscore': vol_zscore,
                'price_change': price_change,
                'current_volume': current_volume
            }
        )
    
    def _generate_microstructure_signal(self, event: MarketDataEvent, market_state: MarketState) -> Optional[FactorSignal]:
        """生成市场微观结构信号"""
        # 买卖价差信号
        spread_signal = 0.0
        if market_state.bid_ask_spread > 0:
            # 价差过大不利于交易
            spread_signal = -min(market_state.bid_ask_spread * 1000, 1.0)
        
        # 订单不平衡信号
        imbalance_signal = market_state.order_imbalance * 0.5
        
        # 综合微观结构信号
        micro_signal = (spread_signal * 0.3 + imbalance_signal * 0.7)
        
        # 转换为信号类型
        if micro_signal > 0.2:
            signal_type = SignalType.LONG
        elif micro_signal < -0.2:
            signal_type = SignalType.SHORT
        else:
            signal_type = SignalType.NEUTRAL
        
        confidence = min(abs(micro_signal), 1.0)
        
        return FactorSignal(
            factor_name='microstructure',
            signal_type=signal_type,
            strength=micro_signal,
            confidence=confidence,
            timestamp=market_state.timestamp,
            raw_value=micro_signal,
            metadata={
                'spread_signal': spread_signal,
                'imbalance_signal': imbalance_signal,
                'bid_ask_spread': market_state.bid_ask_spread
            }
        )
    
    def _combine_signals(self, factor_signals: List[FactorSignal], market_state: MarketState, event: MarketDataEvent) -> Optional[SignalEvent]:
        """融合多因子信号"""
        if not factor_signals:
            return None
        
        # 加权平均信号强度
        total_weight = 0.0
        weighted_strength = 0.0
        weighted_confidence = 0.0
        
        for signal in factor_signals:
            weight = self.config.factor_weights.get(signal.factor_name, 0.0)
            
            # 市场状态调整
            if self.config.enable_regime_adjustment:
                regime_multiplier = self.config.regime_multipliers.get(market_state.regime, 1.0)
                weight *= regime_multiplier
            
            # 置信度加权
            adjusted_weight = weight * signal.confidence
            
            weighted_strength += signal.strength * adjusted_weight
            weighted_confidence += signal.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        # 归一化
        final_strength = weighted_strength / total_weight
        final_confidence = weighted_confidence / total_weight
        
        # 自适应阈值调整
        if self.config.adaptive_threshold:
            threshold = self._get_adaptive_threshold()
        else:
            threshold = self.config.signal_threshold
        
        # 判断信号类型
        if abs(final_strength) < threshold or final_confidence < self.config.confidence_threshold:
            return None
        
        if final_strength > 0:
            signal_type = SignalType.LONG
        else:
            signal_type = SignalType.SHORT
        
        return SignalEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            signal_type=signal_type,
            strength=Decimal(str(abs(final_strength))),
            confidence=Decimal(str(final_confidence)),
            strategy_name=self.strategy_name,
            features={
                'market_regime': market_state.regime.value,
                'volatility': market_state.volatility,
                'trend_strength': market_state.trend_strength,
                'factor_signals': {fs.factor_name: fs.strength for fs in factor_signals}
            }
        )
    
    def _should_generate_signal(self, signal: SignalEvent, event: MarketDataEvent) -> bool:
        """判断是否应该生成信号"""
        # 检查信号间隔
        if (self.last_signal_time and 
            event.timestamp - self.last_signal_time < self.config.min_signal_gap):
            return False
        
        # 检查小时信号数量限制
        hour_ago = event.timestamp - timedelta(hours=1)
        recent_signals = [s for s in self.recent_signals if s.timestamp > hour_ago]
        if len(recent_signals) >= self.config.max_signals_per_hour:
            return False
        
        return True
    
    def _get_adaptive_threshold(self) -> float:
        """获取自适应阈值"""
        if len(self.performance_tracker) < 10:
            return self.config.signal_threshold
        
        # 基于最近表现调整阈值
        recent_performance = self.performance_tracker[-50:]
        avg_performance = np.mean(recent_performance)
        
        if avg_performance > 0:
            # 表现好，降低阈值
            adjustment = -0.1
        else:
            # 表现差，提高阈值
            adjustment = 0.1
        
        new_threshold = self.config.signal_threshold + adjustment
        return max(0.1, min(0.9, new_threshold))
    
    # 技术指标计算辅助方法
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """计算趋势强度"""
        if len(df) < 20:
            return 0.0
        
        # 使用线性回归斜率
        x = np.arange(len(df))
        y = df['close'].values
        slope = np.polyfit(x, y, 1)[0]
        
        # 标准化斜率
        price_range = df['close'].max() - df['close'].min()
        normalized_slope = slope / (price_range / len(df))
        
        return np.tanh(normalized_slope)
    
    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """计算动量"""
        if len(df) < 10:
            return 0.0
        
        # 多周期动量
        mom_5 = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) if len(df) >= 6 else 0
        mom_10 = (df['close'].iloc[-1] / df['close'].iloc[-11] - 1) if len(df) >= 11 else 0
        mom_20 = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1) if len(df) >= 21 else 0
        
        # 加权平均
        momentum = (mom_5 * 0.5 + mom_10 * 0.3 + mom_20 * 0.2)
        return np.tanh(momentum * 10)
    
    def _identify_market_regime(self, df: pd.DataFrame, trend_strength: float, volatility: float, momentum: float) -> MarketRegime:
        """识别市场状态"""
        # 趋势判断
        if abs(trend_strength) > 0.5:
            if trend_strength > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        
        # 波动率判断
        if volatility > 0.05:  # 5%日波动率
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.02:  # 2%日波动率
            return MarketRegime.LOW_VOLATILITY
        
        # 突破判断
        if len(df) >= 20:
            recent_high = df['high'].rolling(20).max().iloc[-1]
            recent_low = df['low'].rolling(20).min().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            if current_price > recent_high * 0.999:
                return MarketRegime.BREAKOUT
            elif current_price < recent_low * 1.001:
                return MarketRegime.BREAKOUT
        
        return MarketRegime.SIDEWAYS
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period + 1:
            return 0.5
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] / 100.0
    
    def _calculate_bb_position(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> float:
        """计算在Bollinger Bands中的位置"""
        if len(prices) < period:
            return 0.5
        
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        current_price = prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        if current_upper == current_lower:
            return 0.5
        
        position = (current_price - current_lower) / (current_upper - current_lower)
        return max(0.0, min(1.0, position))
    
    def _calculate_macd_signal(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """计算MACD信号"""
        if len(prices) < slow + signal:
            return 0.0
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        
        return (macd.iloc[-1] - macd_signal.iloc[-1]) / prices.iloc[-1]
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """分析成交量分布"""
        if len(df) < 10:
            return {}
        
        # 计算成交量相关指标
        vol_ma_10 = df['volume'].rolling(10).mean().iloc[-1]
        vol_current = df['volume'].iloc[-1]
        vol_ratio = vol_current / vol_ma_10 if vol_ma_10 > 0 else 1.0
        
        return {
            'volume_ratio': vol_ratio,
            'volume_trend': df['volume'].pct_change().rolling(5).mean().iloc[-1]
        }