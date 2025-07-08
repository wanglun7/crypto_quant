"""
技术指标计算模块
实现RSI, MACD, Bollinger Bands等常用技术指标
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TechnicalIndicatorResult:
    """技术指标计算结果"""
    indicator_name: str
    values: Dict[str, np.ndarray]
    metadata: Dict[str, any]


class TechnicalIndicators:
    """技术指标计算器"""
    
    def __init__(self):
        self.indicators = {}
    
    def rsi(self, 
            prices: Union[pd.Series, np.ndarray], 
            period: int = 14,
            price_type: str = 'close') -> TechnicalIndicatorResult:
        """
        计算相对强弱指标 (RSI)
        
        Args:
            prices: 价格序列
            period: 计算周期，默认14
            price_type: 价格类型标识
            
        Returns:
            TechnicalIndicatorResult: 包含RSI值的结果对象
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
        
        if len(prices) < period + 1:
            raise ValueError(f"价格序列长度 {len(prices)} 不足以计算RSI (需要至少 {period + 1} 个数据点)")
        
        # 计算价格变化
        price_changes = np.diff(prices)
        
        # 分离上涨和下跌
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)
        
        # 计算初始平均值
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        # 初始化RSI数组
        rsi_values = np.full(len(prices), np.nan)
        
        # 计算RSI
        for i in range(period, len(prices)):
            if i == period:
                # 第一个RSI值
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
                rsi_values[i] = 100 - (100 / (1 + rs))
            else:
                # 指数移动平均
                avg_gain = ((avg_gain * (period - 1)) + gains[i-1]) / period
                avg_loss = ((avg_loss * (period - 1)) + losses[i-1]) / period
                
                rs = avg_gain / avg_loss if avg_loss != 0 else 100
                rsi_values[i] = 100 - (100 / (1 + rs))
        
        return TechnicalIndicatorResult(
            indicator_name="RSI",
            values={"rsi": rsi_values},
            metadata={
                "period": period,
                "price_type": price_type,
                "overbought_level": 70,
                "oversold_level": 30
            }
        )
    
    def macd(self, 
             prices: Union[pd.Series, np.ndarray],
             fast_period: int = 12,
             slow_period: int = 26,
             signal_period: int = 9,
             price_type: str = 'close') -> TechnicalIndicatorResult:
        """
        计算MACD指标
        
        Args:
            prices: 价格序列
            fast_period: 快速EMA周期，默认12
            slow_period: 慢速EMA周期，默认26
            signal_period: 信号线EMA周期，默认9
            price_type: 价格类型标识
            
        Returns:
            TechnicalIndicatorResult: 包含MACD, Signal, Histogram的结果对象
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
        
        if len(prices) < slow_period:
            raise ValueError(f"价格序列长度 {len(prices)} 不足以计算MACD (需要至少 {slow_period} 个数据点)")
        
        # 计算EMA
        def ema(data, period):
            alpha = 2.0 / (period + 1)
            ema_values = np.full(len(data), np.nan)
            ema_values[period-1] = np.mean(data[:period])
            
            for i in range(period, len(data)):
                ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
            
            return ema_values
        
        # 计算快速和慢速EMA
        fast_ema = ema(prices, fast_period)
        slow_ema = ema(prices, slow_period)
        
        # 计算MACD线
        macd_line = fast_ema - slow_ema
        
        # 计算信号线
        signal_line = ema(macd_line[slow_period-1:], signal_period)
        
        # 对齐信号线长度
        signal_aligned = np.full(len(prices), np.nan)
        signal_aligned[slow_period-1:] = signal_line
        
        # 计算柱状图
        histogram = macd_line - signal_aligned
        
        return TechnicalIndicatorResult(
            indicator_name="MACD",
            values={
                "macd": macd_line,
                "signal": signal_aligned,
                "histogram": histogram
            },
            metadata={
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
                "price_type": price_type
            }
        )
    
    def bollinger_bands(self, 
                       prices: Union[pd.Series, np.ndarray],
                       period: int = 20,
                       std_dev: float = 2.0,
                       price_type: str = 'close') -> TechnicalIndicatorResult:
        """
        计算布林带指标
        
        Args:
            prices: 价格序列
            period: 移动平均周期，默认20
            std_dev: 标准差倍数，默认2.0
            price_type: 价格类型标识
            
        Returns:
            TechnicalIndicatorResult: 包含上轨、中轨、下轨的结果对象
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
        
        if len(prices) < period:
            raise ValueError(f"价格序列长度 {len(prices)} 不足以计算布林带 (需要至少 {period} 个数据点)")
        
        # 初始化数组
        middle_band = np.full(len(prices), np.nan)
        upper_band = np.full(len(prices), np.nan)
        lower_band = np.full(len(prices), np.nan)
        
        # 计算移动平均和标准差
        for i in range(period - 1, len(prices)):
            window_data = prices[i - period + 1:i + 1]
            
            # 中轨 (简单移动平均)
            sma = np.mean(window_data)
            middle_band[i] = sma
            
            # 标准差
            std = np.std(window_data, ddof=0)
            
            # 上轨和下轨
            upper_band[i] = sma + (std_dev * std)
            lower_band[i] = sma - (std_dev * std)
        
        # 计算带宽和位置
        bandwidth = (upper_band - lower_band) / middle_band
        position = (prices - lower_band) / (upper_band - lower_band)
        
        return TechnicalIndicatorResult(
            indicator_name="Bollinger_Bands",
            values={
                "upper_band": upper_band,
                "middle_band": middle_band,
                "lower_band": lower_band,
                "bandwidth": bandwidth,
                "position": position
            },
            metadata={
                "period": period,
                "std_dev": std_dev,
                "price_type": price_type
            }
        )
    
    def stochastic_oscillator(self, 
                             high: Union[pd.Series, np.ndarray],
                             low: Union[pd.Series, np.ndarray],
                             close: Union[pd.Series, np.ndarray],
                             k_period: int = 14,
                             d_period: int = 3) -> TechnicalIndicatorResult:
        """
        计算随机振荡器 (KD指标)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            k_period: %K计算周期，默认14
            d_period: %D平滑周期，默认3
            
        Returns:
            TechnicalIndicatorResult: 包含%K和%D的结果对象
        """
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values
        if isinstance(close, pd.Series):
            close = close.values
        
        if not (len(high) == len(low) == len(close)):
            raise ValueError("高价、低价、收盘价序列长度必须相同")
        
        if len(close) < k_period:
            raise ValueError(f"数据长度 {len(close)} 不足以计算随机振荡器 (需要至少 {k_period} 个数据点)")
        
        # 初始化%K数组
        k_percent = np.full(len(close), np.nan)
        
        # 计算%K
        for i in range(k_period - 1, len(close)):
            window_high = np.max(high[i - k_period + 1:i + 1])
            window_low = np.min(low[i - k_period + 1:i + 1])
            
            if window_high != window_low:
                k_percent[i] = 100 * (close[i] - window_low) / (window_high - window_low)
            else:
                k_percent[i] = 50  # 中性值
        
        # 计算%D (对%K的移动平均)
        d_percent = np.full(len(close), np.nan)
        for i in range(k_period + d_period - 2, len(close)):
            d_percent[i] = np.mean(k_percent[i - d_period + 1:i + 1])
        
        return TechnicalIndicatorResult(
            indicator_name="Stochastic",
            values={
                "k_percent": k_percent,
                "d_percent": d_percent
            },
            metadata={
                "k_period": k_period,
                "d_period": d_period,
                "overbought_level": 80,
                "oversold_level": 20
            }
        )
    
    def williams_r(self, 
                   high: Union[pd.Series, np.ndarray],
                   low: Union[pd.Series, np.ndarray],
                   close: Union[pd.Series, np.ndarray],
                   period: int = 14) -> TechnicalIndicatorResult:
        """
        计算威廉指标 (%R)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期，默认14
            
        Returns:
            TechnicalIndicatorResult: 包含%R值的结果对象
        """
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values
        if isinstance(close, pd.Series):
            close = close.values
        
        if len(close) < period:
            raise ValueError(f"数据长度 {len(close)} 不足以计算威廉指标 (需要至少 {period} 个数据点)")
        
        # 初始化%R数组
        williams_r = np.full(len(close), np.nan)
        
        # 计算%R
        for i in range(period - 1, len(close)):
            window_high = np.max(high[i - period + 1:i + 1])
            window_low = np.min(low[i - period + 1:i + 1])
            
            if window_high != window_low:
                williams_r[i] = -100 * (window_high - close[i]) / (window_high - window_low)
            else:
                williams_r[i] = -50  # 中性值
        
        return TechnicalIndicatorResult(
            indicator_name="Williams_R",
            values={"williams_r": williams_r},
            metadata={
                "period": period,
                "overbought_level": -20,
                "oversold_level": -80
            }
        )
    
    def atr(self, 
            high: Union[pd.Series, np.ndarray],
            low: Union[pd.Series, np.ndarray],
            close: Union[pd.Series, np.ndarray],
            period: int = 14) -> TechnicalIndicatorResult:
        """
        计算平均真实波幅 (ATR)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期，默认14
            
        Returns:
            TechnicalIndicatorResult: 包含ATR值的结果对象
        """
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values
        if isinstance(close, pd.Series):
            close = close.values
        
        if len(close) < period + 1:
            raise ValueError(f"数据长度 {len(close)} 不足以计算ATR (需要至少 {period + 1} 个数据点)")
        
        # 计算真实波幅
        true_range = np.full(len(close), np.nan)
        
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]  # 当日最高最低价差
            tr2 = abs(high[i] - close[i-1])  # 当日最高价与前收盘价差
            tr3 = abs(low[i] - close[i-1])   # 当日最低价与前收盘价差
            
            true_range[i] = max(tr1, tr2, tr3)
        
        # 计算ATR (使用指数移动平均)
        atr_values = np.full(len(close), np.nan)
        
        # 第一个ATR值使用简单移动平均
        atr_values[period] = np.mean(true_range[1:period+1])
        
        # 后续ATR值使用指数移动平均
        alpha = 1.0 / period
        for i in range(period + 1, len(close)):
            atr_values[i] = alpha * true_range[i] + (1 - alpha) * atr_values[i-1]
        
        return TechnicalIndicatorResult(
            indicator_name="ATR",
            values={"atr": atr_values},
            metadata={
                "period": period,
                "smoothing_type": "exponential"
            }
        )
    
    def commodity_channel_index(self, 
                               high: Union[pd.Series, np.ndarray],
                               low: Union[pd.Series, np.ndarray],
                               close: Union[pd.Series, np.ndarray],
                               period: int = 20) -> TechnicalIndicatorResult:
        """
        计算商品通道指标 (CCI)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期，默认20
            
        Returns:
            TechnicalIndicatorResult: 包含CCI值的结果对象
        """
        if isinstance(high, pd.Series):
            high = high.values
        if isinstance(low, pd.Series):
            low = low.values
        if isinstance(close, pd.Series):
            close = close.values
        
        if len(close) < period:
            raise ValueError(f"数据长度 {len(close)} 不足以计算CCI (需要至少 {period} 个数据点)")
        
        # 计算典型价格
        typical_price = (high + low + close) / 3
        
        # 初始化CCI数组
        cci_values = np.full(len(close), np.nan)
        
        # 计算CCI
        for i in range(period - 1, len(close)):
            window_tp = typical_price[i - period + 1:i + 1]
            
            # 简单移动平均
            sma_tp = np.mean(window_tp)
            
            # 平均偏差
            mean_deviation = np.mean(np.abs(window_tp - sma_tp))
            
            # CCI计算
            if mean_deviation != 0:
                cci_values[i] = (typical_price[i] - sma_tp) / (0.015 * mean_deviation)
            else:
                cci_values[i] = 0
        
        return TechnicalIndicatorResult(
            indicator_name="CCI",
            values={"cci": cci_values},
            metadata={
                "period": period,
                "overbought_level": 100,
                "oversold_level": -100
            }
        )
    
    def momentum(self, 
                 prices: Union[pd.Series, np.ndarray],
                 period: int = 10,
                 price_type: str = 'close') -> TechnicalIndicatorResult:
        """
        计算动量指标
        
        Args:
            prices: 价格序列
            period: 计算周期，默认10
            price_type: 价格类型标识
            
        Returns:
            TechnicalIndicatorResult: 包含动量值的结果对象
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
        
        if len(prices) < period + 1:
            raise ValueError(f"价格序列长度 {len(prices)} 不足以计算动量 (需要至少 {period + 1} 个数据点)")
        
        # 计算动量
        momentum_values = np.full(len(prices), np.nan)
        
        for i in range(period, len(prices)):
            momentum_values[i] = prices[i] - prices[i - period]
        
        # 计算动量变化率
        momentum_rate = np.full(len(prices), np.nan)
        for i in range(period, len(prices)):
            if prices[i - period] != 0:
                momentum_rate[i] = (prices[i] - prices[i - period]) / prices[i - period] * 100
        
        return TechnicalIndicatorResult(
            indicator_name="Momentum",
            values={
                "momentum": momentum_values,
                "momentum_rate": momentum_rate
            },
            metadata={
                "period": period,
                "price_type": price_type
            }
        )
    
    def calculate_all_indicators(self, 
                               ohlcv_data: pd.DataFrame,
                               config: Optional[Dict[str, Dict]] = None) -> Dict[str, TechnicalIndicatorResult]:
        """
        计算所有技术指标
        
        Args:
            ohlcv_data: 包含OHLCV数据的DataFrame
            config: 指标配置字典
            
        Returns:
            Dict[str, TechnicalIndicatorResult]: 所有指标的计算结果
        """
        # 默认配置
        default_config = {
            'rsi': {'period': 14},
            'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'bollinger_bands': {'period': 20, 'std_dev': 2.0},
            'stochastic': {'k_period': 14, 'd_period': 3},
            'williams_r': {'period': 14},
            'atr': {'period': 14},
            'cci': {'period': 20},
            'momentum': {'period': 10}
        }
        
        # 合并配置
        if config:
            for indicator, params in config.items():
                if indicator in default_config:
                    default_config[indicator].update(params)
        
        results = {}
        
        try:
            # 获取价格数据
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            close = ohlcv_data['close']
            
            # 计算各项指标
            logger.info("开始计算技术指标...")
            
            # RSI
            results['rsi'] = self.rsi(close, **default_config['rsi'])
            
            # MACD
            results['macd'] = self.macd(close, **default_config['macd'])
            
            # 布林带
            results['bollinger_bands'] = self.bollinger_bands(close, **default_config['bollinger_bands'])
            
            # 随机振荡器
            results['stochastic'] = self.stochastic_oscillator(high, low, close, **default_config['stochastic'])
            
            # 威廉指标
            results['williams_r'] = self.williams_r(high, low, close, **default_config['williams_r'])
            
            # ATR
            results['atr'] = self.atr(high, low, close, **default_config['atr'])
            
            # CCI
            results['cci'] = self.commodity_channel_index(high, low, close, **default_config['cci'])
            
            # 动量
            results['momentum'] = self.momentum(close, **default_config['momentum'])
            
            logger.info(f"成功计算 {len(results)} 个技术指标")
            
        except Exception as e:
            logger.error(f"计算技术指标时出错: {e}")
            raise
        
        return results