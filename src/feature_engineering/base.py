"""
Base feature engineering classes
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass
from datetime import datetime
import asyncio
from concurrent.futures import ProcessPoolExecutor
import numba
from decimal import Decimal

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature calculation"""
    lookback_periods: List[int] = None
    min_data_points: int = 100
    parallel_processing: bool = True
    cache_features: bool = True
    normalize: bool = False
    handle_missing: str = 'forward_fill'  # 'forward_fill', 'interpolate', 'drop'
    
    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [5, 10, 20, 50, 100, 200]


class BaseFeatureCalculator(ABC):
    """Abstract base class for feature calculators"""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []
        self.required_columns: List[str] = []
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate features from input data"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data has required columns and sufficient rows"""
        # Check required columns
        missing_cols = set(self.required_columns) - set(data.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
            
        # Check minimum data points
        if len(data) < self.config.min_data_points:
            logger.error(f"Insufficient data points: {len(data)} < {self.config.min_data_points}")
            return False
            
        return True
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to config"""
        if self.config.handle_missing == 'forward_fill':
            return data.fillna(method='ffill').fillna(method='bfill')
        elif self.config.handle_missing == 'interpolate':
            return data.interpolate(method='linear', limit_direction='both')
        elif self.config.handle_missing == 'drop':
            return data.dropna()
        else:
            return data
    
    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features if configured"""
        if not self.config.normalize:
            return features
            
        # Use robust scaling (less sensitive to outliers)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            q75, q25 = np.percentile(features[col].dropna(), [75, 25])
            iqr = q75 - q25
            
            if iqr > 0:
                median = features[col].median()
                features[col] = (features[col] - median) / iqr
                
        return features
    
    def add_feature_prefix(self, features: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Add prefix to feature names"""
        features.columns = [f"{prefix}_{col}" for col in features.columns]
        return features


class MarketMicrostructureCalculator(BaseFeatureCalculator):
    """Calculate market microstructure features"""
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(config)
        self.required_columns = ['bid', 'ask', 'bid_size', 'ask_size', 'price', 'volume']
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate microstructure features"""
        if not self.validate_data(data):
            return pd.DataFrame()
            
        features = pd.DataFrame(index=data.index)
        
        # Basic spread metrics
        features['spread'] = data['ask'] - data['bid']
        features['spread_pct'] = features['spread'] / data['bid'] * 100
        features['mid_price'] = (data['bid'] + data['ask']) / 2
        
        # Order book imbalance
        features['book_imbalance'] = self._calculate_book_imbalance(
            data['bid_size'], data['ask_size']
        )
        
        # Weighted mid price
        features['weighted_mid_price'] = self._calculate_weighted_mid_price(
            data['bid'], data['ask'], data['bid_size'], data['ask_size']
        )
        
        # Price pressure
        features['buy_pressure'] = self._calculate_price_pressure(
            data['price'], data['volume'], side='buy'
        )
        features['sell_pressure'] = self._calculate_price_pressure(
            data['price'], data['volume'], side='sell'
        )
        
        # Microstructure volatility
        features['micro_volatility'] = self._calculate_micro_volatility(
            features['mid_price']
        )
        
        # Order flow toxicity (simplified VPIN)
        features['toxicity'] = self._calculate_toxicity(
            data['price'], data['volume']
        )
        
        # Multi-period features
        for period in self.config.lookback_periods:
            # Rolling spread
            features[f'spread_mean_{period}'] = features['spread'].rolling(period).mean()
            features[f'spread_std_{period}'] = features['spread'].rolling(period).std()
            
            # Rolling imbalance
            features[f'imbalance_mean_{period}'] = features['book_imbalance'].rolling(period).mean()
            
            # Pressure ratios
            features[f'pressure_ratio_{period}'] = (
                features['buy_pressure'].rolling(period).sum() /
                features['sell_pressure'].rolling(period).sum().replace(0, 1)
            )
        
        return self.normalize_features(features)
    
    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_book_imbalance(bid_sizes: np.ndarray, ask_sizes: np.ndarray) -> np.ndarray:
        """Calculate order book imbalance"""
        total = bid_sizes + ask_sizes
        imbalance = np.where(total > 0, (bid_sizes - ask_sizes) / total, 0)
        return imbalance
    
    def _calculate_weighted_mid_price(self, bid: pd.Series, ask: pd.Series,
                                     bid_size: pd.Series, ask_size: pd.Series) -> pd.Series:
        """Calculate size-weighted mid price"""
        total_size = bid_size + ask_size
        weighted_price = (bid * ask_size + ask * bid_size) / total_size.replace(0, 1)
        return weighted_price
    
    def _calculate_price_pressure(self, price: pd.Series, volume: pd.Series,
                                 side: str) -> pd.Series:
        """Calculate buy/sell pressure"""
        price_diff = price.diff()
        
        if side == 'buy':
            pressure = volume.where(price_diff > 0, 0)
        else:
            pressure = volume.where(price_diff < 0, 0)
            
        return pressure
    
    def _calculate_micro_volatility(self, mid_price: pd.Series, 
                                   window: int = 100) -> pd.Series:
        """Calculate microstructure volatility"""
        log_returns = np.log(mid_price / mid_price.shift(1))
        volatility = log_returns.rolling(window).std() * np.sqrt(252 * 24 * 60 * 60)  # Annualized
        return volatility
    
    def _calculate_toxicity(self, price: pd.Series, volume: pd.Series,
                           bucket_size: int = 50) -> pd.Series:
        """Simplified VPIN (Volume-synchronized Probability of Informed Trading)"""
        # Calculate dollar volume
        dollar_volume = price * volume
        
        # Create volume buckets
        cumsum_volume = dollar_volume.cumsum()
        bucket_indices = (cumsum_volume // bucket_size).astype(int)
        
        # Calculate buy/sell imbalance per bucket
        price_diff = price.diff()
        buy_volume = dollar_volume.where(price_diff > 0, 0)
        sell_volume = dollar_volume.where(price_diff < 0, 0)
        
        # Group by bucket and calculate toxicity
        df = pd.DataFrame({
            'bucket': bucket_indices,
            'buy': buy_volume,
            'sell': sell_volume
        })
        
        bucket_toxicity = df.groupby('bucket').apply(
            lambda x: abs(x['buy'].sum() - x['sell'].sum()) / (x['buy'].sum() + x['sell'].sum() + 1)
        )
        
        # Map back to original index
        toxicity = bucket_indices.map(bucket_toxicity).fillna(0)
        
        return pd.Series(toxicity, index=price.index)


class TechnicalIndicatorCalculator(BaseFeatureCalculator):
    """Calculate technical analysis indicators"""
    
    def __init__(self, config: FeatureConfig = None):
        super().__init__(config)
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        if not self.validate_data(data):
            return pd.DataFrame()
            
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Volatility
        for period in self.config.lookback_periods:
            features[f'volatility_{period}'] = features['returns'].rolling(period).std()
            features[f'realized_vol_{period}'] = np.sqrt(
                (features['log_returns'] ** 2).rolling(period).sum()
            )
        
        # Moving averages
        for period in self.config.lookback_periods:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            
        # Price relative to moving averages
        for period in self.config.lookback_periods:
            features[f'price_to_sma_{period}'] = data['close'] / features[f'sma_{period}']
            
        # RSI
        for period in [14, 21, 42]:
            features[f'rsi_{period}'] = self._calculate_rsi(data['close'], period)
            
        # MACD
        features = pd.concat([features, self._calculate_macd(data['close'])], axis=1)
        
        # Bollinger Bands
        for period in [20, 50]:
            bb_features = self._calculate_bollinger_bands(data['close'], period)
            features = pd.concat([features, bb_features], axis=1)
            
        # Volume features
        features['volume_sma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_trend'] = data['volume'].rolling(10).mean().pct_change(5)
        
        # On-Balance Volume (OBV)
        features['obv'] = self._calculate_obv(data['close'], data['volume'])
        
        # Average True Range (ATR)
        for period in [14, 20]:
            features[f'atr_{period}'] = self._calculate_atr(
                data['high'], data['low'], data['close'], period
            )
            
        return self.normalize_features(features)
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        
        rs = avg_gain / avg_loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate MACD indicators"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        return pd.DataFrame({
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram
        })
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        
        return pd.DataFrame({
            f'bb_upper_{period}': upper,
            f'bb_lower_{period}': lower,
            f'bb_width_{period}': upper - lower,
            f'bb_position_{period}': (prices - lower) / (upper - lower).replace(0, 1)
        })
    
    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume"""
        price_diff = prices.diff()
        obv = volume.where(price_diff > 0, -volume).where(price_diff != 0, 0).cumsum()
        return obv
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, 
                      close: pd.Series, period: int) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr