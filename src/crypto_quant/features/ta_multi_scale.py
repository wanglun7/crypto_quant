import pandas as pd
import numpy as np
from typing import List


# Supported time scales
SUPPORTED_SCALES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']


def resample_ohlcv(df_1m: pd.DataFrame, target_scale: str) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV data to target scale.
    
    Args:
        df_1m: 1-minute OHLCV DataFrame
        target_scale: Target timeframe (e.g., '5m', '15m')
    
    Returns:
        Resampled DataFrame
    """
    # Parse scale to pandas frequency
    scale_map = {
        '1m': '1T',
        '5m': '5T',
        '15m': '15T',
        '30m': '30T',
        '1h': '1H',
        '4h': '4H',
        '1d': '1D'
    }
    
    if target_scale not in scale_map:
        raise ValueError(f"Unsupported scale: {target_scale}")
    
    freq = scale_map[target_scale]
    
    # Resample OHLCV data
    resampled = df_1m.resample(freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    # Remove any NaN rows from resampling
    resampled = resampled.dropna()
    
    return resampled


def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average manually."""
    return series.ewm(span=window, adjust=False).mean()


def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return series.rolling(window=window).mean()


def calculate_ema_features(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
    """Calculate EMA indicators."""
    result = pd.DataFrame(index=df.index)
    result['ema_fast'] = calculate_ema(df['close'], fast)
    result['ema_slow'] = calculate_ema(df['close'], slow)
    return result


def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate MACD indicators."""
    result = pd.DataFrame(index=df.index)
    
    # Calculate MACD manually
    ema_fast = calculate_ema(df['close'], 12)
    ema_slow = calculate_ema(df['close'], 26)
    macd_line = ema_fast - ema_slow
    macd_signal = calculate_ema(macd_line, 9)
    
    result['macd'] = macd_line
    result['macd_signal'] = macd_signal
    
    return result


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate RSI indicator manually."""
    result = pd.DataFrame(index=df.index)
    
    # Calculate price changes
    delta = df['close'].diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses using EMA
    avg_gains = calculate_ema(gains, period)
    avg_losses = calculate_ema(losses, period)
    
    # Calculate RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    result['rsi'] = rsi
    return result


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands indicators."""
    result = pd.DataFrame(index=df.index)
    
    # Calculate SMA and standard deviation
    sma = calculate_sma(df['close'], period)
    rolling_std = df['close'].rolling(window=period).std()
    
    result['bb_upper'] = sma + (rolling_std * std)
    result['bb_lower'] = sma - (rolling_std * std)
    result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / sma
    
    return result


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate ADX indicator."""
    result = pd.DataFrame(index=df.index)
    
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Calculate +DM and -DM
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
    
    # Smooth TR, +DM, -DM
    atr = calculate_ema(tr, period)
    plus_di = 100 * calculate_ema(plus_dm, period) / atr
    minus_di = 100 * calculate_ema(minus_dm, period) / atr
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = calculate_ema(dx, period)
    
    result['adx'] = adx
    return result


def generate_features(df_1m: pd.DataFrame, scales: List[str]) -> pd.DataFrame:
    """
    Generate multi-scale technical indicators from 1-minute OHLCV data.
    
    Args:
        df_1m: 1-minute OHLCV DataFrame with DatetimeIndex in UTC
        scales: List of time scales to generate features for
    
    Returns:
        DataFrame with all features, same index as input
    """
    # Validate scales
    for scale in scales:
        if scale not in SUPPORTED_SCALES:
            raise ValueError(f"Unsupported scale: {scale}. Supported scales: {SUPPORTED_SCALES}")
    
    # Store all features
    all_features = []
    
    for scale in scales:
        # Get data for this scale
        if scale == '1m':
            df_scale = df_1m.copy()
        else:
            df_scale = resample_ohlcv(df_1m, scale)
        
        # Calculate all indicators
        features = pd.DataFrame(index=df_scale.index)
        
        # EMA
        ema_features = calculate_ema_features(df_scale)
        features['ema_fast'] = ema_features['ema_fast']
        features['ema_slow'] = ema_features['ema_slow']
        
        # MACD
        macd_features = calculate_macd(df_scale)
        features['macd'] = macd_features['macd']
        features['macd_signal'] = macd_features['macd_signal']
        
        # RSI
        rsi_features = calculate_rsi(df_scale)
        features['rsi'] = rsi_features['rsi']
        
        # Bollinger Bands
        bb_features = calculate_bollinger_bands(df_scale)
        features['bb_upper'] = bb_features['bb_upper']
        features['bb_lower'] = bb_features['bb_lower']
        features['bb_width'] = bb_features['bb_width']
        
        # ADX
        adx_features = calculate_adx(df_scale)
        features['adx'] = adx_features['adx']
        
        # Forward fill NaN values within each indicator
        features = features.fillna(method='ffill')
        
        # If not 1m, resample back to 1m using forward fill
        if scale != '1m':
            # Reindex to 1m frequency and forward fill
            features = features.reindex(df_1m.index, method='ffill')
        
        # Add scale suffix to column names
        features.columns = [f"{col}_{scale}" for col in features.columns]
        
        all_features.append(features)
    
    # Combine all features
    result = pd.concat(all_features, axis=1)
    
    # Ensure result has the same index as input
    result = result.reindex(df_1m.index)
    
    # Forward fill NaN values (fills from valid values backwards in time)
    result = result.fillna(method='ffill')
    
    # Backward fill any remaining NaN at the beginning
    result = result.fillna(method='bfill')
    
    # If there are still NaN values (e.g., insufficient data for some indicators),
    # fill with a reasonable default (like 0 or mean)
    for col in result.columns:
        if result[col].isna().any():
            # For most indicators, fill with the first valid value
            first_valid = result[col].first_valid_index()
            if first_valid is not None:
                fill_value = result.loc[first_valid, col]
                result[col] = result[col].fillna(fill_value)
            else:
                # If completely invalid, fill with 0 or appropriate default
                if 'rsi' in col.lower():
                    result[col] = result[col].fillna(50)  # Neutral RSI
                elif 'bb_width' in col.lower():
                    result[col] = result[col].fillna(0.1)  # Small width
                else:
                    result[col] = result[col].fillna(0)
    
    return result