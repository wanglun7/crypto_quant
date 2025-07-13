"""Technical indicators using pandas."""

import pandas as pd
import numpy as np
from typing import Union, Tuple


def simple_moving_average(data: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=window).mean()


def exponential_moving_average(data: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(span=window).mean()


def rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = exponential_moving_average(data, fast)
    ema_slow = exponential_moving_average(data, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = simple_moving_average(data, window)
    std = data.rolling(window=window).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()


def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return k_percent, d_percent


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    
    return -100 * ((highest_high - close) / (highest_high - lowest_low))


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    direction = np.where(close > close.shift(1), 1, 
                np.where(close < close.shift(1), -1, 0))
    return (direction * volume).cumsum()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def add_cnn_lstm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 18 technical indicators used in Alonso-Monsalve 2020 CNN-LSTM paper.
    
    Returns:
        DataFrame with 18 technical indicators as features
    """
    result = df.copy()
    
    # === Price Indicators (6 features) ===
    # 1. SMA_5: 5-period Simple Moving Average
    result['sma_5'] = simple_moving_average(df['close'], 5)
    
    # 2. SMA_20: 20-period Simple Moving Average  
    result['sma_20'] = simple_moving_average(df['close'], 20)
    
    # 3. EMA_12: 12-period Exponential Moving Average
    result['ema_12'] = exponential_moving_average(df['close'], 12)
    
    # 4. EMA_26: 26-period Exponential Moving Average
    result['ema_26'] = exponential_moving_average(df['close'], 26)
    
    # 5. Price_Change: Price change rate
    result['price_change'] = df['close'].pct_change()
    
    # 6. Price_Volatility: 20-period rolling standard deviation
    result['price_volatility'] = df['close'].rolling(window=20).std()
    
    # === Momentum Indicators (6 features) ===
    # 7. RSI_14: 14-period Relative Strength Index
    result['rsi_14'] = rsi(df['close'], 14)
    
    # 8-10. MACD components
    macd_line, signal_line, histogram = macd(df['close'])
    result['macd'] = macd_line  # 8. MACD line
    result['macd_signal'] = signal_line  # 9. MACD signal line
    result['macd_histogram'] = histogram  # 10. MACD histogram
    
    # 11. Williams_R: Williams %R
    result['williams_r'] = williams_r(df['high'], df['low'], df['close'])
    
    # 12. Stochastic_K: Stochastic %K
    stoch_k, stoch_d = stochastic_oscillator(df['high'], df['low'], df['close'])
    result['stochastic_k'] = stoch_k
    
    # === Volume Indicators (3 features) ===
    # 13. Volume_SMA: Volume simple moving average
    result['volume_sma'] = simple_moving_average(df['volume'], 20)
    
    # 14. Volume_Ratio: Current volume / average volume
    result['volume_ratio'] = df['volume'] / result['volume_sma']
    
    # 15. OBV: On-Balance Volume
    result['obv'] = obv(df['close'], df['volume'])
    
    # === Volatility Indicators (3 features) ===
    # 16-17. Bollinger Bands
    bb_upper, bb_middle, bb_lower = bollinger_bands(df['close'])
    result['bb_upper'] = bb_upper  # 16. Bollinger Band Upper
    result['bb_lower'] = bb_lower  # 17. Bollinger Band Lower
    
    # 18. ATR: Average True Range
    result['atr'] = atr(df['high'], df['low'], df['close'])
    
    return result


def get_cnn_lstm_feature_columns() -> list:
    """Get the list of 18 feature column names for CNN-LSTM model."""
    return [
        # Price indicators (6)
        'sma_5', 'sma_20', 'ema_12', 'ema_26', 'price_change', 'price_volatility',
        # Momentum indicators (6)  
        'rsi_14', 'macd', 'macd_signal', 'macd_histogram', 'williams_r', 'stochastic_k',
        # Volume indicators (3)
        'volume_sma', 'volume_ratio', 'obv',
        # Volatility indicators (3)
        'bb_upper', 'bb_lower', 'atr'
    ]


def prepare_cnn_lstm_data(df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for CNN-LSTM model training.
    
    Args:
        df: DataFrame with OHLCV data and technical indicators
        sequence_length: Number of time steps to look back
        
    Returns:
        X: Feature sequences (samples, timesteps, features)
        y: Binary labels (1 if next minute price increases, 0 otherwise)
    """
    # Add technical indicators
    data = add_cnn_lstm_features(df)
    
    # Get feature columns
    feature_cols = get_cnn_lstm_feature_columns()
    
    # Create target variable (next minute price increase)
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    
    # Remove rows with NaN values
    data = data.dropna()
    
    # Extract features and target
    features = data[feature_cols].values
    target = data['target'].values
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(target[i])
    
    return np.array(X), np.array(y)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to dataframe (legacy function)."""
    # Make a copy to avoid modifying original
    result = df.copy()
    
    # Moving averages
    result['sma_20'] = simple_moving_average(df['close'], 20)
    result['sma_50'] = simple_moving_average(df['close'], 50)
    result['ema_12'] = exponential_moving_average(df['close'], 12)
    result['ema_26'] = exponential_moving_average(df['close'], 26)
    
    # Momentum indicators
    result['rsi'] = rsi(df['close'])
    result['williams_r'] = williams_r(df['high'], df['low'], df['close'])
    
    # MACD
    macd_line, signal_line, histogram = macd(df['close'])
    result['macd'] = macd_line
    result['macd_signal'] = signal_line
    result['macd_histogram'] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = bollinger_bands(df['close'])
    result['bb_upper'] = bb_upper
    result['bb_middle'] = bb_middle
    result['bb_lower'] = bb_lower
    
    # Volatility
    result['atr'] = atr(df['high'], df['low'], df['close'])
    
    # Stochastic
    stoch_k, stoch_d = stochastic_oscillator(df['high'], df['low'], df['close'])
    result['stoch_k'] = stoch_k
    result['stoch_d'] = stoch_d
    
    # Volume indicators
    result['obv'] = obv(df['close'], df['volume'])
    result['vwap'] = vwap(df['high'], df['low'], df['close'], df['volume'])
    
    return result


# ===== 扩展技术指标库 =====

def bollinger_percent_b(close: pd.Series, window: int = 20, std_dev: float = 2.0) -> pd.Series:
    """计算布林带%B指标 - 价格在布林带中的相对位置"""
    bb_upper, bb_middle, bb_lower = bollinger_bands(close, window, std_dev)
    return (close - bb_lower) / (bb_upper - bb_lower)


def vwap_rolling(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
    """计算滚动成交量加权平均价格 (Rolling VWAP)"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """计算商品通道指数 (Commodity Channel Index)"""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=window).mean()
    mad = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    return (typical_price - sma_tp) / (0.015 * mad)


def trix(close: pd.Series, window: int = 14) -> pd.Series:
    """计算TRIX - 三重指数平滑移动平均线的变化率"""
    ema1 = close.ewm(span=window).mean()
    ema2 = ema1.ewm(span=window).mean()
    ema3 = ema2.ewm(span=window).mean()
    return ema3.pct_change() * 10000  # 乘以10000使数值更易处理


def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """计算柴金资金流量 (Chaikin Money Flow)"""
    mfv = volume * ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0)  # 处理high=low的情况
    return mfv.rolling(window=window).sum() / volume.rolling(window=window).sum()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """计算平均方向运动指数 (Average Directional Movement Index)"""
    # 计算真实范围
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 计算方向运动
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    # 平滑处理
    tr_smooth = tr.rolling(window=window).mean()
    plus_dm_smooth = plus_dm.rolling(window=window).mean()
    minus_dm_smooth = minus_dm.rolling(window=window).mean()
    
    # 计算方向指标
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # 计算ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=window).mean()
    
    return adx


def roc(close: pd.Series, window: int = 12) -> pd.Series:
    """计算变化率 (Rate of Change)"""
    return ((close - close.shift(window)) / close.shift(window)) * 100


def add_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加扩展的技术指标特征库 (原18个 + 新增8个 = 26个特征)
    
    新增指标专注于提升预测能力：
    - 市场结构指标：布林带%B, VWAP偏离度
    - 趋势强度指标：ADX, TRIX
    - 资金流向指标：CMF, CCI
    - 动量指标：ROC, 价格动量
    """
    
    # 先添加原有的18个指标
    result = add_cnn_lstm_features(df)
    
    # 新增扩展指标：
    
    # 19. 布林带%B - 价格在布林带中的相对位置 (0-1范围，>1超买，<0超卖)
    result['bb_percent_b'] = bollinger_percent_b(df['close'])
    
    # 20. VWAP偏离度 - 价格相对VWAP的偏离程度
    vwap_14 = vwap_rolling(df['high'], df['low'], df['close'], df['volume'], 14)
    result['vwap_deviation'] = (df['close'] - vwap_14) / vwap_14
    
    # 21. CCI - 商品通道指数 (正常范围±100，超买超卖±200)
    result['cci'] = cci(df['high'], df['low'], df['close'], 20)
    
    # 22. TRIX - 三重指数平滑变化率 (趋势强度和转折)
    result['trix'] = trix(df['close'], 14)
    
    # 23. CMF - 柴金资金流量 (资金流入流出强度)
    result['cmf'] = chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'], 20)
    
    # 24. ADX - 平均方向运动指数 (趋势强度，25+强趋势)
    result['adx'] = adx(df['high'], df['low'], df['close'], 14)
    
    # 25. ROC - 变化率 (12期价格变化百分比)
    result['roc_12'] = roc(df['close'], 12)
    
    # 26. 价格动量 - 相对强度 (14期相对变化)
    result['price_momentum_14'] = df['close'] / df['close'].shift(14) - 1
    
    # 数据清理：限制异常值到合理范围
    result['cci'] = result['cci'].clip(-300, 300)      # CCI限制在合理范围
    result['cmf'] = result['cmf'].clip(-1, 1)          # CMF限制在[-1,1]
    result['bb_percent_b'] = result['bb_percent_b'].clip(-0.5, 1.5)  # 布林带%B合理范围
    result['vwap_deviation'] = result['vwap_deviation'].clip(-0.2, 0.2)  # VWAP偏离度±20%
    result['adx'] = result['adx'].clip(0, 100)         # ADX范围0-100
    
    return result


def get_extended_feature_columns() -> list:
    """获取扩展特征集的列名 (26个特征)"""
    base_features = get_cnn_lstm_feature_columns()  # 原18个特征
    extended_features = [
        # 新增8个特征
        'bb_percent_b',      # 布林带%B
        'vwap_deviation',    # VWAP偏离度
        'cci',               # 商品通道指数
        'trix',              # TRIX
        'cmf',               # 柴金资金流量
        'adx',               # 平均方向运动指数
        'roc_12',            # 变化率
        'price_momentum_14'  # 价格动量
    ]
    
    return base_features + extended_features


def prepare_extended_data(df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """准备扩展特征的训练数据 (26个特征)"""
    
    # 添加扩展特征
    data = add_extended_features(df)
    
    # 获取扩展特征列
    feature_cols = get_extended_feature_columns()
    
    print(f"使用扩展特征集: {len(feature_cols)} 个特征")
    print(f"新增特征: {feature_cols[18:]}")  # 显示新增的8个特征
    
    # 创建目标变量 (下一分钟价格上涨)
    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
    
    # 移除NaN值
    data_clean = data.dropna()
    
    if len(data_clean) < sequence_length + 1:
        raise ValueError(f"清理后数据不足: {len(data_clean)} < {sequence_length + 1}")
    
    # 提取特征和目标
    features = data_clean[feature_cols].values
    target = data_clean['target'].values
    
    # 创建序列
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(target[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"扩展数据集准备完成:")
    print(f"  样本数量: {len(X)}")
    print(f"  特征维度: {X.shape}")
    print(f"  正样本比例: {y.mean():.1%}")
    
    return X, y