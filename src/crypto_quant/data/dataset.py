import numpy as np
import pandas as pd


def build_dataset(df_ohlcv: pd.DataFrame, df_feat: pd.DataFrame, lookback: int = 60, horizon: int = 5):
    """
    Build dataset for GRU model training.
    
    Args:
        df_ohlcv: OHLCV data with 'close' column
        df_feat: Feature data aligned with df_ohlcv
        lookback: Number of past timesteps to use (default: 60)
        horizon: Number of future timesteps for label (default: 5)
        
    Returns:
        X: (N, lookback, num_features) array
        y: (N,) array with labels 0=Down, 1=Hold, 2=Up
    """
    # Calculate additional price-based features
    close = df_ohlcv['close']
    high = df_ohlcv['high'] 
    low = df_ohlcv['low']
    
    # New features
    ret_1m = np.log(close / close.shift(1)).fillna(0)
    ret_5m = np.log(close / close.shift(5)).fillna(0)
    ret_15m = np.log(close / close.shift(15)).fillna(0)
    vol_15m = close.rolling(15).std().fillna(close.std())
    pos_1m = ((close - low) / (high - low)).fillna(0.5)
    
    # Combine original features with new ones
    enhanced_features = df_feat.copy()
    enhanced_features['ret_1m'] = ret_1m
    enhanced_features['ret_5m'] = ret_5m
    enhanced_features['ret_15m'] = ret_15m
    enhanced_features['vol_15m'] = vol_15m
    enhanced_features['pos_1m'] = pos_1m
    
    # Apply z-score normalization to stabilize training
    enhanced_features = (enhanced_features - enhanced_features.mean()) / (enhanced_features.std() + 1e-8)
    
    feat_array = enhanced_features.values
    
    # Prepare samples
    X_list = []
    y_list = []
    
    # Calculate number of samples ensuring no lookahead bias
    # For sample i: we use features [i:i+lookback] and predict close[i+lookback+horizon-1]
    # So we need: i+lookback+horizon-1 < len(df_ohlcv)
    # Therefore: i < len(df_ohlcv) - lookback - horizon + 1
    max_samples = len(df_ohlcv) - lookback - horizon + 1
    
    for i in range(max_samples):
        # Extract lookback window for features
        X_sample = feat_array[i:i+lookback]
        
        # Calculate future return over horizon
        # Current price at end of lookback window
        current_idx = i + lookback - 1
        # Future price after horizon periods
        future_idx = current_idx + horizon
        
        if future_idx < len(df_ohlcv):
            future_return = np.log(df_ohlcv['close'].iloc[future_idx] / 
                                 df_ohlcv['close'].iloc[current_idx])
            
            # Label based on future return - using adjusted thresholds ±0.0005 (±0.05%)
            if future_return > 0.0005:  # +0.05%
                label = 2  # Up
            elif future_return < -0.0005:  # -0.05%
                label = 0  # Down
            else:
                label = 1  # Hold
                
            X_list.append(X_sample)
            y_list.append(label)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    return X, y