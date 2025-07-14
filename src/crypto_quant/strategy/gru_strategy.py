import pandas as pd
import numpy as np
import vectorbt as vbt
import torch

from crypto_quant.models.gru_signal import load_gru
from crypto_quant.data.dataset import build_dataset


def run_backtest(df_ohlcv: pd.DataFrame, df_feat: pd.DataFrame, ckpt_path="latest.pt", 
                 *, fee_maker=0.0002, fee_taker=0.0004):
    """
    Run backtest using trained GRU model.
    
    Args:
        df_ohlcv: OHLCV data
        df_feat: Feature data
        ckpt_path: Path to trained model checkpoint
        fee_maker: Maker fee rate (2 bp)
        fee_taker: Taker fee rate (4 bp)
        
    Returns:
        dict with performance stats
    """
    # Load trained model
    model = load_gru(ckpt_path)
    
    # Build dataset for prediction
    X, y = build_dataset(df_ohlcv, df_feat, lookback=60, horizon=2)
    
    # Apply normalization if available
    if hasattr(model, 'X_mean') and model.X_mean is not None:
        X_norm = (X - model.X_mean) / model.X_std
    else:
        X_norm = X
    
    # Generate predictions
    X_tensor = torch.FloatTensor(X_norm)
    probas = model.predict_proba(X_tensor)
    probas_np = probas.detach().numpy()
    
    # Extract probabilities for each class
    p_down = probas_np[:, 0]   # Class 0: Down
    p_hold = probas_np[:, 1]   # Class 1: Hold  
    p_up = probas_np[:, 2]     # Class 2: Up
    
    # Generate continuous position signals - Long Only Strategy
    signal = p_up - p_down  # Calculate signal strength
    # Only keep long signals, clip short signals to 0
    continuous_pos = np.clip(signal, 0, None)  # Only long positions [0, 1]
    # Apply confidence filter - only trade when signal > 0.3 (adjusted for 5m)
    continuous_pos = np.where(continuous_pos < 0.3, 0, continuous_pos)
    
    # Debug info
    valid_positions = np.count_nonzero(continuous_pos)
    avg_abs_position = np.mean(np.abs(continuous_pos[continuous_pos != 0])) if valid_positions > 0 else 0
    print(f"GRU strategy - Valid positions: {valid_positions}/{len(continuous_pos)}, Avg position: {avg_abs_position:.3f}")
    
    # Align positions with price data
    lookback = 60
    horizon = 2
    start_ix = 62  # lookback + horizon
    
    # Create size array for vectorbt
    size_array = np.zeros(len(df_ohlcv))
    end_ix = start_ix + len(continuous_pos)
    if end_ix <= len(df_ohlcv):
        size_array[start_ix:end_ix] = continuous_pos
    else:
        # Truncate if needed
        available_length = len(df_ohlcv) - start_ix
        size_array[start_ix:] = continuous_pos[:available_length]
    
    close_prices = df_ohlcv['close']
    
    # Run backtest with target percent mode
    pf = vbt.Portfolio.from_orders(
        close_prices,
        size=size_array,
        size_type='targetpercent',
        fees=fee_taker,
        freq='5min'
    )
    
    # Calculate performance metrics with proper maker fee adjustment
    # Deduct maker fee based on position changes (more accurate)
    holdings_pct = pf.asset_value() / pf.value()
    position_changes = holdings_pct.diff().fillna(0)
    maker_cost_pct = np.abs(position_changes) * fee_maker
    
    # Apply maker cost to returns
    returns = pf.returns()
    adjusted_returns = returns - maker_cost_pct.values
    
    # Calculate metrics
    total_return = pf.total_return() 
    sharpe = pf.sharpe_ratio(freq='D')
    max_dd = pf.max_drawdown()
    trade_count = len(pf.orders.records)
    
    stats = {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trade_count': trade_count
    }
    
    return stats


def run_baseline(df_ohlcv: pd.DataFrame, df_feat: pd.DataFrame, 
                 *, fee_maker=0.0002, fee_taker=0.0004):
    """
    Run baseline EMA strategy for comparison.
    
    Args:
        df_ohlcv: OHLCV data
        df_feat: Feature data with EMA columns
        fee_maker: Maker fee rate (2 bp)
        fee_taker: Taker fee rate (4 bp)
        
    Returns:
        dict with performance stats
    """
    required_columns = ['ema_fast_5m', 'ema_slow_5m']
    for col in required_columns:
        if col not in df_feat.columns:
            raise ValueError(f"Missing required column: {col}")
    
    ema_fast = df_feat['ema_fast_5m']
    ema_slow = df_feat['ema_slow_5m']
    
    # Generate continuous position signal based on EMA difference
    ema_avg = (ema_fast + ema_slow) / 2
    ema_diff = (ema_fast - ema_slow) / (ema_avg + 1e-8)  # Avoid division by zero
    continuous_pos = np.tanh(ema_diff * 50)  # Increase multiplier to get stronger signals
    
    # Apply smaller dead zone for baseline strategy
    continuous_pos = np.where(np.abs(continuous_pos) < 0.05, 0, continuous_pos)
    continuous_pos = np.clip(continuous_pos, -1, 1)
    
    # Debug info
    valid_positions = np.count_nonzero(continuous_pos)
    avg_abs_position = np.mean(np.abs(continuous_pos[continuous_pos != 0])) if valid_positions > 0 else 0
    print(f"Baseline strategy - Valid positions: {valid_positions}/{len(continuous_pos)}, Avg position: {avg_abs_position:.3f}")
    
    close_prices = df_ohlcv['close']
    
    # Run backtest with target percent mode (same as GRU)
    pf = vbt.Portfolio.from_orders(
        close_prices,
        size=continuous_pos,
        size_type='targetpercent',
        fees=fee_taker,
        freq='5min'
    )
    
    # Calculate performance metrics with maker fee adjustment
    returns = pf.returns()
    positions = pf.asset_value(group_by=False) / pf.value()
    maker_cost = np.abs(positions.diff().fillna(0)) * fee_maker
    adjusted_returns = returns - maker_cost.values.flatten()
    
    total_return = pf.total_return()
    sharpe = pf.sharpe_ratio(freq='D')
    max_dd = pf.max_drawdown()
    trade_count = len(pf.orders.records)
    
    stats = {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trade_count': trade_count
    }
    
    return stats