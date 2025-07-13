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
    X, y = build_dataset(df_ohlcv, df_feat, lookback=60, horizon=5)
    
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
    
    # Generate position signals based on probability thresholds
    positions = np.zeros(len(probas_np))
    for i in range(len(positions)):
        if p_up[i] >= 0.45:
            positions[i] = 1.0  # Long
        elif p_down[i] >= 0.45:
            positions[i] = -1.0  # Short
        else:
            positions[i] = 0.0  # Hold
    
    # Align positions with price data
    # start_ix = lookback + horizon to avoid lookahead bias
    lookback = 60
    horizon = 5
    start_ix = lookback + horizon
    
    # Create full position array aligned with df_ohlcv
    full_positions = np.zeros(len(df_ohlcv))
    end_ix = start_ix + len(positions)
    if end_ix <= len(df_ohlcv):
        full_positions[start_ix:end_ix] = positions
    else:
        # Truncate if needed
        available_length = len(df_ohlcv) - start_ix
        full_positions[start_ix:] = positions[:available_length]
    
    close_prices = df_ohlcv['close']
    
    # Convert positions to entry/exit signals
    # Only trade when position changes (to minimize fees)
    entries_long = np.zeros(len(full_positions), dtype=bool)
    exits_long = np.zeros(len(full_positions), dtype=bool)
    entries_short = np.zeros(len(full_positions), dtype=bool)
    exits_short = np.zeros(len(full_positions), dtype=bool)
    
    current_pos = 0
    for i in range(1, len(full_positions)):
        new_pos = full_positions[i]
        
        if current_pos != new_pos:
            # Position change
            if current_pos == 1:  # Exit long
                exits_long[i] = True
            elif current_pos == -1:  # Exit short
                exits_short[i] = True
                
            if new_pos == 1:  # Enter long
                entries_long[i] = True
            elif new_pos == -1:  # Enter short
                entries_short[i] = True
                
            current_pos = new_pos
    
    # Run backtest with vectorbt
    pf = vbt.Portfolio.from_signals(
        close_prices,
        entries=entries_long,
        exits=exits_long,
        short_entries=entries_short,
        short_exits=exits_short,
        fees=fee_taker,
        freq='1min',
        slippage=0.0002  # 2 bp slippage
    )
    
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
    required_columns = ['ema_fast_1m', 'ema_slow_1m']
    for col in required_columns:
        if col not in df_feat.columns:
            raise ValueError(f"Missing required column: {col}")
    
    ema_fast = df_feat['ema_fast_1m']
    ema_slow = df_feat['ema_slow_1m']
    
    # Simple EMA crossover strategy - only trade on signal changes
    long_signal = ema_fast > ema_slow
    short_signal = ema_fast < ema_slow
    
    # Generate entry/exit signals
    entries_long = long_signal & ~long_signal.shift(1).fillna(False)
    exits_long = ~long_signal & long_signal.shift(1).fillna(False)
    entries_short = short_signal & ~short_signal.shift(1).fillna(False)
    exits_short = ~short_signal & short_signal.shift(1).fillna(False)
    
    close_prices = df_ohlcv['close']
    
    # Run backtest with vectorbt
    pf = vbt.Portfolio.from_signals(
        close_prices,
        entries=entries_long,
        exits=exits_long,
        short_entries=entries_short,
        short_exits=exits_short,
        fees=fee_taker,
        freq='1min',
        slippage=0.0002  # 2 bp slippage
    )
    
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