import pandas as pd
import numpy as np
import vectorbt as vbt


def run_backtest(df_ohlcv: pd.DataFrame, df_feat: pd.DataFrame, *, fee_maker=0.0001, fee_taker=0.0003):
    required_columns = ['ema_fast_1m', 'ema_slow_1m']
    for col in required_columns:
        if col not in df_feat.columns:
            raise ValueError(f"Missing required column: {col}")
    
    ema_fast = df_feat['ema_fast_1m']
    ema_slow = df_feat['ema_slow_1m']
    
    # Use momentum-based approach
    ema_diff = ema_fast - ema_slow
    ema_diff_momentum = ema_diff.diff()
    
    # Generate position signals based on trend strength
    positions = np.zeros(len(ema_diff))
    for i in range(1, len(positions)):
        if ema_diff.iloc[i] > 0 and ema_diff_momentum.iloc[i] > 0:
            positions[i] = 1.0  # Strong uptrend
        elif ema_diff.iloc[i] < 0 and ema_diff_momentum.iloc[i] < 0:
            positions[i] = -1.0  # Strong downtrend
        else:
            positions[i] = 0.0  # No clear trend
    
    close_prices = df_ohlcv['close']
    
    # Use from_orders approach for better control
    pf = vbt.Portfolio.from_orders(
        close_prices,
        size=positions,
        size_type='targetpercent',
        init_cash=10000,
        fees=fee_taker,
        freq='1min',
        slippage=0
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