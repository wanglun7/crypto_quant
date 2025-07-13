"""Momentum-based trading strategy."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from ..utils.indicators import add_all_indicators


class MomentumStrategy:
    """Multi-timeframe momentum strategy with mean reversion elements."""
    
    def __init__(self, 
                 rsi_oversold: float = 30,
                 rsi_overbought: float = 70,
                 trend_ma_fast: int = 20,
                 trend_ma_slow: int = 50,
                 volume_threshold: float = 1.2):
        """Initialize strategy parameters.
        
        Args:
            rsi_oversold: RSI level for oversold condition
            rsi_overbought: RSI level for overbought condition  
            trend_ma_fast: Fast moving average period
            trend_ma_slow: Slow moving average period
            volume_threshold: Volume spike threshold (multiple of recent average)
        """
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.trend_ma_fast = trend_ma_fast
        self.trend_ma_slow = trend_ma_slow
        self.volume_threshold = volume_threshold
        
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on momentum and mean reversion.
        
        Args:
            df: OHLCV dataframe
            
        Returns:
            DataFrame with signals and analysis
        """
        # Add technical indicators
        data = add_all_indicators(df)
        
        # Calculate additional features
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        data['price_change'] = data['close'].pct_change()
        
        # Trend conditions
        data['trend_bullish'] = data['sma_20'] > data['sma_50']
        data['trend_bearish'] = data['sma_20'] < data['sma_50']
        data['price_above_ma'] = data['close'] > data['sma_20']
        data['price_below_ma'] = data['close'] < data['sma_20']
        
        # Momentum conditions
        data['rsi_oversold'] = data['rsi'] < self.rsi_oversold
        data['rsi_overbought'] = data['rsi'] > self.rsi_overbought
        data['rsi_neutral'] = (data['rsi'] >= 40) & (data['rsi'] <= 60)
        
        # Volume confirmation
        data['volume_spike'] = data['volume_ratio'] > self.volume_threshold
        
        # MACD conditions
        data['macd_bullish'] = (data['macd'] > data['macd_signal']) & (data['macd'].shift(1) <= data['macd_signal'].shift(1))
        data['macd_bearish'] = (data['macd'] < data['macd_signal']) & (data['macd'].shift(1) >= data['macd_signal'].shift(1))
        
        # Generate signals
        data['signal'] = 0
        
        # Long signals (momentum + mean reversion)
        long_momentum = (
            data['trend_bullish'] & 
            data['macd_bullish'] & 
            data['volume_spike'] &
            (data['rsi'] > 30) & (data['rsi'] < 70)  # Not in extreme zones
        )
        
        long_mean_reversion = (
            data['trend_bullish'] &  # Only in uptrend
            data['rsi_oversold'] & 
            (data['close'] > data['bb_lower'])  # Above lower Bollinger band
        )
        
        # Short signals
        short_momentum = (
            data['trend_bearish'] & 
            data['macd_bearish'] & 
            data['volume_spike'] &
            (data['rsi'] > 30) & (data['rsi'] < 70)
        )
        
        short_mean_reversion = (
            data['trend_bearish'] &  # Only in downtrend
            data['rsi_overbought'] & 
            (data['close'] < data['bb_upper'])  # Below upper Bollinger band
        )
        
        # Apply signals
        data.loc[long_momentum | long_mean_reversion, 'signal'] = 1
        data.loc[short_momentum | short_mean_reversion, 'signal'] = -1
        
        # Exit conditions
        data['exit_long'] = (
            (data['rsi'] > 80) |  # Extreme overbought
            (data['close'] < data['sma_20'] * 0.98)  # 2% below MA
        )
        
        data['exit_short'] = (
            (data['rsi'] < 20) |  # Extreme oversold
            (data['close'] > data['sma_20'] * 1.02)  # 2% above MA
        )
        
        # Signal strength (for position sizing)
        data['signal_strength'] = self._calculate_signal_strength(data)
        
        return data
    
    def _calculate_signal_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate signal strength for position sizing."""
        strength = pd.Series(0.0, index=data.index)
        
        # Base strength from momentum alignment
        momentum_score = 0
        momentum_score += np.where(data['macd'] > 0, 0.25, 0)
        momentum_score += np.where(data['rsi'] > 50, 0.25, 0)
        momentum_score += np.where(data['volume_ratio'] > 1, 0.25, 0)
        momentum_score += np.where(data['trend_bullish'], 0.25, 0)
        
        # Reduce strength in choppy conditions
        atr_percentile = data['atr'].rolling(50).apply(lambda x: pd.Series(x).rank().iloc[-1] / len(x))
        momentum_score *= np.where(atr_percentile > 0.7, 0.7, 1.0)  # Reduce in high volatility
        
        return pd.Series(momentum_score, index=data.index)


def backtest_strategy(data: pd.DataFrame, 
                     initial_capital: float = 100000,
                     position_size: float = 0.1,
                     transaction_cost: float = 0.0004) -> Dict[str, Any]:
    """Simple vectorized backtest of the momentum strategy.
    
    Args:
        data: DataFrame with signals
        initial_capital: Starting capital
        position_size: Fraction of capital per trade
        transaction_cost: Transaction cost per trade (0.04% for Binance)
        
    Returns:
        Dictionary with backtest results
    """
    # Initialize strategy
    strategy = MomentumStrategy()
    signals_df = strategy.generate_signals(data)
    
    # Calculate returns
    signals_df['returns'] = signals_df['close'].pct_change()
    signals_df['strategy_returns'] = signals_df['signal'].shift(1) * signals_df['returns']
    
    # Apply transaction costs
    signals_df['position_change'] = signals_df['signal'].diff().abs()
    signals_df['transaction_costs'] = signals_df['position_change'] * transaction_cost * position_size
    signals_df['net_strategy_returns'] = signals_df['strategy_returns'] - signals_df['transaction_costs']
    
    # Calculate cumulative returns
    signals_df['cumulative_returns'] = (1 + signals_df['returns']).cumprod()
    signals_df['cumulative_strategy_returns'] = (1 + signals_df['net_strategy_returns']).cumprod()
    
    # Calculate metrics
    total_return = signals_df['cumulative_strategy_returns'].iloc[-1] - 1
    benchmark_return = signals_df['cumulative_returns'].iloc[-1] - 1
    
    # Calculate Sharpe ratio (assuming 252 trading days)
    strategy_vol = signals_df['net_strategy_returns'].std() * np.sqrt(252)
    strategy_mean = signals_df['net_strategy_returns'].mean() * 252
    sharpe_ratio = strategy_mean / strategy_vol if strategy_vol > 0 else 0
    
    # Calculate maximum drawdown
    rolling_max = signals_df['cumulative_strategy_returns'].expanding().max()
    drawdown = (signals_df['cumulative_strategy_returns'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Calculate win rate
    winning_trades = (signals_df['net_strategy_returns'] > 0).sum()
    total_trades = (signals_df['net_strategy_returns'] != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    return {
        'total_return': total_return,
        'benchmark_return': benchmark_return,
        'excess_return': total_return - benchmark_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'signals_df': signals_df
    }