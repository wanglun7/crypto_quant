#!/usr/bin/env python3
"""Simple backtest for CNN-LSTM returns."""

import numpy as np
import pandas as pd
import torch
from crypto_quant.models.cnn_lstm import CNN_LSTM
from crypto_quant.utils.indicators import prepare_cnn_lstm_data

def calculate_returns():
    """Calculate actual trading returns."""
    
    print("="*60)
    print("CNN-LSTM TRADING RETURNS ANALYSIS")
    print("="*60)
    
    # Load real data
    df = pd.read_csv('data/BTC_USDT_1m_last_3days.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    buy_hold_return = (end_price - start_price) / start_price
    
    print(f"Data: {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price: ${start_price:.2f} → ${end_price:.2f}")
    print(f"Buy & Hold: {buy_hold_return:+.2%}")
    
    # Prepare data for model
    X, y = prepare_cnn_lstm_data(df, sequence_length=30)
    
    # Load model and get predictions (match original training)
    model = CNN_LSTM(
        n_features=18, 
        sequence_length=30,
        cnn_filters=[32, 64, 128],  # Original default
        lstm_hidden_size=64,        # From error: 256/4=64  
        lstm_num_layers=2,          # Original had l1 layers
        fc_hidden_size=32           # From error message
    )
    model.load_state_dict(torch.load('models/cnn_lstm_btc_model.pth'))
    model.eval()
    
    with torch.no_grad():
        probabilities = model(torch.FloatTensor(X)).numpy().flatten()
    
    print(f"\nModel Predictions:")
    print(f"  Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    print(f"  Mean: {probabilities.mean():.4f}, Std: {probabilities.std():.4f}")
    
    # Test different thresholds
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
    
    # Get price changes for strategy calculation
    # Align with predictions (skip first 30 due to sequence length)
    aligned_df = df.iloc[30:30+len(probabilities)].copy()
    aligned_df['price_change'] = aligned_df['close'].pct_change()
    
    print(f"\nStrategy Backtesting:")
    print(f"{'Threshold':<10} {'Signals':<8} {'Return':<10} {'vs B&H':<10} {'Trades':<8}")
    print("-" * 50)
    
    best_return = -1
    best_threshold = 0.5
    
    for threshold in thresholds:
        # Generate signals
        signals = (probabilities > threshold).astype(int)
        
        # Calculate strategy returns
        strategy_returns = []
        position = 0
        trades = 0
        
        for i in range(1, len(aligned_df)):
            if i-1 < len(signals):
                new_signal = signals[i-1]
                
                # Count trades (position changes)
                if new_signal != position:
                    trades += 1
                    position = new_signal
                
                # Calculate return
                if position == 1:  # Long position
                    strategy_returns.append(aligned_df['price_change'].iloc[i])
                else:  # No position
                    strategy_returns.append(0)
        
        # Total strategy return
        strategy_returns = np.array(strategy_returns)
        total_return = np.prod(1 + strategy_returns) - 1
        
        # Apply transaction costs (0.04% per trade)
        transaction_cost = 0.0004 * trades
        net_return = total_return - transaction_cost
        
        excess_return = net_return - buy_hold_return
        
        print(f"{threshold:<10.2f} {signals.sum():<8} {net_return:>9.2%} {excess_return:>9.2%} {trades:<8}")
        
        if net_return > best_return:
            best_return = net_return
            best_threshold = threshold
    
    print("="*60)
    print(f"BEST STRATEGY:")
    print(f"  Threshold: {best_threshold}")
    print(f"  Return: {best_return:+.2%}")
    print(f"  Buy & Hold: {buy_hold_return:+.2%}")
    print(f"  Excess: {best_return - buy_hold_return:+.2%}")
    
    if best_return > buy_hold_return:
        print(f"  ✅ Strategy BEATS buy & hold by {best_return - buy_hold_return:+.2%}")
    else:
        print(f"  ❌ Strategy LOSES to buy & hold by {buy_hold_return - best_return:.2%}")
    
    # Detailed analysis for best threshold
    print(f"\nDETAILED ANALYSIS (threshold={best_threshold}):")
    
    signals = (probabilities > best_threshold).astype(int)
    long_periods = signals.sum()
    total_periods = len(signals)
    time_in_market = long_periods / total_periods
    
    print(f"  Time in market: {time_in_market:.1%}")
    print(f"  Long signals: {long_periods} / {total_periods}")
    
    # Win rate calculation
    strategy_returns = []
    for i in range(1, len(aligned_df)):
        if i-1 < len(signals) and signals[i-1] == 1:
            strategy_returns.append(aligned_df['price_change'].iloc[i])
    
    if strategy_returns:
        strategy_returns = np.array(strategy_returns)
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        win_rate = len(wins) / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        print(f"  Win rate: {win_rate:.1%}")
        print(f"  Avg win: {wins.mean():.3%}" if len(wins) > 0 else "  Avg win: N/A")
        print(f"  Avg loss: {losses.mean():.3%}" if len(losses) > 0 else "  Avg loss: N/A")
    
    print("="*60)
    
    # Summary
    return {
        'best_threshold': best_threshold,
        'strategy_return': best_return,
        'buy_hold_return': buy_hold_return,
        'excess_return': best_return - buy_hold_return,
        'time_in_market': time_in_market
    }

if __name__ == "__main__":
    result = calculate_returns()
    
    print("\n🎯 FINAL ANSWER TO YOUR QUESTION:")
    print(f"Strategy Return: {result['strategy_return']:+.2%}")
    print(f"Buy & Hold Return: {result['buy_hold_return']:+.2%}")
    
    if result['strategy_return'] > result['buy_hold_return']:
        print(f"✅ CNN-LSTM strategy WINS by {result['excess_return']:+.2%}")
    else:
        print(f"❌ CNN-LSTM strategy LOSES by {-result['excess_return']:.2%}")
    print("="*60)