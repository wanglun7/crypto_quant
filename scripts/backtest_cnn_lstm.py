#!/usr/bin/env python3
"""Backtest CNN-LSTM model to calculate real trading returns."""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from crypto_quant.models.cnn_lstm import CNN_LSTM
from crypto_quant.utils.indicators import prepare_cnn_lstm_data


def backtest_strategy(threshold=0.5):
    """Backtest CNN-LSTM trading strategy with real data."""
    
    print("="*70)
    print(f"CNN-LSTM BACKTEST (Threshold={threshold})")
    print("="*70)
    
    # Load real data
    df = pd.read_csv('data/BTC_USDT_1m_last_3days.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"Data loaded: {len(df)} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    
    # Prepare data
    X, y = prepare_cnn_lstm_data(df, sequence_length=30)
    
    # Load trained model
    model = CNN_LSTM(n_features=18, sequence_length=30)
    model.load_state_dict(torch.load('cnn_lstm_btc_model.pth'))
    model.eval()
    
    # Get predictions
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        probabilities = model(X_tensor).numpy().flatten()
    
    # Generate signals with given threshold
    signals = (probabilities > threshold).astype(int)
    
    # Get corresponding prices (need to align with sequence length)
    prices = df['close'].values[30:]  # Skip first 30 for sequence length
    timestamps = df['timestamp'].values[30:]
    
    # Ensure alignment
    min_len = min(len(signals), len(prices))
    signals = signals[:min_len]
    prices = prices[:min_len]
    timestamps = timestamps[:min_len]
    
    # Calculate returns
    price_returns = np.diff(prices) / prices[:-1]
    
    # Strategy returns (only trade when signal=1)
    strategy_returns = []
    positions = []
    
    for i in range(len(price_returns)):
        if i < len(signals) - 1:
            if signals[i] == 1:  # Long position
                strategy_returns.append(price_returns[i])
                positions.append(1)
            else:  # No position
                strategy_returns.append(0)
                positions.append(0)
        else:
            strategy_returns.append(0)
            positions.append(0)
    
    strategy_returns = np.array(strategy_returns)
    positions = np.array(positions)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + price_returns).cumprod()
    cumulative_strategy = (1 + strategy_returns).cumprod()
    
    # Performance metrics
    total_return = cumulative_strategy[-1] - 1
    buy_hold_return = cumulative_returns[-1] - 1
    
    # Calculate Sharpe ratio (1-minute data, annualized)
    minutes_per_year = 365 * 24 * 60
    strategy_sharpe = np.sqrt(minutes_per_year) * strategy_returns.mean() / (strategy_returns.std() + 1e-8)
    
    # Win rate
    winning_trades = strategy_returns[strategy_returns > 0]
    losing_trades = strategy_returns[strategy_returns < 0]
    win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if (len(winning_trades) + len(losing_trades)) > 0 else 0
    
    # Maximum drawdown
    running_max = np.maximum.accumulate(cumulative_strategy)
    drawdown = (cumulative_strategy - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Number of trades
    trades = np.sum(np.diff(positions) != 0) / 2  # Entry and exit
    
    # Print results
    print(f"\nBACKTEST RESULTS:")
    print(f"Strategy Return:      {total_return:+.2%}")
    print(f"Buy & Hold Return:    {buy_hold_return:+.2%}")
    print(f"Excess Return:        {total_return - buy_hold_return:+.2%}")
    print(f"")
    print(f"Sharpe Ratio:         {strategy_sharpe:.2f}")
    print(f"Win Rate:             {win_rate:.1%}")
    print(f"Max Drawdown:         {max_drawdown:.2%}")
    print(f"Number of Trades:     {int(trades)}")
    print(f"")
    print(f"Time in Market:       {positions.mean():.1%}")
    print(f"Total Predictions:    {len(signals)}")
    print(f"Long Signals:         {signals.sum()} ({signals.mean():.1%})")
    
    # Transaction costs analysis
    transaction_cost = 0.0004  # 0.04% Binance fee
    trades_with_costs = strategy_returns - transaction_cost * np.abs(np.diff(np.concatenate([[0], positions])))
    cumulative_with_costs = (1 + trades_with_costs).cumprod()
    return_with_costs = cumulative_with_costs[-1] - 1
    
    print(f"\nWITH TRANSACTION COSTS (0.04%):")
    print(f"Net Return:           {return_with_costs:+.2%}")
    print(f"Cost Impact:          {return_with_costs - total_return:.2%}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Cumulative returns
    ax1.plot(timestamps[1:], cumulative_returns, label='Buy & Hold', linewidth=2)
    ax1.plot(timestamps[1:], cumulative_strategy, label=f'CNN-LSTM (threshold={threshold})', linewidth=2)
    ax1.plot(timestamps[1:], cumulative_with_costs, label='CNN-LSTM (with costs)', linewidth=1, linestyle='--')
    ax1.set_ylabel('Cumulative Return')
    ax1.set_title(f'CNN-LSTM Strategy Performance (3 days)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Signals and price
    ax2.plot(timestamps[1:], prices[1:], color='black', alpha=0.7, linewidth=1)
    
    # Mark buy signals
    buy_points = timestamps[1:][positions[:-1] == 1]
    buy_prices = prices[1:][positions[:-1] == 1]
    ax2.scatter(buy_points, buy_prices, color='green', marker='^', s=30, alpha=0.7, label='Long Signal')
    
    # Mark sell signals (when position changes from 1 to 0)
    position_diff = np.diff(positions)
    sell_idx = np.where(position_diff == -1)[0]
    if len(sell_idx) > 0:
        sell_points = timestamps[1:][sell_idx]
        sell_prices = prices[1:][sell_idx]
        ax2.scatter(sell_points, sell_prices, color='red', marker='v', s=30, alpha=0.7, label='Exit Signal')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('BTC Price ($)')
    ax2.set_title('Trading Signals')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'backtest_threshold_{threshold}.png', dpi=300, bbox_inches='tight')
    
    return {
        'threshold': threshold,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'excess_return': total_return - buy_hold_return,
        'sharpe_ratio': strategy_sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'trades': int(trades),
        'time_in_market': positions.mean(),
        'return_with_costs': return_with_costs
    }


def compare_thresholds():
    """Compare different threshold values."""
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5]
    results = []
    
    for threshold in thresholds:
        result = backtest_strategy(threshold)
        results.append(result)
    
    # Summary table
    print("\n" + "="*70)
    print("THRESHOLD COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Threshold':<10} {'Return':<10} {'Sharpe':<10} {'Win Rate':<10} {'Drawdown':<10}")
    print("-"*50)
    
    for r in results:
        print(f"{r['threshold']:<10.2f} {r['total_return']:>9.2%} {r['sharpe_ratio']:>9.2f} {r['win_rate']:>9.1%} {r['max_drawdown']:>9.2%}")
    
    # Find best threshold
    best_result = max(results, key=lambda x: x['total_return'])
    
    print("\n" + "="*70)
    print(f"BEST THRESHOLD: {best_result['threshold']}")
    print(f"Best Return: {best_result['total_return']:+.2%}")
    print(f"Beats Buy & Hold by: {best_result['excess_return']:+.2%}")
    print("="*70)


if __name__ == "__main__":
    # First test default threshold
    backtest_strategy(threshold=0.5)
    
    # Then test optimal threshold from previous analysis
    backtest_strategy(threshold=0.3)
    
    # Compare all thresholds
    compare_thresholds()