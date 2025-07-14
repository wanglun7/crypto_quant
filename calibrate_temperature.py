#!/usr/bin/env python3
"""Temperature calibration for GRU model."""

import sys
sys.path.append('/Users/lun/code/crypto_quant/src')

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import pickle

from crypto_quant.data.dataset import build_dataset
from crypto_quant.models.gru_signal import load_gru


def calculate_expected_return(probas, actual_returns, threshold=0.2):
    """Calculate expected return for given probabilities and threshold."""
    signal = probas[:, 2] - probas[:, 0]  # p_up - p_down
    
    # Only long positions (clip negative to 0)
    long_signal = np.clip(signal, 0, None)
    
    # Apply threshold filter
    positions = np.where(long_signal > threshold, long_signal, 0)
    
    # Calculate expected return
    expected_return = np.sum(positions[:-1] * actual_returns[1:])
    
    # Count trades (approximate)
    n_trades = np.sum(np.diff(np.concatenate([[0], positions])) != 0)
    
    return expected_return, n_trades


def find_best_temperature(model, X_val, y_val, df_ohlcv, start_ix, horizon=2):
    """Find optimal temperature using validation set."""
    
    # Calculate actual returns for validation period
    close_prices = df_ohlcv['close'].values
    actual_returns = []
    
    for i in range(len(X_val)):
        current_idx = start_ix + i
        future_idx = current_idx + horizon
        
        if future_idx < len(close_prices):
            ret = (close_prices[future_idx] - close_prices[current_idx]) / close_prices[current_idx]
            actual_returns.append(ret)
        else:
            actual_returns.append(0)
    
    actual_returns = np.array(actual_returns)
    
    # Test different temperatures
    temperatures = [1.0, 1.5, 2.0, 2.5, 3.0]
    results = []
    
    print("Testing different temperatures...")
    print("Temp | Expected Return | N Trades | Return/Trade | Signal %")
    print("-" * 60)
    
    for T in temperatures:
        # Get probabilities with temperature scaling
        with torch.no_grad():
            probas = model.predict_proba(torch.FloatTensor(X_val), temperature=T)
            probas_np = probas.numpy()
        
        # Calculate expected return
        expected_ret, n_trades = calculate_expected_return(probas_np, actual_returns)
        
        # Calculate signal statistics
        signal = probas_np[:, 2] - probas_np[:, 0]
        long_signal = np.clip(signal, 0, None)
        signal_pct = np.sum(long_signal > 0.2) / len(long_signal) * 100
        
        ret_per_trade = expected_ret / n_trades if n_trades > 0 else 0
        
        results.append({
            'temperature': T,
            'expected_return': expected_ret,
            'n_trades': n_trades,
            'return_per_trade': ret_per_trade,
            'signal_pct': signal_pct
        })
        
        print(f"{T:4.1f} | {expected_ret*100:13.2f}% | {n_trades:8d} | {ret_per_trade*10000:10.1f}bp | {signal_pct:7.1f}%")
    
    # Find best temperature (highest expected return)
    best_result = max(results, key=lambda x: x['expected_return'])
    best_T = best_result['temperature']
    
    print(f"\nBest temperature: {best_T}")
    print(f"Expected return: {best_result['expected_return']*100:.2f}%")
    print(f"Number of trades: {best_result['n_trades']}")
    print(f"Signal percentage: {best_result['signal_pct']:.1f}%")
    
    return best_T, results


def main():
    """Run temperature calibration."""
    
    # Load cached data
    cache_path = Path("/tmp/btc_ohlcv_feat_5m_72h.pkl")
    if not cache_path.exists():
        print("No cached data found. Please run tests first.")
        return
    
    print("Loading data...")
    df_ohlcv, df_feat = pickle.loads(cache_path.read_bytes())
    
    # Build dataset
    print("Building dataset...")
    X, y = build_dataset(df_ohlcv, df_feat, lookback=60, horizon=2)
    
    # Split into train/val/test (same as training)
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train = X[:n_train]
    X_val = X[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    
    y_train = y[:n_train]
    y_val = y[n_train:n_train+n_val]
    y_test = y[n_train+n_val:]
    
    # Load model
    print("Loading model...")
    model = load_gru()
    
    # Normalize validation data
    if hasattr(model, 'X_mean') and model.X_mean is not None:
        X_val_norm = (X_val - model.X_mean) / model.X_std
    else:
        X_val_norm = X_val
    
    # Find best temperature
    start_ix = 62  # lookback + horizon
    val_start_ix = start_ix + n_train  # Adjust for validation data position
    
    best_T, results = find_best_temperature(model, X_val_norm, y_val, df_ohlcv, val_start_ix)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('temperature_calibration_results.csv', index=False)
    print(f"\nResults saved to temperature_calibration_results.csv")
    
    # Test on validation set with best temperature
    print(f"\n=== Testing with T={best_T} ===")
    with torch.no_grad():
        probas_best = model.predict_proba(torch.FloatTensor(X_val_norm), temperature=best_T)
        probas_np = probas_best.numpy()
    
    # Analyze signal distribution
    signal = probas_np[:, 2] - probas_np[:, 0]
    long_signal = np.clip(signal, 0, None)
    
    print(f"Signal statistics with T={best_T}:")
    print(f"  Mean signal: {np.mean(long_signal):.4f}")
    print(f"  Signals > 0.1: {np.sum(long_signal > 0.1)} ({np.sum(long_signal > 0.1)/len(long_signal)*100:.1f}%)")
    print(f"  Signals > 0.2: {np.sum(long_signal > 0.2)} ({np.sum(long_signal > 0.2)/len(long_signal)*100:.1f}%)")
    print(f"  Signals > 0.5: {np.sum(long_signal > 0.5)} ({np.sum(long_signal > 0.5)/len(long_signal)*100:.1f}%)")
    
    return best_T


if __name__ == "__main__":
    best_temperature = main()
    print(f"\nRecommended temperature for inference: {best_temperature}")