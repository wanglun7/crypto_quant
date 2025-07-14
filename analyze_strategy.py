#!/usr/bin/env python3
"""Strategy diagnosis tool to analyze model performance before backtesting."""

import sys
sys.path.append('/Users/lun/code/crypto_quant/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import torch

from crypto_quant.data.dataset import build_dataset
from crypto_quant.models.gru_signal import load_gru


def analyze_signal_distribution(probas, y_true):
    """Analyze the distribution of prediction signals."""
    print("\n=== Signal Distribution Analysis ===")
    
    # Calculate signal strength
    signal_strength = probas[:, 2] - probas[:, 0]  # p_up - p_down
    
    # Basic statistics
    print(f"Signal strength stats:")
    print(f"  Mean: {np.mean(signal_strength):.4f}")
    print(f"  Std: {np.std(signal_strength):.4f}")
    print(f"  Min: {np.min(signal_strength):.4f}")
    print(f"  Max: {np.max(signal_strength):.4f}")
    
    # Threshold analysis
    print(f"\nSignals by threshold:")
    for threshold in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        n_long = np.sum(signal_strength > threshold)
        n_short = np.sum(signal_strength < -threshold)
        n_total = n_long + n_short
        pct = n_total / len(signal_strength) * 100
        print(f"  |signal| > {threshold}: {n_total:5d} ({pct:5.1f}%) [Long: {n_long}, Short: {n_short}]")
    
    # Accuracy by signal strength
    print(f"\nAccuracy by signal strength:")
    y_pred = probas.argmax(axis=1)
    for threshold in [0.0, 0.2, 0.5, 0.7]:
        mask = np.abs(signal_strength) > threshold
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean()
            print(f"  |signal| > {threshold}: {acc:.4f} (n={mask.sum()})")


def analyze_profit_potential(df_ohlcv, probas, y_true, start_ix, horizon=2):
    """Analyze potential profits by signal strength."""
    print("\n=== Profit Potential Analysis ===")
    
    # Calculate actual returns
    close_prices = df_ohlcv['close'].values
    actual_returns = []
    
    for i in range(len(probas)):
        current_idx = start_ix + i
        future_idx = current_idx + horizon
        
        if future_idx < len(close_prices):
            ret = np.log(close_prices[future_idx] / close_prices[current_idx])
            actual_returns.append(ret)
        else:
            actual_returns.append(0)
    
    actual_returns = np.array(actual_returns)
    signal_strength = probas[:, 2] - probas[:, 0]
    
    # Group by signal strength
    print(f"\nAverage returns by signal strength bins:")
    bins = [-1, -0.7, -0.3, 0, 0.3, 0.7, 1]
    labels = ['Strong Short', 'Weak Short', 'Neutral-', 'Neutral+', 'Weak Long', 'Strong Long']
    
    signal_groups = pd.cut(signal_strength, bins=bins, labels=labels)
    
    for group in labels:
        mask = (signal_groups == group)
        if mask.sum() > 0:
            avg_ret = actual_returns[mask].mean() * 10000  # Convert to bps
            std_ret = actual_returns[mask].std() * 10000
            count = mask.sum()
            print(f"  {group:12s}: {avg_ret:6.1f} bps (±{std_ret:5.1f}), n={count:5d}")
    
    # Expected return calculation
    print(f"\nExpected returns with perfect execution:")
    for threshold in [0.2, 0.5, 0.7, 0.9]:
        positions = np.where(signal_strength > threshold, 1,
                           np.where(signal_strength < -threshold, -1, 0))
        
        expected_ret = np.sum(positions * actual_returns)
        n_trades = np.sum(np.diff(positions) != 0)
        
        print(f"  Threshold {threshold}: Return={expected_ret*100:.2f}%, Trades={n_trades}")


def simulate_cost_impact(df_ohlcv, probas, start_ix, horizon=2, 
                        fee_taker=0.0004, fee_maker=0.0002):
    """Simulate impact of trading costs."""
    print("\n=== Cost Impact Simulation ===")
    
    # Calculate actual returns
    close_prices = df_ohlcv['close'].values
    actual_returns = []
    
    for i in range(len(probas)):
        current_idx = start_ix + i
        future_idx = current_idx + horizon
        
        if future_idx < len(close_prices):
            ret = (close_prices[future_idx] - close_prices[current_idx]) / close_prices[current_idx]
            actual_returns.append(ret)
        else:
            actual_returns.append(0)
    
    actual_returns = np.array(actual_returns)
    signal_strength = probas[:, 2] - probas[:, 0]
    
    # Simulate different thresholds
    results = []
    
    for threshold in np.arange(0.1, 0.95, 0.1):
        # Generate positions
        positions = np.where(signal_strength > threshold, 1,
                           np.where(signal_strength < -threshold, -1, 0))
        
        # Count trades
        position_changes = np.diff(np.concatenate([[0], positions]))
        n_trades = np.sum(position_changes != 0)
        
        # Calculate returns
        position_returns = positions[:-1] * actual_returns[1:]
        gross_return = np.sum(position_returns)
        
        # Calculate costs
        total_taker_cost = n_trades * fee_taker
        # Approximate maker cost (holding cost)
        holding_periods = np.sum(positions != 0)
        total_maker_cost = holding_periods * fee_maker
        
        net_return = gross_return - total_taker_cost - total_maker_cost
        
        results.append({
            'threshold': threshold,
            'n_trades': n_trades,
            'gross_return_%': gross_return * 100,
            'taker_cost_%': total_taker_cost * 100,
            'maker_cost_%': total_maker_cost * 100,
            'net_return_%': net_return * 100,
            'cost_ratio': (total_taker_cost + total_maker_cost) / abs(gross_return) if gross_return != 0 else np.inf
        })
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    
    # Find optimal threshold
    best_idx = df_results['net_return_%'].idxmax()
    best_threshold = df_results.loc[best_idx, 'threshold']
    print(f"\nOptimal threshold: {best_threshold:.2f}")


def main():
    """Run complete strategy diagnosis."""
    
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
    
    # Load model and generate predictions
    print("Loading model and generating predictions...")
    model = load_gru()
    
    # Normalize if needed
    if hasattr(model, 'X_mean') and model.X_mean is not None:
        X_norm = (X - model.X_mean) / model.X_std
    else:
        X_norm = X
    
    # Generate predictions
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_norm)
        probas = model.predict_proba(X_tensor)
        probas_np = probas.numpy()
    
    # Run analyses
    analyze_signal_distribution(probas_np, y)
    
    start_ix = 62  # lookback + horizon
    analyze_profit_potential(df_ohlcv, probas_np, y, start_ix)
    
    simulate_cost_impact(df_ohlcv, probas_np, start_ix)
    
    print("\n=== Diagnosis Summary ===")
    signal_strength = probas_np[:, 2] - probas_np[:, 0]
    strong_signals = np.sum(np.abs(signal_strength) > 0.5) / len(signal_strength) * 100
    print(f"Strong signals (|s|>0.5): {strong_signals:.1f}%")
    
    if strong_signals < 10:
        print("⚠️  WARNING: Too few strong signals. Model may be too uncertain.")
    elif strong_signals > 30:
        print("⚠️  WARNING: Too many strong signals. Model may be overconfident.")
    else:
        print("✓ Signal distribution looks reasonable.")


if __name__ == "__main__":
    main()