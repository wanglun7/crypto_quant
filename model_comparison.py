#!/usr/bin/env python3
"""Comprehensive model comparison framework for crypto trading strategy."""

import sys
sys.path.append('/Users/lun/code/crypto_quant/src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time
from datetime import datetime

from crypto_quant.data.dataset import build_dataset
from crypto_quant.models.gru_signal import train_gru, load_gru
from crypto_quant.models.lstm_model import train_lstm, load_lstm
from crypto_quant.models.simple_nn import train_simple_nn, load_simple_nn
from crypto_quant.models.sklearn_models import run_all_sklearn_models
from crypto_quant.models.technical_baselines import run_all_technical_models
from crypto_quant.strategy.gru_strategy import run_backtest, run_baseline


def analyze_model_signals(model, X, y, model_name):
    """Analyze model signal distribution and accuracy."""
    print(f"\n=== {model_name} Signal Analysis ===")
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        if hasattr(model, 'X_mean') and model.X_mean is not None:
            # Neural network models
            import torch
            X_norm = (X - model.X_mean) / model.X_std
            X_tensor = torch.FloatTensor(X_norm)
            probas = model.predict_proba(X_tensor).detach().numpy()
        else:
            # Sklearn or technical models
            probas = model.predict_proba(X)
    else:
        print(f"Model {model_name} does not support predict_proba")
        return None
    
    # Calculate signal strength
    signal_strength = probas[:, 2] - probas[:, 0]  # p_up - p_down
    
    print(f"Signal strength stats:")
    print(f"  Mean: {np.mean(signal_strength):.4f}")
    print(f"  Std: {np.std(signal_strength):.4f}")
    print(f"  Min: {np.min(signal_strength):.4f}")
    print(f"  Max: {np.max(signal_strength):.4f}")
    
    # Threshold analysis
    for threshold in [0.1, 0.2, 0.5, 0.7]:
        n_strong = np.sum(np.abs(signal_strength) > threshold)
        pct = n_strong / len(signal_strength) * 100
        print(f"  |signal| > {threshold}: {n_strong:5d} ({pct:5.1f}%)")
    
    # Accuracy analysis
    y_pred = probas.argmax(axis=1)
    overall_acc = (y_pred == y).mean()
    print(f"Overall accuracy: {overall_acc:.4f}")
    
    # Accuracy by signal strength
    for threshold in [0.0, 0.2, 0.5, 0.7]:
        mask = np.abs(signal_strength) > threshold
        if mask.sum() > 0:
            acc = (y_pred[mask] == y[mask]).mean()
            print(f"  Accuracy |signal| > {threshold}: {acc:.4f} (n={mask.sum()})")
    
    return {
        'signal_strength': signal_strength,
        'probas': probas,
        'predictions': y_pred,
        'overall_accuracy': overall_acc
    }


def run_backtest_comparison(df_ohlcv, df_feat, model_results):
    """Run backtest comparison across different models."""
    print("\n" + "=" * 60)
    print("BACKTEST COMPARISON")
    print("=" * 60)
    
    backtest_results = {}
    
    # Run baseline
    print("\n--- Running Baseline Strategy ---")
    try:
        baseline_stats = run_baseline(df_ohlcv, df_feat)
        backtest_results['baseline'] = baseline_stats
        print(f"Baseline - Sharpe: {baseline_stats['sharpe']:.4f}, Return: {baseline_stats['total_return']:.4f}")
    except Exception as e:
        print(f"Baseline failed: {e}")
        backtest_results['baseline'] = None
    
    # Test neural network models
    neural_models = ['gru', 'lstm', 'simple_nn', 'simple_conv']
    
    for model_name in neural_models:
        print(f"\n--- Running {model_name.upper()} Strategy ---")
        try:
            if model_name == 'gru':
                ckpt_path = 'latest.pt'
            elif model_name == 'lstm':
                ckpt_path = 'lstm_latest.pt'
            elif model_name == 'simple_nn':
                ckpt_path = 'simple_nn_latest.pt'
            elif model_name == 'simple_conv':
                ckpt_path = 'simple_conv_latest.pt'
            
            # Check if model exists
            if Path(ckpt_path).exists():
                stats = run_backtest(df_ohlcv, df_feat, ckpt_path=ckpt_path)
                backtest_results[model_name] = stats
                print(f"{model_name.upper()} - Sharpe: {stats['sharpe']:.4f}, Return: {stats['total_return']:.4f}")
            else:
                print(f"{model_name.upper()} model not found at {ckpt_path}")
                backtest_results[model_name] = None
        except Exception as e:
            print(f"{model_name.upper()} backtest failed: {e}")
            backtest_results[model_name] = None
    
    return backtest_results


def create_comparison_report(model_results, backtest_results, signal_analyses):
    """Create comprehensive comparison report."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE MODEL COMPARISON REPORT")
    print("=" * 60)
    
    # Classification performance summary
    print("\n1. CLASSIFICATION PERFORMANCE")
    print("-" * 40)
    print(f"{'Model':<20} {'Val BACC':<10} {'Test BACC':<10}")
    print("-" * 40)
    
    for category, models in model_results.items():
        if models is None:
            continue
        print(f"\n{category.upper()}")
        for model_name, result in models.items():
            if result is not None:
                val_bacc = result.get('val_bacc', 0)
                test_bacc = result.get('test_bacc', 0)
                print(f"{model_name:<20} {val_bacc:<10.4f} {test_bacc:<10.4f}")
    
    # Signal quality analysis
    print("\n2. SIGNAL QUALITY ANALYSIS")
    print("-" * 40)
    if signal_analyses:
        print(f"{'Model':<20} {'Mean Signal':<12} {'Strong %':<10} {'Accuracy':<10}")
        print("-" * 40)
        
        for model_name, analysis in signal_analyses.items():
            if analysis is not None:
                mean_signal = np.mean(analysis['signal_strength'])
                strong_pct = np.sum(np.abs(analysis['signal_strength']) > 0.5) / len(analysis['signal_strength']) * 100
                accuracy = analysis['overall_accuracy']
                print(f"{model_name:<20} {mean_signal:<12.4f} {strong_pct:<10.1f}% {accuracy:<10.4f}")
    
    # Backtest performance summary
    print("\n3. BACKTEST PERFORMANCE")
    print("-" * 40)
    print(f"{'Model':<20} {'Sharpe':<10} {'Return':<10} {'Max DD':<10}")
    print("-" * 40)
    
    for model_name, result in backtest_results.items():
        if result is not None:
            sharpe = result.get('sharpe', 0)
            ret = result.get('total_return', 0)
            max_dd = result.get('max_dd', 0)
            print(f"{model_name:<20} {sharpe:<10.4f} {ret:<10.4f} {max_dd:<10.4f}")
    
    # Performance ranking
    print("\n4. PERFORMANCE RANKING")
    print("-" * 40)
    
    # Rank by Sharpe ratio
    valid_backtests = {k: v for k, v in backtest_results.items() if v is not None}
    if valid_backtests:
        sorted_by_sharpe = sorted(valid_backtests.items(), key=lambda x: x[1]['sharpe'], reverse=True)
        print("By Sharpe Ratio:")
        for i, (model_name, result) in enumerate(sorted_by_sharpe, 1):
            print(f"  {i}. {model_name}: {result['sharpe']:.4f}")
    
    # Analysis and recommendations
    print("\n5. ANALYSIS AND RECOMMENDATIONS")
    print("-" * 40)
    
    # Find best classification model
    best_bacc = 0
    best_bacc_model = None
    for category, models in model_results.items():
        if models is None:
            continue
        for model_name, result in models.items():
            if result is not None and result.get('test_bacc', 0) > best_bacc:
                best_bacc = result['test_bacc']
                best_bacc_model = model_name
    
    if best_bacc_model:
        print(f"Best classification model: {best_bacc_model} (BACC: {best_bacc:.4f})")
    
    # Find best backtest model
    best_sharpe = -np.inf
    best_backtest_model = None
    for model_name, result in backtest_results.items():
        if result is not None and result['sharpe'] > best_sharpe:
            best_sharpe = result['sharpe']
            best_backtest_model = model_name
    
    if best_backtest_model:
        print(f"Best backtest model: {best_backtest_model} (Sharpe: {best_sharpe:.4f})")
    
    # Problem diagnosis
    print("\nPROBLEM DIAGNOSIS:")
    
    # Check if any model beats baseline
    baseline_sharpe = backtest_results.get('baseline', {}).get('sharpe', 0)
    better_than_baseline = [name for name, result in backtest_results.items() 
                          if result is not None and result['sharpe'] > baseline_sharpe]
    
    if not better_than_baseline:
        print("❌ No model beats the baseline strategy")
        print("   This suggests the problem is with the task itself, not just the GRU model")
    
    # Check signal quality vs backtest performance correlation
    if signal_analyses and backtest_results:
        print("\nSIGNAL QUALITY vs BACKTEST PERFORMANCE:")
        for model_name in signal_analyses:
            if model_name in backtest_results and backtest_results[model_name] is not None:
                analysis = signal_analyses[model_name]
                backtest = backtest_results[model_name]
                strong_signals = np.sum(np.abs(analysis['signal_strength']) > 0.5) / len(analysis['signal_strength']) * 100
                print(f"  {model_name}: {strong_signals:.1f}% strong signals → {backtest['sharpe']:.4f} Sharpe")
    
    return {
        'best_classification': (best_bacc_model, best_bacc),
        'best_backtest': (best_backtest_model, best_sharpe),
        'baseline_sharpe': baseline_sharpe,
        'models_beat_baseline': better_than_baseline
    }


def main():
    """Run comprehensive model comparison."""
    print("Starting comprehensive model comparison...")
    print(f"Start time: {datetime.now()}")
    
    # Load data
    cache_path = Path("/tmp/btc_ohlcv_feat_5m_72h.pkl")
    if not cache_path.exists():
        print("No cached data found. Please run tests first.")
        return
    
    print("Loading data...")
    df_ohlcv, df_feat = pickle.loads(cache_path.read_bytes())
    
    # Build dataset
    print("Building dataset...")
    X, y = build_dataset(df_ohlcv, df_feat, lookback=60, horizon=2)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Run all model training
    model_results = {}
    
    # 1. Neural network models
    print("\n" + "=" * 60)
    print("TRAINING NEURAL NETWORK MODELS")
    print("=" * 60)
    
    neural_results = {}
    
    # GRU (already trained)
    print("\n--- GRU Model (using existing) ---")
    try:
        if Path('latest.pt').exists():
            neural_results['gru'] = {'val_bacc': 0.52, 'test_bacc': 0.52, 'ckpt_path': 'latest.pt'}
            print("GRU - Using existing model")
        else:
            print("Training GRU model...")
            neural_results['gru'] = train_gru(X, y, epochs=10)
    except Exception as e:
        print(f"GRU training failed: {e}")
        neural_results['gru'] = None
    
    # LSTM
    print("\n--- LSTM Model ---")
    try:
        neural_results['lstm'] = train_lstm(X, y, epochs=10)
    except Exception as e:
        print(f"LSTM training failed: {e}")
        neural_results['lstm'] = None
    
    # Simple NN
    print("\n--- Simple Feedforward NN ---")
    try:
        neural_results['simple_nn'] = train_simple_nn(X, y, epochs=10, model_type="feedforward")
    except Exception as e:
        print(f"Simple NN training failed: {e}")
        neural_results['simple_nn'] = None
    
    # Simple Conv
    print("\n--- Simple Conv Net ---")
    try:
        neural_results['simple_conv'] = train_simple_nn(X, y, epochs=10, model_type="conv")
    except Exception as e:
        print(f"Simple Conv training failed: {e}")
        neural_results['simple_conv'] = None
    
    model_results['neural_networks'] = neural_results
    
    # 2. Sklearn models
    print("\n" + "=" * 60)
    print("TRAINING SKLEARN MODELS")
    print("=" * 60)
    
    try:
        sklearn_results = run_all_sklearn_models(X, y)
        model_results['sklearn'] = sklearn_results
    except Exception as e:
        print(f"Sklearn models failed: {e}")
        model_results['sklearn'] = None
    
    # 3. Technical indicator models
    print("\n" + "=" * 60)
    print("TRAINING TECHNICAL INDICATOR MODELS")
    print("=" * 60)
    
    try:
        technical_results = run_all_technical_models(X, y)
        model_results['technical'] = technical_results
    except Exception as e:
        print(f"Technical models failed: {e}")
        model_results['technical'] = None
    
    # 4. Signal analysis for key models
    print("\n" + "=" * 60)
    print("SIGNAL ANALYSIS")
    print("=" * 60)
    
    signal_analyses = {}
    
    # Analyze GRU
    if Path('latest.pt').exists():
        try:
            gru_model = load_gru('latest.pt')
            signal_analyses['gru'] = analyze_model_signals(gru_model, X, y, "GRU")
        except Exception as e:
            print(f"GRU signal analysis failed: {e}")
    
    # Analyze LSTM
    if Path('lstm_latest.pt').exists():
        try:
            lstm_model = load_lstm('lstm_latest.pt')
            signal_analyses['lstm'] = analyze_model_signals(lstm_model, X, y, "LSTM")
        except Exception as e:
            print(f"LSTM signal analysis failed: {e}")
    
    # 5. Backtest comparison
    backtest_results = run_backtest_comparison(df_ohlcv, df_feat, model_results)
    
    # 6. Create comprehensive report
    report = create_comparison_report(model_results, backtest_results, signal_analyses)
    
    print(f"\nEnd time: {datetime.now()}")
    print("Model comparison completed!")
    
    return {
        'model_results': model_results,
        'backtest_results': backtest_results,
        'signal_analyses': signal_analyses,
        'report': report
    }


if __name__ == "__main__":
    results = main()