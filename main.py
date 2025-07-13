#!/usr/bin/env python3
"""
BTC Quantitative Trading Strategy - Main Entry Point

This script provides a unified interface for:
1. Training CNN-LSTM models
2. Running backtests
3. Analyzing results

Usage:
  python main.py train         # Train new model
  python main.py backtest      # Run backtest with existing model
  python main.py analyze       # Analyze threshold optimization
  python main.py full          # Full pipeline: train + backtest
"""

import sys
import argparse
from pathlib import Path
import structlog

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.simple_backtest import calculate_returns
from scripts.simple_fix import fix_threshold_and_retrain
from scripts.train_quick_fix import train_with_focal_loss
from scripts.train_optimized import main as train_optimized_main
from scripts.rolling_backtest import main as rolling_backtest_main
from scripts.test_intelligent_strategy import main as test_intelligent_strategy_main

logger = structlog.get_logger(__name__)


def setup_logging():
    """Configure structured logging."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def cmd_train(args):
    """Train new CNN-LSTM model with optimized pipeline."""
    logger.info("Starting optimized model training")
    
    try:
        model, metrics, history = train_optimized_main(test_mode=False)
        logger.info("Training completed", metrics=metrics)
        print(f"\n✅ 30天优化训练完成！模型已保存到 models/")
        print(f"最终F1分数: {metrics['f1_score']:.4f}")
        print(f"准确率: {metrics['accuracy']:.2%}")
        print(f"AUC分数: {metrics['auc_score']:.4f}")
        return True
    except Exception as e:
        logger.error("Training failed", error=str(e))
        print(f"❌ 训练失败: {e}")
        return False


def cmd_train_test(args):
    """Test optimized training pipeline with 5 epochs."""
    logger.info("Starting optimized training test")
    
    try:
        model, metrics, history = train_optimized_main(test_mode=True)
        logger.info("Test training completed", metrics=metrics)
        print(f"\n✅ 测试训练完成！(5个epochs)")
        print(f"最终F1分数: {metrics['f1_score']:.4f}")
        print(f"准确率: {metrics['accuracy']:.2%}")
        print(f"AUC分数: {metrics['auc_score']:.4f}")
        print(f"💡 如果结果看起来合理，可运行 'python main.py train' 进行完整训练")
        return True
    except Exception as e:
        logger.error("Test training failed", error=str(e))
        print(f"❌ 测试训练失败: {e}")
        return False


def cmd_train_legacy(args):
    """Train CNN-LSTM model with legacy 3-day pipeline."""
    logger.info("Starting legacy model training")
    
    try:
        model, metrics, history = train_with_focal_loss()
        logger.info("Training completed", metrics=metrics)
        print(f"\n✅ Training completed. Model saved to models/")
        print(f"Final F1 Score: {metrics['f1_score']:.4f}")
        return True
    except Exception as e:
        logger.error("Training failed", error=str(e))
        print(f"❌ Training failed: {e}")
        return False


def cmd_backtest(args):
    """Run backtest with existing model."""
    logger.info("Starting backtest")
    
    try:
        result = calculate_returns()
        logger.info("Backtest completed", result=result)
        
        print(f"\n✅ Backtest completed")
        print(f"Strategy Return: {result['strategy_return']:+.2%}")
        print(f"Buy & Hold Return: {result['buy_hold_return']:+.2%}")
        
        if result['strategy_return'] > result['buy_hold_return']:
            print(f"🎯 Strategy BEATS buy & hold by {result['excess_return']:+.2%}")
        else:
            print(f"📉 Strategy loses to buy & hold by {-result['excess_return']:.2%}")
        
        return True
    except Exception as e:
        logger.error("Backtest failed", error=str(e))
        print(f"❌ Backtest failed: {e}")
        return False


def cmd_analyze(args):
    """Analyze threshold optimization."""
    logger.info("Starting threshold analysis")
    
    try:
        result = fix_threshold_and_retrain()
        if result is None or result[0] is None:
            logger.error("Analysis failed", error="No model or invalid result")
            print("❌ Analysis failed: No trained model available")
            return False
            
        best_threshold, metrics = result
        logger.info("Analysis completed", threshold=best_threshold, metrics=metrics)
        
        print(f"\n✅ Analysis completed")
        print(f"Optimal threshold: {best_threshold}")
        print(f"Best F1 Score: {metrics['f1_score']:.4f}")
        return True
    except Exception as e:
        logger.error("Analysis failed", error=str(e))
        print(f"❌ Analysis failed: {e}")
        return False


def cmd_rolling_backtest(args):
    """Run rolling window backtest."""
    logger.info("Starting rolling window backtest")
    
    try:
        success = rolling_backtest_main()
        if success:
            print(f"\n✅ 滚动窗口回测完成！")
            print(f"📊 详细结果已保存到 results/rolling_backtest_results.json")
            return True
        else:
            print(f"❌ 滚动窗口回测失败")
            return False
    except Exception as e:
        logger.error("Rolling backtest failed", error=str(e))
        print(f"❌ 滚动回测失败: {e}")
        return False


def cmd_test_intelligent_strategy(args):
    """Test intelligent position management strategy."""
    logger.info("Starting intelligent strategy test")
    
    try:
        success = test_intelligent_strategy_main()
        if success:
            print(f"\n✅ 智能策略测试完成！")
            print(f"📊 详细结果已保存到 results/intelligent_strategy_test.json")
            print(f"📈 对比图表已保存到 results/strategy_comparison.png") 
            return True
        else:
            print(f"❌ 智能策略测试失败")
            return False
    except Exception as e:
        logger.error("Intelligent strategy test failed", error=str(e))
        print(f"❌ 智能策略测试失败: {e}")
        return False


def cmd_full(args):
    """Run full pipeline: train + backtest."""
    logger.info("Starting full pipeline")
    
    print("🚀 Running full CNN-LSTM pipeline...")
    print("="*60)
    
    # Step 1: Train
    print("Step 1/3: Training model...")
    if not cmd_train(args):
        return False
    
    # Step 2: Analyze
    print("\nStep 2/3: Analyzing thresholds...")
    if not cmd_analyze(args):
        return False
    
    # Step 3: Backtest
    print("\nStep 3/3: Running backtest...")
    if not cmd_backtest(args):
        return False
    
    print("\n🎉 Full pipeline completed successfully!")
    return True


def main():
    """Main entry point."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="BTC Quantitative Trading Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py test-intelligent  # Test intelligent position management strategy
  python main.py train-test        # Test optimized pipeline (5 epochs)
  python main.py train             # Train with 30-day optimized pipeline
  python main.py rolling-backtest  # Run rolling window backtest
  python main.py backtest          # Run simple backtest
  python main.py analyze           # Analyze thresholds
  python main.py full              # Full pipeline
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command (optimized 30-day)
    train_parser = subparsers.add_parser('train', help='Train CNN-LSTM with 30-day optimized pipeline')
    
    # Test train command (5 epochs)
    train_test_parser = subparsers.add_parser('train-test', help='Test optimized training pipeline (5 epochs)')
    
    # Legacy train command (3-day)
    train_legacy_parser = subparsers.add_parser('train-legacy', help='Train CNN-LSTM with legacy 3-day pipeline')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtest with existing model')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze threshold optimization')
    
    # Rolling backtest command
    rolling_parser = subparsers.add_parser('rolling-backtest', help='Run rolling window backtest')
    
    # Intelligent strategy test command
    intelligent_parser = subparsers.add_parser('test-intelligent', help='Test intelligent position management strategy')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run full pipeline: train + backtest')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Route to appropriate command
    commands = {
        'train': cmd_train,
        'train-test': cmd_train_test,
        'train-legacy': cmd_train_legacy,
        'backtest': cmd_backtest,
        'analyze': cmd_analyze,
        'rolling-backtest': cmd_rolling_backtest,
        'test-intelligent': cmd_test_intelligent_strategy,
        'full': cmd_full
    }
    
    try:
        success = commands[args.command](args)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return 1
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())