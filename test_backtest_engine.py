"""
测试回测引擎
验证基础功能并运行简单策略
"""

import sys
import os
import logging
from datetime import datetime, timedelta
import json

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting.backtest_engine import BacktestEngine
from strategies.simple_moving_average import SimpleMovingAverageStrategy, AdaptiveMovingAverageStrategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_simple_strategy():
    """测试简单移动平均策略"""
    logger.info("🚀 开始测试简单移动平均策略...")
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=100000.0,
        data_path="./data/sample_data"
    )
    
    # 创建策略
    strategy = SimpleMovingAverageStrategy(
        fast_period=20,
        slow_period=50,
        position_size=0.8  # 使用80%资金
    )
    
    # 设置回测时间范围 (2023年前6个月)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 6, 30)
    
    try:
        # 运行回测
        results = engine.run_backtest(
            strategy=strategy,
            symbol="BTCUSDT",
            start_date=start_date,
            end_date=end_date
        )
        
        # 输出结果
        logger.info("📊 简单移动平均策略回测结果:")
        logger.info(f"  - 初始资金: ${engine.portfolio.initial_capital:,.2f}")
        logger.info(f"  - 最终价值: ${results['portfolio_value']:,.2f}")
        logger.info(f"  - 总收益率: {results['total_return']:.2%}")
        logger.info(f"  - 交易次数: {results['total_trades']}")
        
        # 性能指标
        metrics = results['performance_metrics']
        logger.info(f"  - 年化收益率: {metrics['annual_return']:.2%}")
        logger.info(f"  - 夏普比率: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  - 最大回撤: {metrics['max_drawdown']:.2%}")
        logger.info(f"  - 胜率: {metrics['win_rate']:.2%}")
        
        # 策略统计
        strategy_stats = strategy.get_strategy_stats()
        logger.info(f"  - 策略信号数: {strategy_stats['total_signals']}")
        logger.info(f"  - 当前快线MA: {strategy_stats['current_fast_ma']:.2f}")
        logger.info(f"  - 当前慢线MA: {strategy_stats['current_slow_ma']:.2f}")
        
        return results, strategy_stats
        
    except Exception as e:
        logger.error(f"❌ 简单策略回测失败: {e}")
        return None, None


def test_adaptive_strategy():
    """测试自适应移动平均策略"""
    logger.info("🚀 开始测试自适应移动平均策略...")
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=100000.0,
        data_path="./data/sample_data"
    )
    
    # 创建自适应策略
    strategy = AdaptiveMovingAverageStrategy(
        base_fast_period=10,
        base_slow_period=30,
        volatility_lookback=20,
        position_size=0.6  # 使用60%资金
    )
    
    # 设置回测时间范围 (2023年后6个月)
    start_date = datetime(2023, 7, 1)
    end_date = datetime(2023, 12, 31)
    
    try:
        # 运行回测
        results = engine.run_backtest(
            strategy=strategy,
            symbol="BTCUSDT",
            start_date=start_date,
            end_date=end_date
        )
        
        # 输出结果
        logger.info("📊 自适应移动平均策略回测结果:")
        logger.info(f"  - 初始资金: ${engine.portfolio.initial_capital:,.2f}")
        logger.info(f"  - 最终价值: ${results['portfolio_value']:,.2f}")
        logger.info(f"  - 总收益率: {results['total_return']:.2%}")
        logger.info(f"  - 交易次数: {results['total_trades']}")
        
        # 性能指标
        metrics = results['performance_metrics']
        logger.info(f"  - 年化收益率: {metrics['annual_return']:.2%}")
        logger.info(f"  - 夏普比率: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  - 最大回撤: {metrics['max_drawdown']:.2%}")
        logger.info(f"  - 胜率: {metrics['win_rate']:.2%}")
        
        # 策略统计
        strategy_stats = strategy.get_strategy_stats()
        logger.info(f"  - 策略信号数: {strategy_stats['total_signals']}")
        logger.info(f"  - 当前波动率: {strategy_stats['current_volatility']:.3f}")
        logger.info(f"  - 自适应快线周期: {strategy_stats['current_adaptive_fast_period']}")
        logger.info(f"  - 自适应慢线周期: {strategy_stats['current_adaptive_slow_period']}")
        
        return results, strategy_stats
        
    except Exception as e:
        logger.error(f"❌ 自适应策略回测失败: {e}")
        return None, None


def compare_strategies():
    """比较不同策略的表现"""
    logger.info("📈 开始策略比较测试...")
    
    # 测试简单策略
    simple_results, simple_stats = test_simple_strategy()
    
    # 短暂延迟
    import time
    time.sleep(1)
    
    # 测试自适应策略
    adaptive_results, adaptive_stats = test_adaptive_strategy()
    
    if simple_results and adaptive_results:
        logger.info("\n" + "="*60)
        logger.info("📊 策略对比结果")
        logger.info("="*60)
        
        logger.info(f"简单移动平均策略:")
        logger.info(f"  - 总收益率: {simple_results['total_return']:.2%}")
        logger.info(f"  - 年化收益率: {simple_results['performance_metrics']['annual_return']:.2%}")
        logger.info(f"  - 夏普比率: {simple_results['performance_metrics']['sharpe_ratio']:.2f}")
        logger.info(f"  - 最大回撤: {simple_results['performance_metrics']['max_drawdown']:.2%}")
        logger.info(f"  - 交易次数: {simple_results['total_trades']}")
        
        logger.info(f"\n自适应移动平均策略:")
        logger.info(f"  - 总收益率: {adaptive_results['total_return']:.2%}")
        logger.info(f"  - 年化收益率: {adaptive_results['performance_metrics']['annual_return']:.2%}")
        logger.info(f"  - 夏普比率: {adaptive_results['performance_metrics']['sharpe_ratio']:.2f}")
        logger.info(f"  - 最大回撤: {adaptive_results['performance_metrics']['max_drawdown']:.2%}")
        logger.info(f"  - 交易次数: {adaptive_results['total_trades']}")
        
        # 保存详细结果
        comparison_results = {
            'simple_strategy': {
                'results': simple_results,
                'stats': simple_stats
            },
            'adaptive_strategy': {
                'results': adaptive_results,
                'stats': adaptive_stats
            }
        }
        
        # 保存到文件
        with open('backtest_comparison_results.json', 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        logger.info(f"\n📄 详细结果已保存到: backtest_comparison_results.json")
        
        return comparison_results
    else:
        logger.error("❌ 策略比较失败")
        return None


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("🎯 回测引擎测试 - 2023年BTCUSDT数据")
    logger.info("="*60)
    
    try:
        # 运行策略比较
        comparison_results = compare_strategies()
        
        if comparison_results:
            logger.info("\n🎉 回测引擎测试完成!")
            logger.info("回测引擎基本功能验证通过")
            logger.info("下一步建议:")
            logger.info("1. 实现更多技术指标和特征")
            logger.info("2. 构建机器学习策略")
            logger.info("3. 实现Walk-Forward验证")
            logger.info("4. 集成链上数据和微观结构特征")
            return True
        else:
            logger.error("❌ 回测引擎测试失败")
            return False
            
    except Exception as e:
        logger.error(f"测试程序出错: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序错误: {e}")
        sys.exit(1)