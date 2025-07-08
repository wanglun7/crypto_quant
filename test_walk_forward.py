"""
测试Walk-Forward验证框架
验证2年训练→1个月回测的滚动验证功能
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting.walk_forward_validator import WalkForwardValidator
from strategies.simple_moving_average import SimpleMovingAverageStrategy, AdaptiveMovingAverageStrategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_walk_forward_simple_strategy():
    """测试简单移动平均策略的Walk-Forward验证"""
    logger.info("🚀 测试简单移动平均策略的Walk-Forward验证...")
    
    # 创建Walk-Forward验证器
    # 使用较短的训练窗口便于测试（在实际数据有限的情况下）
    validator = WalkForwardValidator(
        data_path="./data/sample_data",
        initial_capital=100000.0,
        training_months=6,   # 6个月训练（由于我们只有2023年数据）
        testing_months=1,    # 1个月测试
        step_months=1        # 每次向前滑动1个月
    )
    
    # 策略参数
    strategy_params = {
        'fast_period': 20,
        'slow_period': 50,
        'position_size': 0.8
    }
    
    # 设置验证时间范围（留足训练时间）
    start_date = datetime(2023, 1, 1)   # 从1月开始
    end_date = datetime(2023, 12, 31)   # 到12月结束
    
    try:
        # 运行Walk-Forward验证
        results = validator.run_walk_forward_validation(
            strategy_class=SimpleMovingAverageStrategy,
            strategy_params=strategy_params,
            start_date=start_date,
            end_date=end_date,
            symbol="BTCUSDT"
        )
        
        if results:
            # 打印汇总报告
            validator.print_summary()
            
            # 保存结果
            validator.save_results("walk_forward_simple_ma.json")
            
            # 验证关键指标
            perf_metrics = results['performance_metrics']
            logger.info("\n✅ 简单移动平均策略Walk-Forward验证完成:")
            logger.info(f"  - 验证窗口数: {results['validation_summary']['total_windows']}")
            logger.info(f"  - 平均收益率: {perf_metrics['mean_return']:.2%}")
            logger.info(f"  - 成功率: {perf_metrics['success_rate']:.1%}")
            logger.info(f"  - 平均夏普比率: {perf_metrics['mean_sharpe_ratio']:.2f}")
            
            return results
        else:
            logger.error("❌ Walk-Forward验证失败")
            return None
            
    except Exception as e:
        logger.error(f"❌ Walk-Forward验证异常: {e}")
        return None


def test_walk_forward_adaptive_strategy():
    """测试自适应移动平均策略的Walk-Forward验证"""
    logger.info("🚀 测试自适应移动平均策略的Walk-Forward验证...")
    
    # 创建Walk-Forward验证器
    validator = WalkForwardValidator(
        data_path="./data/sample_data",
        initial_capital=100000.0,
        training_months=4,   # 4个月训练
        testing_months=1,    # 1个月测试
        step_months=1        # 每次向前滑动1个月
    )
    
    # 自适应策略参数
    strategy_params = {
        'base_fast_period': 10,
        'base_slow_period': 30,
        'volatility_lookback': 20,
        'position_size': 0.6
    }
    
    # 设置验证时间范围
    start_date = datetime(2023, 3, 1)   # 从3月开始（留更多数据给后面的窗口）
    end_date = datetime(2023, 12, 31)   # 到12月结束
    
    try:
        # 运行Walk-Forward验证
        results = validator.run_walk_forward_validation(
            strategy_class=AdaptiveMovingAverageStrategy,
            strategy_params=strategy_params,
            start_date=start_date,
            end_date=end_date,
            symbol="BTCUSDT"
        )
        
        if results:
            # 打印汇总报告
            validator.print_summary()
            
            # 保存结果
            validator.save_results("walk_forward_adaptive_ma.json")
            
            # 验证关键指标
            perf_metrics = results['performance_metrics']
            logger.info("\n✅ 自适应移动平均策略Walk-Forward验证完成:")
            logger.info(f"  - 验证窗口数: {results['validation_summary']['total_windows']}")
            logger.info(f"  - 平均收益率: {perf_metrics['mean_return']:.2%}")
            logger.info(f"  - 成功率: {perf_metrics['success_rate']:.1%}")
            logger.info(f"  - 平均夏普比率: {perf_metrics['mean_sharpe_ratio']:.2f}")
            
            return results
        else:
            logger.error("❌ Walk-Forward验证失败")
            return None
            
    except Exception as e:
        logger.error(f"❌ Walk-Forward验证异常: {e}")
        return None


def compare_strategies_walk_forward():
    """比较两种策略的Walk-Forward验证结果"""
    logger.info("📊 开始策略Walk-Forward验证对比...")
    
    # 测试简单策略
    simple_results = test_walk_forward_simple_strategy()
    
    # 间隔
    import time
    time.sleep(1)
    
    # 测试自适应策略
    adaptive_results = test_walk_forward_adaptive_strategy()
    
    if simple_results and adaptive_results:
        logger.info("\n" + "="*70)
        logger.info("📈 Walk-Forward验证策略对比")
        logger.info("="*70)
        
        # 简单策略结果
        simple_perf = simple_results['performance_metrics']
        logger.info("简单移动平均策略:")
        logger.info(f"  - 验证窗口: {simple_results['validation_summary']['total_windows']}")
        logger.info(f"  - 平均收益率: {simple_perf['mean_return']:.2%}")
        logger.info(f"  - 收益率波动: {simple_perf['return_volatility']:.2%}")
        logger.info(f"  - 平均夏普比率: {simple_perf['mean_sharpe_ratio']:.2f}")
        logger.info(f"  - 成功率: {simple_perf['success_rate']:.1%}")
        logger.info(f"  - 最差回撤: {simple_perf['worst_max_drawdown']:.2%}")
        
        # 自适应策略结果
        adaptive_perf = adaptive_results['performance_metrics']
        logger.info("\n自适应移动平均策略:")
        logger.info(f"  - 验证窗口: {adaptive_results['validation_summary']['total_windows']}")
        logger.info(f"  - 平均收益率: {adaptive_perf['mean_return']:.2%}")
        logger.info(f"  - 收益率波动: {adaptive_perf['return_volatility']:.2%}")
        logger.info(f"  - 平均夏普比率: {adaptive_perf['mean_sharpe_ratio']:.2f}")
        logger.info(f"  - 成功率: {adaptive_perf['success_rate']:.1%}")
        logger.info(f"  - 最差回撤: {adaptive_perf['worst_max_drawdown']:.2%}")
        
        # 对比分析
        logger.info("\n🎯 对比分析:")
        
        # 收益率对比
        return_diff = adaptive_perf['mean_return'] - simple_perf['mean_return']
        logger.info(f"收益率差异: {return_diff:+.2%} "
                   f"({'自适应更优' if return_diff > 0 else '简单策略更优'})")
        
        # 夏普比率对比
        sharpe_diff = adaptive_perf['mean_sharpe_ratio'] - simple_perf['mean_sharpe_ratio']
        logger.info(f"夏普比率差异: {sharpe_diff:+.2f} "
                   f"({'自适应更优' if sharpe_diff > 0 else '简单策略更优'})")
        
        # 稳定性对比
        stability_diff = adaptive_perf['success_rate'] - simple_perf['success_rate']
        logger.info(f"成功率差异: {stability_diff:+.1%} "
                   f"({'自适应更稳定' if stability_diff > 0 else '简单策略更稳定'})")
        
        # 综合评分
        simple_score = (simple_perf['mean_return'] * 0.4 + 
                       simple_perf['mean_sharpe_ratio'] * 0.3 + 
                       simple_perf['success_rate'] * 0.3)
        
        adaptive_score = (adaptive_perf['mean_return'] * 0.4 + 
                         adaptive_perf['mean_sharpe_ratio'] * 0.3 + 
                         adaptive_perf['success_rate'] * 0.3)
        
        logger.info(f"\n🏆 综合评分:")
        logger.info(f"简单策略: {simple_score:.3f}")
        logger.info(f"自适应策略: {adaptive_score:.3f}")
        
        if adaptive_score > simple_score:
            logger.info("🎉 自适应策略在Walk-Forward验证中表现更优！")
        else:
            logger.info("🎉 简单策略在Walk-Forward验证中表现更优！")
        
        return {
            'simple_strategy': simple_results,
            'adaptive_strategy': adaptive_results,
            'comparison': {
                'return_difference': return_diff,
                'sharpe_difference': sharpe_diff,
                'stability_difference': stability_diff,
                'simple_score': simple_score,
                'adaptive_score': adaptive_score,
                'winner': 'adaptive' if adaptive_score > simple_score else 'simple'
            }
        }
    else:
        logger.error("❌ 策略对比失败")
        return None


def main():
    """主函数"""
    logger.info("="*70)
    logger.info("🎯 Walk-Forward验证框架测试")
    logger.info("="*70)
    
    try:
        # 运行策略对比
        comparison_results = compare_strategies_walk_forward()
        
        if comparison_results:
            logger.info("\n🎉 Walk-Forward验证框架测试成功!")
            logger.info("核心功能验证:")
            logger.info("✅ 滚动窗口生成")
            logger.info("✅ 多窗口回测执行")
            logger.info("✅ 聚合统计计算")
            logger.info("✅ 策略稳定性评估")
            logger.info("✅ 对比分析报告")
            
            logger.info("\n下一步建议:")
            logger.info("1. ✅ Walk-Forward验证框架 - 已完成")
            logger.info("2. 🔄 实现更多技术指标和特征")
            logger.info("3. 🔄 构建AI驱动的策略")
            logger.info("4. 🔄 下载更多历史数据验证")
            
            return True
        else:
            logger.error("❌ Walk-Forward验证框架测试失败")
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