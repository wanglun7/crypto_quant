"""
Walk-Forward验证框架
实现2年训练窗口 → 1个月回测窗口的滚动验证
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
import json

from .backtest_engine import BacktestEngine, BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Walk-Forward验证窗口"""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    training_days: int
    testing_days: int


@dataclass
class WalkForwardResult:
    """单个窗口的回测结果"""
    window: WalkForwardWindow
    train_data_points: int
    test_data_points: int
    initial_capital: float
    final_portfolio_value: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    strategy_stats: Dict[str, Any]


class WalkForwardValidator:
    """Walk-Forward验证器"""
    
    def __init__(self, 
                 data_path: str = "./data/sample_data",
                 initial_capital: float = 100000.0,
                 training_months: int = 24,  # 2年 = 24个月
                 testing_months: int = 1,    # 1个月测试
                 step_months: int = 1):      # 每次向前滑动1个月
        
        self.data_path = data_path
        self.initial_capital = initial_capital
        self.training_months = training_months
        self.testing_months = testing_months
        self.step_months = step_months
        
        # 验证结果
        self.walk_forward_results: List[WalkForwardResult] = []
        self.aggregate_stats: Dict[str, float] = {}
    
    def generate_windows(self, 
                        overall_start: datetime, 
                        overall_end: datetime) -> List[WalkForwardWindow]:
        """生成Walk-Forward验证窗口"""
        
        windows = []
        window_id = 1
        
        # 第一个窗口的训练开始时间
        current_start = overall_start
        
        while True:
            # 计算训练窗口
            train_start = current_start
            train_end = train_start + timedelta(days=self.training_months * 30)
            
            # 计算测试窗口
            test_start = train_end
            test_end = test_start + timedelta(days=self.testing_months * 30)
            
            # 检查是否超出数据范围
            if test_end > overall_end:
                logger.info(f"Reached end of data. Generated {len(windows)} windows.")
                break
            
            # 创建窗口
            window = WalkForwardWindow(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                training_days=self.training_months * 30,
                testing_days=self.testing_months * 30
            )
            
            windows.append(window)
            
            logger.info(f"Window {window_id}: Train {train_start.date()} to {train_end.date()}, "
                       f"Test {test_start.date()} to {test_end.date()}")
            
            # 向前滑动
            current_start += timedelta(days=self.step_months * 30)
            window_id += 1
        
        return windows
    
    def run_single_window(self, 
                         window: WalkForwardWindow,
                         strategy_class,
                         strategy_params: Dict[str, Any],
                         symbol: str = "BTCUSDT") -> Optional[WalkForwardResult]:
        """运行单个窗口的验证"""
        
        logger.info(f"Running Window {window.window_id}...")
        
        try:
            # 第一步：在训练数据上"训练"策略（对于简单策略，这主要是参数验证）
            logger.debug(f"  Training period: {window.train_start} to {window.train_end}")
            
            # 创建策略实例
            strategy = strategy_class(**strategy_params)
            
            # 第二步：在测试数据上进行回测
            logger.debug(f"  Testing period: {window.test_start} to {window.test_end}")
            
            # 创建回测引擎
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                data_path=self.data_path
            )
            
            # 运行回测
            results = engine.run_backtest(
                strategy=strategy,
                symbol=symbol,
                start_date=window.test_start,
                end_date=window.test_end
            )
            
            # 获取策略统计
            strategy_stats = {}
            if hasattr(strategy, 'get_strategy_stats'):
                strategy_stats = strategy.get_strategy_stats()
            
            # 创建结果对象
            wf_result = WalkForwardResult(
                window=window,
                train_data_points=0,  # 对于简单策略，暂时不需要训练数据点数
                test_data_points=len(engine.portfolio.equity_curve),
                initial_capital=self.initial_capital,
                final_portfolio_value=results['portfolio_value'],
                total_return=results['total_return'],
                annual_return=results['performance_metrics']['annual_return'],
                sharpe_ratio=results['performance_metrics']['sharpe_ratio'],
                max_drawdown=results['performance_metrics']['max_drawdown'],
                total_trades=results['total_trades'],
                win_rate=results['performance_metrics']['win_rate'],
                strategy_stats=strategy_stats
            )
            
            logger.info(f"  Window {window.window_id} completed: "
                       f"Return {wf_result.total_return:.2%}, "
                       f"Sharpe {wf_result.sharpe_ratio:.2f}, "
                       f"Trades {wf_result.total_trades}")
            
            return wf_result
            
        except Exception as e:
            logger.error(f"Window {window.window_id} failed: {e}")
            return None
    
    def run_walk_forward_validation(self,
                                   strategy_class,
                                   strategy_params: Dict[str, Any],
                                   start_date: datetime,
                                   end_date: datetime,
                                   symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """运行完整的Walk-Forward验证"""
        
        logger.info("="*60)
        logger.info("🚀 开始Walk-Forward验证")
        logger.info("="*60)
        logger.info(f"策略: {strategy_class.__name__}")
        logger.info(f"参数: {strategy_params}")
        logger.info(f"时间范围: {start_date.date()} 到 {end_date.date()}")
        logger.info(f"训练窗口: {self.training_months}个月")
        logger.info(f"测试窗口: {self.testing_months}个月")
        
        # 生成验证窗口
        windows = self.generate_windows(start_date, end_date)
        logger.info(f"总共生成 {len(windows)} 个验证窗口")
        
        if len(windows) == 0:
            logger.error("没有足够的数据进行Walk-Forward验证")
            return {}
        
        # 运行每个窗口
        self.walk_forward_results = []
        successful_windows = 0
        
        for window in windows:
            result = self.run_single_window(window, strategy_class, strategy_params, symbol)
            
            if result:
                self.walk_forward_results.append(result)
                successful_windows += 1
            else:
                logger.warning(f"窗口 {window.window_id} 失败，跳过")
        
        logger.info(f"成功完成 {successful_windows}/{len(windows)} 个窗口的验证")
        
        # 计算聚合统计
        if self.walk_forward_results:
            self._calculate_aggregate_stats()
            return self._generate_summary_report()
        else:
            logger.error("所有窗口都失败了")
            return {}
    
    def _calculate_aggregate_stats(self):
        """计算聚合统计指标"""
        
        if not self.walk_forward_results:
            return
        
        # 提取各个指标
        returns = [r.total_return for r in self.walk_forward_results]
        annual_returns = [r.annual_return for r in self.walk_forward_results]
        sharpe_ratios = [r.sharpe_ratio for r in self.walk_forward_results if not np.isnan(r.sharpe_ratio)]
        max_drawdowns = [r.max_drawdown for r in self.walk_forward_results]
        trade_counts = [r.total_trades for r in self.walk_forward_results]
        win_rates = [r.win_rate for r in self.walk_forward_results]
        
        # 计算统计量
        self.aggregate_stats = {
            # 收益率统计
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'min_return': np.min(returns),
            'max_return': np.max(returns),
            
            # 年化收益率统计
            'mean_annual_return': np.mean(annual_returns),
            'median_annual_return': np.median(annual_returns),
            'std_annual_return': np.std(annual_returns),
            
            # 夏普比率统计
            'mean_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'median_sharpe': np.median(sharpe_ratios) if sharpe_ratios else 0,
            'std_sharpe': np.std(sharpe_ratios) if sharpe_ratios else 0,
            
            # 最大回撤统计
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns),  # 最差回撤（最负）
            'best_max_drawdown': np.max(max_drawdowns),   # 最好回撤（最接近0）
            
            # 交易统计
            'mean_trades': np.mean(trade_counts),
            'total_trades': np.sum(trade_counts),
            'mean_win_rate': np.mean(win_rates),
            
            # 稳定性指标
            'positive_windows': sum(1 for r in returns if r > 0),
            'negative_windows': sum(1 for r in returns if r <= 0),
            'success_rate': sum(1 for r in returns if r > 0) / len(returns),
            
            # 复合统计
            'total_windows': len(self.walk_forward_results),
            'return_volatility': np.std(returns),
            'sharpe_consistency': len([s for s in sharpe_ratios if s > 1]) / len(sharpe_ratios) if sharpe_ratios else 0
        }
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """生成汇总报告"""
        
        summary = {
            'validation_summary': {
                'strategy_name': self.walk_forward_results[0].strategy_stats.get('strategy_name', 'Unknown') if self.walk_forward_results else 'Unknown',
                'total_windows': self.aggregate_stats['total_windows'],
                'successful_windows': len(self.walk_forward_results),
                'training_months': self.training_months,
                'testing_months': self.testing_months,
            },
            
            'performance_metrics': {
                'mean_return': self.aggregate_stats['mean_return'],
                'return_volatility': self.aggregate_stats['return_volatility'],
                'mean_annual_return': self.aggregate_stats['mean_annual_return'],
                'mean_sharpe_ratio': self.aggregate_stats['mean_sharpe'],
                'mean_max_drawdown': self.aggregate_stats['mean_max_drawdown'],
                'worst_max_drawdown': self.aggregate_stats['worst_max_drawdown'],
                'success_rate': self.aggregate_stats['success_rate'],
                'sharpe_consistency': self.aggregate_stats['sharpe_consistency']
            },
            
            'detailed_results': [
                {
                    'window_id': r.window.window_id,
                    'test_period': f"{r.window.test_start.date()} to {r.window.test_end.date()}",
                    'total_return': r.total_return,
                    'annual_return': r.annual_return,
                    'sharpe_ratio': r.sharpe_ratio,
                    'max_drawdown': r.max_drawdown,
                    'total_trades': r.total_trades,
                    'win_rate': r.win_rate
                } for r in self.walk_forward_results
            ],
            
            'aggregate_stats': self.aggregate_stats
        }
        
        return summary
    
    def save_results(self, filename: str = "walk_forward_results.json"):
        """保存验证结果"""
        
        if not self.walk_forward_results:
            logger.warning("没有结果可以保存")
            return
        
        # 生成完整报告
        report = self._generate_summary_report()
        
        # 保存到文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Walk-Forward验证结果已保存到: {filename}")
    
    def print_summary(self):
        """打印汇总结果"""
        
        if not self.walk_forward_results:
            logger.warning("没有结果可以显示")
            return
        
        print("\n" + "="*60)
        print("📊 Walk-Forward验证汇总报告")
        print("="*60)
        
        # 基本信息
        strategy_name = self.walk_forward_results[0].strategy_stats.get('strategy_name', 'Unknown')
        print(f"策略名称: {strategy_name}")
        print(f"验证窗口: {self.aggregate_stats['total_windows']} 个")
        print(f"成功率: {self.aggregate_stats['success_rate']:.1%}")
        print()
        
        # 性能指标
        print("🎯 核心性能指标:")
        print(f"  平均收益率: {self.aggregate_stats['mean_return']:.2%} ± {self.aggregate_stats['std_return']:.2%}")
        print(f"  平均年化收益率: {self.aggregate_stats['mean_annual_return']:.2%}")
        print(f"  平均夏普比率: {self.aggregate_stats['mean_sharpe']:.2f}")
        print(f"  平均最大回撤: {self.aggregate_stats['mean_max_drawdown']:.2%}")
        print(f"  最差回撤: {self.aggregate_stats['worst_max_drawdown']:.2%}")
        print()
        
        # 稳定性指标
        print("📈 稳定性分析:")
        print(f"  盈利窗口: {self.aggregate_stats['positive_windows']}/{self.aggregate_stats['total_windows']}")
        print(f"  夏普比率>1的窗口占比: {self.aggregate_stats['sharpe_consistency']:.1%}")
        print(f"  收益率波动性: {self.aggregate_stats['return_volatility']:.2%}")
        print()
        
        # 最佳和最差窗口
        best_window = max(self.walk_forward_results, key=lambda x: x.total_return)
        worst_window = min(self.walk_forward_results, key=lambda x: x.total_return)
        
        print("🏆 最佳窗口:")
        print(f"  窗口 {best_window.window.window_id}: {best_window.total_return:.2%} "
              f"({best_window.window.test_start.date()} - {best_window.window.test_end.date()})")
        
        print("⚠️ 最差窗口:")
        print(f"  窗口 {worst_window.window.window_id}: {worst_window.total_return:.2%} "
              f"({worst_window.window.test_start.date()} - {worst_window.window.test_end.date()})")
        
        print("\n" + "="*60)