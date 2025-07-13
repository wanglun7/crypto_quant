#!/usr/bin/env python3
"""
滚动窗口回测工具

实现时间序列交叉验证，评估模型在不同时间段的稳定性：
1. 固定大小训练窗口（如7天）
2. 固定预测窗口（如1天）  
3. 向前滚动评估
4. 聚合多个窗口的表现

目标：检测模型是否真正具备预测能力，还是仅仅过拟合了特定时间段
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import structlog
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_quant.models.cnn_lstm import CNN_LSTM
from crypto_quant.utils.indicators import prepare_extended_data, add_extended_features
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = structlog.get_logger(__name__)


class RollingBacktester:
    """滚动窗口回测器"""
    
    def __init__(self, train_days=7, test_days=1, step_days=1):
        """
        Args:
            train_days: 训练窗口天数
            test_days: 测试窗口天数  
            step_days: 滚动步长天数
        """
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.results = []
        
    def create_time_windows(self, df):
        """创建滚动时间窗口"""
        
        # 确保数据按时间排序
        df = df.sort_values('timestamp')
        
        # 计算每天的分钟数（假设1分钟数据）
        minutes_per_day = 24 * 60
        train_size = self.train_days * minutes_per_day
        test_size = self.test_days * minutes_per_day
        step_size = self.step_days * minutes_per_day
        
        windows = []
        start_idx = 0
        
        while start_idx + train_size + test_size <= len(df):
            train_end = start_idx + train_size
            test_end = train_end + test_size
            
            train_data = df.iloc[start_idx:train_end].copy()
            test_data = df.iloc[train_end:test_end].copy()
            
            # 记录时间窗口信息
            window_info = {
                'window_id': len(windows),
                'train_start': train_data['timestamp'].min(),
                'train_end': train_data['timestamp'].max(),
                'test_start': test_data['timestamp'].min(),
                'test_end': test_data['timestamp'].max(),
                'train_size': len(train_data),
                'test_size': len(test_data)
            }
            
            windows.append((train_data, test_data, window_info))
            start_idx += step_size
        
        logger.info(f"创建滚动窗口", total_windows=len(windows), 
                   train_days=self.train_days, test_days=self.test_days)
        
        return windows
    
    def train_model_for_window(self, train_data, test_data, sequence_length=60):
        """为单个窗口训练模型"""
        
        try:
            # 准备训练数据
            X_train, y_train = prepare_extended_data(train_data, sequence_length)
            X_test, y_test = prepare_extended_data(test_data, sequence_length)
            
            if len(X_train) < 100:  # 最少需要100个样本
                logger.warning("训练样本不足", samples=len(X_train))
                return None, None, None
            
            # 创建模型
            model = CNN_LSTM(
                n_features=26,  # 扩展特征集
                sequence_length=sequence_length,
                cnn_filters=[32, 64, 128],
                lstm_hidden_size=64,
                lstm_num_layers=2,
                dropout_rate=0.3,
                fc_hidden_size=64
            )
            
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            model.to(device)
            
            # 快速训练（少量epochs，针对滚动回测）
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            # 转换为PyTorch张量
            X_train_tensor = torch.FloatTensor(X_train).to(device)
            y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1).to(device)
            
            # 训练循环（快速版本）
            model.train()
            epochs = 10  # 较少epochs以节省时间
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 预测
            model.eval()
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            
            with torch.no_grad():
                test_outputs = model(X_test_tensor).cpu().numpy().flatten()
            
            return model, test_outputs, y_test
            
        except Exception as e:
            logger.error("模型训练失败", error=str(e))
            return None, None, None
    
    def calculate_trading_metrics(self, probabilities, y_true, prices_data, threshold=0.3):
        """计算交易指标"""
        
        if probabilities is None or len(probabilities) == 0:
            return {}
        
        # 生成信号
        signals = (probabilities > threshold).astype(int)
        
        # 计算分类指标
        accuracy = accuracy_score(y_true, signals)
        precision = precision_score(y_true, signals, zero_division=0)
        recall = recall_score(y_true, signals, zero_division=0)
        f1 = f1_score(y_true, signals, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, probabilities)
        except:
            auc = 0.5
        
        # 计算交易收益（简化版）
        if len(prices_data) >= len(signals):
            price_returns = prices_data['close'].pct_change().iloc[1:len(signals)+1].values
            signal_returns = signals[:-1] * price_returns  # 前一个信号对应下一个收益
            
            strategy_return = (1 + signal_returns).prod() - 1
            buy_hold_return = (prices_data['close'].iloc[-1] / prices_data['close'].iloc[0]) - 1
            
            # 简单交易成本
            trades = np.sum(np.diff(np.concatenate([[0], signals])) != 0)
            transaction_costs = trades * 0.0004  # 0.04% per trade
            net_return = strategy_return - transaction_costs
            
        else:
            strategy_return = 0
            buy_hold_return = 0
            net_return = 0
            trades = 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'strategy_return': strategy_return,
            'buy_hold_return': buy_hold_return,
            'net_return': net_return,
            'trades': trades,
            'prob_mean': probabilities.mean(),
            'prob_std': probabilities.std(),
            'signals_ratio': signals.mean()
        }
    
    def run_rolling_backtest(self, df):
        """运行完整的滚动回测"""
        
        print(f"🚀 开始滚动窗口回测")
        print(f"训练窗口: {self.train_days}天, 测试窗口: {self.test_days}天, 步长: {self.step_days}天")
        print("="*70)
        
        # 创建时间窗口
        windows = self.create_time_windows(df)
        
        if len(windows) == 0:
            print("❌ 无法创建时间窗口，数据可能不足")
            return {}
        
        # 处理每个窗口
        for i, (train_data, test_data, window_info) in enumerate(windows):
            
            print(f"\n🔄 窗口 {i+1}/{len(windows)}")
            print(f"   训练: {window_info['train_start'].date()} ~ {window_info['train_end'].date()}")
            print(f"   测试: {window_info['test_start'].date()} ~ {window_info['test_end'].date()}")
            
            # 训练并预测
            model, probabilities, y_true = self.train_model_for_window(train_data, test_data)
            
            if model is None:
                print(f"   ❌ 窗口 {i+1} 训练失败")
                continue
            
            # 计算指标
            metrics = self.calculate_trading_metrics(probabilities, y_true, test_data)
            
            # 添加窗口信息
            result = {**window_info, **metrics}
            self.results.append(result)
            
            # 显示结果
            print(f"   📊 准确率: {metrics['accuracy']:.3f}, F1: {metrics['f1_score']:.3f}")
            print(f"   💰 策略收益: {metrics['net_return']:+.2%}, 买入持有: {metrics['buy_hold_return']:+.2%}")
            
            # 内存管理
            del model, probabilities
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 聚合结果
        return self.aggregate_results()
    
    def aggregate_results(self):
        """聚合多个窗口的结果"""
        
        if not self.results:
            return {}
        
        # 数值指标的平均值
        numeric_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score', 
                          'strategy_return', 'buy_hold_return', 'net_return']
        
        aggregated = {}
        
        for metric in numeric_metrics:
            values = [r[metric] for r in self.results if metric in r and not np.isnan(r[metric])]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
        
        # 胜率统计
        strategy_wins = sum(1 for r in self.results if r.get('net_return', 0) > r.get('buy_hold_return', 0))
        aggregated['strategy_win_rate'] = strategy_wins / len(self.results)
        
        # 累积收益
        cumulative_strategy = 1
        cumulative_buy_hold = 1
        
        for r in self.results:
            cumulative_strategy *= (1 + r.get('net_return', 0))
            cumulative_buy_hold *= (1 + r.get('buy_hold_return', 0))
        
        aggregated['cumulative_strategy_return'] = cumulative_strategy - 1
        aggregated['cumulative_buy_hold_return'] = cumulative_buy_hold - 1
        
        # 总体统计
        aggregated['total_windows'] = len(self.results)
        aggregated['successful_windows'] = len([r for r in self.results if 'accuracy' in r])
        
        return aggregated
    
    def generate_report(self, aggregated_results):
        """生成详细报告"""
        
        print("\n" + "="*70)
        print("📋 滚动窗口回测报告")
        print("="*70)
        
        if not aggregated_results:
            print("❌ 无有效结果")
            return
        
        # 基本统计
        print(f"测试窗口数量: {aggregated_results['total_windows']}")
        print(f"成功窗口数量: {aggregated_results['successful_windows']}")
        print(f"成功率: {aggregated_results['successful_windows']/aggregated_results['total_windows']:.1%}")
        
        # 模型表现
        print(f"\n🤖 模型表现 (平均值±标准差):")
        print(f"   准确率: {aggregated_results.get('accuracy_mean', 0):.3f} ± {aggregated_results.get('accuracy_std', 0):.3f}")
        print(f"   F1分数: {aggregated_results.get('f1_score_mean', 0):.3f} ± {aggregated_results.get('f1_score_std', 0):.3f}")
        print(f"   AUC分数: {aggregated_results.get('auc_score_mean', 0):.3f} ± {aggregated_results.get('auc_score_std', 0):.3f}")
        
        # 交易表现
        print(f"\n💰 交易表现:")
        print(f"   策略胜率: {aggregated_results['strategy_win_rate']:.1%}")
        print(f"   累积策略收益: {aggregated_results['cumulative_strategy_return']:+.2%}")
        print(f"   累积买入持有: {aggregated_results['cumulative_buy_hold_return']:+.2%}")
        
        excess_return = aggregated_results['cumulative_strategy_return'] - aggregated_results['cumulative_buy_hold_return']
        print(f"   超额收益: {excess_return:+.2%}")
        
        # 稳定性分析
        strategy_std = aggregated_results.get('net_return_std', 0)
        buy_hold_std = aggregated_results.get('buy_hold_return_std', 0)
        
        print(f"\n📊 稳定性分析:")
        print(f"   策略收益波动: {strategy_std:.3f}")
        print(f"   买入持有波动: {buy_hold_std:.3f}")
        
        # 结论
        print(f"\n🎯 结论:")
        if aggregated_results['strategy_win_rate'] > 0.5:
            print(f"   ✅ 策略在{aggregated_results['strategy_win_rate']:.1%}的时间窗口中超越买入持有")
        else:
            print(f"   ❌ 策略仅在{aggregated_results['strategy_win_rate']:.1%}的时间窗口中超越买入持有")
            
        if abs(excess_return) < 0.01:  # 1%以内
            print(f"   ⚖️  策略与买入持有表现接近，可能存在过拟合")
        elif excess_return > 0:
            print(f"   🚀 策略显示出持续的超额收益能力")
        else:
            print(f"   📉 策略未能持续超越买入持有")
        
        print("="*70)


def main():
    """主函数"""
    
    # 配置日志
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
    
    # 加载数据
    try:
        df = pd.read_csv('data/BTC_USDT_1m_extended_30days.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"📊 数据加载完成: {len(df)} 行")
        print(f"时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        
    except FileNotFoundError:
        print("❌ 30天数据文件不存在")
        return False
    
    # 创建回测器
    backtester = RollingBacktester(
        train_days=7,   # 7天训练窗口
        test_days=1,    # 1天测试窗口
        step_days=1     # 1天滚动步长
    )
    
    # 运行回测
    aggregated_results = backtester.run_rolling_backtest(df)
    
    # 生成报告
    backtester.generate_report(aggregated_results)
    
    # 保存详细结果
    results_data = {
        'aggregated_results': aggregated_results,
        'individual_windows': backtester.results,
        'backtest_params': {
            'train_days': backtester.train_days,
            'test_days': backtester.test_days,
            'step_days': backtester.step_days
        },
        'completed_at': datetime.now().isoformat()
    }
    
    with open('results/rolling_backtest_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    print(f"\n💾 详细结果已保存到 results/rolling_backtest_results.json")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)