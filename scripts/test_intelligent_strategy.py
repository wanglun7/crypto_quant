#!/usr/bin/env python3
"""
测试智能仓位管理策略

对比不同策略的表现：
1. 原始策略：简单二元分类 + 固定仓位
2. 智能策略：动态仓位 + 趋势过滤 + 风险控制

目标：验证智能策略是否能将+7.93%基础收益放大到70%+年化收益
"""

import pandas as pd
import numpy as np
import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_quant.models.cnn_lstm import CNN_LSTM
from crypto_quant.utils.indicators import prepare_cnn_lstm_data
from crypto_quant.strategy.intelligent_position_manager import (
    IntelligentPositionManager, PortfolioState, TrendDirection
)


def load_model_and_predict(df, model_path='models/cnn_lstm_btc_model.pth'):
    """加载模型并生成预测"""
    
    print("🤖 加载模型并生成预测...")
    
    # 准备数据（使用原始序列长度）
    X, y = prepare_cnn_lstm_data(df, sequence_length=30)
    
    # 加载模型（匹配原始架构）
    model = CNN_LSTM(
        n_features=18,  # 原始特征集
        sequence_length=30,  # 原始序列长度
        cnn_filters=[32, 64, 128],
        lstm_hidden_size=64,
        lstm_num_layers=2,
        dropout_rate=0.3,
        fc_hidden_size=32  # 原始架构
    )
    
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"✅ 模型加载成功: {model_path}")
    except:
        print(f"❌ 模型加载失败，使用随机预测")
        # 使用随机预测进行概念验证
        np.random.seed(42)
        probabilities = np.random.beta(2, 2, len(y))  # Beta分布生成更真实的概率
        return probabilities, y
    
    # 生成预测
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        probabilities = model(X_tensor).cpu().numpy().flatten()
    
    print(f"✅ 预测生成完成: {len(probabilities)} 个预测")
    print(f"   概率范围: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
    
    return probabilities, y


def run_simple_strategy(df, probabilities, threshold=0.3):
    """运行原始简单策略"""
    
    print(f"\n📊 运行简单策略 (阈值={threshold})...")
    
    # 生成信号
    signals = (probabilities > threshold).astype(int)
    
    # 计算收益
    aligned_df = df.iloc[30:30+len(probabilities)].copy()  # 对齐序列长度
    price_returns = aligned_df['close'].pct_change().iloc[1:].values
    
    # 策略收益（前一个信号对应下一个收益）
    strategy_returns = signals[:-1] * price_returns
    
    # 累积收益
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # 买入持有收益
    buy_hold_return = (aligned_df['close'].iloc[-1] / aligned_df['close'].iloc[0]) - 1
    strategy_return = cumulative_returns[-1] - 1
    
    # 交易成本
    trades = np.sum(np.diff(np.concatenate([[0], signals])) != 0)
    transaction_costs = trades * 0.0004
    net_return = strategy_return - transaction_costs
    
    print(f"   策略收益: {net_return:+.2%}")
    print(f"   买入持有: {buy_hold_return:+.2%}")
    print(f"   超额收益: {net_return - buy_hold_return:+.2%}")
    print(f"   交易次数: {trades}")
    
    return {
        'strategy_return': net_return,
        'buy_hold_return': buy_hold_return,
        'trades': trades,
        'cumulative_returns': cumulative_returns,
        'daily_returns': strategy_returns
    }


def run_intelligent_strategy(df, probabilities, initial_capital=10000):
    """运行智能仓位管理策略"""
    
    print(f"\n🧠 运行智能策略 (初始资金=${initial_capital:,.0f})...")
    
    # 创建智能管理器
    manager = IntelligentPositionManager(
        initial_capital=initial_capital,
        confidence_threshold=0.6
    )
    
    # 初始化投资组合
    current_portfolio = PortfolioState(
        timestamp=df.iloc[30]['timestamp'],
        cash=initial_capital,
        position_size=0,
        position_value=0,
        total_value=initial_capital,
        unrealized_pnl=0,
        realized_pnl=0,
        max_drawdown=0,
        current_drawdown=0
    )
    
    # 对齐数据（使用序列长度30）
    aligned_df = df.iloc[30:30+len(probabilities)].copy()
    
    # 逐步执行策略
    portfolio_values = []
    signals_log = []
    
    for i, (idx, row) in enumerate(aligned_df.iterrows()):
        if i >= len(probabilities):
            break
            
        current_price = row['close']
        model_prob = probabilities[i]
        
        # 生成信号
        signal = manager.generate_signal(aligned_df.reset_index(drop=True), i, model_prob)
        signals_log.append(signal)
        
        # 执行交易
        current_portfolio = manager.execute_trade(signal, current_price, current_portfolio)
        portfolio_values.append(current_portfolio.total_value)
    
    # 计算最终指标
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    # 计算年化收益率（基于实际时间）
    time_diff = aligned_df['timestamp'].iloc[-1] - aligned_df['timestamp'].iloc[0]
    days = time_diff.total_seconds() / (24 * 3600)  # 精确到小数天
    
    # 防止除零错误
    if days < 0.1:  # 少于2.4小时
        annual_return = 0
        print(f"   警告：数据时间过短({days:.2f}天)，无法计算有意义的年化收益")
    else:
        annual_return = (1 + total_return) ** (365/days) - 1
        # 限制在合理范围内
        if annual_return > 10:  # 1000%以上视为异常
            print(f"   警告：年化收益计算异常({annual_return:.1%})，可能是数据时间过短")
            annual_return = min(annual_return, 10)  # 限制最大1000%
    
    # 买入持有对比
    buy_hold_return = (aligned_df['close'].iloc[-1] / aligned_df['close'].iloc[0]) - 1
    
    print(f"   最终价值: ${final_value:,.2f}")
    print(f"   总收益: {total_return:+.2%}")
    print(f"   数据时间跨度: {days:.2f}天")
    print(f"   年化收益: {annual_return:+.2%}")
    print(f"   买入持有: {buy_hold_return:+.2%}")
    print(f"   超额收益: {total_return - buy_hold_return:+.2%}")
    print(f"   最大回撤: {current_portfolio.max_drawdown:.2%}")
    
    # 获取详细表现
    performance = manager.get_performance_metrics()
    
    return {
        'manager': manager,
        'portfolio_values': portfolio_values,
        'signals_log': signals_log,
        'performance': performance,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'buy_hold_return': buy_hold_return
    }


def analyze_signals(signals_log):
    """分析交易信号"""
    
    print(f"\n🔍 信号分析:")
    
    # 统计不同趋势下的信号
    trend_stats = {}
    for signal in signals_log:
        trend = signal.trend.value
        if trend not in trend_stats:
            trend_stats[trend] = {'count': 0, 'avg_size': 0, 'total_confidence': 0}
        
        trend_stats[trend]['count'] += 1
        trend_stats[trend]['avg_size'] += signal.recommended_size
        trend_stats[trend]['total_confidence'] += signal.confidence
    
    for trend, stats in trend_stats.items():
        if stats['count'] > 0:
            avg_size = stats['avg_size'] / stats['count']
            avg_confidence = stats['total_confidence'] / stats['count']
            print(f"   {trend}: {stats['count']}次, 平均仓位{avg_size:.1%}, 平均置信度{avg_confidence:.1%}")
    
    # 高置信度信号统计
    high_confidence_signals = [s for s in signals_log if s.confidence > 0.8]
    if high_confidence_signals:
        avg_size = np.mean([s.recommended_size for s in high_confidence_signals])
        print(f"   高置信度信号(>80%): {len(high_confidence_signals)}次, 平均仓位{avg_size:.1%}")


def create_comparison_chart(simple_results, intelligent_results, df):
    """创建策略对比图表"""
    
    print(f"\n📈 生成对比图表...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 对齐时间序列
    aligned_df = df.iloc[30:30+len(simple_results['cumulative_returns'])].copy()
    timestamps = aligned_df['timestamp']
    
    # 1. 累积收益对比
    ax1.plot(timestamps, simple_results['cumulative_returns'], 
            label='简单策略', linewidth=2, alpha=0.8)
    
    # 智能策略需要标准化到同样的起始点
    intel_returns = np.array(intelligent_results['portfolio_values'])
    intel_returns = intel_returns / intel_returns[0]  # 标准化到1开始
    
    # 确保长度匹配
    min_len = min(len(timestamps), len(intel_returns))
    ax1.plot(timestamps[:min_len], intel_returns[:min_len], 
            label='智能策略', linewidth=2, alpha=0.8)
    
    # 买入持有基准
    buy_hold_values = aligned_df['close'] / aligned_df['close'].iloc[0]
    ax1.plot(timestamps, buy_hold_values, 
            label='买入持有', linewidth=1, linestyle='--', alpha=0.6)
    
    ax1.set_title('累积收益对比')
    ax1.set_ylabel('累积收益')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 仓位大小变化（智能策略）
    if intelligent_results['signals_log']:
        position_sizes = [s.recommended_size for s in intelligent_results['signals_log']]
        min_len_pos = min(len(timestamps), len(position_sizes))
        ax2.plot(timestamps[:min_len_pos], position_sizes[:min_len_pos], 
                linewidth=1, alpha=0.7, color='orange')
        ax2.fill_between(timestamps[:min_len_pos], position_sizes[:min_len_pos], 
                        alpha=0.3, color='orange')
        ax2.set_title('智能策略仓位变化')
        ax2.set_ylabel('仓位大小')
        ax2.grid(True, alpha=0.3)
    
    # 3. 趋势分析
    if intelligent_results['signals_log']:
        trends = [s.trend.value for s in intelligent_results['signals_log']]
        trend_colors = {
            'strong_bull': 'darkgreen',
            'weak_bull': 'lightgreen', 
            'sideways': 'gray',
            'weak_bear': 'lightcoral',
            'strong_bear': 'darkred'
        }
        
        y_pos = 0
        min_len_trend = min(len(timestamps), len(trends))
        for i, trend in enumerate(trends[:min_len_trend]):
            color = trend_colors.get(trend, 'gray')
            ax3.scatter(timestamps.iloc[i], y_pos, c=color, s=20, alpha=0.6)
        
        ax3.set_title('市场趋势识别')
        ax3.set_ylabel('趋势')
        ax3.grid(True, alpha=0.3)
    
    # 4. 收益分布对比
    simple_daily = simple_results['daily_returns']
    intel_daily = np.diff(intel_returns) / intel_returns[:-1]
    
    ax4.hist(simple_daily, bins=50, alpha=0.5, label='简单策略', density=True)
    ax4.hist(intel_daily, bins=50, alpha=0.5, label='智能策略', density=True)
    ax4.set_title('日收益分布')
    ax4.set_xlabel('日收益率')
    ax4.set_ylabel('密度')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/strategy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   图表已保存: results/strategy_comparison.png")


def main():
    """主函数"""
    
    print("🚀 智能仓位管理策略测试")
    print("="*60)
    
    # 1. 加载数据
    try:
        df = pd.read_csv('data/BTC_USDT_1m_last_3days.csv')  # 使用3天数据快速测试
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"✅ 数据加载: {len(df)} 行")
        print(f"   时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    except FileNotFoundError:
        print("❌ 数据文件不存在")
        return False
    
    # 2. 生成模型预测
    probabilities, y_true = load_model_and_predict(df)
    
    # 3. 运行简单策略
    simple_results = run_simple_strategy(df, probabilities)
    
    # 4. 运行智能策略
    intelligent_results = run_intelligent_strategy(df, probabilities)
    
    # 5. 分析信号
    analyze_signals(intelligent_results['signals_log'])
    
    # 6. 对比结果
    print(f"\n🏆 策略对比总结:")
    print(f"="*60)
    print(f"{'指标':<20} {'简单策略':<15} {'智能策略':<15} {'提升幅度':<15}")
    print(f"-"*65)
    
    simple_return = simple_results['strategy_return']
    intel_return = intelligent_results['total_return']
    improvement = (intel_return - simple_return) / abs(simple_return) * 100 if simple_return != 0 else 0
    
    print(f"{'总收益率':<20} {simple_return:>13.2%} {intel_return:>13.2%} {improvement:>13.1f}%")
    
    # 年化收益率（使用相同的时间基准）
    intel_annual = intelligent_results['annual_return']
    
    # 计算简单策略的实际时间跨度
    aligned_df_simple = df.iloc[30:30+len(simple_results['cumulative_returns'])].copy()
    time_diff_simple = aligned_df_simple['timestamp'].iloc[-1] - aligned_df_simple['timestamp'].iloc[0]
    days_simple = time_diff_simple.total_seconds() / (24 * 3600)
    
    if days_simple < 0.1:
        simple_annual = 0
    else:
        simple_annual = (1 + simple_return) ** (365/days_simple) - 1
        if simple_annual > 10:
            simple_annual = min(simple_annual, 10)  # 限制最大1000%
    
    annual_improvement = (intel_annual - simple_annual) / abs(simple_annual) * 100 if simple_annual != 0 else 0
    
    print(f"{'年化收益率':<20} {simple_annual:>13.2%} {intel_annual:>13.2%} {annual_improvement:>13.1f}%")
    
    # 最大回撤
    intel_dd = intelligent_results['manager'].portfolio_history[-1].max_drawdown
    print(f"{'最大回撤':<20} {'N/A':<15} {intel_dd:>13.2%} {'N/A':<15}")
    
    # 交易次数
    intel_trades = len(intelligent_results['signals_log'])
    print(f"{'交易信号数':<20} {simple_results['trades']:>13} {intel_trades:>13} {intel_trades-simple_results['trades']:>13}")
    
    print(f"="*60)
    
    # 7. 生成图表
    create_comparison_chart(simple_results, intelligent_results, df)
    
    # 8. 保存详细结果
    results = {
        'simple_strategy': {
            'return': simple_results['strategy_return'],
            'annual_return': simple_annual,
            'trades': simple_results['trades']
        },
        'intelligent_strategy': {
            'return': intelligent_results['total_return'],
            'annual_return': intelligent_results['annual_return'],
            'max_drawdown': intel_dd,
            'trades': intel_trades,
            'performance': intelligent_results['performance']
        },
        'improvement': {
            'return_improvement': improvement,
            'annual_return_improvement': annual_improvement
        }
    }
    
    with open('results/intelligent_strategy_test.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 详细结果已保存: results/intelligent_strategy_test.json")
    
    # 9. 结论
    print(f"\n🎯 结论:")
    if intel_annual > 0.7:  # 70%年化收益目标
        print(f"✅ 智能策略达到年化收益目标: {intel_annual:.1%} > 70%")
    else:
        print(f"⚠️  智能策略年化收益: {intel_annual:.1%}，距离70%目标还有差距")
    
    if improvement > 50:  # 50%以上提升
        print(f"✅ 智能策略相比简单策略提升显著: +{improvement:.1f}%")
    else:
        print(f"⚠️  智能策略提升有限: +{improvement:.1f}%，需要进一步优化")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)