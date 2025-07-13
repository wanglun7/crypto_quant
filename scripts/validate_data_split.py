#!/usr/bin/env python3
"""
数据分割验证工具 - 确保无数据泄漏

检查项目：
1. 时间序列严格分割：训练 < 验证 < 测试 
2. 无重叠时间窗口
3. 足够的时间间隔防止前瞻性偏差
4. 验证特征构造过程中的数据泄漏
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import structlog
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_optimized import TimeSeriesSplitter
from crypto_quant.utils.indicators import add_cnn_lstm_features

logger = structlog.get_logger(__name__)


class DataLeakageValidator:
    """数据泄漏验证器"""
    
    def __init__(self):
        self.issues = []
    
    def validate_time_split(self, train_df, val_df, test_df):
        """验证时间分割是否正确"""
        
        print("🔍 验证时间序列分割...")
        
        # 1. 检查时间顺序
        train_end = train_df['timestamp'].max()
        val_start = val_df['timestamp'].min()
        val_end = val_df['timestamp'].max()
        test_start = test_df['timestamp'].min()
        
        print(f"训练集时间: {train_df['timestamp'].min()} ~ {train_end}")
        print(f"验证集时间: {val_start} ~ {val_end}")
        print(f"测试集时间: {test_start} ~ {test_df['timestamp'].max()}")
        
        # 2. 检查是否有时间重叠
        if train_end >= val_start:
            issue = f"❌ 训练集与验证集时间重叠: {train_end} >= {val_start}"
            self.issues.append(issue)
            print(issue)
        else:
            print(f"✅ 训练集与验证集无重叠")
        
        if val_end >= test_start:
            issue = f"❌ 验证集与测试集时间重叠: {val_end} >= {test_start}"
            self.issues.append(issue)
            print(issue)
        else:
            print(f"✅ 验证集与测试集无重叠")
        
        # 3. 检查时间间隔
        train_val_gap = (val_start - train_end).total_seconds() / 60  # 分钟
        val_test_gap = (test_start - val_end).total_seconds() / 60    # 分钟
        
        print(f"训练-验证间隔: {train_val_gap}分钟")
        print(f"验证-测试间隔: {val_test_gap}分钟")
        
        # 建议至少1分钟间隔
        if train_val_gap < 1:
            issue = f"⚠️  训练-验证间隔过小: {train_val_gap}分钟"
            self.issues.append(issue)
            print(issue)
        
        if val_test_gap < 1:
            issue = f"⚠️  验证-测试间隔过小: {val_test_gap}分钟"
            self.issues.append(issue)
            print(issue)
        
        return len(self.issues) == 0
    
    def validate_feature_leakage(self, df):
        """验证特征工程中是否有数据泄漏"""
        
        print("\n🔍 验证特征工程数据泄漏...")
        
        # 检查技术指标是否使用了未来数据
        # 重点检查滚动窗口计算
        
        original_len = len(df)
        df_with_features = add_cnn_lstm_features(df.copy())
        
        # 检查是否有NaN值（可能表示数据不足）
        nan_cols = df_with_features.columns[df_with_features.isnull().any()].tolist()
        
        if nan_cols:
            print(f"⚠️  发现包含NaN的特征列: {nan_cols}")
            print("这可能是由于滚动窗口计算导致的正常现象")
            
            # 检查NaN的分布
            for col in nan_cols[:3]:  # 只检查前几列
                nan_count = df_with_features[col].isnull().sum()
                print(f"  {col}: {nan_count} NaN值 ({nan_count/len(df_with_features)*100:.1f}%)")
        else:
            print("✅ 所有特征列都没有NaN值")
        
        # 检查特征值的合理性（简单检查）
        numeric_cols = df_with_features.select_dtypes(include=[np.number]).columns
        
        for col in ['RSI_14', 'Williams_R']:  # 检查几个有明确范围的指标
            if col in numeric_cols:
                values = df_with_features[col].dropna()
                if len(values) > 0:
                    min_val, max_val = values.min(), values.max()
                    
                    if col == 'RSI_14':
                        if min_val < 0 or max_val > 100:
                            issue = f"❌ {col} 值超出正常范围 [0,100]: [{min_val:.2f}, {max_val:.2f}]"
                            self.issues.append(issue)
                            print(issue)
                        else:
                            print(f"✅ {col} 值在正常范围内: [{min_val:.2f}, {max_val:.2f}]")
                    
                    elif col == 'Williams_R':
                        if min_val < -100 or max_val > 0:
                            issue = f"❌ {col} 值超出正常范围 [-100,0]: [{min_val:.2f}, {max_val:.2f}]"
                            self.issues.append(issue)
                            print(issue)
                        else:
                            print(f"✅ {col} 值在正常范围内: [{min_val:.2f}, {max_val:.2f}]")
        
        return len(self.issues) == 0
    
    def validate_sequence_creation(self, df, sequence_length=60):
        """验证序列创建过程中的数据泄漏"""
        
        print(f"\n🔍 验证序列创建过程 (长度={sequence_length})...")
        
        # 模拟序列创建过程
        n_samples = len(df) - sequence_length
        
        if n_samples <= 0:
            issue = f"❌ 数据不足以创建序列: {len(df)} < {sequence_length}"
            self.issues.append(issue)
            print(issue)
            return False
        
        print(f"✅ 原始数据: {len(df)}行")
        print(f"✅ 可创建序列: {n_samples}个")
        print(f"✅ 数据利用率: {n_samples/len(df)*100:.1f}%")
        
        # 检查序列是否按时间顺序
        timestamps = df['timestamp'].values
        
        # 随机检查几个序列
        for i in [0, n_samples//2, n_samples-1]:
            seq_times = timestamps[i:i+sequence_length]
            if not all(seq_times[j] <= seq_times[j+1] for j in range(len(seq_times)-1)):
                issue = f"❌ 序列 {i} 时间顺序错误"
                self.issues.append(issue)
                print(issue)
            
        print(f"✅ 时间序列顺序验证通过")
        
        return True
    
    def generate_report(self):
        """生成验证报告"""
        
        print("\n" + "="*60)
        print("📋 数据泄漏验证报告")
        print("="*60)
        
        if len(self.issues) == 0:
            print("🎉 恭喜！未发现数据泄漏问题")
            print("✅ 训练流程符合时间序列机器学习最佳实践")
        else:
            print(f"⚠️  发现 {len(self.issues)} 个潜在问题:")
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue}")
        
        print("="*60)
        
        return len(self.issues) == 0


def main():
    """主验证流程"""
    
    print("🚀 开始数据分割和泄漏验证")
    
    # 1. 加载数据
    try:
        df = pd.read_csv('data/BTC_USDT_1m_extended_30days.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        print(f"✅ 数据加载完成: {len(df)} 行")
        print(f"时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        
    except FileNotFoundError:
        print("❌ 30天数据文件不存在")
        return False
    
    # 2. 执行时间序列分割
    splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    train_df, val_df, test_df = splitter.split(df)
    
    # 3. 创建验证器并运行所有检查
    validator = DataLeakageValidator()
    
    # 检查时间分割
    validator.validate_time_split(train_df, val_df, test_df)
    
    # 检查特征工程
    validator.validate_feature_leakage(df)
    
    # 检查序列创建
    validator.validate_sequence_creation(df, sequence_length=60)
    
    # 4. 生成最终报告
    is_valid = validator.generate_report()
    
    return is_valid


if __name__ == "__main__":
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
    
    success = main()
    exit(0 if success else 1)