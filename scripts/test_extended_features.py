#!/usr/bin/env python3
"""测试扩展技术指标库"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from crypto_quant.utils.indicators import (
    add_extended_features, 
    get_extended_feature_columns,
    prepare_extended_data
)


def test_extended_features():
    """测试扩展特征集"""
    
    print("🧪 测试扩展技术指标库")
    print("="*50)
    
    # 1. 加载少量数据进行测试
    try:
        df = pd.read_csv('data/BTC_USDT_1m_last_3days.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        print(f"✅ 测试数据加载: {len(df)} 行")
        
    except FileNotFoundError:
        print("❌ 测试数据文件不存在")
        return False
    
    # 2. 测试扩展特征生成
    try:
        df_extended = add_extended_features(df)
        
        # 获取特征列
        feature_cols = get_extended_feature_columns()
        
        print(f"✅ 扩展特征生成成功")
        print(f"   总特征数: {len(feature_cols)}")
        print(f"   原始18个特征: {feature_cols[:18]}")
        print(f"   新增8个特征: {feature_cols[18:]}")
        
        # 检查特征是否都存在
        missing_cols = [col for col in feature_cols if col not in df_extended.columns]
        if missing_cols:
            print(f"❌ 缺失特征列: {missing_cols}")
            return False
        else:
            print("✅ 所有特征列都存在")
            
    except Exception as e:
        print(f"❌ 扩展特征生成失败: {e}")
        return False
    
    # 3. 检查特征数据质量
    print(f"\n📊 特征数据质量检查:")
    
    for col in feature_cols[18:]:  # 只检查新增特征
        if col in df_extended.columns:
            values = df_extended[col].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                std_val = values.std()
                min_val = values.min()
                max_val = values.max()
                nan_count = df_extended[col].isnull().sum()
                
                print(f"   {col:<18}: 均值={mean_val:>8.3f}, 标准差={std_val:>8.3f}, 范围=[{min_val:>7.3f}, {max_val:>7.3f}], NaN={nan_count}")
                
                # 检查是否有异常值
                if np.isinf(mean_val) or np.isnan(mean_val):
                    print(f"      ⚠️  {col} 包含异常值")
            else:
                print(f"   {col:<18}: 全部为NaN")
    
    # 4. 测试数据准备功能
    try:
        print(f"\n🔧 测试数据准备功能:")
        X, y = prepare_extended_data(df, sequence_length=30)  # 使用较短序列进行测试
        
        print(f"✅ 数据准备成功")
        print(f"   样本形状: {X.shape}")
        print(f"   标签形状: {y.shape}")
        print(f"   正样本比例: {y.mean():.1%}")
        
        # 检查数据完整性
        if np.isnan(X).any():
            print(f"⚠️  特征矩阵包含NaN值")
        else:
            print(f"✅ 特征矩阵无NaN值")
            
        if np.isnan(y).any():
            print(f"⚠️  标签向量包含NaN值")
        else:
            print(f"✅ 标签向量无NaN值")
            
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        return False
    
    # 5. 特征重要性简单检查
    print(f"\n📈 特征统计摘要:")
    
    # 检查各个特征的变化幅度
    feature_stats = {}
    for i, col in enumerate(feature_cols):
        if col in df_extended.columns:
            values = df_extended[col].dropna()
            if len(values) > 1:
                cv = values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                feature_stats[col] = cv
    
    # 显示变化系数最大的特征（可能最有信息量）
    sorted_features = sorted(feature_stats.items(), key=lambda x: x[1], reverse=True)
    
    print("   变化系数最大的前5个特征（可能信息量最大）:")
    for col, cv in sorted_features[:5]:
        if not np.isinf(cv):
            print(f"     {col:<18}: {cv:.3f}")
    
    print("\n" + "="*50)
    print("🎉 扩展技术指标库测试完成！")
    print("💡 可以运行 'python main.py train-test' 使用扩展特征进行训练")
    
    return True


if __name__ == "__main__":
    success = test_extended_features()
    exit(0 if success else 1)