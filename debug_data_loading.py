"""
调试数据加载问题
"""

import sys
import os
from datetime import datetime
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtesting.backtest_engine import DataProvider

def debug_data_loading():
    """调试数据加载"""
    print("🔍 调试数据加载...")
    
    # 创建数据提供器
    data_provider = DataProvider("./data/sample_data")
    
    # 测试单个文件解析
    from pathlib import Path
    test_file = Path("./data/sample_data/klines/BTCUSDT/1h/BTCUSDT-1h-2023-01-01.zip")
    
    if test_file.exists():
        print(f"📄 测试文件: {test_file}")
        df = data_provider._parse_klines_file(test_file)
        
        if not df.empty:
            print(f"✅ 文件解析成功: {len(df)} 条记录")
            print(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
            print(f"前5行数据:")
            print(df.head())
            
            # 测试数据加载范围
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2023, 1, 2)
            
            print(f"\n📊 测试加载数据: {start_date} 到 {end_date}")
            loaded_data = data_provider.load_data("BTCUSDT", start_date, end_date)
            print(f"加载结果: {len(loaded_data)} 条记录")
            
            if not loaded_data.empty:
                print(f"加载的时间范围: {loaded_data['timestamp'].min()} 到 {loaded_data['timestamp'].max()}")
            
        else:
            print("❌ 文件解析失败")
    else:
        print(f"❌ 测试文件不存在: {test_file}")

if __name__ == "__main__":
    debug_data_loading()