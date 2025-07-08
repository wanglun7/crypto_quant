"""
简单的数据质量测试
验证下载的历史数据并规划下一步
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys
import os
import zipfile
import io

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.historical_downloader import BinanceHistoricalDownloader, DataInterval

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_downloaded_files():
    """检查已下载的文件"""
    data_path = Path("./data/test_basic")
    
    logger.info("🔍 检查已下载的历史数据文件...")
    
    # 检查BTCUSDT 1小时数据
    klines_path = data_path / "klines" / "BTCUSDT" / "1h"
    
    if klines_path.exists():
        files = list(klines_path.glob("*.zip"))
        logger.info(f"找到 {len(files)} 个数据文件:")
        
        total_size = 0
        for file in files:
            file_size = file.stat().st_size
            total_size += file_size
            logger.info(f"  - {file.name}: {file_size} bytes")
        
        logger.info(f"总文件大小: {total_size} bytes ({total_size/1024:.2f} KB)")
        return files
    else:
        logger.warning("未找到数据文件目录")
        return []


def parse_single_file(file_path: Path):
    """解析单个数据文件"""
    logger.info(f"📊 解析文件: {file_path.name}")
    
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            files_in_zip = zip_file.namelist()
            logger.info(f"  ZIP文件内容: {files_in_zip}")
            
            # 查找CSV文件
            csv_files = [f for f in files_in_zip if f.endswith('.csv')]
            if not csv_files:
                logger.error("  未找到CSV文件")
                return None
            
            # 读取第一个CSV文件
            with zip_file.open(csv_files[0]) as csv_file:
                # 读取前几行看看数据格式
                lines = []
                for i, line in enumerate(csv_file):
                    if i < 5:  # 只读前5行
                        lines.append(line.decode('utf-8').strip())
                    else:
                        break
                
                logger.info("  前5行数据:")
                for i, line in enumerate(lines):
                    logger.info(f"    {i+1}: {line}")
                
                # 重新读取完整数据
                csv_file.seek(0)
                df = pd.read_csv(csv_file, header=None)
                
                logger.info(f"  数据维度: {df.shape}")
                logger.info(f"  列数: {len(df.columns)}")
                
                # 显示数据类型
                logger.info("  数据类型:")
                for i, dtype in enumerate(df.dtypes):
                    logger.info(f"    列{i}: {dtype}")
                
                return df
                
    except Exception as e:
        logger.error(f"  解析文件失败: {e}")
        return None


async def download_more_data_if_needed():
    """如果需要，下载更多数据"""
    logger.info("📥 评估是否需要下载更多数据...")
    
    # 检查是否有足够的数据进行回测
    files = check_downloaded_files()
    
    if len(files) < 10:  # 少于10个文件
        logger.warning("⚠️ 当前数据量不足以进行有效回测")
        logger.info("建议下载更多历史数据...")
        
        # 下载最近30天的数据
        downloader = BinanceHistoricalDownloader("./data/extended")
        
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)
        
        logger.info(f"下载时间范围: {start_date.date()} 到 {end_date.date()}")
        
        try:
            # 下载1小时数据
            await downloader.download_klines(
                'BTCUSDT', DataInterval.HOUR_1, start_date, end_date
            )
            
            # 下载日数据
            await downloader.download_klines(
                'BTCUSDT', DataInterval.DAY_1, start_date, end_date
            )
            
            stats = downloader.get_download_statistics()
            logger.info("✅ 扩展数据下载完成:")
            logger.info(f"  - 下载文件数: {stats['total_downloaded']}")
            logger.info(f"  - 总大小: {stats['total_size_mb']:.2f} MB")
            logger.info(f"  - 成功率: {stats['success_rate']:.2%}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 扩展数据下载失败: {e}")
            return False
    else:
        logger.info("✅ 当前数据量足够进行基本测试")
        return True


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("🎯 历史数据质量检查和规划")
    logger.info("=" * 60)
    
    # 步骤1: 检查已下载的文件
    logger.info("\n📋 步骤1: 检查已下载的文件")
    logger.info("-" * 40)
    files = check_downloaded_files()
    
    # 步骤2: 解析一个文件样本
    logger.info("\n📊 步骤2: 解析文件样本")
    logger.info("-" * 40)
    if files:
        sample_file = files[0]
        df = parse_single_file(sample_file)
        
        if df is not None:
            logger.info("✅ 文件解析成功")
        else:
            logger.error("❌ 文件解析失败")
    else:
        logger.error("❌ 没有文件可以解析")
    
    # 步骤3: 评估数据充足性
    logger.info("\n🔍 步骤3: 评估数据充足性")
    logger.info("-" * 40)
    
    if len(files) >= 4:  # 至少4天的数据
        logger.info("✅ 数据量足够进行基本回测测试")
        data_sufficient = True
    else:
        logger.warning("⚠️ 数据量不足，需要下载更多数据")
        data_sufficient = False
    
    # 步骤4: 下一步规划
    logger.info("\n🚀 步骤4: 下一步规划")
    logger.info("-" * 40)
    
    if data_sufficient:
        logger.info("建议下一步行动:")
        logger.info("1. ✅ 数据采集基础功能 - 已完成")
        logger.info("2. 🔄 开始构建回测引擎")
        logger.info("3. 🔄 实现基础特征工程")
        logger.info("4. 🔄 构建简单的交易策略")
        logger.info("5. 🔄 进行2年数据的Walk-Forward回测")
        
        next_action = "构建回测引擎"
    else:
        logger.info("建议下一步行动:")
        logger.info("1. 🔄 下载更多历史数据 (至少2020-2024年)")
        logger.info("2. 🔄 验证数据质量和完整性")
        logger.info("3. 🔄 然后构建回测引擎")
        
        next_action = "下载更多历史数据"
    
    logger.info(f"\n🎯 推荐的下一步: {next_action}")
    
    # 步骤5: 提供具体的实施建议
    logger.info("\n📝 步骤5: 具体实施建议")
    logger.info("-" * 40)
    
    logger.info("为了实现原始CRYPTO_QUANT_ARCHITECTURE.md的目标，我们需要:")
    logger.info("1. 📊 数据基础设施:")
    logger.info("   - ✅ 基本历史数据下载 (已完成)")
    logger.info("   - 🔄 扩展到2020-2024年完整数据")
    logger.info("   - 🔄 添加1分钟级别的高频数据")
    logger.info("   - 🔄 集成链上数据 (Glassnode)")
    
    logger.info("2. 🛠️ 回测系统:")
    logger.info("   - 🔄 构建事件驱动的回测引擎")
    logger.info("   - 🔄 实现Walk-Forward验证框架")
    logger.info("   - 🔄 添加真实的交易成本和滑点模型")
    
    logger.info("3. 🧠 AI模型:")
    logger.info("   - 🔄 实现330+特征工程")
    logger.info("   - 🔄 构建Transformer+LSTM+CNN混合模型")
    logger.info("   - 🔄 实现模型集成和动态权重")
    
    logger.info("4. 📈 性能目标:")
    logger.info("   - 🎯 年化收益率 > BTC HODL × 2")
    logger.info("   - 🎯 Sharpe Ratio > 2.0")
    logger.info("   - 🎯 最大回撤 < BTC HODL × 30%")
    
    logger.info("\n" + "=" * 60)
    logger.info("🏁 数据质量检查完成")
    logger.info("=" * 60)
    
    return data_sufficient


if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("✅ 可以继续下一步开发")
        else:
            logger.warning("⚠️ 需要先完善数据基础设施")
    except Exception as e:
        logger.error(f"测试程序出错: {e}")
        sys.exit(1)