"""
下载样本数据进行快速测试
下载2023年数据作为Walk-Forward回测的基础
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.historical_downloader import BinanceHistoricalDownloader, DataInterval

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def download_sample_data():
    """下载2023年样本数据进行快速测试"""
    logger.info("🚀 开始下载2023年样本数据...")
    
    # 创建下载器
    downloader = BinanceHistoricalDownloader("./data/sample_data")
    
    # 设置时间范围 (2023年)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    logger.info(f"📅 数据时间范围: {start_date.date()} 到 {end_date.date()}")
    
    # 下载关键数据间隔
    intervals = [
        DataInterval.HOUR_1,     # 1小时数据 (主要用于回测)
        DataInterval.DAY_1       # 日数据 (用于长期分析)
    ]
    
    download_results = {}
    total_start_time = datetime.now()
    
    for interval in intervals:
        logger.info(f"📊 下载 {interval.value} 间隔数据...")
        interval_start_time = datetime.now()
        
        try:
            success = await downloader.download_klines(
                'BTCUSDT', interval, start_date, end_date
            )
            
            if success:
                stats = downloader.get_download_statistics()
                interval_duration = datetime.now() - interval_start_time
                
                download_results[interval.value] = {
                    'success': True,
                    'files_downloaded': stats['total_downloaded'],
                    'total_size_mb': stats['total_size_mb'],
                    'success_rate': stats['success_rate'],
                    'duration_minutes': interval_duration.total_seconds() / 60
                }
                
                logger.info(f"✅ {interval.value} 完成: {stats['total_downloaded']} 文件, "
                          f"{stats['total_size_mb']:.2f} MB, "
                          f"{interval_duration.total_seconds()/60:.1f} 分钟")
            else:
                download_results[interval.value] = {
                    'success': False,
                    'error': '下载失败'
                }
                logger.error(f"❌ {interval.value} 数据下载失败")
                
        except Exception as e:
            download_results[interval.value] = {
                'success': False,
                'error': str(e)
            }
            logger.error(f"❌ {interval.value} 数据下载异常: {e}")
    
    total_duration = datetime.now() - total_start_time
    
    # 汇总结果
    logger.info(f"\n{'-'*50}")
    logger.info("📊 下载完成汇总")
    logger.info(f"{'-'*50}")
    logger.info(f"总下载时间: {total_duration.total_seconds()/60:.1f} 分钟")
    
    total_files = 0
    total_size_mb = 0
    successful_intervals = 0
    
    for interval, result in download_results.items():
        if result['success']:
            successful_intervals += 1
            total_files += result['files_downloaded']
            total_size_mb += result['total_size_mb']
            logger.info(f"✅ {interval}: {result['files_downloaded']} 文件, {result['total_size_mb']:.2f} MB")
        else:
            logger.error(f"❌ {interval}: {result['error']}")
    
    logger.info(f"\n🎯 总体统计:")
    logger.info(f"  - 总文件数: {total_files}")
    logger.info(f"  - 总大小: {total_size_mb:.2f} MB")
    logger.info(f"  - 成功率: {successful_intervals}/{len(intervals)} ({successful_intervals/len(intervals)*100:.1f}%)")
    
    return successful_intervals == len(intervals)


async def quick_data_validation():
    """快速验证下载的数据"""
    logger.info("\n🔍 快速验证下载的数据...")
    
    data_path = Path("./data/sample_data/klines/BTCUSDT/1h")
    
    if data_path.exists():
        files = list(data_path.glob("*.zip"))
        logger.info(f"📊 1小时数据文件数: {len(files)}")
        
        # 计算预期文件数 (365天)
        expected_files = 365
        coverage_rate = len(files) / expected_files
        
        logger.info(f"📈 数据覆盖率: {coverage_rate*100:.1f}% ({len(files)}/{expected_files})")
        
        if coverage_rate > 0.8:  # 80%覆盖率就足够测试
            logger.info("✅ 样本数据足够进行回测系统开发")
            return True
        else:
            logger.warning("⚠️ 样本数据覆盖率不足")
            return False
    else:
        logger.error("❌ 未找到数据目录")
        return False


async def main():
    """主函数"""
    logger.info("="*60)
    logger.info("🎯 样本数据下载 - 2023年BTCUSDT数据")
    logger.info("="*60)
    
    try:
        # 下载样本数据
        download_success = await download_sample_data()
        
        if download_success:
            # 验证数据
            validation_success = await quick_data_validation()
            
            if validation_success:
                logger.info("\n🎉 样本数据准备完成!")
                logger.info("下一步建议:")
                logger.info("1. 开始构建回测引擎")
                logger.info("2. 实现基础特征工程")
                logger.info("3. 测试简单的交易策略")
                return True
            else:
                logger.warning("\n⚠️ 数据验证有问题")
                return False
        else:
            logger.error("\n❌ 数据下载失败")
            return False
            
    except Exception as e:
        logger.error(f"程序异常: {e}")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("任务被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序错误: {e}")
        sys.exit(1)