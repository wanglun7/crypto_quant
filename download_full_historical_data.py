"""
下载完整的2020-2024年历史数据
用于Walk-Forward回测验证
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


async def download_full_historical_data():
    """下载2020-2024年完整历史数据"""
    logger.info("🚀 开始下载2020-2024年完整历史数据...")
    
    # 创建下载器
    downloader = BinanceHistoricalDownloader("./data/full_historical")
    
    # 设置时间范围 (2020-2024年)
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    logger.info(f"📅 数据时间范围: {start_date.date()} 到 {end_date.date()}")
    
    # 需要的数据间隔
    intervals = [
        DataInterval.MINUTE_1,   # 1分钟数据 (最重要)
        DataInterval.MINUTE_5,   # 5分钟数据
        DataInterval.HOUR_1,     # 1小时数据
        DataInterval.DAY_1       # 日数据
    ]
    
    # 币种列表
    symbols = ['BTCUSDT']  # 先从BTC开始
    
    download_results = {}
    total_start_time = datetime.now()
    
    for symbol in symbols:
        logger.info(f"\n💰 开始下载 {symbol} 数据...")
        symbol_results = {}
        
        for interval in intervals:
            logger.info(f"  📊 下载 {interval.value} 间隔数据...")
            interval_start_time = datetime.now()
            
            try:
                success = await downloader.download_klines(
                    symbol, interval, start_date, end_date
                )
                
                if success:
                    stats = downloader.get_download_statistics()
                    interval_duration = datetime.now() - interval_start_time
                    
                    symbol_results[interval.value] = {
                        'success': True,
                        'files_downloaded': stats['total_downloaded'],
                        'total_size_mb': stats['total_size_mb'],
                        'success_rate': stats['success_rate'],
                        'duration_minutes': interval_duration.total_seconds() / 60
                    }
                    
                    logger.info(f"    ✅ 完成: {stats['total_downloaded']} 文件, "
                              f"{stats['total_size_mb']:.2f} MB, "
                              f"{interval_duration.total_seconds()/60:.1f} 分钟")
                else:
                    symbol_results[interval.value] = {
                        'success': False,
                        'error': '下载失败'
                    }
                    logger.error(f"    ❌ {interval.value} 数据下载失败")
                    
            except Exception as e:
                symbol_results[interval.value] = {
                    'success': False,
                    'error': str(e)
                }
                logger.error(f"    ❌ {interval.value} 数据下载异常: {e}")
        
        download_results[symbol] = symbol_results
    
    total_duration = datetime.now() - total_start_time
    
    # 生成下载报告
    logger.info(f"\n{'-'*60}")
    logger.info("📊 下载完成汇总报告")
    logger.info(f"{'-'*60}")
    logger.info(f"总下载时间: {total_duration.total_seconds()/60:.1f} 分钟")
    
    total_files = 0
    total_size_mb = 0
    successful_intervals = 0
    total_intervals = 0
    
    for symbol, symbol_data in download_results.items():
        logger.info(f"\n💰 {symbol}:")
        
        for interval, result in symbol_data.items():
            total_intervals += 1
            if result['success']:
                successful_intervals += 1
                total_files += result['files_downloaded']
                total_size_mb += result['total_size_mb']
                
                logger.info(f"  ✅ {interval}: {result['files_downloaded']} 文件, "
                          f"{result['total_size_mb']:.2f} MB")
            else:
                logger.error(f"  ❌ {interval}: {result['error']}")
    
    logger.info(f"\n🎯 总体统计:")
    logger.info(f"  - 总文件数: {total_files:,}")
    logger.info(f"  - 总大小: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    logger.info(f"  - 成功率: {successful_intervals}/{total_intervals} ({successful_intervals/total_intervals*100:.1f}%)")
    
    # 保存下载报告
    report_path = Path("./data/download_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 历史数据下载报告\n\n")
        f.write(f"下载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据范围: 2020-2024年\n")
        f.write(f"总下载时间: {total_duration.total_seconds()/60:.1f} 分钟\n\n")
        
        f.write(f"## 下载统计\n")
        f.write(f"- 总文件数: {total_files:,}\n")
        f.write(f"- 总大小: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)\n")
        f.write(f"- 成功率: {successful_intervals}/{total_intervals} ({successful_intervals/total_intervals*100:.1f}%)\n\n")
        
        f.write(f"## 详细结果\n")
        for symbol, symbol_data in download_results.items():
            f.write(f"### {symbol}\n")
            for interval, result in symbol_data.items():
                if result['success']:
                    f.write(f"- ✅ {interval}: {result['files_downloaded']} 文件, {result['total_size_mb']:.2f} MB\n")
                else:
                    f.write(f"- ❌ {interval}: {result['error']}\n")
            f.write("\n")
    
    logger.info(f"📄 下载报告已保存到: {report_path}")
    
    # 检查是否足够进行Walk-Forward回测
    if successful_intervals >= len(intervals) * 0.8:  # 80%成功率
        logger.info("\n🎉 数据下载基本成功! 可以开始构建回测系统")
        return True
    else:
        logger.error("\n⚠️ 数据下载成功率不足，需要检查网络或API限制")
        return False


async def verify_downloaded_data():
    """验证下载的数据质量"""
    logger.info("\n🔍 开始验证下载的数据质量...")
    
    # 检查关键数据文件
    data_path = Path("./data/full_historical")
    
    # 检查1分钟数据 (最重要)
    minute_data_path = data_path / "klines" / "BTCUSDT" / "1m"
    if minute_data_path.exists():
        files = list(minute_data_path.glob("*.zip"))
        logger.info(f"📊 1分钟数据: {len(files)} 个文件")
        
        # 计算数据覆盖度
        expected_days = (datetime(2024, 12, 31) - datetime(2020, 1, 1)).days + 1
        coverage_rate = len(files) / expected_days
        logger.info(f"📈 数据覆盖率: {coverage_rate*100:.1f}% ({len(files)}/{expected_days} 天)")
        
        if coverage_rate > 0.95:  # 95%覆盖率
            logger.info("✅ 数据覆盖率良好，足够进行Walk-Forward回测")
            return True
        else:
            logger.warning("⚠️ 数据覆盖率不足，可能影响回测质量")
            return False
    else:
        logger.error("❌ 未找到1分钟数据目录")
        return False


async def main():
    """主函数"""
    logger.info("="*60)
    logger.info("🎯 加密货币量化交易系统 - 完整历史数据下载")
    logger.info("="*60)
    
    try:
        # 步骤1: 下载完整历史数据
        download_success = await download_full_historical_data()
        
        if download_success:
            # 步骤2: 验证数据质量
            verify_success = await verify_downloaded_data()
            
            if verify_success:
                logger.info("\n🎉 历史数据准备完成! 下一步可以开始构建回测系统")
                return True
            else:
                logger.warning("\n⚠️ 数据验证有问题，需要补充下载")
                return False
        else:
            logger.error("\n❌ 数据下载失败，需要检查网络或API配置")
            return False
            
    except Exception as e:
        logger.error(f"程序异常: {e}")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        if success:
            logger.info("✅ 历史数据下载任务完成")
        else:
            logger.error("❌ 历史数据下载任务失败")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("任务被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序错误: {e}")
        sys.exit(1)