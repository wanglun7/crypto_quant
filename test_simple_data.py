"""
简化的数据管道测试
验证基本的数据采集功能
"""

import asyncio
import sys
import os
import logging
from datetime import datetime, timedelta

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_historical_download_basic():
    """基本历史数据下载测试"""
    logger.info("🚀 开始基本历史数据下载测试...")
    
    try:
        from data_pipeline.historical_downloader import BinanceHistoricalDownloader, DataInterval
        
        # 创建下载器
        downloader = BinanceHistoricalDownloader("./data/test_basic")
        
        # 下载最近3天的1小时数据
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=3)
        
        logger.info(f"下载时间范围: {start_date.date()} 到 {end_date.date()}")
        
        # 下载BTCUSDT 1小时数据
        success = await downloader.download_klines(
            'BTCUSDT', 
            DataInterval.HOUR_1, 
            start_date, 
            end_date
        )
        
        if success:
            stats = downloader.get_download_statistics()
            logger.info("✅ 历史数据下载测试通过!")
            logger.info(f"  - 下载文件数: {stats['total_downloaded']}")
            logger.info(f"  - 总大小: {stats['total_size_mb']:.2f} MB")
            logger.info(f"  - 成功率: {stats['success_rate']:.2%}")
            return True
        else:
            logger.error("❌ 历史数据下载失败")
            return False
            
    except Exception as e:
        logger.error(f"❌ 历史数据下载测试失败: {e}")
        return False


async def test_websocket_basic():
    """基本WebSocket测试"""
    logger.info("🚀 开始基本WebSocket测试...")
    
    try:
        from data_pipeline.binance_websocket import BinanceWebSocketManager, DataType
        
        # 创建WebSocket管理器
        ws_manager = BinanceWebSocketManager(['btcusdt'])
        
        # 数据计数器
        message_count = 0
        
        # 简单的数据处理回调
        async def on_ticker(ticker):
            nonlocal message_count
            message_count += 1
            if message_count <= 5:  # 只显示前5条
                logger.info(f"📊 收到ticker数据: {ticker.symbol} = ${ticker.price}")
        
        # 注册回调
        ws_manager.register_callback(DataType.TICKER, on_ticker)
        
        # 创建连接任务
        connection_task = asyncio.create_task(ws_manager.start())
        
        # 等待10秒
        try:
            await asyncio.wait_for(connection_task, timeout=10)
        except asyncio.TimeoutError:
            await ws_manager.stop()
        
        if message_count > 0:
            logger.info(f"✅ WebSocket测试通过! 收到 {message_count} 条消息")
            return True
        else:
            logger.error("❌ WebSocket测试失败: 没有收到数据")
            return False
            
    except Exception as e:
        logger.error(f"❌ WebSocket测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("🎯 加密货币量化交易系统 - 数据管道基础测试")
    logger.info("=" * 60)
    
    results = []
    
    # 测试1: 历史数据下载
    logger.info("\n📥 测试1: 历史数据下载")
    logger.info("-" * 40)
    result1 = await test_historical_download_basic()
    results.append(("历史数据下载", result1))
    
    # 测试2: WebSocket实时数据
    logger.info("\n📡 测试2: WebSocket实时数据")
    logger.info("-" * 40)
    result2 = await test_websocket_basic()
    results.append(("WebSocket实时数据", result2))
    
    # 输出最终结果
    logger.info("\n" + "=" * 60)
    logger.info("📊 测试结果汇总")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\n🏆 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过! 数据管道基础功能正常")
        return True
    else:
        logger.error("⚠️ 部分测试失败，需要检查数据管道配置")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试程序出错: {e}")
        sys.exit(1)