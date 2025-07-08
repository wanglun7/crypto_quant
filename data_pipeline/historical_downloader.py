"""
历史数据下载器
下载2020-2024年完整的Binance历史数据
支持多种数据类型：K线、深度、成交等
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path
import json
import time
import zipfile
import io
from decimal import Decimal
import sqlite3
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataInterval(Enum):
    """数据间隔枚举"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    HOUR_12 = "12h"
    DAY_1 = "1d"


class DataType(Enum):
    """数据类型枚举"""
    KLINES = "klines"
    TRADES = "trades"
    AGG_TRADES = "aggTrades"
    DEPTH = "depth"


@dataclass
class DownloadTask:
    """下载任务数据结构"""
    symbol: str
    data_type: DataType
    interval: Optional[DataInterval]
    start_date: datetime
    end_date: datetime
    file_path: str
    status: str = "pending"  # pending, downloading, completed, failed
    progress: float = 0.0
    error_msg: Optional[str] = None


class BinanceHistoricalDownloader:
    """Binance历史数据下载器"""
    
    def __init__(self, base_path: str = "./data/historical"):
        """
        初始化历史数据下载器
        
        Args:
            base_path: 数据存储基础路径
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Binance历史数据API URLs
        self.base_url = "https://api.binance.com/api/v3"
        self.data_url = "https://data.binance.vision/data"
        
        # 下载配置
        self.max_concurrent_downloads = 5
        self.retry_attempts = 3
        self.retry_delay = 5
        self.timeout = 30
        
        # 下载任务队列
        self.download_tasks: List[DownloadTask] = []
        self.completed_tasks: List[DownloadTask] = []
        self.failed_tasks: List[DownloadTask] = []
        
        # 统计信息
        self.total_downloaded = 0
        self.total_size = 0
        self.start_time = None
        
        logger.info(f"Historical downloader initialized, base path: {self.base_path}")
    
    async def get_available_symbols(self) -> List[str]:
        """获取可用的交易对列表"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/exchangeInfo") as response:
                    data = await response.json()
                    
                    symbols = [s['symbol'] for s in data['symbols'] 
                             if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT']
                    
                    logger.info(f"Found {len(symbols)} active USDT symbols")
                    return symbols
                    
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []
    
    def _get_klines_url(self, symbol: str, interval: DataInterval, 
                       date: datetime) -> str:
        """构建K线数据下载URL"""
        date_str = date.strftime("%Y-%m-%d")
        return f"{self.data_url}/spot/daily/klines/{symbol}/{interval.value}/{symbol}-{interval.value}-{date_str}.zip"
    
    def _get_trades_url(self, symbol: str, date: datetime) -> str:
        """构建成交数据下载URL"""
        date_str = date.strftime("%Y-%m-%d")
        return f"{self.data_url}/spot/daily/trades/{symbol}/{symbol}-trades-{date_str}.zip"
    
    def _get_agg_trades_url(self, symbol: str, date: datetime) -> str:
        """构建聚合成交数据下载URL"""
        date_str = date.strftime("%Y-%m-%d")
        return f"{self.data_url}/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
    
    async def _download_file(self, url: str, file_path: Path, 
                           task: DownloadTask) -> bool:
        """下载单个文件"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.read()
                        
                        # 创建目录
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # 写入文件
                        with open(file_path, 'wb') as f:
                            f.write(content)
                        
                        # 更新统计
                        self.total_downloaded += 1
                        self.total_size += len(content)
                        
                        logger.info(f"Downloaded: {file_path.name} ({len(content)} bytes)")
                        return True
                    else:
                        logger.warning(f"Failed to download {url}: HTTP {response.status}")
                        return False
                        
        except asyncio.TimeoutError:
            logger.error(f"Timeout downloading {url}")
            return False
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False
    
    async def _download_with_retry(self, url: str, file_path: Path, 
                                 task: DownloadTask) -> bool:
        """带重试的下载"""
        for attempt in range(self.retry_attempts):
            try:
                if await self._download_file(url, file_path, task):
                    return True
                    
                if attempt < self.retry_attempts - 1:
                    logger.info(f"Retry {attempt + 1}/{self.retry_attempts} for {url}")
                    await asyncio.sleep(self.retry_delay)
                    
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
        
        return False
    
    def _parse_klines_data(self, zip_content: bytes) -> pd.DataFrame:
        """解析K线数据"""
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                # 获取zip文件中的CSV文件
                csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    logger.error("No CSV files found in zip")
                    return pd.DataFrame()
                
                # 读取第一个CSV文件
                with zip_file.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file, header=None)
                    
                    # 设置列名
                    df.columns = [
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                        'taker_buy_quote_volume', 'ignore'
                    ]
                    
                    # 转换时间戳
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                    
                    # 转换数值类型
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                                     'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return df
                    
        except Exception as e:
            logger.error(f"Error parsing klines data: {e}")
            return pd.DataFrame()
    
    def _parse_trades_data(self, zip_content: bytes) -> pd.DataFrame:
        """解析成交数据"""
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    logger.error("No CSV files found in zip")
                    return pd.DataFrame()
                
                with zip_file.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file, header=None)
                    
                    # 设置列名
                    df.columns = [
                        'trade_id', 'price', 'quantity', 'quote_quantity',
                        'timestamp', 'buyer_maker', 'best_match'
                    ]
                    
                    # 转换时间戳
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # 转换数值类型
                    numeric_columns = ['price', 'quantity', 'quote_quantity']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return df
                    
        except Exception as e:
            logger.error(f"Error parsing trades data: {e}")
            return pd.DataFrame()
    
    async def download_klines(self, symbol: str, interval: DataInterval,
                            start_date: datetime, end_date: datetime) -> bool:
        """下载K线数据"""
        logger.info(f"Starting klines download for {symbol} {interval.value} "
                   f"from {start_date.date()} to {end_date.date()}")
        
        current_date = start_date
        tasks = []
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            url = self._get_klines_url(symbol, interval, current_date)
            file_path = self.base_path / "klines" / symbol / interval.value / f"{symbol}-{interval.value}-{date_str}.zip"
            
            # 检查文件是否已存在
            if not file_path.exists():
                task = DownloadTask(
                    symbol=symbol,
                    data_type=DataType.KLINES,
                    interval=interval,
                    start_date=current_date,
                    end_date=current_date,
                    file_path=str(file_path)
                )
                tasks.append(task)
            else:
                logger.info(f"File already exists: {file_path.name}")
            
            current_date += timedelta(days=1)
        
        # 批量下载
        if tasks:
            await self._batch_download(tasks)
            
        logger.info(f"Completed klines download for {symbol} {interval.value}")
        return True
    
    async def download_trades(self, symbol: str, start_date: datetime, 
                            end_date: datetime) -> bool:
        """下载成交数据"""
        logger.info(f"Starting trades download for {symbol} "
                   f"from {start_date.date()} to {end_date.date()}")
        
        current_date = start_date
        tasks = []
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            url = self._get_trades_url(symbol, current_date)
            file_path = self.base_path / "trades" / symbol / f"{symbol}-trades-{date_str}.zip"
            
            if not file_path.exists():
                task = DownloadTask(
                    symbol=symbol,
                    data_type=DataType.TRADES,
                    interval=None,
                    start_date=current_date,
                    end_date=current_date,
                    file_path=str(file_path)
                )
                tasks.append(task)
            else:
                logger.info(f"File already exists: {file_path.name}")
            
            current_date += timedelta(days=1)
        
        if tasks:
            await self._batch_download(tasks)
            
        logger.info(f"Completed trades download for {symbol}")
        return True
    
    async def _batch_download(self, tasks: List[DownloadTask]):
        """批量下载任务"""
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        
        async def download_task(task: DownloadTask):
            async with semaphore:
                task.status = "downloading"
                
                # 构建URL
                if task.data_type == DataType.KLINES:
                    url = self._get_klines_url(task.symbol, task.interval, task.start_date)
                elif task.data_type == DataType.TRADES:
                    url = self._get_trades_url(task.symbol, task.start_date)
                elif task.data_type == DataType.AGG_TRADES:
                    url = self._get_agg_trades_url(task.symbol, task.start_date)
                else:
                    task.status = "failed"
                    task.error_msg = f"Unsupported data type: {task.data_type}"
                    return
                
                # 下载文件
                file_path = Path(task.file_path)
                success = await self._download_with_retry(url, file_path, task)
                
                if success:
                    task.status = "completed"
                    task.progress = 1.0
                    self.completed_tasks.append(task)
                else:
                    task.status = "failed"
                    task.error_msg = f"Failed to download after {self.retry_attempts} attempts"
                    self.failed_tasks.append(task)
        
        # 执行所有下载任务
        await asyncio.gather(*[download_task(task) for task in tasks])
    
    async def download_comprehensive_data(self, symbols: List[str],
                                        start_date: datetime, end_date: datetime):
        """下载综合数据集"""
        logger.info(f"Starting comprehensive data download for {len(symbols)} symbols")
        self.start_time = time.time()
        
        # 下载1分钟K线数据（最重要）
        for symbol in symbols:
            await self.download_klines(symbol, DataInterval.MINUTE_1, start_date, end_date)
        
        # 下载5分钟K线数据
        for symbol in symbols:
            await self.download_klines(symbol, DataInterval.MINUTE_5, start_date, end_date)
        
        # 下载1小时K线数据
        for symbol in symbols:
            await self.download_klines(symbol, DataInterval.HOUR_1, start_date, end_date)
        
        # 下载日K线数据
        for symbol in symbols:
            await self.download_klines(symbol, DataInterval.DAY_1, start_date, end_date)
        
        # 下载成交数据（仅主要交易对）
        main_symbols = [s for s in symbols if s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']]
        for symbol in main_symbols:
            await self.download_trades(symbol, start_date, end_date)
        
        # 输出统计信息
        elapsed_time = time.time() - self.start_time
        logger.info(f"Download completed in {elapsed_time:.2f} seconds")
        logger.info(f"Total files downloaded: {self.total_downloaded}")
        logger.info(f"Total size: {self.total_size / (1024*1024):.2f} MB")
        logger.info(f"Completed tasks: {len(self.completed_tasks)}")
        logger.info(f"Failed tasks: {len(self.failed_tasks)}")
    
    def save_download_report(self, report_path: str = None):
        """保存下载报告"""
        if report_path is None:
            report_path = self.base_path / "download_report.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_downloaded': self.total_downloaded,
            'total_size': self.total_size,
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'failed_task_details': [
                {
                    'symbol': task.symbol,
                    'data_type': task.data_type.value,
                    'interval': task.interval.value if task.interval else None,
                    'date': task.start_date.isoformat(),
                    'error': task.error_msg
                }
                for task in self.failed_tasks
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Download report saved to {report_path}")
    
    def get_download_statistics(self) -> Dict[str, Any]:
        """获取下载统计信息"""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_downloaded': self.total_downloaded,
            'total_size_mb': self.total_size / (1024*1024),
            'elapsed_time_seconds': elapsed_time,
            'download_rate_mbps': (self.total_size / (1024*1024)) / elapsed_time if elapsed_time > 0 else 0,
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'success_rate': len(self.completed_tasks) / (len(self.completed_tasks) + len(self.failed_tasks)) if (len(self.completed_tasks) + len(self.failed_tasks)) > 0 else 0
        }


# 便捷函数
async def download_btc_historical_data(start_year: int = 2020, end_year: int = 2024):
    """下载BTC历史数据的便捷函数"""
    downloader = BinanceHistoricalDownloader()
    
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    symbols = ['BTCUSDT']
    
    await downloader.download_comprehensive_data(symbols, start_date, end_date)
    
    # 保存报告
    downloader.save_download_report()
    
    # 输出统计
    stats = downloader.get_download_statistics()
    print(f"Download Statistics:")
    print(f"- Files downloaded: {stats['total_downloaded']}")
    print(f"- Total size: {stats['total_size_mb']:.2f} MB")
    print(f"- Time elapsed: {stats['elapsed_time_seconds']:.2f} seconds")
    print(f"- Success rate: {stats['success_rate']:.2%}")


async def download_top_crypto_data(start_year: int = 2020, end_year: int = 2024):
    """下载主要加密货币历史数据"""
    downloader = BinanceHistoricalDownloader()
    
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    # 主要加密货币
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT',
        'SOLUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT'
    ]
    
    await downloader.download_comprehensive_data(symbols, start_date, end_date)
    
    downloader.save_download_report()
    
    return downloader.get_download_statistics()


if __name__ == "__main__":
    # 示例使用
    print("Starting BTC historical data download...")
    asyncio.run(download_btc_historical_data(2020, 2024))
    print("Download completed!")