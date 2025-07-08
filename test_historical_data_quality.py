"""
历史数据质量验证测试
验证下载的历史数据是否完整、准确，能否支持回测
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
from typing import Dict, Any

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_pipeline.historical_downloader import BinanceHistoricalDownloader, DataInterval

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoricalDataValidator:
    """历史数据验证器"""
    
    def __init__(self, data_path: str = "./data/test_basic"):
        self.data_path = Path(data_path)
        self.downloader = BinanceHistoricalDownloader(str(self.data_path))
        
    def parse_klines_file(self, file_path: Path) -> pd.DataFrame:
        """解析K线数据文件"""
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                # 获取CSV文件
                csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                if not csv_files:
                    logger.error(f"No CSV files found in {file_path}")
                    return pd.DataFrame()
                
                # 读取CSV数据
                with zip_file.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file, header=None)
                    
                    # 设置列名
                    df.columns = [
                        'open_time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                        'taker_buy_quote_volume', 'ignore'
                    ]
                    
                    # 转换时间戳 (处理可能的时间戳格式问题)
                    try:
                        # Binance的时间戳是微秒级 (microseconds)
                        df['open_time'] = pd.to_datetime(df['open_time'], unit='us')
                        df['close_time'] = pd.to_datetime(df['close_time'], unit='us')
                    except Exception as e:
                        try:
                            # 如果微秒转换失败，尝试毫秒转换
                            logger.warning(f"时间戳微秒转换失败，尝试毫秒转换: {e}")
                            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                        except Exception as e2:
                            # 最后尝试秒转换
                            logger.warning(f"时间戳毫秒转换失败，尝试秒转换: {e2}")
                            df['open_time'] = pd.to_datetime(df['open_time'], unit='s')
                            df['close_time'] = pd.to_datetime(df['close_time'], unit='s')
                    
                    # 转换数值类型
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                                     'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    return df
                    
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return pd.DataFrame()
    
    def validate_ohlc_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """验证OHLC数据的合理性"""
        results = {}
        
        # 检查1: 价格合理性 (High >= Low, Open/Close在High/Low之间)
        price_logic = (
            (df['high'] >= df['low']) & 
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) & 
            (df['low'] <= df['close'])
        )
        results['price_logic'] = price_logic.all()
        
        # 检查2: 价格不能为0或负数
        price_positive = (
            (df['open'] > 0) & 
            (df['high'] > 0) & 
            (df['low'] > 0) & 
            (df['close'] > 0)
        )
        results['price_positive'] = price_positive.all()
        
        # 检查3: 成交量不能为负数
        volume_positive = (df['volume'] >= 0)
        results['volume_positive'] = volume_positive.all()
        
        # 检查4: 没有重复的时间戳
        no_duplicates = not df['open_time'].duplicated().any()
        results['no_duplicates'] = no_duplicates
        
        # 检查5: 时间戳是连续的
        if len(df) > 1:
            time_diff = df['open_time'].diff().dt.total_seconds()
            expected_interval = 3600  # 1小时 = 3600秒
            time_continuous = (time_diff[1:] == expected_interval).all()
            results['time_continuous'] = time_continuous
        else:
            results['time_continuous'] = True
        
        # 检查6: 没有异常的价格跳跃 (超过10%的单小时涨跌)
        price_changes = df['close'].pct_change().abs()
        reasonable_changes = (price_changes < 0.1).all()  # 10%阈值
        results['reasonable_changes'] = reasonable_changes
        
        return results
    
    def analyze_data_completeness(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """分析数据完整性"""
        results = {
            'total_files': 0,
            'parsed_files': 0,
            'total_records': 0,
            'date_coverage': {},
            'missing_dates': [],
            'data_quality': {}
        }
        
        # 检查每一天的数据文件
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            file_path = self.data_path / "klines" / "BTCUSDT" / "1h" / f"BTCUSDT-1h-{date_str}.zip"
            
            results['total_files'] += 1
            
            if file_path.exists():
                # 解析文件
                df = self.parse_klines_file(file_path)
                
                if not df.empty:
                    results['parsed_files'] += 1
                    results['total_records'] += len(df)
                    
                    # 验证数据质量
                    quality_checks = self.validate_ohlc_data(df)
                    results['data_quality'][date_str] = quality_checks
                    
                    # 记录日期覆盖情况
                    results['date_coverage'][date_str] = {
                        'records': len(df),
                        'start_time': df['open_time'].min(),
                        'end_time': df['open_time'].max(),
                        'quality_score': sum(quality_checks.values()) / len(quality_checks)
                    }
                else:
                    results['missing_dates'].append(date_str)
            else:
                results['missing_dates'].append(date_str)
            
            current_date += timedelta(days=1)
        
        return results
    
    async def download_extended_data(self, start_year: int = 2020, end_year: int = 2024):
        """下载扩展的历史数据用于回测"""
        logger.info(f"开始下载 {start_year}-{end_year} 年的BTCUSDT历史数据...")
        
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        # 下载不同间隔的数据
        intervals = [
            DataInterval.MINUTE_1,   # 1分钟数据 (最重要)
            DataInterval.MINUTE_5,   # 5分钟数据
            DataInterval.HOUR_1,     # 1小时数据
            DataInterval.DAY_1       # 日数据
        ]
        
        download_results = {}
        
        for interval in intervals:
            logger.info(f"下载 {interval.value} 间隔数据...")
            
            try:
                await self.downloader.download_klines(
                    'BTCUSDT', interval, start_date, end_date
                )
                
                stats = self.downloader.get_download_statistics()
                download_results[interval.value] = {
                    'files_downloaded': stats['total_downloaded'],
                    'total_size_mb': stats['total_size_mb'],
                    'success_rate': stats['success_rate']
                }
                
                logger.info(f"✅ {interval.value} 数据下载完成: "
                          f"{stats['total_downloaded']} 文件, "
                          f"{stats['total_size_mb']:.2f} MB")
                
            except Exception as e:
                logger.error(f"❌ {interval.value} 数据下载失败: {e}")
                download_results[interval.value] = {'error': str(e)}
        
        return download_results
    
    def generate_data_report(self, analysis_results: Dict[str, Any]) -> str:
        """生成数据质量报告"""
        report = []
        report.append("# 历史数据质量分析报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 总体统计
        report.append("## 总体统计")
        report.append(f"- 总文件数: {analysis_results['total_files']}")
        report.append(f"- 成功解析文件数: {analysis_results['parsed_files']}")
        report.append(f"- 总记录数: {analysis_results['total_records']:,}")
        report.append(f"- 缺失日期数: {len(analysis_results['missing_dates'])}")
        report.append("")
        
        # 数据完整性
        if analysis_results['missing_dates']:
            report.append("## 缺失数据")
            for date in analysis_results['missing_dates']:
                report.append(f"- {date}")
            report.append("")
        
        # 数据质量
        report.append("## 数据质量检查")
        quality_scores = []
        for date, quality in analysis_results['data_quality'].items():
            score = sum(quality.values()) / len(quality)
            quality_scores.append(score)
            
            if score < 1.0:  # 有质量问题
                report.append(f"### {date} (质量分: {score:.2f})")
                for check, passed in quality.items():
                    status = "✅" if passed else "❌"
                    report.append(f"  - {check}: {status}")
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            report.append(f"平均质量分: {avg_quality:.2f}")
        
        return "\n".join(report)


async def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("🔍 历史数据质量验证测试")
    logger.info("=" * 60)
    
    validator = HistoricalDataValidator()
    
    # 测试1: 验证现有测试数据
    logger.info("\n📊 测试1: 验证现有测试数据质量")
    logger.info("-" * 40)
    
    test_start = datetime(2025, 7, 4)
    test_end = datetime(2025, 7, 7)
    
    analysis = validator.analyze_data_completeness(test_start, test_end)
    
    # 生成报告
    report = validator.generate_data_report(analysis)
    
    # 保存报告
    report_path = Path("./data/data_quality_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"📄 数据质量报告已保存到: {report_path}")
    
    # 输出关键统计
    logger.info("📊 关键统计:")
    logger.info(f"  - 文件覆盖率: {analysis['parsed_files']}/{analysis['total_files']} ({analysis['parsed_files']/analysis['total_files']*100:.1f}%)")
    logger.info(f"  - 总记录数: {analysis['total_records']:,}")
    logger.info(f"  - 缺失日期: {len(analysis['missing_dates'])}")
    
    # 计算平均质量分
    if analysis['data_quality']:
        quality_scores = []
        for quality in analysis['data_quality'].values():
            score = sum(quality.values()) / len(quality.values())
            quality_scores.append(score)
        avg_quality = sum(quality_scores) / len(quality_scores)
        logger.info(f"  - 平均质量分: {avg_quality:.2f}/1.00")
    
    # 测试2: 是否需要下载更多数据
    logger.info("\n📥 测试2: 评估是否需要下载更多数据")
    logger.info("-" * 40)
    
    # 检查是否有足够的数据进行回测
    if analysis['total_records'] < 1000:  # 少于1000条记录
        logger.warning("⚠️ 当前数据量不足以进行有效回测")
        logger.info("建议下载更多历史数据...")
        
        # 询问是否下载更多数据
        download_more = input("是否下载2022-2024年的数据? (y/n): ").lower().strip()
        
        if download_more == 'y':
            logger.info("开始下载2022-2024年数据...")
            download_results = await validator.download_extended_data(2022, 2024)
            
            logger.info("📊 下载结果:")
            for interval, result in download_results.items():
                if 'error' in result:
                    logger.error(f"  - {interval}: ❌ {result['error']}")
                else:
                    logger.info(f"  - {interval}: ✅ {result['files_downloaded']} 文件, "
                              f"{result['total_size_mb']:.2f} MB")
        else:
            logger.info("跳过数据下载")
    else:
        logger.info("✅ 当前数据量足够进行基本回测测试")
    
    # 测试3: 数据解析和预处理测试
    logger.info("\n🔧 测试3: 数据解析和预处理测试")
    logger.info("-" * 40)
    
    # 测试解析一个文件
    test_file = Path("./data/test_basic/klines/BTCUSDT/1h/BTCUSDT-1h-2025-07-04.zip")
    if test_file.exists():
        df = validator.parse_klines_file(test_file)
        if not df.empty:
            logger.info(f"✅ 成功解析测试文件: {len(df)} 条记录")
            logger.info(f"  - 时间范围: {df['open_time'].min()} 到 {df['open_time'].max()}")
            logger.info(f"  - 价格范围: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
            logger.info(f"  - 平均成交量: {df['volume'].mean():,.2f}")
            
            # 数据质量检查
            quality = validator.validate_ohlc_data(df)
            logger.info("  - 数据质量检查:")
            for check, passed in quality.items():
                status = "✅" if passed else "❌"
                logger.info(f"    {check}: {status}")
        else:
            logger.error("❌ 文件解析失败")
    else:
        logger.error("❌ 测试文件不存在")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 历史数据验证完成")
    logger.info("=" * 60)
    
    # 根据测试结果给出建议
    if analysis['parsed_files'] == analysis['total_files'] and analysis['total_records'] > 0:
        logger.info("✅ 数据验证通过，可以进行下一步回测系统开发")
        return True
    else:
        logger.error("❌ 数据验证失败，需要解决数据问题")
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