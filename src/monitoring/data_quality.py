"""
Data Quality Monitoring System

This module provides comprehensive data quality checks for market data,
including completeness, accuracy, consistency, and timeliness validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from sqlalchemy import text
from prometheus_client import Counter, Histogram, Gauge

from src.data_pipeline.database import db_manager
from src.data_pipeline.cache import cache_manager
from src.data_pipeline.models import (
    MarketTick, OrderBookSnapshot, OHLCV, 
    FundingRate, OpenInterest, Liquidation
)
from config.settings import settings

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    """Data quality severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class QualityIssue:
    """Data quality issue representation"""
    issue_type: str
    severity: QualityLevel
    description: str
    affected_records: int
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class QualityReport:
    """Data quality assessment report"""
    dataset_name: str
    check_timestamp: datetime
    total_records: int
    issues: List[QualityIssue]
    quality_score: float
    metadata: Dict[str, Any]
    
    @property
    def has_critical_issues(self) -> bool:
        return any(issue.severity == QualityLevel.CRITICAL for issue in self.issues)
    
    @property
    def issue_count_by_severity(self) -> Dict[QualityLevel, int]:
        counts = {level: 0 for level in QualityLevel}
        for issue in self.issues:
            counts[issue.severity] += 1
        return counts


class DataQualityChecker:
    """Main data quality monitoring class"""
    
    def __init__(self):
        # Prometheus metrics
        self.quality_checks_total = Counter(
            'data_quality_checks_total',
            'Total number of data quality checks',
            ['dataset', 'check_type']
        )
        
        self.quality_issues_total = Counter(
            'data_quality_issues_total',
            'Total number of quality issues found',
            ['dataset', 'severity', 'issue_type']
        )
        
        self.quality_score_gauge = Gauge(
            'data_quality_score',
            'Current data quality score',
            ['dataset']
        )
        
        self.check_duration = Histogram(
            'data_quality_check_duration_seconds',
            'Time spent on quality checks',
            ['dataset', 'check_type']
        )
        
        # Quality thresholds
        self.thresholds = {
            'completeness_min': 0.95,
            'timeliness_max_delay': 300,  # 5 minutes
            'accuracy_min': 0.98,
            'consistency_min': 0.99,
            'max_price_change': 0.1,  # 10% max price change
            'min_volume_threshold': 0.001
        }
    
    def check_market_data_quality(self, 
                                 symbol: str, 
                                 exchange: str,
                                 timeframe: str = "1h",
                                 lookback_hours: int = 24) -> QualityReport:
        """
        Comprehensive market data quality check
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe
            lookback_hours: Hours to look back
            
        Returns:
            QualityReport with all findings
        """
        start_time = datetime.now()
        issues = []
        
        # Get data for analysis
        end_time = datetime.now()
        start_time_data = end_time - timedelta(hours=lookback_hours)
        
        with db_manager.get_session() as session:
            # Get OHLCV data
            ohlcv_data = session.query(OHLCV).filter(
                OHLCV.symbol == symbol,
                OHLCV.exchange == exchange,
                OHLCV.timeframe == timeframe,
                OHLCV.timestamp >= int(start_time_data.timestamp() * 1000000)
            ).order_by(OHLCV.timestamp).all()
            
            # Get tick data
            tick_data = session.query(MarketTick).filter(
                MarketTick.symbol == symbol,
                MarketTick.exchange == exchange,
                MarketTick.timestamp >= int(start_time_data.timestamp() * 1000000)
            ).order_by(MarketTick.timestamp).all()
            
            # Get order book data
            orderbook_data = session.query(OrderBookSnapshot).filter(
                OrderBookSnapshot.symbol == symbol,
                OrderBookSnapshot.exchange == exchange,
                OrderBookSnapshot.timestamp >= int(start_time_data.timestamp() * 1000000)
            ).order_by(OrderBookSnapshot.timestamp).all()
        
        total_records = len(ohlcv_data) + len(tick_data) + len(orderbook_data)
        
        # Run quality checks
        issues.extend(self._check_data_completeness(ohlcv_data, tick_data, orderbook_data, 
                                                  symbol, exchange, timeframe))
        issues.extend(self._check_data_timeliness(ohlcv_data, tick_data, orderbook_data))
        issues.extend(self._check_data_accuracy(ohlcv_data, tick_data))
        issues.extend(self._check_data_consistency(ohlcv_data, tick_data))
        issues.extend(self._check_price_anomalies(ohlcv_data, tick_data))
        issues.extend(self._check_volume_anomalies(ohlcv_data, tick_data))
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(issues, total_records)
        
        # Update metrics
        self.quality_score_gauge.labels(dataset=f"{exchange}_{symbol}").set(quality_score)
        for issue in issues:
            self.quality_issues_total.labels(
                dataset=f"{exchange}_{symbol}",
                severity=issue.severity.value,
                issue_type=issue.issue_type
            ).inc()
        
        # Record check duration
        duration = (datetime.now() - start_time).total_seconds()
        self.check_duration.labels(
            dataset=f"{exchange}_{symbol}",
            check_type="comprehensive"
        ).observe(duration)
        
        return QualityReport(
            dataset_name=f"{exchange}_{symbol}_{timeframe}",
            check_timestamp=datetime.now(),
            total_records=total_records,
            issues=issues,
            quality_score=quality_score,
            metadata={
                'lookback_hours': lookback_hours,
                'duration_seconds': duration,
                'thresholds': self.thresholds
            }
        )
    
    def _check_data_completeness(self, 
                               ohlcv_data: List[OHLCV], 
                               tick_data: List[MarketTick],
                               orderbook_data: List[OrderBookSnapshot],
                               symbol: str,
                               exchange: str,
                               timeframe: str) -> List[QualityIssue]:
        """Check for missing data gaps"""
        issues = []
        
        # Check OHLCV completeness
        if ohlcv_data:
            df = pd.DataFrame([{
                'timestamp': ohlcv.timestamp,
                'open': ohlcv.open,
                'high': ohlcv.high,
                'low': ohlcv.low,
                'close': ohlcv.close,
                'volume': ohlcv.volume
            } for ohlcv in ohlcv_data])
            
            # Check for missing candles
            df['timestamp'] = pd.to_datetime(df['timestamp'] / 1000000, unit='s')
            df = df.set_index('timestamp')
            
            # Expected frequency based on timeframe
            freq_map = {'1m': '1T', '5m': '5T', '15m': '15T', '1h': '1H', '4h': '4H', '1d': '1D'}
            expected_freq = freq_map.get(timeframe, '1H')
            
            # Create expected timestamp range
            expected_range = pd.date_range(
                start=df.index.min(),
                end=df.index.max(),
                freq=expected_freq
            )
            
            missing_count = len(expected_range) - len(df)
            if missing_count > 0:
                completeness_ratio = len(df) / len(expected_range)
                if completeness_ratio < self.thresholds['completeness_min']:
                    issues.append(QualityIssue(
                        issue_type="missing_ohlcv_data",
                        severity=QualityLevel.HIGH if completeness_ratio < 0.9 else QualityLevel.MEDIUM,
                        description=f"Missing {missing_count} OHLCV candles ({completeness_ratio:.2%} complete)",
                        affected_records=missing_count,
                        timestamp=datetime.now(),
                        metadata={
                            'completeness_ratio': completeness_ratio,
                            'expected_records': len(expected_range),
                            'actual_records': len(df)
                        }
                    ))
        
        # Check for null values
        null_checks = [
            (ohlcv_data, 'ohlcv', ['open', 'high', 'low', 'close', 'volume']),
            (tick_data, 'tick', ['price', 'volume']),
            (orderbook_data, 'orderbook', ['bids', 'asks'])
        ]
        
        for data, data_type, fields in null_checks:
            if data:
                for field in fields:
                    null_count = sum(1 for record in data if getattr(record, field) is None)
                    if null_count > 0:
                        issues.append(QualityIssue(
                            issue_type=f"null_{field}",
                            severity=QualityLevel.HIGH,
                            description=f"Found {null_count} null values in {data_type}.{field}",
                            affected_records=null_count,
                            timestamp=datetime.now(),
                            metadata={'field': field, 'data_type': data_type}
                        ))
        
        return issues
    
    def _check_data_timeliness(self, 
                             ohlcv_data: List[OHLCV], 
                             tick_data: List[MarketTick],
                             orderbook_data: List[OrderBookSnapshot]) -> List[QualityIssue]:
        """Check data freshness and update delays"""
        issues = []
        current_time = datetime.now()
        max_delay = self.thresholds['timeliness_max_delay']
        
        # Check latest data timestamps
        data_checks = [
            (ohlcv_data, 'ohlcv'),
            (tick_data, 'tick'),
            (orderbook_data, 'orderbook')
        ]
        
        for data, data_type in data_checks:
            if data:
                latest_timestamp = max(record.timestamp for record in data)
                latest_time = datetime.fromtimestamp(latest_timestamp / 1000000)
                delay_seconds = (current_time - latest_time).total_seconds()
                
                if delay_seconds > max_delay:
                    severity = QualityLevel.CRITICAL if delay_seconds > max_delay * 2 else QualityLevel.HIGH
                    issues.append(QualityIssue(
                        issue_type=f"stale_{data_type}_data",
                        severity=severity,
                        description=f"{data_type} data is {delay_seconds:.1f} seconds old",
                        affected_records=1,
                        timestamp=datetime.now(),
                        metadata={
                            'delay_seconds': delay_seconds,
                            'latest_timestamp': latest_timestamp,
                            'threshold': max_delay
                        }
                    ))
        
        return issues
    
    def _check_data_accuracy(self, 
                           ohlcv_data: List[OHLCV], 
                           tick_data: List[MarketTick]) -> List[QualityIssue]:
        """Check data accuracy and logical consistency"""
        issues = []
        
        # Check OHLCV accuracy
        for ohlcv in ohlcv_data:
            # OHLC relationship validation
            if not (ohlcv.low <= ohlcv.open <= ohlcv.high and
                   ohlcv.low <= ohlcv.close <= ohlcv.high):
                issues.append(QualityIssue(
                    issue_type="invalid_ohlc_relationship",
                    severity=QualityLevel.HIGH,
                    description=f"Invalid OHLC relationship: O={ohlcv.open}, H={ohlcv.high}, L={ohlcv.low}, C={ohlcv.close}",
                    affected_records=1,
                    timestamp=datetime.now(),
                    metadata={
                        'ohlcv_id': str(ohlcv.id),
                        'timestamp': ohlcv.timestamp,
                        'values': {
                            'open': ohlcv.open,
                            'high': ohlcv.high,
                            'low': ohlcv.low,
                            'close': ohlcv.close
                        }
                    }
                ))
            
            # Volume validation
            if ohlcv.volume < 0:
                issues.append(QualityIssue(
                    issue_type="negative_volume",
                    severity=QualityLevel.HIGH,
                    description=f"Negative volume: {ohlcv.volume}",
                    affected_records=1,
                    timestamp=datetime.now(),
                    metadata={'ohlcv_id': str(ohlcv.id), 'volume': ohlcv.volume}
                ))
        
        # Check tick data accuracy
        for tick in tick_data:
            if tick.price <= 0:
                issues.append(QualityIssue(
                    issue_type="invalid_price",
                    severity=QualityLevel.HIGH,
                    description=f"Invalid price: {tick.price}",
                    affected_records=1,
                    timestamp=datetime.now(),
                    metadata={'tick_id': str(tick.id), 'price': tick.price}
                ))
            
            if tick.volume <= 0:
                issues.append(QualityIssue(
                    issue_type="invalid_volume",
                    severity=QualityLevel.HIGH,
                    description=f"Invalid volume: {tick.volume}",
                    affected_records=1,
                    timestamp=datetime.now(),
                    metadata={'tick_id': str(tick.id), 'volume': tick.volume}
                ))
        
        return issues
    
    def _check_data_consistency(self, 
                              ohlcv_data: List[OHLCV], 
                              tick_data: List[MarketTick]) -> List[QualityIssue]:
        """Check cross-dataset consistency"""
        issues = []
        
        if not ohlcv_data or not tick_data:
            return issues
        
        # Group tick data by time windows matching OHLCV
        tick_df = pd.DataFrame([{
            'timestamp': tick.timestamp,
            'price': tick.price,
            'volume': tick.volume
        } for tick in tick_data])
        
        if tick_df.empty:
            return issues
        
        tick_df['timestamp'] = pd.to_datetime(tick_df['timestamp'] / 1000000, unit='s')
        
        # Check consistency between OHLCV and tick data
        for ohlcv in ohlcv_data:
            ohlcv_time = pd.to_datetime(ohlcv.timestamp / 1000000, unit='s')
            
            # Find ticks within the OHLCV candle timeframe
            window_start = ohlcv_time - pd.Timedelta(hours=1)  # Assuming 1h candles
            window_end = ohlcv_time
            
            window_ticks = tick_df[
                (tick_df['timestamp'] >= window_start) & 
                (tick_df['timestamp'] <= window_end)
            ]
            
            if not window_ticks.empty:
                tick_high = window_ticks['price'].max()
                tick_low = window_ticks['price'].min()
                tick_volume = window_ticks['volume'].sum()
                
                # Check price consistency
                if abs(tick_high - ohlcv.high) / ohlcv.high > 0.01:  # 1% tolerance
                    issues.append(QualityIssue(
                        issue_type="price_inconsistency",
                        severity=QualityLevel.MEDIUM,
                        description=f"High price mismatch: OHLCV={ohlcv.high}, Tick={tick_high}",
                        affected_records=1,
                        timestamp=datetime.now(),
                        metadata={
                            'ohlcv_high': ohlcv.high,
                            'tick_high': tick_high,
                            'difference_pct': abs(tick_high - ohlcv.high) / ohlcv.high
                        }
                    ))
                
                if abs(tick_low - ohlcv.low) / ohlcv.low > 0.01:  # 1% tolerance
                    issues.append(QualityIssue(
                        issue_type="price_inconsistency",
                        severity=QualityLevel.MEDIUM,
                        description=f"Low price mismatch: OHLCV={ohlcv.low}, Tick={tick_low}",
                        affected_records=1,
                        timestamp=datetime.now(),
                        metadata={
                            'ohlcv_low': ohlcv.low,
                            'tick_low': tick_low,
                            'difference_pct': abs(tick_low - ohlcv.low) / ohlcv.low
                        }
                    ))
                
                # Check volume consistency
                if abs(tick_volume - ohlcv.volume) / ohlcv.volume > 0.05:  # 5% tolerance
                    issues.append(QualityIssue(
                        issue_type="volume_inconsistency",
                        severity=QualityLevel.MEDIUM,
                        description=f"Volume mismatch: OHLCV={ohlcv.volume}, Tick={tick_volume}",
                        affected_records=1,
                        timestamp=datetime.now(),
                        metadata={
                            'ohlcv_volume': ohlcv.volume,
                            'tick_volume': tick_volume,
                            'difference_pct': abs(tick_volume - ohlcv.volume) / ohlcv.volume
                        }
                    ))
        
        return issues
    
    def _check_price_anomalies(self, 
                             ohlcv_data: List[OHLCV], 
                             tick_data: List[MarketTick]) -> List[QualityIssue]:
        """Detect price anomalies and outliers"""
        issues = []
        
        # Check OHLCV price anomalies
        if len(ohlcv_data) > 1:
            prices = [ohlcv.close for ohlcv in ohlcv_data]
            
            for i in range(1, len(prices)):
                price_change = abs(prices[i] - prices[i-1]) / prices[i-1]
                if price_change > self.thresholds['max_price_change']:
                    issues.append(QualityIssue(
                        issue_type="extreme_price_change",
                        severity=QualityLevel.HIGH,
                        description=f"Extreme price change: {price_change:.2%}",
                        affected_records=1,
                        timestamp=datetime.now(),
                        metadata={
                            'price_change_pct': price_change,
                            'previous_price': prices[i-1],
                            'current_price': prices[i],
                            'ohlcv_id': str(ohlcv_data[i].id)
                        }
                    ))
        
        # Check tick price anomalies
        if len(tick_data) > 1:
            prices = [tick.price for tick in tick_data]
            
            # Use statistical outlier detection
            if len(prices) >= 30:
                q25, q75 = np.percentile(prices, [25, 75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                
                for i, price in enumerate(prices):
                    if price < lower_bound or price > upper_bound:
                        issues.append(QualityIssue(
                            issue_type="price_outlier",
                            severity=QualityLevel.MEDIUM,
                            description=f"Price outlier detected: {price}",
                            affected_records=1,
                            timestamp=datetime.now(),
                            metadata={
                                'price': price,
                                'lower_bound': lower_bound,
                                'upper_bound': upper_bound,
                                'tick_id': str(tick_data[i].id)
                            }
                        ))
        
        return issues
    
    def _check_volume_anomalies(self, 
                              ohlcv_data: List[OHLCV], 
                              tick_data: List[MarketTick]) -> List[QualityIssue]:
        """Detect volume anomalies"""
        issues = []
        
        # Check for abnormally low volume
        for ohlcv in ohlcv_data:
            if ohlcv.volume < self.thresholds['min_volume_threshold']:
                issues.append(QualityIssue(
                    issue_type="low_volume",
                    severity=QualityLevel.LOW,
                    description=f"Very low volume: {ohlcv.volume}",
                    affected_records=1,
                    timestamp=datetime.now(),
                    metadata={
                        'volume': ohlcv.volume,
                        'threshold': self.thresholds['min_volume_threshold'],
                        'ohlcv_id': str(ohlcv.id)
                    }
                ))
        
        # Check for volume spikes
        if len(ohlcv_data) > 5:
            volumes = [ohlcv.volume for ohlcv in ohlcv_data]
            median_volume = np.median(volumes)
            
            for i, volume in enumerate(volumes):
                if volume > median_volume * 10:  # 10x spike
                    issues.append(QualityIssue(
                        issue_type="volume_spike",
                        severity=QualityLevel.MEDIUM,
                        description=f"Volume spike: {volume:.2f} (median: {median_volume:.2f})",
                        affected_records=1,
                        timestamp=datetime.now(),
                        metadata={
                            'volume': volume,
                            'median_volume': median_volume,
                            'spike_ratio': volume / median_volume,
                            'ohlcv_id': str(ohlcv_data[i].id)
                        }
                    ))
        
        return issues
    
    def _calculate_quality_score(self, 
                               issues: List[QualityIssue], 
                               total_records: int) -> float:
        """Calculate overall quality score (0-1)"""
        if total_records == 0:
            return 0.0
        
        # Weight issues by severity
        severity_weights = {
            QualityLevel.CRITICAL: 1.0,
            QualityLevel.HIGH: 0.7,
            QualityLevel.MEDIUM: 0.4,
            QualityLevel.LOW: 0.2,
            QualityLevel.INFO: 0.1
        }
        
        total_penalty = 0
        for issue in issues:
            penalty = severity_weights[issue.severity] * issue.affected_records
            total_penalty += penalty
        
        # Normalize to 0-1 scale
        max_possible_penalty = total_records
        quality_score = max(0, 1 - (total_penalty / max_possible_penalty))
        
        return quality_score
    
    def check_real_time_data(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Check real-time data quality from cache"""
        issues = []
        
        # Check cached orderbook
        orderbook = cache_manager.get_orderbook(symbol, exchange)
        if orderbook:
            issues.extend(self._validate_orderbook(orderbook))
        else:
            issues.append(QualityIssue(
                issue_type="missing_orderbook",
                severity=QualityLevel.HIGH,
                description="No orderbook data in cache",
                affected_records=1,
                timestamp=datetime.now(),
                metadata={'symbol': symbol, 'exchange': exchange}
            ))
        
        # Check cached tick
        tick = cache_manager.get_tick(symbol, exchange)
        if tick:
            issues.extend(self._validate_tick(tick))
        else:
            issues.append(QualityIssue(
                issue_type="missing_tick",
                severity=QualityLevel.HIGH,
                description="No tick data in cache",
                affected_records=1,
                timestamp=datetime.now(),
                metadata={'symbol': symbol, 'exchange': exchange}
            ))
        
        return {
            'issues': issues,
            'quality_score': self._calculate_quality_score(issues, 1),
            'timestamp': datetime.now()
        }
    
    def _validate_orderbook(self, orderbook: Dict[str, Any]) -> List[QualityIssue]:
        """Validate orderbook data structure"""
        issues = []
        
        required_fields = ['bids', 'asks', 'timestamp']
        for field in required_fields:
            if field not in orderbook:
                issues.append(QualityIssue(
                    issue_type="missing_orderbook_field",
                    severity=QualityLevel.HIGH,
                    description=f"Missing {field} in orderbook",
                    affected_records=1,
                    timestamp=datetime.now(),
                    metadata={'missing_field': field}
                ))
        
        # Check bid/ask structure
        if 'bids' in orderbook and 'asks' in orderbook:
            if not orderbook['bids'] or not orderbook['asks']:
                issues.append(QualityIssue(
                    issue_type="empty_orderbook",
                    severity=QualityLevel.HIGH,
                    description="Empty bids or asks in orderbook",
                    affected_records=1,
                    timestamp=datetime.now(),
                    metadata={'bids_count': len(orderbook['bids']), 'asks_count': len(orderbook['asks'])}
                ))
            
            # Check spread
            if orderbook['bids'] and orderbook['asks']:
                best_bid = float(orderbook['bids'][0][0])
                best_ask = float(orderbook['asks'][0][0])
                
                if best_bid >= best_ask:
                    issues.append(QualityIssue(
                        issue_type="crossed_spread",
                        severity=QualityLevel.CRITICAL,
                        description=f"Crossed spread: bid={best_bid}, ask={best_ask}",
                        affected_records=1,
                        timestamp=datetime.now(),
                        metadata={'best_bid': best_bid, 'best_ask': best_ask}
                    ))
        
        return issues
    
    def _validate_tick(self, tick: Dict[str, Any]) -> List[QualityIssue]:
        """Validate tick data structure"""
        issues = []
        
        required_fields = ['price', 'volume', 'timestamp']
        for field in required_fields:
            if field not in tick:
                issues.append(QualityIssue(
                    issue_type="missing_tick_field",
                    severity=QualityLevel.HIGH,
                    description=f"Missing {field} in tick",
                    affected_records=1,
                    timestamp=datetime.now(),
                    metadata={'missing_field': field}
                ))
        
        # Validate values
        if 'price' in tick and (tick['price'] <= 0 or not isinstance(tick['price'], (int, float))):
            issues.append(QualityIssue(
                issue_type="invalid_tick_price",
                severity=QualityLevel.HIGH,
                description=f"Invalid tick price: {tick['price']}",
                affected_records=1,
                timestamp=datetime.now(),
                metadata={'price': tick['price']}
            ))
        
        if 'volume' in tick and (tick['volume'] <= 0 or not isinstance(tick['volume'], (int, float))):
            issues.append(QualityIssue(
                issue_type="invalid_tick_volume",
                severity=QualityLevel.HIGH,
                description=f"Invalid tick volume: {tick['volume']}",
                affected_records=1,
                timestamp=datetime.now(),
                metadata={'volume': tick['volume']}
            ))
        
        return issues