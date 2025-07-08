"""
Database models for market data storage
"""
from sqlalchemy import Column, Integer, BigInteger, Float, String, DateTime, Boolean, Index, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime
import uuid

Base = declarative_base()


class MarketTick(Base):
    """Store tick-level market data"""
    __tablename__ = 'market_ticks'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(20), nullable=False)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(BigInteger, nullable=False)  # Microsecond precision
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    side = Column(String(4), nullable=False)  # 'buy' or 'sell'
    
    __table_args__ = (
        Index('idx_market_ticks_timestamp', 'timestamp'),
        Index('idx_market_ticks_symbol_timestamp', 'symbol', 'timestamp'),
    )


class OrderBookSnapshot(Base):
    """Store order book snapshots"""
    __tablename__ = 'orderbook_snapshots'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(20), nullable=False)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    bids = Column(JSON, nullable=False)  # [[price, volume], ...]
    asks = Column(JSON, nullable=False)  # [[price, volume], ...]
    
    __table_args__ = (
        Index('idx_orderbook_timestamp', 'timestamp'),
        Index('idx_orderbook_symbol_timestamp', 'symbol', 'timestamp'),
    )


class OHLCV(Base):
    """Store OHLCV candle data"""
    __tablename__ = 'ohlcv'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(20), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)  # '1m', '5m', '1h', etc.
    timestamp = Column(BigInteger, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    trades = Column(Integer)
    
    __table_args__ = (
        Index('idx_ohlcv_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
    )


class FundingRate(Base):
    """Store perpetual contract funding rates"""
    __tablename__ = 'funding_rates'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(20), nullable=False)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    funding_rate = Column(Float, nullable=False)
    funding_time = Column(BigInteger, nullable=False)
    
    __table_args__ = (
        Index('idx_funding_symbol_timestamp', 'symbol', 'timestamp'),
    )


class OpenInterest(Base):
    """Store open interest data"""
    __tablename__ = 'open_interest'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(20), nullable=False)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    open_interest = Column(Float, nullable=False)  # In contracts
    open_interest_value = Column(Float, nullable=False)  # In USD
    
    __table_args__ = (
        Index('idx_oi_symbol_timestamp', 'symbol', 'timestamp'),
    )


class Liquidation(Base):
    """Store liquidation events"""
    __tablename__ = 'liquidations'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    exchange = Column(String(20), nullable=False)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    side = Column(String(5), nullable=False)  # 'long' or 'short'
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    value = Column(Float, nullable=False)  # In USD
    
    __table_args__ = (
        Index('idx_liquidation_timestamp', 'timestamp'),
        Index('idx_liquidation_symbol_timestamp', 'symbol', 'timestamp'),
    )


class OnChainMetric(Base):
    """Store on-chain metrics"""
    __tablename__ = 'onchain_metrics'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(50), nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    value = Column(Float, nullable=False)
    metadata = Column(JSON)  # Additional context
    
    __table_args__ = (
        Index('idx_onchain_metric_timestamp', 'metric_name', 'timestamp'),
    )


class ModelPrediction(Base):
    """Store model predictions for analysis"""
    __tablename__ = 'model_predictions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version = Column(String(50), nullable=False)
    timestamp = Column(BigInteger, nullable=False)
    prediction_type = Column(String(20), nullable=False)  # 'direction', 'price', etc.
    prediction_value = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    features_used = Column(JSON)
    
    __table_args__ = (
        Index('idx_prediction_timestamp', 'timestamp'),
        Index('idx_prediction_model_timestamp', 'model_version', 'timestamp'),
    )


class TradingSignal(Base):
    """Store generated trading signals"""
    __tablename__ = 'trading_signals'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    timestamp = Column(BigInteger, nullable=False)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(10), nullable=False)  # 'long', 'short', 'close'
    strength = Column(Float, nullable=False)  # 0-1
    entry_price = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_signal_timestamp', 'timestamp'),
        Index('idx_signal_symbol_timestamp', 'symbol', 'timestamp'),
    )