"""
Database connection and session management
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging
from typing import Generator
import asyncpg
import asyncio
from config.settings import settings
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database connections and sessions"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.async_pool = None
        
    def initialize(self):
        """Initialize database connections"""
        # Create synchronous engine
        self.engine = create_engine(
            settings.database.timescale_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
            echo=settings.debug
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
        
        # Enable TimescaleDB extensions
        self._setup_timescale()
        
        logger.info("Database initialized successfully")
    
    async def initialize_async(self):
        """Initialize async connection pool"""
        self.async_pool = await asyncpg.create_pool(
            settings.database.timescale_url,
            min_size=10,
            max_size=50,
            command_timeout=60
        )
        logger.info("Async database pool initialized")
    
    def _setup_timescale(self):
        """Setup TimescaleDB extensions and hypertables"""
        with self.engine.connect() as conn:
            # Enable TimescaleDB extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            conn.commit()
            
            # Create hypertables for time-series data
            hypertables = [
                ('market_ticks', 'timestamp'),
                ('orderbook_snapshots', 'timestamp'),
                ('ohlcv', 'timestamp'),
                ('funding_rates', 'timestamp'),
                ('open_interest', 'timestamp'),
                ('liquidations', 'timestamp'),
                ('onchain_metrics', 'timestamp'),
                ('model_predictions', 'timestamp'),
                ('trading_signals', 'timestamp')
            ]
            
            for table, time_column in hypertables:
                try:
                    conn.execute(text(
                        f"SELECT create_hypertable('{table}', '{time_column}', "
                        f"if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day')"
                    ))
                    conn.commit()
                    logger.info(f"Created hypertable for {table}")
                except Exception as e:
                    logger.warning(f"Could not create hypertable for {table}: {e}")
            
            # Create continuous aggregates for OHLCV data
            self._create_continuous_aggregates(conn)
    
    def _create_continuous_aggregates(self, conn):
        """Create continuous aggregates for faster queries"""
        # 5-minute aggregates
        try:
            conn.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_5m
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('5 minutes', to_timestamp(timestamp/1000000)) AS bucket,
                    symbol,
                    exchange,
                    first(open, timestamp) AS open,
                    max(high) AS high,
                    min(low) AS low,
                    last(close, timestamp) AS close,
                    sum(volume) AS volume
                FROM ohlcv
                WHERE timeframe = '1m'
                GROUP BY bucket, symbol, exchange
            """))
            conn.commit()
        except Exception as e:
            logger.warning(f"Could not create 5m continuous aggregate: {e}")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def get_async_connection(self):
        """Get async database connection from pool"""
        async with self.async_pool.acquire() as connection:
            yield connection
    
    def close(self):
        """Close all database connections"""
        if self.engine:
            self.engine.dispose()
        
        if self.async_pool:
            asyncio.create_task(self.async_pool.close())
        
        logger.info("Database connections closed")


# Global database manager instance
db_manager = DatabaseManager()