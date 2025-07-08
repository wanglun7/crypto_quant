"""TimescaleDB writer for trade data."""

import asyncio
import os
from datetime import datetime
from decimal import Decimal
from typing import Any, List, Dict, Union

import asyncpg
import structlog
from asyncpg.pool import Pool

logger = structlog.get_logger(__name__)


class TimescaleWriter:
    """Writer for inserting trade data into TimescaleDB."""

    def __init__(
        self,
        database_url: Union[str, None] = None,
        pool_size: int = 10,
        batch_size: int = 1000,
    ):
        """Initialize TimescaleWriter.

        Args:
            database_url: PostgreSQL connection URL
            pool_size: Connection pool size
            batch_size: Number of records to batch before inserting
        """
        self.database_url = database_url or os.getenv(
            "TIMESCALE_URL",
            "postgresql://localhost:5432/crypto_quant"
        )
        self.pool_size = pool_size
        self.batch_size = batch_size
        self._pool: Union[Pool, None] = None
        self._batch: List[Dict[str, Any]] = []
        self._batch_lock = asyncio.Lock()

    async def connect(self) -> None:
        """Create connection pool and initialize database."""
        try:
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=self.pool_size,
            )
            logger.info(
                "Connected to TimescaleDB",
                pool_size=self.pool_size,
            )

            # Create table if not exists
            await self._create_tables()

        except Exception as e:
            logger.error("Failed to connect to database", error=str(e))
            raise

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()
            logger.info("Disconnected from TimescaleDB")

    async def _create_tables(self) -> None:
        """Create tables and hypertables."""
        if not self._pool:
            raise RuntimeError("Not connected to database")

        async with self._pool.acquire() as conn:
            # Create trades table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    price NUMERIC(20, 8) NOT NULL,
                    quantity NUMERIC(20, 8) NOT NULL,
                    trade_id BIGINT,
                    first_trade_id BIGINT,
                    last_trade_id BIGINT,
                    is_buyer_maker BOOLEAN,
                    PRIMARY KEY (time, symbol, trade_id)
                )
            """)

            # Create hypertable (ignore if already exists)
            try:
                await conn.execute("""
                    SELECT create_hypertable('trades', 'time', if_not_exists => TRUE)
                """)
                logger.info("Created hypertable for trades")
            except asyncpg.UndefinedFunctionError:
                logger.warning("TimescaleDB extension not installed, using regular table")

            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_time
                ON trades (symbol, time DESC)
            """)

    async def write_trade(self, trade: dict[str, Any]) -> None:
        """Write a single trade to the batch.

        Args:
            trade: Trade data from Binance aggTrade stream
        """
        async with self._batch_lock:
            self._batch.append(trade)

            if len(self._batch) >= self.batch_size:
                await self._flush_batch()

    async def _flush_batch(self) -> None:
        """Flush current batch to database."""
        if not self._batch or not self._pool:
            return

        batch_to_insert = self._batch
        self._batch = []

        try:
            async with self._pool.acquire() as conn:
                # Prepare data for insertion
                records = []
                for trade in batch_to_insert:
                    records.append((
                        # Convert milliseconds to timestamp
                        datetime.fromtimestamp(
                            trade["E"] / 1000
                        ),
                        trade["s"],  # symbol
                        Decimal(trade["p"]),  # price
                        Decimal(trade["q"]),  # quantity
                        trade["a"],  # trade_id
                        trade["f"],  # first_trade_id
                        trade["l"],  # last_trade_id
                        trade["m"],  # is_buyer_maker
                    ))

                # Bulk insert
                await conn.executemany(
                    """
                    INSERT INTO trades (
                        time, symbol, price, quantity,
                        trade_id, first_trade_id, last_trade_id, is_buyer_maker
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (time, symbol, trade_id) DO NOTHING
                    """,
                    records
                )

                logger.info(
                    "Inserted batch of trades",
                    count=len(records),
                )

        except Exception as e:
            logger.error(
                "Failed to insert batch",
                error=str(e),
                batch_size=len(batch_to_insert),
            )
            # Re-add failed batch to the beginning
            async with self._batch_lock:
                self._batch = batch_to_insert + self._batch

    async def flush(self) -> None:
        """Force flush any pending trades."""
        async with self._batch_lock:
            await self._flush_batch()

    async def __aenter__(self) -> "TimescaleWriter":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.flush()
        await self.disconnect()
