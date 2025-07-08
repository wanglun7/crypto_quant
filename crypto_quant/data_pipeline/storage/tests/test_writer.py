"""Tests for TimescaleDB writer."""

import asyncio
import os
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from ..writer import TimescaleWriter


@pytest.fixture
def sample_trade():
    """Sample trade data from Binance aggTrade."""
    return {
        "e": "aggTrade",
        "E": 1672531200123,  # Event time in milliseconds
        "s": "BTCUSDT",
        "a": 1234567890,     # Aggregate trade ID
        "p": "43250.50",     # Price
        "q": "0.00123",      # Quantity
        "f": 9876543210,     # First trade ID
        "l": 9876543211,     # Last trade ID
        "T": 1672531200122,  # Trade time
        "m": False,          # Is buyer the maker
        "M": True,           # Ignore
    }


@pytest.mark.asyncio
async def test_writer_init():
    """Test writer initialization."""
    writer = TimescaleWriter(
        database_url="postgresql://test:test@localhost:5432/test",
        pool_size=5,
        batch_size=100,
    )
    assert writer.pool_size == 5
    assert writer.batch_size == 100
    assert writer._pool is None
    assert writer._batch == []


@pytest.mark.asyncio
async def test_writer_connect_disconnect():
    """Test connection and disconnection."""
    mock_pool = AsyncMock()

    with patch("asyncpg.create_pool", return_value=mock_pool) as mock_create:
        writer = TimescaleWriter()

        # Mock _create_tables to avoid database operations
        with patch.object(writer, '_create_tables', new_callable=AsyncMock):
            await writer.connect()

            mock_create.assert_called_once()
            assert writer._pool == mock_pool

            await writer.disconnect()
            mock_pool.close.assert_called_once()


@pytest.mark.asyncio
async def test_writer_batch_logic(sample_trade):
    """Test batch accumulation and flushing."""
    writer = TimescaleWriter(batch_size=2)
    writer._pool = AsyncMock()

    # Mock connection and execute
    mock_conn = AsyncMock()
    writer._pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Add first trade - should not flush yet
    await writer.write_trade(sample_trade)
    assert len(writer._batch) == 1  # Batch has one item
    mock_conn.executemany.assert_not_called()

    # Add second trade - should trigger flush
    await writer.write_trade(sample_trade)
    await asyncio.sleep(0.01)  # Allow async flush to complete

    assert len(writer._batch) == 0  # Batch was flushed
    mock_conn.executemany.assert_called_once()

    # Check the inserted data format
    call_args = mock_conn.executemany.call_args
    sql = call_args[0][0]
    records = call_args[0][1]

    assert "INSERT INTO trades" in sql
    assert len(records) == 2

    # Verify record format
    record = records[0]
    assert isinstance(record[0], datetime)  # time
    assert record[1] == "BTCUSDT"  # symbol
    assert record[2] == Decimal("43250.50")  # price
    assert record[3] == Decimal("0.00123")  # quantity


@pytest.mark.asyncio
async def test_writer_manual_flush(sample_trade):
    """Test manual flush."""
    writer = TimescaleWriter(batch_size=10)
    writer._pool = AsyncMock()

    mock_conn = AsyncMock()
    writer._pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Add one trade
    await writer.write_trade(sample_trade)

    # Manual flush
    await writer.flush()

    mock_conn.executemany.assert_called_once()
    assert len(writer._batch) == 0


@pytest.mark.asyncio
async def test_writer_error_handling(sample_trade):
    """Test error handling during insert."""
    writer = TimescaleWriter(batch_size=1)
    writer._pool = AsyncMock()

    mock_conn = AsyncMock()
    mock_conn.executemany.side_effect = asyncpg.PostgresError("Connection error")
    writer._pool.acquire.return_value.__aenter__.return_value = mock_conn

    # Add trade - should fail but re-add to batch
    await writer.write_trade(sample_trade)
    await asyncio.sleep(0.01)  # Allow async operations

    # Batch should be restored after failure
    assert len(writer._batch) == 1
    assert writer._batch[0] == sample_trade


@pytest.mark.asyncio
async def test_writer_context_manager():
    """Test async context manager."""
    with patch("asyncpg.create_pool") as mock_create:
        mock_pool = AsyncMock()
        mock_create.return_value = mock_pool

        async with TimescaleWriter() as writer:
            writer._create_tables = AsyncMock()
            assert writer._pool is not None

        mock_pool.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="Integration test requires TEST_DATABASE_URL"
)
async def test_writer_integration():
    """Integration test with real PostgreSQL."""
    database_url = os.getenv("TEST_DATABASE_URL")

    async with TimescaleWriter(database_url=database_url, batch_size=2) as writer:
        # Insert test trades
        trades = [
            {
                "e": "aggTrade",
                "E": 1672531200000 + i,
                "s": "BTCUSDT",
                "a": 1000 + i,
                "p": f"{43000 + i}.50",
                "q": "0.001",
                "f": 2000 + i,
                "l": 2001 + i,
                "T": 1672531200000 + i,
                "m": i % 2 == 0,
                "M": True,
            }
            for i in range(5)
        ]

        for trade in trades:
            await writer.write_trade(trade)

        # Force flush remaining
        await writer.flush()

        # Verify data was inserted
        async with writer._pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM trades WHERE symbol = 'BTCUSDT'"
            )
            assert count >= 5
