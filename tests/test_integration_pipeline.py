"""Integration test for BinanceWSCollector + TimescaleWriter pipeline."""

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from crypto_quant.data_pipeline.collectors.binance_ws import BinanceWSCollector
from crypto_quant.data_pipeline.storage.writer import TimescaleWriter


@pytest.fixture
def sample_trades():
    """Sample trade data for testing."""
    return [
        {
            "e": "aggTrade",
            "E": 1672531200000 + i * 1000,
            "s": "BTCUSDT",
            "a": 1000000 + i,
            "p": f"{43000 + i}.50",
            "q": "0.001",
            "f": 2000000 + i,
            "l": 2000001 + i,
            "T": 1672531200000 + i * 1000,
            "m": i % 2 == 0,
            "M": True,
        }
        for i in range(10)
    ]


@pytest.fixture
def test_database():
    """Set up test database connection."""
    database_url = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://test:test@localhost:5433/crypto_quant_test"
    )
    return database_url


@pytest.mark.asyncio
async def test_pipeline_integration_with_mocked_websocket(sample_trades, test_database):
    """Test full pipeline with mocked WebSocket and real TimescaleDB."""

    # Mock WebSocket connection
    mock_ws = AsyncMock()
    mock_ws.closed = False
    mock_ws.__aiter__.return_value = [json.dumps(trade) for trade in sample_trades]

    with patch("websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__.return_value = mock_ws

        # Initialize components
        collector = BinanceWSCollector()
        writer = TimescaleWriter(
            database_url=test_database,
            batch_size=5,  # Small batch size for testing
        )

        await collector.start()
        await writer.connect()

        try:
            # Process trades
            trade_count = 0
            async for trade in collector.produce():
                await writer.write_trade(trade)
                trade_count += 1

                # Stop after processing all sample trades
                if trade_count >= len(sample_trades):
                    break

            # Flush any remaining trades
            await writer.flush()

            # Verify data was written to database
            async with writer._pool.acquire() as conn:
                count = await conn.fetchval("SELECT COUNT(*) FROM trades")
                assert count == len(sample_trades)

                # Verify data integrity
                rows = await conn.fetch(
                    "SELECT * FROM trades ORDER BY time ASC"
                )

                for i, row in enumerate(rows):
                    expected_trade = sample_trades[i]
                    assert row["symbol"] == expected_trade["s"]
                    assert float(row["price"]) == float(expected_trade["p"])
                    assert float(row["quantity"]) == float(expected_trade["q"])
                    assert row["trade_id"] == expected_trade["a"]
                    assert row["is_buyer_maker"] == expected_trade["m"]

        finally:
            await collector.stop()
            await writer.disconnect()


@pytest.mark.asyncio
async def test_pipeline_error_handling(test_database):
    """Test pipeline error handling with connection failures."""

    # Mock WebSocket that fails after a few messages
    mock_ws = AsyncMock()
    mock_ws.closed = False

    # First call succeeds with one message, second call fails
    call_count = 0
    async def mock_aiter():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield json.dumps({
                "e": "aggTrade",
                "E": 1672531200000,
                "s": "BTCUSDT",
                "a": 1000000,
                "p": "43000.50",
                "q": "0.001",
                "f": 2000000,
                "l": 2000001,
                "T": 1672531200000,
                "m": False,
                "M": True,
            })
        else:
            raise ConnectionError("WebSocket connection failed")

    mock_ws.__aiter__ = mock_aiter

    with patch("websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__.return_value = mock_ws

        collector = BinanceWSCollector()
        writer = TimescaleWriter(
            database_url=test_database,
            batch_size=1,
        )

        await collector.start()
        await writer.connect()

        try:
            # Process trades until error occurs
            trade_count = 0

            with patch("asyncio.sleep") as mock_sleep:
                mock_sleep.return_value = None  # Skip sleep delays

                async for trade in collector.produce():
                    await writer.write_trade(trade)
                    trade_count += 1

                    # Stop after first trade to test error handling
                    if trade_count >= 1:
                        break

                # Verify that one trade was processed successfully
                await writer.flush()

                async with writer._pool.acquire() as conn:
                    count = await conn.fetchval("SELECT COUNT(*) FROM trades")
                    assert count == 1

        finally:
            await collector.stop()
            await writer.disconnect()


@pytest.mark.asyncio
async def test_pipeline_batch_processing(test_database):
    """Test that batching works correctly."""

    # Create more trades than batch size
    trades = [
        {
            "e": "aggTrade",
            "E": 1672531200000 + i * 1000,
            "s": "BTCUSDT",
            "a": 1000000 + i,
            "p": f"{43000 + i}.50",
            "q": "0.001",
            "f": 2000000 + i,
            "l": 2000001 + i,
            "T": 1672531200000 + i * 1000,
            "m": i % 2 == 0,
            "M": True,
        }
        for i in range(15)  # 15 trades
    ]

    mock_ws = AsyncMock()
    mock_ws.closed = False
    mock_ws.__aiter__.return_value = [json.dumps(trade) for trade in trades]

    with patch("websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__.return_value = mock_ws

        collector = BinanceWSCollector()
        writer = TimescaleWriter(
            database_url=test_database,
            batch_size=6,  # Should create 3 batches: 6 + 6 + 3
        )

        await collector.start()
        await writer.connect()

        try:
            # Process all trades
            trade_count = 0
            async for trade in collector.produce():
                await writer.write_trade(trade)
                trade_count += 1

                if trade_count >= len(trades):
                    break

            # Flush final batch
            await writer.flush()

            # Verify all trades were written
            async with writer._pool.acquire() as conn:
                count = await conn.fetchval("SELECT COUNT(*) FROM trades")
                assert count == len(trades)

                # Verify data integrity with ORDER BY to ensure consistent ordering
                rows = await conn.fetch(
                    "SELECT * FROM trades ORDER BY trade_id ASC"
                )

                for i, row in enumerate(rows):
                    expected_trade = trades[i]
                    assert row["trade_id"] == expected_trade["a"]
                    assert row["symbol"] == expected_trade["s"]

        finally:
            await collector.stop()
            await writer.disconnect()


@pytest.mark.asyncio
async def test_pipeline_graceful_shutdown(test_database):
    """Test graceful shutdown of the pipeline."""

    trades = [
        {
            "e": "aggTrade",
            "E": 1672531200000 + i * 1000,
            "s": "BTCUSDT",
            "a": 1000000 + i,
            "p": f"{43000}.50",
            "q": "0.001",
            "f": 2000000 + i,
            "l": 2000001 + i,
            "T": 1672531200000 + i * 1000,
            "m": False,
            "M": True,
        }
        for i in range(8)
    ]

    mock_ws = AsyncMock()
    mock_ws.closed = False
    mock_ws.__aiter__.return_value = [json.dumps(trade) for trade in trades]

    with patch("websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__.return_value = mock_ws

        collector = BinanceWSCollector()
        writer = TimescaleWriter(
            database_url=test_database,
            batch_size=10,  # Larger than number of trades
        )

        await collector.start()
        await writer.connect()

        try:
            # Process some trades but not all
            trade_count = 0
            async for trade in collector.produce():
                await writer.write_trade(trade)
                trade_count += 1

                # Stop early to test graceful shutdown
                if trade_count >= 5:
                    break

            # Graceful shutdown should flush pending trades
            await writer.flush()

            # Verify that processed trades were written
            async with writer._pool.acquire() as conn:
                count = await conn.fetchval("SELECT COUNT(*) FROM trades")
                assert count == 5

        finally:
            await collector.stop()
            await writer.disconnect()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
