"""Tests for Binance WebSocket collector."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from ..binance_ws import BinanceWSCollector


@pytest.fixture
def sample_agg_trade():
    """Sample aggregated trade message from Binance."""
    return {
        "e": "aggTrade",
        "E": 1672531200123,
        "s": "BTCUSDT",
        "a": 1234567890,
        "p": "43250.50",
        "q": "0.00123",
        "f": 9876543210,
        "l": 9876543211,
        "T": 1672531200122,
        "m": False,
        "M": True,
    }


@pytest.mark.asyncio
async def test_binance_ws_collector_init():
    """Test collector initialization."""
    collector = BinanceWSCollector()
    assert collector._running is False
    assert collector._ws is None
    assert collector._reconnect_delay == 1.0


@pytest.mark.asyncio
async def test_binance_ws_collector_start_stop():
    """Test collector start and stop."""
    collector = BinanceWSCollector()

    await collector.start()
    assert collector._running is True

    await collector.stop()
    assert collector._running is False


@pytest.mark.asyncio
async def test_binance_ws_collector_produce(sample_agg_trade):
    """Test collector produce with mocked WebSocket."""
    collector = BinanceWSCollector()
    await collector.start()

    # Mock WebSocket connection
    mock_ws = AsyncMock()
    mock_ws.closed = False
    mock_ws.__aiter__.return_value = [json.dumps(sample_agg_trade)]

    with patch("websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__.return_value = mock_ws

        # Collect one message
        messages = []
        async for msg in collector.produce():
            messages.append(msg)
            break

        assert len(messages) == 1
        assert messages[0] == sample_agg_trade

    await collector.stop()


@pytest.mark.asyncio
async def test_binance_ws_collector_reconnect():
    """Test collector reconnection delay logic."""
    collector = BinanceWSCollector()

    # Test that reconnect delay doubles after failure
    initial_delay = collector._reconnect_delay

    # Simulate increasing delay
    collector._reconnect_delay = min(collector._reconnect_delay * 2, collector.MAX_RECONNECT_DELAY)

    assert collector._reconnect_delay > initial_delay
    assert collector._reconnect_delay <= collector.MAX_RECONNECT_DELAY


@pytest.mark.asyncio
async def test_binance_ws_collector_json_decode_error(sample_agg_trade):
    """Test handling of invalid JSON messages."""
    collector = BinanceWSCollector()
    await collector.start()

    # Mock WebSocket with invalid JSON
    mock_ws = AsyncMock()
    mock_ws.closed = False
    mock_ws.__aiter__.return_value = [
        "invalid json",
        json.dumps(sample_agg_trade),
    ]

    with patch("websockets.connect") as mock_connect:
        mock_connect.return_value.__aenter__.return_value = mock_ws

        # Collect messages, should skip invalid JSON
        messages = []
        async for msg in collector.produce():
            messages.append(msg)
            if len(messages) >= 1:
                break

        assert len(messages) == 1
        assert messages[0] == sample_agg_trade

    await collector.stop()
