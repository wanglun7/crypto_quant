"""Binance WebSocket collector for aggregated trades."""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
import websockets
from websockets.exceptions import WebSocketException

from .base import AsyncProducer

logger = structlog.get_logger(__name__)


class BinanceWSCollector(AsyncProducer):
    """Binance WebSocket collector for aggregated trades stream."""

    WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
    PING_INTERVAL = 30  # seconds
    MAX_RECONNECT_DELAY = 60  # seconds

    def __init__(self) -> None:
        """Initialize the collector."""
        self._running = False
        self._ws: Any = None
        self._reconnect_delay = 1.0

    async def start(self) -> None:
        """Start the collector."""
        self._running = True
        logger.info("Starting BinanceWSCollector")

    async def stop(self) -> None:
        """Stop the collector gracefully."""
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info("Stopped BinanceWSCollector")

    async def produce(self) -> AsyncGenerator[dict[str, Any], None]:
        """Yield aggregated trade data from Binance WebSocket."""
        while self._running:
            try:
                async with websockets.connect(self.WS_URL) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1.0  # Reset delay on successful connection
                    logger.info("Connected to Binance WebSocket", url=self.WS_URL)

                    # Create ping task
                    ping_task = asyncio.create_task(self._ping_loop())

                    try:
                        async for message in ws:
                            if not self._running:
                                break  # type: ignore[unreachable]

                            try:
                                data = json.loads(message)
                                yield data
                            except json.JSONDecodeError as e:
                                logger.error("Failed to parse message", error=str(e), message=message)
                    finally:
                        ping_task.cancel()
                        try:
                            await ping_task
                        except asyncio.CancelledError:
                            pass

            except (WebSocketException, ConnectionError, OSError) as e:
                if not self._running:
                    break  # type: ignore[unreachable]

                logger.warning(
                    "WebSocket connection error, will reconnect",
                    error=str(e),
                    reconnect_delay=self._reconnect_delay,
                )

                await asyncio.sleep(self._reconnect_delay)
                # Exponential backoff with max delay
                self._reconnect_delay = min(self._reconnect_delay * 2, self.MAX_RECONNECT_DELAY)

            except Exception as e:
                logger.exception("Unexpected error in produce loop", error=str(e))
                if self._running:
                    await asyncio.sleep(self._reconnect_delay)

    async def _ping_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        try:
            while self._running and self._ws:
                await asyncio.sleep(self.PING_INTERVAL)
                if self._ws and not self._ws.closed:
                    await self._ws.ping()
                    logger.debug("Sent ping to Binance WebSocket")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Error in ping loop", error=str(e))
