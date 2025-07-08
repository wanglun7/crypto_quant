#!/usr/bin/env python3
"""Demo script for BinanceWSCollector + TimescaleWriter integration.

This script demonstrates real-time data collection from Binance WebSocket
and writing to TimescaleDB with batching.

Usage:
    python demo_pipeline.py
    python demo_pipeline.py --database-url postgresql://user:pass@host:5432/crypto_quant
    python demo_pipeline.py --duration 60 --batch-size 500
"""

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timedelta

import structlog

from crypto_quant.data_pipeline.collectors.binance_ws import BinanceWSCollector
from crypto_quant.data_pipeline.storage.writer import TimescaleWriter

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class PipelineStats:
    """Track pipeline statistics."""

    def __init__(self):
        self.trades_received = 0
        self.trades_written = 0
        self.start_time = datetime.now()
        self.last_report = datetime.now()

    def record_trade_received(self):
        """Record a trade received from WebSocket."""
        self.trades_received += 1

    def record_trade_written(self):
        """Record a trade written to database."""
        self.trades_written += 1

    def should_report(self, interval_seconds: int = 10) -> bool:
        """Check if it's time to report stats."""
        return (datetime.now() - self.last_report).total_seconds() >= interval_seconds

    def report(self):
        """Report current statistics."""
        now = datetime.now()
        elapsed = (now - self.start_time).total_seconds()

        logger.info(
            "Pipeline statistics",
            trades_received=self.trades_received,
            trades_written=self.trades_written,
            pending_trades=self.trades_received - self.trades_written,
            elapsed_seconds=elapsed,
            trades_per_second=self.trades_received / elapsed if elapsed > 0 else 0,
        )

        self.last_report = now


async def run_pipeline(
    database_url: str,
    batch_size: int,
    duration: int | None = None,
) -> None:
    """Run the data collection pipeline.
    
    Args:
        database_url: TimescaleDB connection URL
        batch_size: Number of trades to batch before writing
        duration: Optional duration in seconds to run the pipeline
    """
    stats = PipelineStats()

    # Initialize collector and writer
    collector = BinanceWSCollector()
    writer = TimescaleWriter(
        database_url=database_url,
        batch_size=batch_size,
    )

    logger.info(
        "Starting pipeline",
        database_url=database_url,
        batch_size=batch_size,
        duration=duration,
    )

    try:
        # Start collector and connect to database
        await collector.start()
        await writer.connect()

        # Set up termination time if duration is specified
        end_time = None
        if duration:
            end_time = datetime.now() + timedelta(seconds=duration)

        # Process trades
        async for trade in collector.produce():
            stats.record_trade_received()

            # Write trade to database
            await writer.write_trade(trade)
            stats.record_trade_written()

            # Report stats periodically
            if stats.should_report():
                stats.report()

            # Check if we should stop
            if end_time and datetime.now() >= end_time:
                logger.info("Duration reached, stopping pipeline")
                break

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping pipeline")
    except Exception as e:
        logger.error("Pipeline error", error=str(e))
        raise
    finally:
        # Clean shutdown
        logger.info("Shutting down pipeline")
        await collector.stop()
        await writer.flush()  # Ensure all pending writes complete
        await writer.disconnect()

        # Final stats report
        stats.report()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demo pipeline for BinanceWSCollector + TimescaleWriter"
    )

    parser.add_argument(
        "--database-url",
        default="postgresql://localhost:5432/crypto_quant",
        help="TimescaleDB connection URL (default: postgresql://localhost:5432/crypto_quant)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of trades to batch before writing (default: 1000)"
    )

    parser.add_argument(
        "--duration",
        type=int,
        help="Duration in seconds to run the pipeline (default: run indefinitely)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)"
    )

    return parser.parse_args()


def setup_signal_handlers():
    """Set up signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info("Received signal", signal=signum)
        # Let the KeyboardInterrupt propagate
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    args = parse_arguments()

    # Set up signal handlers
    setup_signal_handlers()

    # Configure logging level
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    try:
        await run_pipeline(
            database_url=args.database_url,
            batch_size=args.batch_size,
            duration=args.duration,
        )
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    except Exception as e:
        logger.error("Pipeline failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
