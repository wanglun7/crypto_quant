"""
Main application entry point
"""
import asyncio
import logging
import signal
import sys
from typing import Dict, List
import click
from loguru import logger

from config.settings import settings
from src.data_pipeline.database import db_manager
from src.data_pipeline.cache import cache_manager
from src.data_pipeline.collectors.binance import BinanceCollector


class CryptoQuantApp:
    """Main application class"""
    
    def __init__(self):
        self.collectors: Dict[str, List] = {
            'binance': [],
            'okx': [],
            'bybit': []
        }
        self.is_running = False
        self.tasks: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Crypto Quant System...")
        
        # Initialize database
        db_manager.initialize()
        await db_manager.initialize_async()
        
        # Initialize cache
        cache_manager.initialize()
        await cache_manager.initialize_async()
        
        # Initialize collectors
        await self.initialize_collectors()
        
        logger.success("System initialized successfully")
    
    async def initialize_collectors(self):
        """Initialize data collectors"""
        # For now, just Binance with BTC/USDT
        symbols = ['BTC/USDT']
        
        # Create Binance collector
        binance_collector = BinanceCollector(symbols)
        self.collectors['binance'].append(binance_collector)
        
        # Register callbacks
        binance_collector.register_callback('tick', self.on_tick)
        binance_collector.register_callback('orderbook', self.on_orderbook)
        
        logger.info(f"Initialized collectors for symbols: {symbols}")
    
    async def on_tick(self, tick: Dict):
        """Handle new tick data"""
        logger.debug(f"New tick: {tick['symbol']} @ {tick['price']}")
    
    async def on_orderbook(self, orderbook: Dict):
        """Handle new orderbook data"""
        bid = orderbook['bids'][0][0] if orderbook['bids'] else 0
        ask = orderbook['asks'][0][0] if orderbook['asks'] else 0
        spread = ask - bid if bid and ask else 0
        logger.debug(f"Orderbook update: {orderbook['symbol']} spread={spread:.2f}")
    
    async def start(self):
        """Start the application"""
        self.is_running = True
        logger.info("Starting data collection...")
        
        # Start all collectors
        for exchange, collectors in self.collectors.items():
            for collector in collectors:
                task = asyncio.create_task(collector.start())
                self.tasks.append(task)
        
        # Wait for all tasks
        await asyncio.gather(*self.tasks, return_exceptions=True)
    
    async def stop(self):
        """Stop the application"""
        logger.info("Stopping application...")
        self.is_running = False
        
        # Stop all collectors
        for exchange, collectors in self.collectors.items():
            for collector in collectors:
                await collector.stop()
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Close connections
        db_manager.close()
        cache_manager.close()
        
        logger.success("Application stopped")
    
    def signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {sig}")
        asyncio.create_task(self.stop())


@click.group()
def cli():
    """Crypto Quant Trading System"""
    pass


@cli.command()
@click.option('--debug', is_flag=True, help='Enable debug mode')
def collect(debug):
    """Start data collection"""
    # Configure logging
    logger.remove()
    log_level = "DEBUG" if debug else settings.monitoring.log_level
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Create and run app
    app = CryptoQuantApp()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, app.signal_handler)
    signal.signal(signal.SIGTERM, app.signal_handler)
    
    # Run event loop
    try:
        asyncio.run(run_app(app))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


async def run_app(app: CryptoQuantApp):
    """Run the application"""
    await app.initialize()
    await app.start()


@cli.command()
def init_db():
    """Initialize database"""
    logger.info("Initializing database...")
    db_manager.initialize()
    logger.success("Database initialized successfully")


@cli.command()
@click.option('--symbol', default='BTC/USDT', help='Symbol to backtest')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
def backtest(symbol, start_date, end_date):
    """Run backtest"""
    logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
    # TODO: Implement backtest command
    

@cli.command()
def train():
    """Train models"""
    logger.info("Training models...")
    # TODO: Implement training command


if __name__ == '__main__':
    cli()