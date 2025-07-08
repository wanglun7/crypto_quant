"""
Batch data writer for efficient database writes
"""
import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
import asyncpg
from collections import defaultdict
import time
from config.settings import settings
from .database import db_manager
from .models import MarketTick, OrderBookSnapshot, OHLCV, FundingRate

logger = logging.getLogger(__name__)


class BatchWriter:
    """Efficiently write data to database in batches"""
    
    def __init__(self, batch_size: int = 1000, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.buffers: Dict[str, List] = defaultdict(list)
        self.last_flush: Dict[str, float] = defaultdict(float)
        self.is_running = False
        self.flush_task = None
        self.stats = {
            'total_writes': 0,
            'total_errors': 0,
            'last_write_time': 0
        }
        
    async def start(self):
        """Start the batch writer"""
        self.is_running = True
        self.flush_task = asyncio.create_task(self.flush_loop())
        logger.info("Batch writer started")
        
    async def stop(self):
        """Stop the batch writer"""
        self.is_running = False
        
        # Final flush
        await self.flush_all()
        
        if self.flush_task:
            self.flush_task.cancel()
            
        logger.info(f"Batch writer stopped. Stats: {self.stats}")
        
    async def flush_loop(self):
        """Periodically flush buffers"""
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush_all()
            except Exception as e:
                logger.error(f"Flush loop error: {e}")
                
    async def add_tick(self, tick: Dict):
        """Add market tick to buffer"""
        self.buffers['ticks'].append(tick)
        await self.check_flush('ticks')
        
    async def add_orderbook(self, orderbook: Dict):
        """Add orderbook snapshot to buffer"""
        self.buffers['orderbooks'].append(orderbook)
        await self.check_flush('orderbooks')
        
    async def add_ohlcv(self, ohlcv: Dict):
        """Add OHLCV candle to buffer"""
        self.buffers['ohlcv'].append(ohlcv)
        await self.check_flush('ohlcv')
        
    async def add_funding(self, funding: Dict):
        """Add funding rate to buffer"""
        self.buffers['funding'].append(funding)
        await self.check_flush('funding')
        
    async def check_flush(self, buffer_name: str):
        """Check if buffer needs flushing"""
        if len(self.buffers[buffer_name]) >= self.batch_size:
            await self.flush_buffer(buffer_name)
            
    async def flush_all(self):
        """Flush all buffers"""
        tasks = []
        for buffer_name in list(self.buffers.keys()):
            if self.buffers[buffer_name]:
                tasks.append(self.flush_buffer(buffer_name))
                
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def flush_buffer(self, buffer_name: str):
        """Flush specific buffer to database"""
        if not self.buffers[buffer_name]:
            return
            
        # Get data and clear buffer
        data = self.buffers[buffer_name]
        self.buffers[buffer_name] = []
        
        try:
            async for conn in db_manager.get_async_connection():
                if buffer_name == 'ticks':
                    await self.write_ticks(conn, data)
                elif buffer_name == 'orderbooks':
                    await self.write_orderbooks(conn, data)
                elif buffer_name == 'ohlcv':
                    await self.write_ohlcv(conn, data)
                elif buffer_name == 'funding':
                    await self.write_funding(conn, data)
                    
            self.stats['total_writes'] += len(data)
            self.stats['last_write_time'] = time.time()
            
            logger.debug(f"Flushed {len(data)} {buffer_name} to database")
            
        except Exception as e:
            logger.error(f"Error flushing {buffer_name}: {e}")
            self.stats['total_errors'] += 1
            
            # Put data back in buffer for retry
            self.buffers[buffer_name].extend(data)
            
    async def write_ticks(self, conn: asyncpg.Connection, ticks: List[Dict]):
        """Write market ticks to database"""
        # Prepare data for COPY
        records = []
        for tick in ticks:
            records.append((
                tick['exchange'],
                tick['symbol'],
                tick['timestamp'],
                tick['price'],
                tick['volume'],
                tick['side']
            ))
            
        # Use COPY for fast insertion
        await conn.copy_records_to_table(
            'market_ticks',
            records=records,
            columns=['exchange', 'symbol', 'timestamp', 'price', 'volume', 'side']
        )
        
    async def write_orderbooks(self, conn: asyncpg.Connection, orderbooks: List[Dict]):
        """Write orderbook snapshots to database"""
        # Prepare data
        records = []
        for ob in orderbooks:
            records.append((
                ob['exchange'],
                ob['symbol'],
                ob['timestamp'],
                ob['bids'],  # JSON
                ob['asks']   # JSON
            ))
            
        # Use prepared statement for JSON columns
        stmt = await conn.prepare("""
            INSERT INTO orderbook_snapshots 
            (exchange, symbol, timestamp, bids, asks)
            VALUES ($1, $2, $3, $4::json, $5::json)
        """)
        
        await stmt.executemany(records)
        
    async def write_ohlcv(self, conn: asyncpg.Connection, candles: List[Dict]):
        """Write OHLCV data to database"""
        records = []
        for candle in candles:
            records.append((
                candle['exchange'],
                candle['symbol'],
                candle['timeframe'],
                candle['timestamp'],
                candle['open'],
                candle['high'],
                candle['low'],
                candle['close'],
                candle['volume'],
                candle.get('trades')
            ))
            
        await conn.copy_records_to_table(
            'ohlcv',
            records=records,
            columns=['exchange', 'symbol', 'timeframe', 'timestamp', 
                    'open', 'high', 'low', 'close', 'volume', 'trades']
        )
        
    async def write_funding(self, conn: asyncpg.Connection, funding_rates: List[Dict]):
        """Write funding rates to database"""
        records = []
        for funding in funding_rates:
            records.append((
                funding['exchange'],
                funding['symbol'],
                funding['timestamp'],
                funding['funding_rate'],
                funding['next_funding_time']
            ))
            
        await conn.copy_records_to_table(
            'funding_rates',
            records=records,
            columns=['exchange', 'symbol', 'timestamp', 
                    'funding_rate', 'funding_time']
        )


# Global batch writer instance
batch_writer = BatchWriter()