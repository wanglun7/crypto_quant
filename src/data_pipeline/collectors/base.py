"""
Base collector class for all data sources
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import time
from contextlib import asynccontextmanager
import aiohttp
import websockets
from ..cache import cache_manager
from ..database import db_manager

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base class for data collectors"""
    
    def __init__(self, exchange: str, symbols: List[str]):
        self.exchange = exchange
        self.symbols = symbols
        self.is_running = False
        self.callbacks: Dict[str, List[Callable]] = {}
        self.error_count = 0
        self.max_errors = 10
        self.reconnect_delay = 5
        self.session: Optional[aiohttp.ClientSession] = None
        
    @abstractmethod
    async def connect(self):
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to data source"""
        pass
    
    @abstractmethod
    async def subscribe(self, channels: List[str]):
        """Subscribe to data channels"""
        pass
    
    @abstractmethod
    async def process_message(self, message: Any):
        """Process incoming message"""
        pass
    
    async def start(self):
        """Start data collection"""
        self.is_running = True
        self.error_count = 0
        
        while self.is_running:
            try:
                await self.connect()
                await self.collect()
            except Exception as e:
                self.error_count += 1
                logger.error(f"{self.exchange} collector error ({self.error_count}/{self.max_errors}): {e}")
                
                if self.error_count >= self.max_errors:
                    logger.error(f"{self.exchange} collector exceeded max errors, stopping")
                    break
                
                await asyncio.sleep(self.reconnect_delay * self.error_count)
            finally:
                await self.disconnect()
    
    async def stop(self):
        """Stop data collection"""
        self.is_running = False
        await self.disconnect()
    
    @abstractmethod
    async def collect(self):
        """Main collection loop"""
        pass
    
    def register_callback(self, event: str, callback: Callable):
        """Register callback for event"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    async def emit(self, event: str, data: Any):
        """Emit event to registered callbacks"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Callback error for {event}: {e}")
    
    @asynccontextmanager
    async def get_session(self):
        """Get aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        yield self.session
    
    def normalize_timestamp(self, timestamp: Any) -> int:
        """Normalize timestamp to microseconds"""
        if isinstance(timestamp, str):
            # Parse ISO format
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1_000_000)
        elif isinstance(timestamp, (int, float)):
            # Detect precision and convert
            if timestamp > 10**12:  # Already in microseconds
                return int(timestamp)
            elif timestamp > 10**9:  # Milliseconds
                return int(timestamp * 1000)
            else:  # Seconds
                return int(timestamp * 1_000_000)
        else:
            raise ValueError(f"Unknown timestamp format: {timestamp}")
    
    async def store_tick(self, symbol: str, price: float, volume: float, 
                        side: str, timestamp: int):
        """Store market tick"""
        tick = {
            'exchange': self.exchange,
            'symbol': symbol,
            'price': price,
            'volume': volume,
            'side': side,
            'timestamp': timestamp
        }
        
        # Store in cache
        cache_manager.store_tick(symbol, self.exchange, tick)
        cache_manager.push_tick_buffer(symbol, self.exchange, tick)
        
        # Emit event
        await self.emit('tick', tick)
        
        # Store in database (batch for efficiency)
        # This would be handled by a separate batch writer
        
    async def store_orderbook(self, symbol: str, bids: List[List[float]], 
                             asks: List[List[float]], timestamp: int):
        """Store orderbook snapshot"""
        orderbook = {
            'exchange': self.exchange,
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': timestamp
        }
        
        # Store in cache
        cache_manager.store_orderbook(symbol, self.exchange, orderbook)
        
        # Emit event
        await self.emit('orderbook', orderbook)
    
    async def store_ohlcv(self, symbol: str, timeframe: str, ohlcv: Dict):
        """Store OHLCV candle"""
        candle = {
            'exchange': self.exchange,
            'symbol': symbol,
            'timeframe': timeframe,
            **ohlcv
        }
        
        # Emit event
        await self.emit('ohlcv', candle)


class WebSocketCollector(BaseCollector):
    """Base class for WebSocket-based collectors"""
    
    def __init__(self, exchange: str, symbols: List[str], ws_url: str):
        super().__init__(exchange, symbols)
        self.ws_url = ws_url
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.ping_interval = 30
        self.ping_task = None
        
    async def connect(self):
        """Connect to WebSocket"""
        self.ws = await websockets.connect(
            self.ws_url,
            ping_interval=self.ping_interval,
            ping_timeout=10,
            close_timeout=10
        )
        logger.info(f"Connected to {self.exchange} WebSocket")
        
        # Start ping task
        self.ping_task = asyncio.create_task(self.ping_loop())
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.ping_task:
            self.ping_task.cancel()
            
        if self.ws:
            await self.ws.close()
            self.ws = None
            
        logger.info(f"Disconnected from {self.exchange} WebSocket")
    
    async def ping_loop(self):
        """Keep connection alive"""
        while self.is_running and self.ws:
            try:
                await asyncio.sleep(self.ping_interval)
                if self.ws:
                    await self.ws.ping()
            except Exception as e:
                logger.error(f"Ping error: {e}")
                break
    
    async def send_message(self, message: Dict):
        """Send message to WebSocket"""
        if self.ws:
            import json
            await self.ws.send(json.dumps(message))
    
    async def collect(self):
        """Main WebSocket collection loop"""
        async for message in self.ws:
            if not self.is_running:
                break
                
            try:
                await self.process_message(message)
            except Exception as e:
                logger.error(f"Message processing error: {e}")


class RestCollector(BaseCollector):
    """Base class for REST API-based collectors"""
    
    def __init__(self, exchange: str, symbols: List[str], base_url: str):
        super().__init__(exchange, symbols)
        self.base_url = base_url
        self.rate_limit = 10  # requests per second
        self.last_request_time = 0
        
    async def connect(self):
        """Initialize REST connection"""
        # REST doesn't need persistent connection
        pass
    
    async def disconnect(self):
        """Close REST connection"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def rate_limit_wait(self):
        """Wait for rate limit"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make GET request"""
        await self.rate_limit_wait()
        
        async with self.get_session() as session:
            url = f"{self.base_url}{endpoint}"
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
    
    async def subscribe(self, channels: List[str]):
        """REST doesn't use subscriptions"""
        pass