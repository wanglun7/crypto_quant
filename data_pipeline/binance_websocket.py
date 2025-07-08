"""
Binance WebSocket实时数据流接收器
实现1ms级tick数据、20档订单簿、成交明细的实时采集
"""

import asyncio
import websockets
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timezone
import pandas as pd
from decimal import Decimal
import time
import aiohttp
from dataclasses import dataclass
from enum import Enum
import threading
import queue

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataType(Enum):
    """数据类型枚举"""
    TICKER = "ticker"
    DEPTH = "depth"
    TRADE = "trade"
    KLINE = "kline"
    AGG_TRADE = "aggTrade"


@dataclass
class TickerData:
    """Ticker数据结构"""
    symbol: str
    price: Decimal
    price_change: Decimal
    price_change_percent: Decimal
    volume: Decimal
    quote_volume: Decimal
    high: Decimal
    low: Decimal
    open: Decimal
    timestamp: int
    
    @classmethod
    def from_websocket(cls, data: Dict) -> 'TickerData':
        """从WebSocket数据创建TickerData对象"""
        return cls(
            symbol=data.get('s', ''),
            price=Decimal(data.get('c', '0')),
            price_change=Decimal(data.get('P', '0')),
            price_change_percent=Decimal(data.get('p', '0')),
            volume=Decimal(data.get('v', '0')),
            quote_volume=Decimal(data.get('q', '0')),
            high=Decimal(data.get('h', '0')),
            low=Decimal(data.get('l', '0')),
            open=Decimal(data.get('o', '0')),
            timestamp=int(data.get('E', 0))
        )


@dataclass
class DepthData:
    """订单簿深度数据结构"""
    symbol: str
    bids: List[tuple]  # [(price, quantity), ...]
    asks: List[tuple]  # [(price, quantity), ...]
    timestamp: int
    last_update_id: int
    
    @classmethod
    def from_websocket(cls, data: Dict) -> 'DepthData':
        """从WebSocket数据创建DepthData对象"""
        return cls(
            symbol=data.get('s', ''),
            bids=[(Decimal(bid[0]), Decimal(bid[1])) for bid in data.get('b', [])],
            asks=[(Decimal(ask[0]), Decimal(ask[1])) for ask in data.get('a', [])],
            timestamp=int(data.get('E', 0)),
            last_update_id=int(data.get('u', 0))
        )


@dataclass
class TradeData:
    """成交数据结构"""
    symbol: str
    price: Decimal
    quantity: Decimal
    timestamp: int
    trade_id: int
    buyer_maker: bool
    
    @classmethod
    def from_websocket(cls, data: Dict) -> 'TradeData':
        """从WebSocket数据创建TradeData对象"""
        return cls(
            symbol=data.get('s', ''),
            price=Decimal(data.get('p', '0')),
            quantity=Decimal(data.get('q', '0')),
            timestamp=int(data.get('T', 0)),
            trade_id=int(data.get('t', 0)),
            buyer_maker=bool(data.get('m', False))
        )


class BinanceWebSocketManager:
    """Binance WebSocket管理器"""
    
    def __init__(self, symbols: List[str] = None):
        """
        初始化WebSocket管理器
        
        Args:
            symbols: 订阅的交易对列表，默认为['btcusdt']
        """
        self.symbols = symbols or ['btcusdt']
        self.base_url = "wss://fstream.binance.com/ws/"
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.callbacks: Dict[DataType, List[Callable]] = {
            DataType.TICKER: [],
            DataType.DEPTH: [],
            DataType.TRADE: [],
            DataType.KLINE: [],
            DataType.AGG_TRADE: []
        }
        
        # 数据缓存
        self.latest_ticker: Dict[str, TickerData] = {}
        self.latest_depth: Dict[str, DepthData] = {}
        self.trade_buffer: queue.Queue = queue.Queue(maxsize=10000)
        
        # 统计信息
        self.message_count = 0
        self.start_time = time.time()
        self.last_heartbeat = time.time()
        
        # 控制标志
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        logger.info(f"Binance WebSocket Manager initialized for symbols: {self.symbols}")
    
    def register_callback(self, data_type: DataType, callback: Callable):
        """注册数据回调函数"""
        self.callbacks[data_type].append(callback)
        logger.info(f"Registered callback for {data_type.value}")
    
    def remove_callback(self, data_type: DataType, callback: Callable):
        """移除数据回调函数"""
        if callback in self.callbacks[data_type]:
            self.callbacks[data_type].remove(callback)
            logger.info(f"Removed callback for {data_type.value}")
    
    async def _create_stream_url(self, symbol: str) -> str:
        """创建数据流URL"""
        # 订阅多个数据流
        streams = [
            f"{symbol.lower()}@ticker",        # 24小时ticker
            f"{symbol.lower()}@depth20@100ms", # 20档深度数据，100ms更新
            f"{symbol.lower()}@trade",         # 实时成交
            f"{symbol.lower()}@aggTrade",      # 聚合成交
            f"{symbol.lower()}@kline_1m"       # 1分钟K线
        ]
        
        # 使用combined stream
        stream_names = "/".join(streams)
        return f"{self.base_url}{stream_names}"
    
    async def _handle_message(self, message: str):
        """处理WebSocket消息"""
        try:
            data = json.loads(message)
            self.message_count += 1
            self.last_heartbeat = time.time()
            
            # 处理不同类型的数据
            if 'stream' in data:
                stream_name = data['stream']
                stream_data = data['data']
                
                if '@ticker' in stream_name:
                    await self._handle_ticker_data(stream_data)
                elif '@depth' in stream_name:
                    await self._handle_depth_data(stream_data)
                elif '@trade' in stream_name:
                    await self._handle_trade_data(stream_data)
                elif '@aggTrade' in stream_name:
                    await self._handle_agg_trade_data(stream_data)
                elif '@kline' in stream_name:
                    await self._handle_kline_data(stream_data)
            
            # 每1000条消息记录一次统计
            if self.message_count % 1000 == 0:
                elapsed = time.time() - self.start_time
                rate = self.message_count / elapsed
                logger.info(f"Processed {self.message_count} messages, rate: {rate:.2f} msg/s")
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def _handle_ticker_data(self, data: Dict):
        """处理ticker数据"""
        try:
            ticker = TickerData.from_websocket(data)
            self.latest_ticker[ticker.symbol] = ticker
            
            # 触发回调
            for callback in self.callbacks[DataType.TICKER]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(ticker)
                    else:
                        callback(ticker)
                except Exception as e:
                    logger.error(f"Error in ticker callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing ticker data: {e}")
    
    async def _handle_depth_data(self, data: Dict):
        """处理深度数据"""
        try:
            depth = DepthData.from_websocket(data)
            self.latest_depth[depth.symbol] = depth
            
            # 触发回调
            for callback in self.callbacks[DataType.DEPTH]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(depth)
                    else:
                        callback(depth)
                except Exception as e:
                    logger.error(f"Error in depth callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing depth data: {e}")
    
    async def _handle_trade_data(self, data: Dict):
        """处理成交数据"""
        try:
            trade = TradeData.from_websocket(data)
            
            # 添加到缓冲区
            if not self.trade_buffer.full():
                self.trade_buffer.put(trade)
            
            # 触发回调
            for callback in self.callbacks[DataType.TRADE]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(trade)
                    else:
                        callback(trade)
                except Exception as e:
                    logger.error(f"Error in trade callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
    
    async def _handle_agg_trade_data(self, data: Dict):
        """处理聚合成交数据"""
        try:
            # 聚合成交数据处理逻辑
            for callback in self.callbacks[DataType.AGG_TRADE]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.error(f"Error in agg trade callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing agg trade data: {e}")
    
    async def _handle_kline_data(self, data: Dict):
        """处理K线数据"""
        try:
            kline_data = data.get('k', {})
            
            # 触发回调
            for callback in self.callbacks[DataType.KLINE]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(kline_data)
                    else:
                        callback(kline_data)
                except Exception as e:
                    logger.error(f"Error in kline callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing kline data: {e}")
    
    async def _connect_symbol(self, symbol: str):
        """连接单个交易对的WebSocket"""
        url = await self._create_stream_url(symbol)
        
        try:
            logger.info(f"Connecting to WebSocket for {symbol}: {url}")
            
            async with websockets.connect(
                url,
                ping_interval=20,  # 20秒ping间隔
                ping_timeout=10,   # 10秒ping超时
                close_timeout=5    # 5秒关闭超时
            ) as websocket:
                
                self.connections[symbol] = websocket
                logger.info(f"Successfully connected to {symbol} WebSocket")
                
                # 消息处理循环
                async for message in websocket:
                    if not self.running:
                        break
                    await self._handle_message(message)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"WebSocket connection closed for {symbol}")
        except Exception as e:
            logger.error(f"WebSocket connection error for {symbol}: {e}")
        finally:
            if symbol in self.connections:
                del self.connections[symbol]
    
    async def _monitor_connections(self):
        """监控连接状态"""
        while self.running:
            await asyncio.sleep(30)  # 每30秒检查一次
            
            current_time = time.time()
            if current_time - self.last_heartbeat > 60:  # 60秒没有数据
                logger.warning("No data received in 60 seconds, checking connections...")
                
                for symbol in self.symbols:
                    if symbol not in self.connections:
                        logger.info(f"Reconnecting to {symbol}...")
                        asyncio.create_task(self._connect_symbol(symbol))
            
            # 输出统计信息
            if self.message_count > 0:
                elapsed = current_time - self.start_time
                rate = self.message_count / elapsed
                logger.info(f"Connection status - Messages: {self.message_count}, "
                          f"Rate: {rate:.2f} msg/s, "
                          f"Active connections: {len(self.connections)}")
    
    async def start(self):
        """启动WebSocket连接"""
        self.running = True
        self.start_time = time.time()
        
        logger.info("Starting Binance WebSocket connections...")
        
        # 为每个交易对创建连接任务
        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(self._connect_symbol(symbol))
            tasks.append(task)
        
        # 添加连接监控任务
        monitor_task = asyncio.create_task(self._monitor_connections())
        tasks.append(monitor_task)
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            await self.stop()
        except Exception as e:
            logger.error(f"Error in WebSocket manager: {e}")
    
    async def stop(self):
        """停止WebSocket连接"""
        self.running = False
        
        logger.info("Stopping Binance WebSocket connections...")
        
        # 关闭所有连接
        for symbol, websocket in self.connections.items():
            try:
                await websocket.close()
                logger.info(f"Closed WebSocket for {symbol}")
            except Exception as e:
                logger.error(f"Error closing WebSocket for {symbol}: {e}")
        
        self.connections.clear()
        logger.info("All WebSocket connections closed")
    
    def get_latest_ticker(self, symbol: str) -> Optional[TickerData]:
        """获取最新的ticker数据"""
        return self.latest_ticker.get(symbol.upper())
    
    def get_latest_depth(self, symbol: str) -> Optional[DepthData]:
        """获取最新的深度数据"""
        return self.latest_depth.get(symbol.upper())
    
    def get_trade_buffer(self) -> queue.Queue:
        """获取成交数据缓冲区"""
        return self.trade_buffer
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        return {
            'message_count': self.message_count,
            'elapsed_time': elapsed,
            'message_rate': self.message_count / elapsed if elapsed > 0 else 0,
            'active_connections': len(self.connections),
            'symbols': self.symbols,
            'last_heartbeat': self.last_heartbeat,
            'trade_buffer_size': self.trade_buffer.qsize()
        }


# 便捷的异步上下文管理器
class BinanceWebSocketContext:
    """Binance WebSocket异步上下文管理器"""
    
    def __init__(self, symbols: List[str] = None):
        self.manager = BinanceWebSocketManager(symbols)
    
    async def __aenter__(self):
        return self.manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.manager.stop()


# 示例使用函数
async def example_usage():
    """示例使用方法"""
    
    # 定义数据处理回调
    async def on_ticker(ticker: TickerData):
        print(f"Ticker: {ticker.symbol} - Price: {ticker.price}")
    
    async def on_depth(depth: DepthData):
        if depth.bids and depth.asks:
            best_bid = depth.bids[0][0]
            best_ask = depth.asks[0][0]
            spread = best_ask - best_bid
            print(f"Depth: {depth.symbol} - Spread: {spread}")
    
    async def on_trade(trade: TradeData):
        print(f"Trade: {trade.symbol} - Price: {trade.price}, Qty: {trade.quantity}")
    
    # 使用上下文管理器
    async with BinanceWebSocketContext(['btcusdt']) as ws_manager:
        # 注册回调
        ws_manager.register_callback(DataType.TICKER, on_ticker)
        ws_manager.register_callback(DataType.DEPTH, on_depth)
        ws_manager.register_callback(DataType.TRADE, on_trade)
        
        # 启动连接
        await ws_manager.start()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())