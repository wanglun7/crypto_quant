"""
Binance futures data collector
"""
import json
import logging
from typing import List, Dict, Any
import asyncio
from decimal import Decimal
from .base import WebSocketCollector
from config.settings import settings

logger = logging.getLogger(__name__)


class BinanceCollector(WebSocketCollector):
    """Collect real-time data from Binance Futures"""
    
    def __init__(self, symbols: List[str]):
        # Convert symbols to Binance format (e.g., BTCUSDT)
        binance_symbols = [s.replace('/', '').upper() for s in symbols]
        
        # WebSocket URL
        base_url = "wss://fstream.binance.com" if not settings.exchange.binance_testnet else "wss://stream.binancefuture.com"
        ws_url = f"{base_url}/stream"
        
        super().__init__("binance", symbols, ws_url)
        self.binance_symbols = binance_symbols
        self.stream_id = 1
        
    async def subscribe(self, channels: List[str]):
        """Subscribe to Binance streams"""
        streams = []
        
        for symbol in self.binance_symbols:
            symbol_lower = symbol.lower()
            
            # Add required streams
            if 'trade' in channels:
                streams.append(f"{symbol_lower}@aggTrade")
            
            if 'orderbook' in channels:
                streams.append(f"{symbol_lower}@depth20@100ms")
            
            if 'ticker' in channels:
                streams.append(f"{symbol_lower}@ticker")
            
            if 'kline' in channels:
                streams.append(f"{symbol_lower}@kline_1m")
            
            if 'funding' in channels:
                streams.append(f"{symbol_lower}@markPrice@1s")
        
        # Subscribe to streams
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": self.stream_id
        }
        
        await self.send_message(subscribe_message)
        self.stream_id += 1
        
        logger.info(f"Subscribed to Binance streams: {streams}")
    
    async def process_message(self, message: Any):
        """Process Binance WebSocket message"""
        try:
            data = json.loads(message)
            
            # Skip subscription confirmations
            if 'result' in data or 'id' in data:
                return
            
            # Process stream data
            if 'stream' in data and 'data' in data:
                stream_name = data['stream']
                stream_data = data['data']
                
                if '@aggTrade' in stream_name:
                    await self.process_trade(stream_data)
                elif '@depth' in stream_name:
                    await self.process_orderbook(stream_data)
                elif '@ticker' in stream_name:
                    await self.process_ticker(stream_data)
                elif '@kline' in stream_name:
                    await self.process_kline(stream_data)
                elif '@markPrice' in stream_name:
                    await self.process_funding(stream_data)
                    
        except Exception as e:
            logger.error(f"Error processing Binance message: {e}")
    
    async def process_trade(self, data: Dict):
        """Process aggregated trade data"""
        symbol = self.normalize_symbol(data['s'])
        
        await self.store_tick(
            symbol=symbol,
            price=float(data['p']),
            volume=float(data['q']),
            side='buy' if data['m'] else 'sell',  # m = true means buyer is maker
            timestamp=self.normalize_timestamp(data['T'])
        )
    
    async def process_orderbook(self, data: Dict):
        """Process orderbook snapshot"""
        symbol = self.normalize_symbol(data['s'])
        
        # Convert to standard format [[price, volume], ...]
        bids = [[float(bid[0]), float(bid[1])] for bid in data['bids']]
        asks = [[float(ask[0]), float(ask[1])] for ask in data['asks']]
        
        await self.store_orderbook(
            symbol=symbol,
            bids=bids[:settings.data.orderbook_levels],
            asks=asks[:settings.data.orderbook_levels],
            timestamp=self.normalize_timestamp(data['T'])
        )
    
    async def process_ticker(self, data: Dict):
        """Process 24hr ticker data"""
        ticker = {
            'symbol': self.normalize_symbol(data['s']),
            'last_price': float(data['c']),
            'bid_price': float(data['b']),
            'ask_price': float(data['a']),
            'volume_24h': float(data['v']),
            'quote_volume_24h': float(data['q']),
            'high_24h': float(data['h']),
            'low_24h': float(data['l']),
            'price_change_24h': float(data['p']),
            'price_change_pct_24h': float(data['P']),
            'timestamp': self.normalize_timestamp(data['E'])
        }
        
        await self.emit('ticker', ticker)
    
    async def process_kline(self, data: Dict):
        """Process kline/candlestick data"""
        kline = data['k']
        
        ohlcv = {
            'timestamp': self.normalize_timestamp(kline['t']),
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v']),
            'trades': int(kline['n'])
        }
        
        await self.store_ohlcv(
            symbol=self.normalize_symbol(kline['s']),
            timeframe=kline['i'],
            ohlcv=ohlcv
        )
    
    async def process_funding(self, data: Dict):
        """Process funding rate data"""
        funding = {
            'symbol': self.normalize_symbol(data['s']),
            'mark_price': float(data['p']),
            'index_price': float(data['i']),
            'funding_rate': float(data['r']),
            'next_funding_time': self.normalize_timestamp(data['T']),
            'timestamp': self.normalize_timestamp(data['E'])
        }
        
        await self.emit('funding', funding)
    
    def normalize_symbol(self, binance_symbol: str) -> str:
        """Convert Binance symbol to standard format"""
        # BTCUSDT -> BTC/USDT
        if binance_symbol.endswith('USDT'):
            base = binance_symbol[:-4]
            return f"{base}/USDT"
        elif binance_symbol.endswith('BUSD'):
            base = binance_symbol[:-4]
            return f"{base}/BUSD"
        else:
            return binance_symbol
    
    async def collect(self):
        """Override to add initial subscriptions"""
        # Subscribe to all channels
        await self.subscribe(['trade', 'orderbook', 'ticker', 'kline', 'funding'])
        
        # Call parent collect method
        await super().collect()


class BinanceRestCollector:
    """Collect historical and reference data from Binance REST API"""
    
    def __init__(self):
        self.base_url = "https://fapi.binance.com" if not settings.exchange.binance_testnet else "https://testnet.binancefuture.com"
        self.session = None
        
    async def get_historical_klines(self, symbol: str, interval: str, 
                                   start_time: int, end_time: int) -> List[List]:
        """Get historical kline data"""
        endpoint = "/fapi/v1/klines"
        params = {
            'symbol': symbol.replace('/', '').upper(),
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1500  # Max limit
        }
        
        # Implementation would fetch data
        # This is a placeholder
        return []
    
    async def get_exchange_info(self) -> Dict:
        """Get exchange trading rules and symbols"""
        endpoint = "/fapi/v1/exchangeInfo"
        # Implementation would fetch data
        return {}
    
    async def get_open_interest(self, symbol: str) -> Dict:
        """Get open interest for symbol"""
        endpoint = "/fapi/v1/openInterest"
        params = {'symbol': symbol.replace('/', '').upper()}
        # Implementation would fetch data
        return {}
    
    async def get_funding_rate_history(self, symbol: str, start_time: int, 
                                      end_time: int) -> List[Dict]:
        """Get historical funding rates"""
        endpoint = "/fapi/v1/fundingRate"
        params = {
            'symbol': symbol.replace('/', '').upper(),
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        # Implementation would fetch data
        return []