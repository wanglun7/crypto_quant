"""
Redis cache manager for real-time data
"""
import redis
import aioredis
import json
import pickle
import logging
from typing import Any, Optional, List, Dict
from datetime import timedelta
import asyncio
from config.settings import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Manage Redis cache for real-time data"""
    
    def __init__(self):
        self.redis_client = None
        self.async_redis = None
        self.pubsub = None
        
    def initialize(self):
        """Initialize Redis connections"""
        # Synchronous client
        self.redis_client = redis.Redis.from_url(
            settings.database.redis_url,
            decode_responses=False,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 1,  # TCP_KEEPINTVL
                3: 5,  # TCP_KEEPCNT
            }
        )
        
        # Test connection
        self.redis_client.ping()
        logger.info("Redis connection established")
        
    async def initialize_async(self):
        """Initialize async Redis connection"""
        self.async_redis = await aioredis.from_url(
            settings.database.redis_url,
            decode_responses=False
        )
        await self.async_redis.ping()
        logger.info("Async Redis connection established")
    
    # Synchronous methods
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            serialized = pickle.dumps(value)
            if ttl:
                return self.redis_client.setex(key, ttl, serialized)
            return self.redis_client.set(key, serialized)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis_client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def set_json(self, key: str, value: Dict, ttl: Optional[int] = None) -> bool:
        """Set JSON value in cache"""
        try:
            json_str = json.dumps(value)
            if ttl:
                return self.redis_client.setex(key, ttl, json_str)
            return self.redis_client.set(key, json_str)
        except Exception as e:
            logger.error(f"Cache set_json error: {e}")
            return False
    
    def get_json(self, key: str) -> Optional[Dict]:
        """Get JSON value from cache"""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value.decode('utf-8'))
            return None
        except Exception as e:
            logger.error(f"Cache get_json error: {e}")
            return None
    
    # Async methods
    async def async_set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Async set value in cache"""
        try:
            serialized = pickle.dumps(value)
            if ttl:
                return await self.async_redis.setex(key, ttl, serialized)
            return await self.async_redis.set(key, serialized)
        except Exception as e:
            logger.error(f"Async cache set error: {e}")
            return False
    
    async def async_get(self, key: str) -> Optional[Any]:
        """Async get value from cache"""
        try:
            value = await self.async_redis.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Async cache get error: {e}")
            return None
    
    # Real-time data methods
    def store_orderbook(self, symbol: str, exchange: str, orderbook: Dict) -> bool:
        """Store orderbook snapshot"""
        key = f"orderbook:{exchange}:{symbol}"
        return self.set_json(key, orderbook, ttl=60)  # 1 minute TTL
    
    def get_orderbook(self, symbol: str, exchange: str) -> Optional[Dict]:
        """Get latest orderbook snapshot"""
        key = f"orderbook:{exchange}:{symbol}"
        return self.get_json(key)
    
    def store_tick(self, symbol: str, exchange: str, tick: Dict) -> bool:
        """Store market tick"""
        key = f"tick:{exchange}:{symbol}"
        return self.set_json(key, tick, ttl=60)
    
    def get_tick(self, symbol: str, exchange: str) -> Optional[Dict]:
        """Get latest market tick"""
        key = f"tick:{exchange}:{symbol}"
        return self.get_json(key)
    
    # Pub/Sub methods
    def publish(self, channel: str, message: Dict) -> int:
        """Publish message to channel"""
        try:
            json_message = json.dumps(message)
            return self.redis_client.publish(channel, json_message)
        except Exception as e:
            logger.error(f"Publish error: {e}")
            return 0
    
    def subscribe(self, channels: List[str]):
        """Subscribe to channels"""
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(*channels)
        return self.pubsub
    
    # List operations for tick buffers
    def push_tick_buffer(self, symbol: str, exchange: str, tick: Dict) -> int:
        """Push tick to buffer"""
        key = f"tick_buffer:{exchange}:{symbol}"
        json_tick = json.dumps(tick)
        
        # Push to list and trim to maintain buffer size
        pipe = self.redis_client.pipeline()
        pipe.lpush(key, json_tick)
        pipe.ltrim(key, 0, settings.data.tick_buffer_size - 1)
        pipe.expire(key, 3600)  # 1 hour TTL
        results = pipe.execute()
        
        return results[0]  # Return list length
    
    def get_tick_buffer(self, symbol: str, exchange: str, limit: int = 100) -> List[Dict]:
        """Get recent ticks from buffer"""
        key = f"tick_buffer:{exchange}:{symbol}"
        ticks = self.redis_client.lrange(key, 0, limit - 1)
        return [json.loads(tick.decode('utf-8')) for tick in ticks]
    
    # Feature cache methods
    def cache_features(self, symbol: str, features: Dict, ttl: int = 300) -> bool:
        """Cache calculated features"""
        key = f"features:{symbol}"
        return self.set_json(key, features, ttl=ttl)
    
    def get_cached_features(self, symbol: str) -> Optional[Dict]:
        """Get cached features"""
        key = f"features:{symbol}"
        return self.get_json(key)
    
    def close(self):
        """Close Redis connections"""
        if self.pubsub:
            self.pubsub.close()
        
        if self.redis_client:
            self.redis_client.close()
        
        if self.async_redis:
            asyncio.create_task(self.async_redis.close())
        
        logger.info("Redis connections closed")


# Global cache manager instance
cache_manager = CacheManager()