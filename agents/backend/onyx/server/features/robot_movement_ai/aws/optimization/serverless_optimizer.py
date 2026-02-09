"""
Serverless Optimizer
====================

Optimizations for serverless deployments (Lambda, Azure Functions):
- Cold start reduction
- Connection pooling
- Lazy loading
- Memory optimization
"""

import logging
import os
from typing import Dict, Any, Optional
from functools import lru_cache
import asyncio

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Connection pool for serverless environments."""
    
    def __init__(self):
        self._pools: Dict[str, Any] = {}
        self._initialized = False
    
    async def get_redis_pool(self, redis_url: str):
        """Get Redis connection pool."""
        if "redis" not in self._pools:
            try:
                import redis.asyncio as aioredis
                self._pools["redis"] = await aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=10
                )
            except Exception as e:
                logger.warning(f"Failed to create Redis pool: {e}")
                return None
        return self._pools.get("redis")
    
    async def get_http_pool(self):
        """Get HTTP connection pool."""
        if "http" not in self._pools:
            import httpx
            self._pools["http"] = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
            )
        return self._pools.get("http")
    
    async def close_all(self):
        """Close all connection pools."""
        for pool in self._pools.values():
            if hasattr(pool, "aclose"):
                await pool.aclose()
            elif hasattr(pool, "close"):
                pool.close()
        self._pools.clear()


class LazyLoader:
    """Lazy loading for heavy dependencies."""
    
    _cache: Dict[str, Any] = {}
    
    @classmethod
    def load(cls, module_name: str):
        """Lazy load a module."""
        if module_name not in cls._cache:
            import importlib
            cls._cache[module_name] = importlib.import_module(module_name)
        return cls._cache[module_name]
    
    @classmethod
    def clear_cache(cls):
        """Clear lazy load cache."""
        cls._cache.clear()


class ServerlessOptimizer:
    """Optimizer for serverless deployments."""
    
    def __init__(self):
        self.connection_pool = ConnectionPool()
        self.lazy_loader = LazyLoader()
        self._warm_up_done = False
    
    async def warm_up(self, config: Dict[str, Any]):
        """Warm up connections and caches."""
        if self._warm_up_done:
            return
        
        logger.info("Warming up serverless environment...")
        
        # Warm up Redis if configured
        redis_url = config.get("redis_url")
        if redis_url:
            try:
                await self.connection_pool.get_redis_pool(redis_url)
                logger.info("Redis pool warmed up")
            except Exception as e:
                logger.warning(f"Failed to warm up Redis: {e}")
        
        # Warm up HTTP pool
        try:
            await self.connection_pool.get_http_pool()
            logger.info("HTTP pool warmed up")
        except Exception as e:
            logger.warning(f"Failed to warm up HTTP pool: {e}")
        
        self._warm_up_done = True
        logger.info("Warm up completed")
    
    async def cleanup(self):
        """Cleanup connections."""
        await self.connection_pool.close_all()
        self.lazy_loader.clear_cache()
        logger.info("Serverless cleanup completed")
    
    @staticmethod
    def optimize_imports():
        """Optimize imports for cold start reduction."""
        # Import only what's needed
        # Use lazy loading for heavy dependencies
        pass
    
    @staticmethod
    def minimize_memory():
        """Minimize memory usage."""
        # Use generators instead of lists
        # Clear caches
        # Use __slots__ for classes
        pass


# Global optimizer instance
_optimizer: Optional[ServerlessOptimizer] = None


def get_serverless_optimizer() -> ServerlessOptimizer:
    """Get global serverless optimizer."""
    global _optimizer
    if _optimizer is None:
        _optimizer = ServerlessOptimizer()
    return _optimizer















