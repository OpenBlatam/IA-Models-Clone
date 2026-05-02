"""
Cache pool management.

Provides connection pooling and resource management.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from queue import Queue, Empty
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Pool configuration."""
    min_size: int = 1
    max_size: int = 10
    timeout: float = 30.0
    idle_timeout: float = 300.0
    max_idle: int = 5


class CachePool:
    """
    Cache connection pool.
    
    Manages pool of cache instances.
    """
    
    def __init__(
        self,
        cache_factory: Callable[[], Any],
        config: Optional[PoolConfig] = None
    ):
        """
        Initialize cache pool.
        
        Args:
            cache_factory: Factory function to create cache instances
            config: Optional pool configuration
        """
        self.cache_factory = cache_factory
        self.config = config or PoolConfig()
        
        self.pool: Queue = Queue(maxsize=self.config.max_size)
        self.active: List[Any] = []
        self.lock = threading.Lock()
        
        # Initialize pool
        for _ in range(self.config.min_size):
            cache = self.cache_factory()
            self.pool.put(cache)
    
    def acquire(self, timeout: Optional[float] = None) -> Any:
        """
        Acquire cache from pool.
        
        Args:
            timeout: Optional timeout
            
        Returns:
            Cache instance
        """
        timeout = timeout or self.config.timeout
        
        try:
            cache = self.pool.get(timeout=timeout)
        except Empty:
            # Create new if pool is empty and under max size
            with self.lock:
                if len(self.active) < self.config.max_size:
                    cache = self.cache_factory()
                else:
                    raise TimeoutError("Pool exhausted")
        
        with self.lock:
            self.active.append(cache)
        
        return cache
    
    def release(self, cache: Any) -> None:
        """
        Release cache back to pool.
        
        Args:
            cache: Cache instance to release
        """
        with self.lock:
            if cache in self.active:
                self.active.remove(cache)
        
        try:
            self.pool.put_nowait(cache)
        except:
            # Pool is full, discard
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.lock:
            return {
                "pool_size": self.pool.qsize(),
                "active_count": len(self.active),
                "max_size": self.config.max_size
            }
    
    def clear(self) -> None:
        """Clear pool."""
        with self.lock:
            while not self.pool.empty():
                try:
                    cache = self.pool.get_nowait()
                    cache.clear()
                except Empty:
                    break
            
            self.active.clear()
    
    def __enter__(self):
        """Context manager entry."""
        self.cache = self.acquire()
        return self.cache
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release(self.cache)


class CachePoolManager:
    """
    Cache pool manager.
    
    Manages multiple cache pools.
    """
    
    def __init__(self):
        """Initialize pool manager."""
        self.pools: Dict[str, CachePool] = {}
        self.lock = threading.Lock()
    
    def create_pool(
        self,
        pool_name: str,
        cache_factory: Callable[[], Any],
        config: Optional[PoolConfig] = None
    ) -> CachePool:
        """
        Create a new pool.
        
        Args:
            pool_name: Name of pool
            cache_factory: Factory function
            config: Optional pool configuration
            
        Returns:
            Created pool
        """
        with self.lock:
            if pool_name in self.pools:
                raise ValueError(f"Pool {pool_name} already exists")
            
            pool = CachePool(cache_factory, config)
            self.pools[pool_name] = pool
            
            return pool
    
    def get_pool(self, pool_name: str) -> Optional[CachePool]:
        """
        Get pool by name.
        
        Args:
            pool_name: Name of pool
            
        Returns:
            Pool instance or None
        """
        with self.lock:
            return self.pools.get(pool_name)
    
    def remove_pool(self, pool_name: str) -> bool:
        """
        Remove pool.
        
        Args:
            pool_name: Name of pool
            
        Returns:
            True if removed
        """
        with self.lock:
            if pool_name in self.pools:
                pool = self.pools[pool_name]
                pool.clear()
                del self.pools[pool_name]
                return True
            return False
    
    def list_pools(self) -> List[str]:
        """
        List all pools.
        
        Returns:
            List of pool names
        """
        with self.lock:
            return list(self.pools.keys())
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all pools.
        
        Returns:
            Dictionary of pool stats
        """
        with self.lock:
            return {
                name: pool.get_stats()
                for name, pool in self.pools.items()
            }

