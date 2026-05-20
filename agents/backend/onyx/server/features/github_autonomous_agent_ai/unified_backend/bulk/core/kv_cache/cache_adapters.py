"""
Cache adapter patterns.

Provides adapters for different cache interfaces.
"""
from __future__ import annotations

import logging
from typing import Dict, Any, Optional, List
import torch

logger = logging.getLogger(__name__)


class CacheAdapter:
    """
    Base cache adapter.
    
    Adapts cache to different interfaces.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize adapter.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        return self.cache.get(key)
    
    def put(self, key: Any, value: Any) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache.put(key, value)
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


class DictAdapter(CacheAdapter):
    """
    Dictionary-like adapter.
    
    Makes cache behave like a dictionary.
    """
    
    def __getitem__(self, key: Any) -> Any:
        """
        Get item using [] operator.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value
            
        Raises:
            KeyError: If key not found
        """
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value
    
    def __setitem__(self, key: Any, value: Any) -> None:
        """
        Set item using [] operator.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.put(key, value)
    
    def __delitem__(self, key: Any) -> None:
        """
        Delete item using del operator.
        
        Args:
            key: Cache key
            
        Raises:
            KeyError: If key not found
        """
        # Implementation depends on cache API
        pass
    
    def __contains__(self, key: Any) -> bool:
        """
        Check if key exists.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        return self.get(key) is not None
    
    def keys(self) -> List[Any]:
        """
        Get all keys.
        
        Returns:
            List of keys
        """
        stats = self.cache.get_stats()
        return list(range(stats.get("cache_size", 0)))
    
    def values(self) -> List[Any]:
        """
        Get all values.
        
        Returns:
            List of values
        """
        keys = self.keys()
        return [self.get(k) for k in keys]
    
    def items(self) -> List[tuple]:
        """
        Get all key-value pairs.
        
        Returns:
            List of (key, value) tuples
        """
        keys = self.keys()
        return [(k, self.get(k)) for k in keys]


class ContextManagerAdapter(CacheAdapter):
    """
    Context manager adapter.
    
    Makes cache usable as context manager.
    """
    
    def __enter__(self) -> CacheAdapter:
        """
        Enter context.
        
        Returns:
            Self
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit context.
        
        Args:
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        # Cleanup if needed
        pass


class AsyncAdapter:
    """
    Async adapter for cache.
    
    Provides async interface for cache operations.
    """
    
    def __init__(self, cache: Any):
        """
        Initialize async adapter.
        
        Args:
            cache: Cache instance
        """
        self.cache = cache
    
    async def get_async(self, key: Any) -> Optional[Any]:
        """
        Get value asynchronously.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.cache.get, key)
    
    async def put_async(self, key: Any, value: Any) -> None:
        """
        Put value asynchronously.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.cache.put, key, value)
    
    async def clear_async(self) -> None:
        """Clear cache asynchronously."""
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.cache.clear)


class BatchAdapter(CacheAdapter):
    """
    Batch operation adapter.
    
    Provides batch operations for cache.
    """
    
    def batch_get(self, keys: List[Any]) -> Dict[Any, Any]:
        """
        Get multiple values.
        
        Args:
            keys: List of keys
            
        Returns:
            Dictionary of key-value pairs
        """
        results = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                results[key] = value
        return results
    
    def batch_put(self, items: Dict[Any, Any]) -> None:
        """
        Put multiple values.
        
        Args:
            items: Dictionary of key-value pairs
        """
        for key, value in items.items():
            self.put(key, value)
    
    def batch_delete(self, keys: List[Any]) -> None:
        """
        Delete multiple values.
        
        Args:
            keys: List of keys to delete
        """
        # Implementation depends on cache API
        pass


class TransformerAdapter(CacheAdapter):
    """
    Transformer-specific adapter.
    
    Adapts cache for transformer model usage.
    """
    
    def __init__(self, cache: Any, num_heads: int, head_dim: int):
        """
        Initialize transformer adapter.
        
        Args:
            cache: Cache instance
            num_heads: Number of attention heads
            head_dim: Dimension of each head
        """
        super().__init__(cache)
        self.num_heads = num_heads
        self.head_dim = head_dim
    
    def get_kv_cache(
        self,
        position: int,
        layer: int
    ) -> Optional[torch.Tensor]:
        """
        Get KV cache for position and layer.
        
        Args:
            position: Token position
            layer: Layer index
            
        Returns:
            KV cache tensor or None
        """
        key = f"layer_{layer}_pos_{position}"
        return self.get(key)
    
    def put_kv_cache(
        self,
        position: int,
        layer: int,
        kv: torch.Tensor
    ) -> None:
        """
        Put KV cache for position and layer.
        
        Args:
            position: Token position
            layer: Layer index
            kv: KV cache tensor
        """
        key = f"layer_{layer}_pos_{position}"
        self.put(key, kv)
    
    def get_all_layers(self, position: int) -> Dict[int, torch.Tensor]:
        """
        Get KV cache for all layers at position.
        
        Args:
            position: Token position
            
        Returns:
            Dictionary of layer -> KV cache
        """
        results = {}
        stats = self.cache.get_stats()
        num_layers = stats.get("num_layers", 0)
        
        for layer in range(num_layers):
            kv = self.get_kv_cache(position, layer)
            if kv is not None:
                results[layer] = kv
        
        return results

