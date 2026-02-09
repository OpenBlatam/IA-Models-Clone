"""
Inference Caching

Utilities for caching inference results.
"""

import logging
import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)


class InferenceCache:
    """Cache for inference results."""
    
    def __init__(self, cache_dir: str = "./cache/inference", max_size: int = 1000):
        """
        Initialize inference cache.
        
        Args:
            cache_dir: Cache directory
            max_size: Maximum cache size
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key
        """
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cache(
        self,
        result: Any,
        *args,
        **kwargs
    ) -> str:
        """
        Cache inference result.
        
        Args:
            result: Result to cache
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cache key
        """
        cache_key = self._get_cache_key(*args, **kwargs)
        
        # Check cache size
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        # Cache result
        self.cache[cache_key] = result
        
        # Also save to disk
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            logger.warning(f"Could not save to disk: {e}")
        
        return cache_key
    
    def get(
        self,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Get cached result.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Cached result or None
        """
        cache_key = self._get_cache_key(*args, **kwargs)
        
        # Check memory cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check disk cache
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                self.cache[cache_key] = result  # Load into memory
                return result
            except Exception as e:
                logger.warning(f"Could not load from disk: {e}")
        
        return None
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        logger.info("Cleared inference cache")


def cache_inference(
    result: Any,
    *args,
    **kwargs
) -> str:
    """Cache inference result."""
    cache = InferenceCache()
    return cache.cache(result, *args, **kwargs)


def get_cached_inference(
    *args,
    **kwargs
) -> Optional[Any]:
    """Get cached inference result."""
    cache = InferenceCache()
    return cache.get(*args, **kwargs)

