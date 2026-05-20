"""
Precomputation Utilities
========================

Utilities for precomputing and caching expensive operations.
"""

import torch
import functools
import hashlib
import pickle
from typing import Any, Callable, Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PrecomputationCache:
    """Cache for precomputed values."""
    
    def __init__(self, cache_dir: str = ".cache"):
        """
        Initialize precomputation cache.
        
        Args:
            cache_dir: Cache directory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logger
    
    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key."""
        key_data = {
            "func": func_name,
            "args": args,
            "kwargs": kwargs
        }
        key_str = pickle.dumps(key_data)
        key_hash = hashlib.sha256(key_str).hexdigest()
        return f"{func_name}_{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self._logger.warning(f"Cache read error: {str(e)}")
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value."""
        cache_path = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            self._logger.warning(f"Cache write error: {str(e)}")


def precompute(
    cache_dir: str = ".cache",
    use_cache: bool = True
):
    """
    Decorator for precomputation.
    
    Args:
        cache_dir: Cache directory
        use_cache: Whether to use cache
    
    Returns:
        Decorator
    """
    cache = PrecomputationCache(cache_dir)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not use_cache:
                return func(*args, **kwargs)
            
            # Generate cache key
            key = cache._get_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            cached = cache.get(key)
            if cached is not None:
                return cached
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result)
            
            return result
        
        return wrapper
    return decorator


class FeaturePrecomputer:
    """Precompute features for faster inference."""
    
    def __init__(self, cache_dir: str = ".feature_cache"):
        """
        Initialize feature precomputer.
        
        Args:
            cache_dir: Cache directory
        """
        self.cache = PrecomputationCache(cache_dir)
        self._logger = logger
    
    def precompute_features(
        self,
        feature_extractor: Callable,
        data: list,
        batch_size: int = 32
    ) -> Dict[str, torch.Tensor]:
        """
        Precompute features for dataset.
        
        Args:
            feature_extractor: Feature extraction function
            data: List of data items
            batch_size: Batch size
        
        Returns:
            Dictionary of precomputed features
        """
        precomputed = {}
        
        for i, item in enumerate(data):
            cache_key = f"feature_{hash(str(item))}"
            
            # Check cache
            cached = self.cache.get(cache_key)
            if cached is not None:
                precomputed[i] = cached
                continue
            
            # Compute
            features = feature_extractor(item)
            self.cache.set(cache_key, features)
            precomputed[i] = features
        
        self._logger.info(f"Precomputed {len(precomputed)} features")
        return precomputed




