"""
Model Cache
Cache model predictions and intermediate results
"""

from typing import Dict, Any, Optional, Callable
import logging
import hashlib
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelCache:
    """Cache for model predictions"""
    
    def __init__(self, cache_dir: str = "./cache", max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
    
    def _get_key(self, data: Any) -> str:
        """Generate cache key from data"""
        data_str = pickle.dumps(data)
        return hashlib.md5(data_str).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key in self.cache:
            return self.cache[key]
        
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                self.cache[key] = value
                return value
            except Exception as e:
                logger.warning(f"Error loading cache: {str(e)}")
        
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value"""
        self.cache[key] = value
        
        # Save to disk
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Error saving cache: {str(e)}")
        
        # Limit cache size
        if len(self.cache) > self.max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
    
    def cached(self, func: Callable) -> Callable:
        """Decorator for caching function results"""
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = (args, tuple(sorted(kwargs.items())))
            key = self._get_key(key_data)
            
            # Check cache
            cached_value = self.get(key)
            if cached_value is not None:
                return cached_value
            
            # Compute and cache
            result = func(*args, **kwargs)
            self.set(key, result)
            return result
        
        return wrapper
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()



