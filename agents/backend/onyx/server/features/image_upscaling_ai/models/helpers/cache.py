"""
Upscaling Cache
===============

LRU cache for upscaled images with TTL support.
"""

import logging
import time
import hashlib
from typing import Optional, Dict, Any
from collections import OrderedDict
from PIL import Image

logger = logging.getLogger(__name__)


class UpscalingCache:
    """LRU cache for upscaled images."""
    
    def __init__(self, max_size: int = 64, ttl: int = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached images
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
    
    def _get_key(self, image_path: str, scale_factor: float, method: str) -> str:
        """Generate cache key."""
        key_str = f"{image_path}:{scale_factor}:{method}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, image_path: str, scale_factor: float, method: str) -> Optional[Image.Image]:
        """Get cached upscaled image."""
        key = self._get_key(image_path, scale_factor, method)
        
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key].copy()
    
    def set(self, image_path: str, scale_factor: float, method: str, image: Image.Image) -> None:
        """Cache upscaled image."""
        key = self._get_key(image_path, scale_factor, method)
        
        # Remove if exists
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
        
        # Add new entry
        self.cache[key] = image.copy()
        self.timestamps[key] = time.time()
        
        # Evict if over size
        if len(self.cache) > self.max_size:
            self._evict_oldest()
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self.cache:
            return
        
        oldest_key = next(iter(self.cache))
        del self.cache[oldest_key]
        del self.timestamps[oldest_key]
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
        }


