"""
LRU Embedding Cache
===================

In-memory LRU cache for character and clothing embeddings with TTL support.
"""

import time
import hashlib
import logging
from typing import Optional, Dict, Any
from collections import OrderedDict
import torch

logger = logging.getLogger(__name__)


class LRUEmbeddingCache:
    """LRU cache for character and clothing embeddings."""
    
    def __init__(self, max_size: int = 128, ttl: int = 3600):
        """
        Initialize LRU embedding cache.
        
        Args:
            max_size: Maximum number of cached items
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
    
    def _get_key(self, image_path: str, description: str) -> str:
        """Generate cache key."""
        key_str = f"{image_path}:{description}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, image_path: str, description: str) -> Optional[torch.Tensor]:
        """Get cached embedding."""
        key = self._get_key(image_path, description)
        
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def set(self, image_path: str, description: str, embedding: torch.Tensor) -> None:
        """Cache embedding."""
        key = self._get_key(image_path, description)
        
        # Remove if exists
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
        
        # Add new entry
        self.cache[key] = embedding.cpu().clone()
        self.timestamps[key] = time.time()
        
        # Evict if over size
        if len(self.cache) > self.max_size:
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

