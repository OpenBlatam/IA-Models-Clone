"""
Cache Manager
=============

Handles embedding cache operations.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

from ...models.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages embedding cache operations."""
    
    def __init__(self, cache_dir: str = "./embedding_cache", enabled: bool = True):
        """
        Initialize Cache Manager.
        
        Args:
            cache_dir: Cache directory path
            enabled: Whether cache is enabled
        """
        self.enabled = enabled
        self.cache: Optional[EmbeddingCache] = None
        
        if enabled:
            self.cache = EmbeddingCache(cache_dir=cache_dir)
            logger.info(f"Cache enabled at: {cache_dir}")
        else:
            logger.info("Cache disabled")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get embedding cache statistics.
        
        Returns:
            Cache statistics
        """
        if not self.enabled or self.cache is None:
            return {"cache_enabled": False}
        
        stats = self.cache.get_cache_stats()
        stats["cache_enabled"] = True
        return stats
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        if self.cache:
            self.cache.clear_cache()
            logger.info("Cache cleared")


