"""
Cache Management Mixin

Contains cache management and optimization methods.
"""

import logging
import gc
from typing import Dict, Any, Optional
from PIL import Image

from ..helpers import (
    UpscalingCache,
    StatisticsManager,
)

logger = logging.getLogger(__name__)


class CacheManagementMixin:
    """
    Mixin providing cache management functionality.
    
    This mixin contains:
    - Cache operations
    - Memory optimization
    - Statistics management
    - Cache statistics
    """
    
    def clear_cache(self) -> None:
        """Clear upscaling cache."""
        if hasattr(self, 'cache') and self.cache:
            self.cache.clear()
            logger.info("Upscaling cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not hasattr(self, 'cache') or not self.cache:
            return {
                "cache_size": 0,
                "cache_max_size": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "hit_rate": 0.0,
            }
        
        hits = getattr(self.stats, 'get', lambda k, d: d)("cache_hits", 0)
        misses = getattr(self.stats, 'get', lambda k, d: d)("cache_misses", 0)
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0
        
        return {
            "cache_size": len(self.cache.cache) if hasattr(self.cache, 'cache') else 0,
            "cache_max_size": self.cache.max_size if hasattr(self.cache, 'max_size') else 0,
            "cache_hits": hits,
            "cache_misses": misses,
            "hit_rate": hit_rate,
        }
    
    def optimize_memory(self) -> None:
        """Optimize memory usage."""
        # Clear cache if too large
        if hasattr(self, 'cache') and self.cache:
            if hasattr(self.cache, 'cache') and len(self.cache.cache) > self.cache.max_size * 0.8:
                items_to_keep = len(self.cache.cache) // 2
                keys_to_remove = list(self.cache.cache.keys())[:-items_to_keep]
                for key in keys_to_remove:
                    if key in self.cache.cache:
                        del self.cache.cache[key]
                    if hasattr(self.cache, 'timestamps') and key in self.cache.timestamps:
                        del self.cache.timestamps[key]
                logger.info(f"Cleared {len(keys_to_remove)} cache entries")
        
        # Garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("Memory optimization completed")
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        if hasattr(self, 'stats'):
            self.stats = StatisticsManager.reset_stats()
            logger.info("Statistics reset")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            
            cache_size = 0
            cache_max_size = 0
            if hasattr(self, 'cache') and self.cache:
                cache_size = len(self.cache.cache) if hasattr(self.cache, 'cache') else 0
                cache_max_size = self.cache.max_size if hasattr(self.cache, 'max_size') else 0
            
            return {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
                "cache_size": cache_size,
                "cache_max_size": cache_max_size,
            }
        except ImportError:
            return {
                "rss_mb": 0,
                "vms_mb": 0,
                "cache_size": len(self.cache.cache) if hasattr(self, 'cache') and self.cache and hasattr(self.cache, 'cache') else 0,
                "cache_max_size": self.cache.max_size if hasattr(self, 'cache') and self.cache and hasattr(self.cache, 'max_size') else 0,
                "note": "psutil not available",
            }


