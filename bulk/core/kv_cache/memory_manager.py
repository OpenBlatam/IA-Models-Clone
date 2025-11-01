"""
Memory management module for KV Cache.

Handles memory monitoring, eviction strategies, and garbage collection.
"""
import logging
import gc
import threading
from typing import Dict, List, Tuple, Optional
import torch

from kv_cache.config import CacheStrategy, KVCacheConfig

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages memory for KV cache operations.
    
    Responsibilities:
    - Monitor memory usage (GPU/CPU)
    - Determine when eviction is needed
    - Trigger garbage collection
    - Provide memory statistics
    """
    
    def __init__(self, config: KVCacheConfig, device: torch.device):
        """
        Initialize memory manager.
        
        Args:
            config: KV cache configuration
            device: Device being used
        """
        self.config = config
        self.device = device
        self._lock = threading.Lock()
        
        logger.info(f"Initialized MemoryManager for device {device}")
    
    def should_evict(self, cache_size: int) -> bool:
        """
        Check if eviction is needed based on memory or token limits.
        
        Args:
            cache_size: Current number of cache entries
            
        Returns:
            True if eviction should be performed
        """
        # Always check token limit first (cheaper check)
        if cache_size >= self.config.max_tokens:
            logger.debug(
                f"Eviction needed: cache size ({cache_size}) >= "
                f"max_tokens ({self.config.max_tokens})"
            )
            return True
        
        # Check memory limit if configured
        if self.config.max_memory_mb is not None:
            try:
                if torch.cuda.is_available() and self.device.type == "cuda":
                    memory_allocated = torch.cuda.memory_allocated(self.device) / (1024**2)  # MB
                    memory_reserved = torch.cuda.memory_reserved(self.device) / (1024**2)  # MB
                    
                    # Use reserved memory as it's more accurate for cache
                    threshold = self.config.max_memory_mb * self.config.gc_threshold
                    if memory_reserved > threshold:
                        logger.debug(
                            f"Eviction needed: memory ({memory_reserved:.2f} MB) > "
                            f"threshold ({threshold:.2f} MB)"
                        )
                        return True
                elif self.device.type == "cpu":
                    # For CPU, estimate based on cache size
                    estimated_mb = cache_size * self.config.head_dim * 4 / (1024**2)  # Rough estimate
                    threshold = self.config.max_memory_mb * self.config.gc_threshold
                    if estimated_mb > threshold:
                        return True
            except Exception as e:
                logger.warning(f"Error checking memory usage: {e}, using token-based eviction")
                return cache_size >= self.config.max_tokens
        
        return False
    
    def collect_garbage(self) -> None:
        """
        Trigger garbage collection if enabled.
        
        Should be called outside locks to avoid blocking.
        """
        if not self.config.enable_gc:
            return
        
        try:
            gc.collect()
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure operations complete
        except Exception as e:
            logger.debug(f"Error during garbage collection: {e}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        stats = {}
        
        try:
            if torch.cuda.is_available() and self.device.type == "cuda":
                stats["allocated_mb"] = torch.cuda.memory_allocated(self.device) / (1024**2)
                stats["reserved_mb"] = torch.cuda.memory_reserved(self.device) / (1024**2)
                stats["max_allocated_mb"] = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                stats["max_reserved_mb"] = torch.cuda.max_memory_reserved(self.device) / (1024**2)
        except Exception as e:
            logger.warning(f"Error getting memory stats: {e}")
        
        return stats

