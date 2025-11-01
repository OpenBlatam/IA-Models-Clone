"""
Paged cache adapter for efficient memory management.

Implements paged storage for large-scale caching.
"""
import logging
from typing import Dict, Any, Optional, Tuple, List
import torch

from kv_cache.base import BaseKVCache
from kv_cache.config import KVCacheConfig, CacheStrategy

logger = logging.getLogger(__name__)


class PagedKVCache(BaseKVCache):
    """
    Paged KV cache for efficient memory management.
    
    Organizes cache entries into pages for better memory locality
    and faster batch operations.
    """
    
    def __init__(self, config: KVCacheConfig):
        """
        Initialize paged cache.
        
        Args:
            config: KV cache configuration
        """
        # Ensure paged strategy
        if config.cache_strategy != CacheStrategy.PAGED:
            logger.warning(
                f"PagedKVCache requires PAGED strategy, "
                f"got {config.cache_strategy}. Overriding."
            )
            config.cache_strategy = CacheStrategy.PAGED
        
        super().__init__(config)
        
        # Page management
        self._pages: Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor]]] = {}
        self._page_size = config.block_size
        self._page_map: Dict[int, int] = {}  # position -> page_id mapping
        
        logger.info(
            f"Initialized PagedKVCache with page_size={self._page_size}, "
            f"max_tokens={config.max_tokens}"
        )
    
    def get_page(self, page_id: int) -> Optional[List[Tuple[int, torch.Tensor, torch.Tensor]]]:
        """
        Get entire page.
        
        Args:
            page_id: Page identifier
            
        Returns:
            List of (position, key, value) tuples in page, or None
        """
        return self._pages.get(page_id)
    
    def put(
        self,
        position: int,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> None:
        """
        Put KV into paged cache.
        
        Args:
            position: Cache position
            key: Key tensor
            value: Value tensor
        """
        # Calculate page ID
        page_id = position // self._page_size
        
        # Initialize page if needed
        if page_id not in self._pages:
            self._pages[page_id] = []
        
        # Add to page
        self._pages[page_id].append((position, key, value))
        self._page_map[position] = page_id
        
        # Also store in regular cache for backward compatibility
        super().put(position, key, value)
    
    def get(self, position: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get from paged cache (uses base implementation).
        
        Args:
            position: Cache position
            
        Returns:
            Tuple of (key, value) if found, None otherwise
        """
        return super().get(position)
    
    def evict_page(self, page_id: int) -> int:
        """
        Evict entire page.
        
        Args:
            page_id: Page to evict
            
        Returns:
            Number of entries evicted
        """
        if page_id not in self._pages:
            return 0
        
        page_entries = self._pages[page_id]
        positions = [pos for pos, _, _ in page_entries]
        
        # Remove from base cache
        evicted = self.storage.remove(positions)
        
        # Clean up page
        del self._pages[page_id]
        for pos in positions:
            self._page_map.pop(pos, None)
        
        logger.debug(f"Evicted page {page_id} with {evicted} entries")
        return evicted
    
    def get_page_stats(self) -> Dict[str, Any]:
        """
        Get page statistics.
        
        Returns:
            Dictionary with page stats
        """
        page_sizes = {pid: len(entries) for pid, entries in self._pages.items()}
        
        return {
            "num_pages": len(self._pages),
            "total_entries": sum(len(entries) for entries in self._pages.values()),
            "page_size": self._page_size,
            "page_sizes": page_sizes,
            "avg_page_size": sum(page_sizes.values()) / len(page_sizes) if page_sizes else 0.0,
        }

