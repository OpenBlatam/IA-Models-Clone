"""
Agent Cache
Manages caching for agent operations (reasoning and knowledge retrieval)
Follows Single Responsibility Principle
"""

import logging
from typing import Dict, Any, Optional, List
from collections import OrderedDict

logger = logging.getLogger(__name__)


class AgentCache:
    """
    Manages caching for agent operations.
    
    Responsibilities:
    - Cache reasoning results
    - Cache knowledge retrieval results
    - Track cache statistics
    - Implement LRU eviction policy
    """
    
    def __init__(self, max_reasoning_size: int = 100, max_knowledge_size: int = 50):
        """
        Initialize agent cache.
        
        Args:
            max_reasoning_size: Maximum number of reasoning results to cache
            max_knowledge_size: Maximum number of knowledge queries to cache
        """
        # Use OrderedDict for LRU eviction
        self._reasoning_cache: OrderedDict[str, Any] = OrderedDict()
        self._knowledge_cache: OrderedDict[str, List] = OrderedDict()
        self._max_reasoning_size = max_reasoning_size
        self._max_knowledge_size = max_knowledge_size
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_reasoning(self, key: str) -> Optional[Any]:
        """
        Get cached reasoning result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached result or None if not found
        """
        if key in self._reasoning_cache:
            # Move to end (most recently used)
            self._reasoning_cache.move_to_end(key)
            self._cache_hits += 1
            return self._reasoning_cache[key]
        self._cache_misses += 1
        return None
    
    def set_reasoning(self, key: str, value: Any) -> None:
        """
        Cache reasoning result with LRU eviction.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        if len(self._reasoning_cache) >= self._max_reasoning_size:
            # Remove oldest entry (first in OrderedDict)
            oldest_key = next(iter(self._reasoning_cache))
            del self._reasoning_cache[oldest_key]
            logger.debug(f"Evicted reasoning cache entry: {oldest_key}")
        
        self._reasoning_cache[key] = value
        # Move to end (most recently used)
        self._reasoning_cache.move_to_end(key)
    
    def get_knowledge(self, key: str) -> Optional[List]:
        """
        Get cached knowledge retrieval result.
        
        Args:
            key: Cache key
            
        Returns:
            Cached knowledge list or None if not found
        """
        if key in self._knowledge_cache:
            # Move to end (most recently used)
            self._knowledge_cache.move_to_end(key)
            self._cache_hits += 1
            return self._knowledge_cache[key]
        self._cache_misses += 1
        return None
    
    def set_knowledge(self, key: str, value: List) -> None:
        """
        Cache knowledge retrieval result with LRU eviction.
        
        Args:
            key: Cache key
            value: Knowledge list to cache
        """
        if len(self._knowledge_cache) >= self._max_knowledge_size:
            # Remove oldest entry (first in OrderedDict)
            oldest_key = next(iter(self._knowledge_cache))
            del self._knowledge_cache[oldest_key]
            logger.debug(f"Evicted knowledge cache entry: {oldest_key}")
        
        self._knowledge_cache[key] = value
        # Move to end (most recently used)
        self._knowledge_cache.move_to_end(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_ops = self._cache_hits + self._cache_misses
        return {
            "reasoning_cache_size": len(self._reasoning_cache),
            "knowledge_cache_size": len(self._knowledge_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": self._cache_hits / total_ops if total_ops > 0 else 0.0,
        }
    
    def clear(self) -> None:
        """Clear all caches and reset statistics"""
        self._reasoning_cache.clear()
        self._knowledge_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Agent cache cleared")
    
    def evict_oldest_reasoning(self, count: int = 1) -> int:
        """
        Evict oldest reasoning cache entries.
        
        Args:
            count: Number of entries to evict
            
        Returns:
            Number of entries actually evicted
        """
        evicted = 0
        for _ in range(min(count, len(self._reasoning_cache))):
            if self._reasoning_cache:
                oldest_key = next(iter(self._reasoning_cache))
                del self._reasoning_cache[oldest_key]
                evicted += 1
        return evicted

