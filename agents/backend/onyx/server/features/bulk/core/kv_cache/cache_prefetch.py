"""
Cache prefetching strategies.

Provides intelligent prefetching for better cache performance.
"""
from __future__ import annotations

import logging
from typing import List, Callable, Optional, Dict
import threading

from kv_cache.types import TensorPair

logger = logging.getLogger(__name__)


class CachePrefetcher:
    """
    Intelligent cache prefetcher.
    
    Prefetches likely-to-be-needed entries to improve hit rate.
    """
    
    def __init__(
        self,
        cache: Any,
        prefetch_size: int = 10,
        enabled: bool = True
    ):
        """
        Initialize cache prefetcher.
        
        Args:
            cache: Cache instance
            prefetch_size: Number of entries to prefetch
            enabled: Whether prefetching is enabled
        """
        self.cache = cache
        self.prefetch_size = prefetch_size
        self.enabled = enabled
        
        # Access patterns
        self.access_patterns: Dict[int, List[int]] = {}
        self._lock = threading.Lock()
    
    def record_access(self, position: int, next_positions: List[int]) -> None:
        """
        Record access pattern.
        
        Args:
            position: Current position accessed
            next_positions: Positions accessed after this one
        """
        if not self.enabled:
            return
        
        with self._lock:
            if position not in self.access_patterns:
                self.access_patterns[position] = []
            self.access_patterns[position].extend(next_positions)
            
            # Keep only recent patterns
            if len(self.access_patterns[position]) > 100:
                self.access_patterns[position] = self.access_patterns[position][-100:]
    
    def predict_next(self, position: int) -> List[int]:
        """
        Predict next positions likely to be accessed.
        
        Args:
            position: Current position
            
        Returns:
            List of predicted positions
        """
        if not self.enabled:
            return []
        
        with self._lock:
            if position not in self.access_patterns:
                return []
            
            # Count frequency of next positions
            next_positions = self.access_patterns[position]
            frequency: Dict[int, int] = {}
            
            for pos in next_positions:
                frequency[pos] = frequency.get(pos, 0) + 1
            
            # Sort by frequency
            sorted_positions = sorted(
                frequency.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.prefetch_size]
            
            return [pos for pos, _ in sorted_positions]
    
    def prefetch(
        self,
        position: int,
        compute_fn: Callable[[int], TensorPair],
        background: bool = True
    ) -> Optional[List[TensorPair]]:
        """
        Prefetch predicted positions.
        
        Args:
            position: Current position
            compute_fn: Function to compute values
            background: Whether to prefetch in background thread
            
        Returns:
            List of prefetched entries (if not background)
        """
        if not self.enabled:
            return None
        
        predicted = self.predict_next(position)
        if not predicted:
            return None
        
        # Filter out already cached
        to_prefetch = [pos for pos in predicted if self.cache.get(pos) is None]
        
        if not to_prefetch:
            return None
        
        def prefetch_worker():
            """Background prefetch worker."""
            for pos in to_prefetch:
                try:
                    key, value = compute_fn(pos)
                    self.cache.put(pos, key, value)
                    logger.debug(f"Prefetched position {pos}")
                except Exception as e:
                    logger.warning(f"Prefetch failed for position {pos}: {e}")
        
        if background:
            thread = threading.Thread(target=prefetch_worker, daemon=True)
            thread.start()
            return None
        else:
            # Synchronous prefetch
            results = []
            for pos in to_prefetch:
                try:
                    key, value = compute_fn(pos)
                    self.cache.put(pos, key, value)
                    results.append((key, value))
                except Exception as e:
                    logger.warning(f"Prefetch failed for position {pos}: {e}")
            return results
    
    def get_prefetch_stats(self) -> Dict[str, Any]:
        """
        Get prefetch statistics.
        
        Returns:
            Dictionary with prefetch stats
        """
        with self._lock:
            total_patterns = sum(len(p) for p in self.access_patterns.values())
            return {
                "enabled": self.enabled,
                "prefetch_size": self.prefetch_size,
                "tracked_positions": len(self.access_patterns),
                "total_patterns": total_patterns,
                "avg_patterns_per_position": (
                    total_patterns / len(self.access_patterns)
                    if self.access_patterns else 0
                )
            }


class SequentialPrefetcher:
    """
    Sequential prefetcher.
    
    Prefetches sequential positions.
    """
    
    def __init__(self, cache: Any, lookahead: int = 5):
        """
        Initialize sequential prefetcher.
        
        Args:
            cache: Cache instance
            lookahead: Number of positions ahead to prefetch
        """
        self.cache = cache
        self.lookahead = lookahead
    
    def prefetch_sequential(
        self,
        current_position: int,
        compute_fn: Callable[[int], TensorPair],
        background: bool = True
    ) -> None:
        """
        Prefetch sequential positions.
        
        Args:
            current_position: Current position
            compute_fn: Function to compute values
            background: Whether to prefetch in background
        """
        positions_to_prefetch = [
            current_position + i + 1
            for i in range(self.lookahead)
        ]
        
        def prefetch_worker():
            """Background prefetch worker."""
            for pos in positions_to_prefetch:
                if self.cache.get(pos) is None:
                    try:
                        key, value = compute_fn(pos)
                        self.cache.put(pos, key, value)
                    except Exception as e:
                        logger.warning(f"Sequential prefetch failed for {pos}: {e}")
        
        if background:
            thread = threading.Thread(target=prefetch_worker, daemon=True)
            thread.start()
        else:
            prefetch_worker()

