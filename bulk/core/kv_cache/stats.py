"""
Statistics tracking module for KV Cache.

Separates statistics tracking from cache implementation for better modularity.
"""
import logging
import threading
from typing import Dict, Any, Optional
from collections import deque
import time

logger = logging.getLogger(__name__)


class CacheStatsTracker:
    """
    Tracks cache statistics and performance metrics.
    
    Provides thread-safe statistics tracking with history.
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize statistics tracker.
        
        Args:
            history_size: Maximum number of history entries to keep
        """
        self.history_size = history_size
        self._lock = threading.Lock()
        
        # Current statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_size = 0
        
        # History for analysis
        self._hit_rate_history: deque = deque(maxlen=history_size)
        self._timestamp_history: deque = deque(maxlen=history_size)
        
        logger.debug(f"Initialized CacheStatsTracker with history_size={history_size}")
    
    def record_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self.hits += 1
            self._update_history()
    
    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self.misses += 1
            self._update_history()
    
    def record_eviction(self, count: int = 1) -> None:
        """Record evictions."""
        with self._lock:
            self.evictions += count
    
    def update_size(self, size: int) -> None:
        """Update total cache size."""
        with self._lock:
            self.total_size = size
    
    def _update_history(self) -> None:
        """Update hit rate history."""
        total = self.hits + self.misses
        if total > 0:
            hit_rate = self.hits / total
            self._hit_rate_history.append(hit_rate)
            self._timestamp_history.append(time.time())
    
    def get_stats(self, include_history: bool = False) -> Dict[str, Any]:
        """
        Get current statistics.
        
        Args:
            include_history: Whether to include historical data
            
        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = (
                self.hits / total_requests
                if total_requests > 0
                else 0.0
            )
            
            stats = {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "total_size": self.total_size,
                "total_requests": total_requests,
            }
            
            if include_history:
                stats["history"] = {
                    "hit_rates": list(self._hit_rate_history),
                    "timestamps": list(self._timestamp_history),
                }
            
            return stats
    
    def get_average_hit_rate(self, window: Optional[int] = None) -> float:
        """
        Get average hit rate over a window.
        
        Args:
            window: Number of recent entries to average (None = all)
            
        Returns:
            Average hit rate
        """
        with self._lock:
            if not self._hit_rate_history:
                return 0.0
            
            if window is None:
                rates = list(self._hit_rate_history)
            else:
                rates = list(self._hit_rate_history)[-window:]
            
            return sum(rates) / len(rates) if rates else 0.0
    
    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.total_size = 0
            self._hit_rate_history.clear()
            self._timestamp_history.clear()
    
    def get_trend(self) -> str:
        """
        Get trend of hit rate (improving/declining/stable).
        
        Returns:
            "improving", "declining", or "stable"
        """
        with self._lock:
            if len(self._hit_rate_history) < 2:
                return "stable"
            
            recent = list(self._hit_rate_history)[-10:]
            if len(recent) < 2:
                return "stable"
            
            first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
            second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            
            diff = second_half - first_half
            if abs(diff) < 0.01:
                return "stable"
            elif diff > 0:
                return "improving"
            else:
                return "declining"

