"""
Priority queue system for KV cache.

This module provides priority-based caching and eviction,
allowing important data to be retained longer.
"""

import time
import threading
import heapq
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Priority(Enum):
    """Cache priority levels."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    NONE = 4


@dataclass
class PriorityEntry:
    """Entry with priority information."""
    key: str
    value: Any
    priority: Priority
    priority_score: float
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    created_at: float = field(default_factory=time.time)


class PriorityCache:
    """Priority-based cache."""
    
    def __init__(self, cache: Any, max_size: int = 1000):
        self.cache = cache
        self.max_size = max_size
        self._priority_entries: Dict[str, PriorityEntry] = {}
        self._priority_queue: List[Tuple[float, str]] = []  # Min-heap
        self._lock = threading.Lock()
        
    def _calculate_priority_score(
        self,
        priority: Priority,
        access_count: int,
        last_accessed: float,
        created_at: float
    ) -> float:
        """Calculate priority score (lower is better for eviction)."""
        base_score = priority.value
        
        # Adjust based on access frequency
        time_since_access = time.time() - last_accessed
        access_factor = access_count / (time_since_access + 1)
        
        # Lower score = higher priority (less likely to evict)
        score = base_score - (access_factor * 0.1)
        
        return score
        
    def get(self, key: str) -> Any:
        """Get value and update priority."""
        value = self.cache.get(key)
        
        if value is not None:
            with self._lock:
                if key in self._priority_entries:
                    entry = self._priority_entries[key]
                    entry.last_accessed = time.time()
                    entry.access_count += 1
                    
                    # Recalculate priority score
                    entry.priority_score = self._calculate_priority_score(
                        entry.priority,
                        entry.access_count,
                        entry.last_accessed,
                        entry.created_at
                    )
                    
        return value
        
    def put(
        self,
        key: str,
        value: Any,
        priority: Priority = Priority.MEDIUM
    ) -> bool:
        """Put value with priority."""
        with self._lock:
            # Check if we need to evict
            if len(self._priority_entries) >= self.max_size and key not in self._priority_entries:
                self._evict_lowest_priority()
                
            # Add or update entry
            if key in self._priority_entries:
                entry = self._priority_entries[key]
                entry.value = value
                entry.priority = priority
            else:
                entry = PriorityEntry(
                    key=key,
                    value=value,
                    priority=priority,
                    priority_score=self._calculate_priority_score(priority, 0, time.time(), time.time())
                )
                self._priority_entries[key] = entry
                
            # Update priority queue
            heapq.heappush(self._priority_queue, (entry.priority_score, key))
            
        return self.cache.put(key, value)
        
    def _evict_lowest_priority(self) -> None:
        """Evict entry with lowest priority."""
        while self._priority_queue:
            score, key = heapq.heappop(self._priority_queue)
            
            # Check if key still exists and score matches
            if key in self._priority_entries:
                entry = self._priority_entries[key]
                if entry.priority_score == score:
                    # This is the lowest priority, evict it
                    del self._priority_entries[key]
                    self.cache.delete(key)
                    return
                    
    def set_priority(self, key: str, priority: Priority) -> bool:
        """Update priority for a key."""
        with self._lock:
            if key in self._priority_entries:
                entry = self._priority_entries[key]
                entry.priority = priority
                entry.priority_score = self._calculate_priority_score(
                    priority,
                    entry.access_count,
                    entry.last_accessed,
                    entry.created_at
                )
                heapq.heappush(self._priority_queue, (entry.priority_score, key))
                return True
            return False
            
    def get_priority_stats(self) -> Dict[str, int]:
        """Get statistics by priority."""
        stats = {p.value: 0 for p in Priority}
        
        with self._lock:
            for entry in self._priority_entries.values():
                stats[entry.priority.value] += 1
                
        return stats


