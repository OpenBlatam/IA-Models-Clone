"""
Data Structure Utilities for Piel Mejorador AI SAM3
===================================================

Unified data structure utilities.
"""

import logging
from typing import Any, List, Dict, Optional, Callable, TypeVar, Deque
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class TimedItem:
    """Item with timestamp."""
    value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataStructureUtils:
    """Unified data structure utilities."""
    
    @staticmethod
    def create_bounded_deque(maxlen: int) -> deque:
        """
        Create bounded deque.
        
        Args:
            maxlen: Maximum length
            
        Returns:
            Bounded deque
        """
        return deque(maxlen=maxlen)
    
    @staticmethod
    def create_timed_deque(max_age_seconds: float) -> deque:
        """
        Create deque that automatically expires old items.
        
        Args:
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Timed deque
        """
        return deque(maxlen=None)  # Will be managed manually
    
    @staticmethod
    def expire_old_items(
        items: List[TimedItem],
        max_age_seconds: float
    ) -> List[TimedItem]:
        """
        Remove items older than max_age.
        
        Args:
            items: List of timed items
            max_age_seconds: Maximum age in seconds
            
        Returns:
            Filtered items
        """
        cutoff = datetime.now() - timedelta(seconds=max_age_seconds)
        return [item for item in items if item.timestamp > cutoff]
    
    @staticmethod
    def create_lru_dict(max_size: int) -> Dict[str, Any]:
        """
        Create LRU-like dictionary (simple implementation).
        
        Args:
            max_size: Maximum size
            
        Returns:
            Dictionary (will need manual LRU management)
        """
        return {}
    
    @staticmethod
    def create_default_dict(default_factory: Callable) -> defaultdict:
        """
        Create defaultdict with factory.
        
        Args:
            default_factory: Factory function
            
        Returns:
            Defaultdict
        """
        return defaultdict(default_factory)
    
    @staticmethod
    def create_nested_dict() -> Dict[str, Any]:
        """
        Create nested dictionary (auto-creates nested dicts).
        
        Returns:
            Nested dictionary
        """
        return defaultdict(lambda: defaultdict(dict))


class TimedQueue:
    """Queue with automatic expiration."""
    
    def __init__(self, max_age_seconds: float):
        """
        Initialize timed queue.
        
        Args:
            max_age_seconds: Maximum age in seconds
        """
        self.max_age = timedelta(seconds=max_age_seconds)
        self._items: List[TimedItem] = []
    
    def add(self, value: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Add item to queue.
        
        Args:
            value: Item value
            metadata: Optional metadata
        """
        item = TimedItem(value=value, metadata=metadata or {})
        self._items.append(item)
        self._cleanup()
    
    def get_all(self) -> List[Any]:
        """
        Get all non-expired items.
        
        Returns:
            List of values
        """
        self._cleanup()
        return [item.value for item in self._items]
    
    def get_recent(self, seconds: float) -> List[Any]:
        """
        Get items from last N seconds.
        
        Args:
            seconds: Number of seconds
            
        Returns:
            List of values
        """
        cutoff = datetime.now() - timedelta(seconds=seconds)
        return [
            item.value
            for item in self._items
            if item.timestamp > cutoff
        ]
    
    def _cleanup(self):
        """Remove expired items."""
        cutoff = datetime.now() - self.max_age
        self._items = [item for item in self._items if item.timestamp > cutoff]
    
    def size(self) -> int:
        """
        Get current size.
        
        Returns:
            Number of items
        """
        self._cleanup()
        return len(self._items)
    
    def clear(self):
        """Clear all items."""
        self._items.clear()


class BoundedQueue:
    """Bounded queue with automatic overflow handling."""
    
    def __init__(self, max_size: int, overflow_strategy: str = "drop_oldest"):
        """
        Initialize bounded queue.
        
        Args:
            max_size: Maximum size
            overflow_strategy: "drop_oldest" or "drop_newest"
        """
        self.max_size = max_size
        self.overflow_strategy = overflow_strategy
        self._items: deque = deque(maxlen=max_size)
    
    def add(self, value: Any):
        """
        Add item to queue.
        
        Args:
            value: Item value
        """
        if len(self._items) >= self.max_size:
            if self.overflow_strategy == "drop_oldest":
                # deque with maxlen automatically drops oldest
                pass
            elif self.overflow_strategy == "drop_newest":
                # Don't add if full
                return
        
        self._items.append(value)
    
    def get(self) -> Optional[Any]:
        """
        Get item from queue (FIFO).
        
        Returns:
            Item or None
        """
        if not self._items:
            return None
        return self._items.popleft()
    
    def peek(self) -> Optional[Any]:
        """
        Peek at next item without removing.
        
        Returns:
            Item or None
        """
        if not self._items:
            return None
        return self._items[0]
    
    def size(self) -> int:
        """
        Get current size.
        
        Returns:
            Number of items
        """
        return len(self._items)
    
    def clear(self):
        """Clear all items."""
        self._items.clear()
    
    def get_all(self) -> List[Any]:
        """
        Get all items.
        
        Returns:
            List of items
        """
        return list(self._items)


# Convenience functions
def create_bounded_deque(maxlen: int) -> deque:
    """Create bounded deque."""
    return DataStructureUtils.create_bounded_deque(maxlen)


def create_timed_deque(max_age_seconds: float) -> deque:
    """Create timed deque."""
    return DataStructureUtils.create_timed_deque(max_age_seconds)


def expire_old_items(items: List[TimedItem], max_age_seconds: float) -> List[TimedItem]:
    """Expire old items."""
    return DataStructureUtils.expire_old_items(items, max_age_seconds)




