"""
Intelligent Cache System for Flux2 Clothing Changer
====================================================

Advanced intelligent caching with prediction and optimization.
"""

import hashlib
import time
import json
from typing import Dict, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict
import logging
import pickle

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    size: int = 0
    tags: list = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.last_access == 0.0:
            self.last_access = self.timestamp


class IntelligentCache:
    """Intelligent caching system with prediction."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 1024.0,
        ttl: Optional[float] = None,
        enable_persistence: bool = False,
        persistence_path: Optional[Path] = None,
    ):
        """
        Initialize intelligent cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
            ttl: Time to live in seconds (None for no expiration)
            enable_persistence: Enable disk persistence
            persistence_path: Path for persistence
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.ttl = ttl
        self.enable_persistence = enable_persistence
        self.persistence_path = persistence_path or Path("cache")
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_patterns: Dict[str, float] = {}  # For prediction
        self.total_memory_mb = 0.0
        
        if enable_persistence:
            self.persistence_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items()),
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(
        self,
        key: Optional[str] = None,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Optional explicit key
            *args, **kwargs: Arguments to generate key if key not provided
            
        Returns:
            Cached value or None
        """
        if key is None:
            key = self._generate_key(*args, **kwargs)
        
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if self.ttl and (time.time() - entry.timestamp) > self.ttl:
            del self.cache[key]
            self.total_memory_mb -= entry.size / (1024 * 1024)
            return None
        
        # Update access info
        entry.access_count += 1
        entry.last_access = time.time()
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        
        # Update access pattern for prediction
        self.access_patterns[key] = time.time()
        
        return entry.value
    
    def set(
        self,
        value: Any,
        key: Optional[str] = None,
        tags: Optional[list] = None,
        *args,
        **kwargs
    ) -> str:
        """
        Set value in cache.
        
        Args:
            value: Value to cache
            key: Optional explicit key
            tags: Optional tags for categorization
            *args, **kwargs: Arguments to generate key if key not provided
            
        Returns:
            Cache key
        """
        if key is None:
            key = self._generate_key(*args, **kwargs)
        
        # Calculate size (approximate)
        try:
            size = len(pickle.dumps(value))
        except Exception:
            size = 1024  # Default estimate
        
        size_mb = size / (1024 * 1024)
        
        # Remove if exists
        if key in self.cache:
            old_entry = self.cache[key]
            self.total_memory_mb -= old_entry.size / (1024 * 1024)
        
        # Check if we need to evict
        while (
            len(self.cache) >= self.max_size or
            self.total_memory_mb + size_mb > self.max_memory_mb
        ):
            self._evict_least_valuable()
        
        # Create entry
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            size=size,
            tags=tags or [],
        )
        
        self.cache[key] = entry
        self.total_memory_mb += size_mb
        
        # Persist if enabled
        if self.enable_persistence:
            self._save_to_disk(key, value)
        
        return key
    
    def _evict_least_valuable(self) -> None:
        """Evict least valuable entry using intelligent algorithm."""
        if not self.cache:
            return
        
        # Calculate value score for each entry
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Score based on:
            # - Recency (more recent = higher score)
            # - Frequency (more accesses = higher score)
            # - Size (smaller = higher score, easier to replace)
            
            recency_score = 1.0 / (1.0 + (current_time - entry.last_access))
            frequency_score = entry.access_count / 10.0  # Normalize
            size_score = 1.0 / (1.0 + entry.size / (1024 * 1024))  # Prefer smaller
            
            # Weighted combination
            score = (
                recency_score * 0.4 +
                frequency_score * 0.4 +
                size_score * 0.2
            )
            
            scores[key] = score
        
        # Evict lowest score
        if scores:
            least_valuable = min(scores.items(), key=lambda x: x[1])[0]
            entry = self.cache[least_valuable]
            del self.cache[least_valuable]
            self.total_memory_mb -= entry.size / (1024 * 1024)
            logger.debug(f"Evicted cache entry: {least_valuable}")
    
    def predict_access(self, key: str) -> float:
        """
        Predict likelihood of access for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Access probability (0.0 to 1.0)
        """
        if key not in self.access_patterns:
            return 0.0
        
        # Simple prediction based on access pattern
        last_access = self.access_patterns[key]
        time_since_access = time.time() - last_access
        
        # Exponential decay
        probability = 1.0 / (1.0 + time_since_access / 3600.0)  # Decay over 1 hour
        
        return min(1.0, probability)
    
    def get_by_tags(self, tags: list) -> Dict[str, Any]:
        """
        Get entries by tags.
        
        Args:
            tags: List of tags
            
        Returns:
            Dictionary of matching entries
        """
        results = {}
        for key, entry in self.cache.items():
            if any(tag in entry.tags for tag in tags):
                results[key] = entry.value
        return results
    
    def clear_tags(self, tags: list) -> int:
        """
        Clear entries with specific tags.
        
        Args:
            tags: Tags to clear
            
        Returns:
            Number of entries cleared
        """
        to_remove = []
        for key, entry in self.cache.items():
            if any(tag in entry.tags for tag in tags):
                to_remove.append(key)
        
        for key in to_remove:
            entry = self.cache[key]
            del self.cache[key]
            self.total_memory_mb -= entry.size / (1024 * 1024)
        
        return len(to_remove)
    
    def _save_to_disk(self, key: str, value: Any) -> None:
        """Save entry to disk."""
        try:
            file_path = self.persistence_path / f"{key}.pkl"
            with open(file_path, "wb") as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to persist cache entry: {e}")
    
    def _load_from_disk(self) -> None:
        """Load entries from disk."""
        if not self.persistence_path.exists():
            return
        
        try:
            for file_path in self.persistence_path.glob("*.pkl"):
                key = file_path.stem
                with open(file_path, "rb") as f:
                    value = pickle.load(f)
                # Don't load all at once, just mark as available
                # Could implement lazy loading here
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        avg_accesses = (
            total_accesses / len(self.cache)
            if self.cache else 0.0
        )
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "memory_mb": self.total_memory_mb,
            "max_memory_mb": self.max_memory_mb,
            "total_accesses": total_accesses,
            "avg_accesses_per_entry": avg_accesses,
            "hit_rate": 0.0,  # Would need to track hits/misses
        }
    
    def clear(self) -> None:
        """Clear all cache."""
        self.cache.clear()
        self.total_memory_mb = 0.0
        self.access_patterns.clear()
        
        if self.enable_persistence and self.persistence_path.exists():
            for file_path in self.persistence_path.glob("*.pkl"):
                file_path.unlink()
        
        logger.info("Cache cleared")


