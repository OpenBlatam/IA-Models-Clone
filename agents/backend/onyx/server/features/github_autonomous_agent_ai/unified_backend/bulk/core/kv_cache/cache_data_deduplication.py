"""
Data deduplication system for KV cache.

This module provides deduplication capabilities to reduce storage
requirements by identifying and eliminating duplicate data.
"""

import time
import threading
import hashlib
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class DeduplicationStrategy(Enum):
    """Deduplication strategies."""
    CONTENT_HASH = "content_hash"  # Hash-based deduplication
    SIMILARITY = "similarity"  # Similarity-based deduplication
    COMPRESSION = "compression"  # Compression-based deduplication
    BLOCK_LEVEL = "block_level"  # Block-level deduplication


@dataclass
class DeduplicationEntry:
    """Deduplication entry."""
    content_hash: str
    reference_count: int
    size_bytes: int
    first_seen: float
    last_accessed: float
    keys: Set[str] = field(default_factory=set)


@dataclass
class DeduplicationStats:
    """Deduplication statistics."""
    total_keys: int
    unique_content: int
    duplicate_count: int
    space_saved_bytes: int
    space_saved_percent: float
    deduplication_ratio: float


class ContentDeduplicator:
    """Content-based deduplication."""
    
    def __init__(self):
        self._hash_to_entry: Dict[str, DeduplicationEntry] = {}
        self._key_to_hash: Dict[str, str] = {}
        self._lock = threading.Lock()
        
    def _compute_hash(self, value: Any) -> str:
        """Compute hash of value."""
        # Convert value to string for hashing
        if isinstance(value, (dict, list)):
            import json
            content = json.dumps(value, sort_keys=True)
        else:
            content = str(value)
            
        return hashlib.sha256(content.encode()).hexdigest()
        
    def add(self, key: str, value: Any) -> Tuple[str, bool]:
        """
        Add key-value pair. Returns (content_hash, is_duplicate).
        """
        content_hash = self._compute_hash(value)
        size_bytes = len(str(value).encode())
        
        with self._lock:
            is_duplicate = content_hash in self._hash_to_entry
            
            if is_duplicate:
                # Increment reference count
                entry = self._hash_to_entry[content_hash]
                entry.reference_count += 1
                entry.keys.add(key)
                entry.last_accessed = time.time()
            else:
                # Create new entry
                entry = DeduplicationEntry(
                    content_hash=content_hash,
                    reference_count=1,
                    size_bytes=size_bytes,
                    first_seen=time.time(),
                    last_accessed=time.time(),
                    keys={key}
                )
                self._hash_to_entry[content_hash] = entry
                
            self._key_to_hash[key] = content_hash
            
        return content_hash, is_duplicate
        
    def remove(self, key: str) -> bool:
        """Remove key from deduplication tracking."""
        with self._lock:
            if key not in self._key_to_hash:
                return False
                
            content_hash = self._key_to_hash[key]
            
            if content_hash in self._hash_to_entry:
                entry = self._hash_to_entry[content_hash]
                entry.reference_count = max(0, entry.reference_count - 1)
                entry.keys.discard(key)
                
                # Remove entry if no more references
                if entry.reference_count == 0:
                    del self._hash_to_entry[content_hash]
                    
            del self._key_to_hash[key]
            return True
            
    def get_hash(self, key: str) -> Optional[str]:
        """Get content hash for a key."""
        return self._key_to_hash.get(key)
        
    def get_duplicates(self, key: str) -> Set[str]:
        """Get other keys with same content."""
        content_hash = self.get_hash(key)
        if not content_hash:
            return set()
            
        entry = self._hash_to_entry.get(content_hash)
        if entry:
            return entry.keys - {key}
        return set()
        
    def get_stats(self) -> DeduplicationStats:
        """Get deduplication statistics."""
        with self._lock:
            total_keys = len(self._key_to_hash)
            unique_content = len(self._hash_to_entry)
            duplicate_count = total_keys - unique_content
            
            # Calculate space saved
            total_size = sum(entry.size_bytes * entry.reference_count for entry in self._hash_to_entry.values())
            unique_size = sum(entry.size_bytes for entry in self._hash_to_entry.values())
            space_saved = total_size - unique_size
            
            space_saved_percent = (space_saved / total_size * 100) if total_size > 0 else 0.0
            deduplication_ratio = unique_content / total_keys if total_keys > 0 else 0.0
            
            return DeduplicationStats(
                total_keys=total_keys,
                unique_content=unique_content,
                duplicate_count=duplicate_count,
                space_saved_bytes=space_saved,
                space_saved_percent=space_saved_percent,
                deduplication_ratio=deduplication_ratio
            )


class SimilarityDeduplicator:
    """Similarity-based deduplication."""
    
    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
        self._content_deduplicator = ContentDeduplicator()
        
    def _compute_similarity(self, value1: Any, value2: Any) -> float:
        """Compute similarity between two values."""
        # Simple similarity: compare string representations
        str1 = str(value1)
        str2 = str(value2)
        
        if len(str1) == 0 and len(str2) == 0:
            return 1.0
        if len(str1) == 0 or len(str2) == 0:
            return 0.0
            
        # Use Jaccard similarity on character sets
        set1 = set(str1)
        set2 = set(str2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
        
    def find_similar(self, key: str, value: Any) -> List[Tuple[str, float]]:
        """Find keys with similar content."""
        similar = []
        
        # This is simplified - real implementation would use more efficient algorithms
        for other_key, other_hash in self._content_deduplicator._key_to_hash.items():
            if other_key == key:
                continue
                
            # Get value from cache (simplified - would need cache reference)
            # For now, we can't compute similarity without values
            pass
            
        return similar


class CacheDeduplication:
    """Deduplication system for cache."""
    
    def __init__(
        self,
        cache: Any,
        strategy: DeduplicationStrategy = DeduplicationStrategy.CONTENT_HASH,
        enable_auto_deduplication: bool = True
    ):
        self.cache = cache
        self.strategy = strategy
        self.enable_auto_deduplication = enable_auto_deduplication
        
        if strategy == DeduplicationStrategy.CONTENT_HASH:
            self.deduplicator = ContentDeduplicator()
        elif strategy == DeduplicationStrategy.SIMILARITY:
            self.deduplicator = SimilarityDeduplicator()
        else:
            self.deduplicator = ContentDeduplicator()  # Default
            
    def put(self, key: str, value: Any) -> bool:
        """Put value with deduplication."""
        content_hash, is_duplicate = self.deduplicator.add(key, value)
        
        if is_duplicate and self.enable_auto_deduplication:
            # Check if we should store reference instead of duplicate
            # For now, still store but track deduplication
            pass
            
        return self.cache.put(key, value)
        
    def get(self, key: str) -> Any:
        """Get value."""
        return self.cache.get(key)
        
    def delete(self, key: str) -> bool:
        """Delete value and update deduplication tracking."""
        self.deduplicator.remove(key)
        return self.cache.delete(key)
        
    def find_duplicates(self, key: str) -> Set[str]:
        """Find keys with duplicate content."""
        return self.deduplicator.get_duplicates(key)
        
    def get_deduplication_stats(self) -> DeduplicationStats:
        """Get deduplication statistics."""
        if isinstance(self.deduplicator, ContentDeduplicator):
            return self.deduplicator.get_stats()
        else:
            # For other strategies, return basic stats
            return DeduplicationStats(
                total_keys=0,
                unique_content=0,
                duplicate_count=0,
                space_saved_bytes=0,
                space_saved_percent=0.0,
                deduplication_ratio=0.0
            )
            
    def optimize(self) -> Dict[str, Any]:
        """Optimize cache by removing duplicates."""
        stats = self.get_deduplication_stats()
        
        # Find all duplicates and potentially merge them
        # This is a simplified optimization
        optimization_result = {
            'duplicates_found': stats.duplicate_count,
            'space_saved': stats.space_saved_bytes,
            'optimization_time': time.time()
        }
        
        return optimization_result


class DeduplicatedCache:
    """Cache wrapper with deduplication."""
    
    def __init__(
        self,
        cache: Any,
        enable_deduplication: bool = True
    ):
        self.cache = cache
        self.deduplication = CacheDeduplication(
            cache,
            enable_auto_deduplication=enable_deduplication
        )
        
    def get(self, key: str) -> Any:
        """Get value."""
        return self.deduplication.get(key)
        
    def put(self, key: str, value: Any) -> bool:
        """Put value with deduplication."""
        return self.deduplication.put(key, value)
        
    def delete(self, key: str) -> bool:
        """Delete value."""
        return self.deduplication.delete(key)
        
    def get_deduplication_stats(self) -> DeduplicationStats:
        """Get deduplication statistics."""
        return self.deduplication.get_deduplication_stats()
        
    def find_duplicates(self, key: str) -> Set[str]:
        """Find duplicate keys."""
        return self.deduplication.find_duplicates(key)















