"""
Advanced indexing system for KV cache.

This module provides indexing capabilities for efficient querying
and searching within the cache.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import bisect


class IndexType(Enum):
    """Types of indexes."""
    HASH = "hash"  # Hash index
    BTREE = "btree"  # B-tree index
    SORTED = "sorted"  # Sorted index
    FULLTEXT = "fulltext"  # Full-text index
    SPATIAL = "spatial"  # Spatial index
    COMPOSITE = "composite"  # Composite index


@dataclass
class IndexDefinition:
    """Definition of an index."""
    index_id: str
    index_type: IndexType
    fields: List[str]
    unique: bool = False
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HashIndex:
    """Hash-based index."""
    
    def __init__(self, field: str):
        self.field = field
        self._index: Dict[Any, Set[str]] = defaultdict(set)
        
    def add(self, key: str, value: Any) -> None:
        """Add entry to index."""
        field_value = self._extract_field(value)
        self._index[field_value].add(key)
        
    def remove(self, key: str, value: Any) -> None:
        """Remove entry from index."""
        field_value = self._extract_field(value)
        if field_value in self._index:
            self._index[field_value].discard(key)
            if not self._index[field_value]:
                del self._index[field_value]
                
    def find(self, value: Any) -> Set[str]:
        """Find keys matching value."""
        return self._index.get(value, set())
        
    def _extract_field(self, value: Any) -> Any:
        """Extract field value from object."""
        if isinstance(value, dict):
            return value.get(self.field)
        elif hasattr(value, self.field):
            return getattr(value, self.field)
        return None


class SortedIndex:
    """Sorted index for range queries."""
    
    def __init__(self, field: str):
        self.field = field
        self._sorted_keys: List[Tuple[Any, str]] = []
        self._lock = threading.Lock()
        
    def add(self, key: str, value: Any) -> None:
        """Add entry to index."""
        field_value = self._extract_field(value)
        with self._lock:
            self._remove_key(key)
            bisect.insort(self._sorted_keys, (field_value, key))
            
    def remove(self, key: str, value: Any) -> None:
        """Remove entry from index."""
        with self._lock:
            self._remove_key(key)
            
    def _remove_key(self, key: str) -> None:
        """Remove key from index."""
        self._sorted_keys = [kv for kv in self._sorted_keys if kv[1] != key]
        
    def range_query(self, start: Any, end: Any) -> List[str]:
        """Find keys in range [start, end)."""
        with self._lock:
            result = []
            for field_value, key in self._sorted_keys:
                if start <= field_value < end:
                    result.append(key)
            return result
            
    def _extract_field(self, value: Any) -> Any:
        """Extract field value from object."""
        if isinstance(value, dict):
            return value.get(self.field)
        elif hasattr(value, self.field):
            return getattr(value, self.field)
        return None


class FullTextIndex:
    """Full-text search index."""
    
    def __init__(self, field: str):
        self.field = field
        self._inverted_index: Dict[str, Set[str]] = defaultdict(set)
        
    def add(self, key: str, value: Any) -> None:
        """Add entry to index."""
        text = self._extract_text(value)
        if text:
            words = self._tokenize(text)
            for word in words:
                self._inverted_index[word].add(key)
                
    def remove(self, key: str, value: Any) -> None:
        """Remove entry from index."""
        text = self._extract_text(value)
        if text:
            words = self._tokenize(text)
            for word in words:
                if word in self._inverted_index:
                    self._inverted_index[word].discard(key)
                    if not self._inverted_index[word]:
                        del self._inverted_index[word]
                        
    def search(self, query: str) -> Set[str]:
        """Search for keys matching query."""
        words = self._tokenize(query)
        if not words:
            return set()
            
        result = self._inverted_index.get(words[0], set())
        for word in words[1:]:
            result = result & self._inverted_index.get(word, set())
            
        return result
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return text.lower().split()
        
    def _extract_text(self, value: Any) -> str:
        """Extract text from value."""
        if isinstance(value, dict):
            return str(value.get(self.field, ''))
        elif hasattr(value, self.field):
            return str(getattr(value, self.field))
        return str(value)


class CacheIndexManager:
    """Manages indexes for cache."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self._indexes: Dict[str, Any] = {}
        self._index_definitions: Dict[str, IndexDefinition] = {}
        self._lock = threading.Lock()
        
    def create_index(
        self,
        index_id: str,
        index_type: IndexType,
        fields: List[str],
        unique: bool = False
    ) -> IndexDefinition:
        """Create a new index."""
        with self._lock:
            if index_id in self._indexes:
                raise ValueError(f"Index '{index_id}' already exists")
                
            definition = IndexDefinition(
                index_id=index_id,
                index_type=index_type,
                fields=fields,
                unique=unique
            )
            
            if index_type == IndexType.HASH:
                index = HashIndex(fields[0])
            elif index_type == IndexType.SORTED:
                index = SortedIndex(fields[0])
            elif index_type == IndexType.FULLTEXT:
                index = FullTextIndex(fields[0])
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
                
            self._indexes[index_id] = index
            self._index_definitions[index_id] = definition
            
            self._build_index(index_id)
            
            return definition
            
    def _build_index(self, index_id: str) -> None:
        """Build index from existing cache entries."""
        index = self._indexes[index_id]
        
        if hasattr(self.cache, '_cache'):
            for key, value in self.cache._cache.items():
                index.add(key, value)
                
    def drop_index(self, index_id: str) -> bool:
        """Drop an index."""
        with self._lock:
            if index_id in self._indexes:
                del self._indexes[index_id]
                del self._index_definitions[index_id]
                return True
            return False
            
    def get_index(self, index_id: str) -> Optional[Any]:
        """Get index by ID."""
        return self._indexes.get(index_id)
        
    def list_indexes(self) -> List[str]:
        """List all index IDs."""
        return list(self._indexes.keys())
        
    def update_index(self, key: str, old_value: Any, new_value: Any) -> None:
        """Update index when cache entry changes."""
        with self._lock:
            for index in self._indexes.values():
                if old_value is not None:
                    index.remove(key, old_value)
                if new_value is not None:
                    index.add(key, new_value)
                    
    def query_index(
        self,
        index_id: str,
        query: Any,
        query_type: str = "exact"
    ) -> Set[str]:
        """Query an index."""
        index = self._indexes.get(index_id)
        if not index:
            return set()
            
        if isinstance(index, HashIndex):
            if query_type == "exact":
                return index.find(query)
        elif isinstance(index, SortedIndex):
            if query_type == "range" and isinstance(query, tuple):
                start, end = query
                return set(index.range_query(start, end))
        elif isinstance(index, FullTextIndex):
            if query_type == "search":
                return index.search(query)
                
        return set()


class IndexedCache:
    """Cache wrapper with indexing."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self.index_manager = CacheIndexManager(cache)
        
    def create_index(
        self,
        index_id: str,
        index_type: IndexType,
        fields: List[str],
        unique: bool = False
    ) -> IndexDefinition:
        """Create an index."""
        return self.index_manager.create_index(index_id, index_type, fields, unique)
        
    def get(self, key: str) -> Any:
        """Get value."""
        return self.cache.get(key)
        
    def put(self, key: str, value: Any) -> bool:
        """Put value and update indexes."""
        old_value = self.cache.get(key)
        result = self.cache.put(key, value)
        
        if result:
            self.index_manager.update_index(key, old_value, value)
            
        return result
        
    def delete(self, key: str) -> bool:
        """Delete value and update indexes."""
        old_value = self.cache.get(key)
        result = self.cache.delete(key)
        
        if result:
            self.index_manager.update_index(key, old_value, None)
            
        return result
        
    def query_index(
        self,
        index_id: str,
        query: Any,
        query_type: str = "exact"
    ) -> Set[str]:
        """Query an index."""
        return self.index_manager.query_index(index_id, query, query_type)
















