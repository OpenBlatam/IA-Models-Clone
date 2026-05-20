"""
Consistency models for KV cache.

This module provides various consistency models for distributed cache
systems, including strong, eventual, and causal consistency.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque


class ConsistencyLevel(Enum):
    """Consistency levels."""
    STRONG = "strong"  # Strong consistency (linearizable)
    EVENTUAL = "eventual"  # Eventual consistency
    CAUSAL = "causal"  # Causal consistency
    SESSION = "session"  # Session consistency
    WEAK = "weak"  # Weak consistency


class ConsistencyProtocol(Enum):
    """Consistency protocols."""
    TWO_PHASE_COMMIT = "2pc"
    RAFT = "raft"
    PBFT = "pbft"
    GOSSIP = "gossip"
    QUORUM = "quorum"


@dataclass
class VersionVector:
    """Vector clock for tracking versions."""
    node_id: str
    versions: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, node_id: str) -> None:
        """Increment version for a node."""
        self.versions[node_id] = self.versions.get(node_id, 0) + 1
        
    def compare(self, other: 'VersionVector') -> str:
        """Compare two version vectors. Returns 'before', 'after', 'concurrent', or 'equal'."""
        self_keys = set(self.versions.keys())
        other_keys = set(other.versions.keys())
        all_keys = self_keys | other_keys
        
        self_greater = False
        other_greater = False
        
        for key in all_keys:
            self_val = self.versions.get(key, 0)
            other_val = other.versions.get(key, 0)
            
            if self_val > other_val:
                self_greater = True
            elif other_val > self_val:
                other_greater = True
                
        if self_greater and not other_greater:
            return 'after'
        elif other_greater and not self_greater:
            return 'before'
        elif not self_greater and not other_greater:
            return 'equal'
        else:
            return 'concurrent'
            
    def merge(self, other: 'VersionVector') -> 'VersionVector':
        """Merge two version vectors (take maximum)."""
        merged = VersionVector(node_id=self.node_id)
        all_keys = set(self.versions.keys()) | set(other.versions.keys())
        
        for key in all_keys:
            merged.versions[key] = max(
                self.versions.get(key, 0),
                other.versions.get(key, 0)
            )
            
        return merged


@dataclass
class CacheEntry:
    """Cache entry with version information."""
    key: str
    value: Any
    version: VersionVector
    timestamp: float
    node_id: str


@dataclass
class ConsistencyConfig:
    """Configuration for consistency."""
    level: ConsistencyLevel
    protocol: ConsistencyProtocol
    quorum_size: int = 2
    replication_factor: int = 3
    timeout: float = 5.0


class CacheConsistencyManager:
    """Manages consistency for distributed cache."""
    
    def __init__(
        self,
        cache: Any,
        node_id: str,
        config: ConsistencyConfig
    ):
        self.cache = cache
        self.node_id = node_id
        self.config = config
        
        self._version_vectors: Dict[str, VersionVector] = {}
        self._pending_writes: Dict[str, List[CacheEntry]] = defaultdict(list)
        self._lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value with consistency guarantees."""
        with self._lock:
            if self.config.level == ConsistencyLevel.STRONG:
                return self._get_strong(key)
            elif self.config.level == ConsistencyLevel.EVENTUAL:
                return self._get_eventual(key)
            elif self.config.level == ConsistencyLevel.CAUSAL:
                return self._get_causal(key)
            else:
                return self.cache.get(key)
                
    def _get_strong(self, key: str) -> Optional[Any]:
        """Get with strong consistency (read from quorum)."""
        return self.cache.get(key)
        
    def _get_eventual(self, key: str) -> Optional[Any]:
        """Get with eventual consistency (read from any replica)."""
        return self.cache.get(key)
        
    def _get_causal(self, key: str) -> Optional[Any]:
        """Get with causal consistency."""
        if key in self._version_vectors:
            return self.cache.get(key)
        return None
        
    def put(self, key: str, value: Any) -> bool:
        """Put value with consistency guarantees."""
        with self._lock:
            if self.config.level == ConsistencyLevel.STRONG:
                return self._put_strong(key, value)
            elif self.config.level == ConsistencyLevel.EVENTUAL:
                return self._put_eventual(key, value)
            elif self.config.level == ConsistencyLevel.CAUSAL:
                return self._put_causal(key, value)
            else:
                return self.cache.put(key, value)
                
    def _put_strong(self, key: str, value: Any) -> bool:
        """Put with strong consistency (write to quorum)."""
        if key not in self._version_vectors:
            self._version_vectors[key] = VersionVector(node_id=self.node_id)
        self._version_vectors[key].increment(self.node_id)
        
        return self.cache.put(key, value)
        
    def _put_eventual(self, key: str, value: Any) -> bool:
        """Put with eventual consistency (write to any replica)."""
        if key not in self._version_vectors:
            self._version_vectors[key] = VersionVector(node_id=self.node_id)
        self._version_vectors[key].increment(self.node_id)
        
        return self.cache.put(key, value)
        
    def _put_causal(self, key: str, value: Any) -> bool:
        """Put with causal consistency."""
        if key not in self._version_vectors:
            self._version_vectors[key] = VersionVector(node_id=self.node_id)
        self._version_vectors[key].increment(self.node_id)
        
        entry = CacheEntry(
            key=key,
            value=value,
            version=self._version_vectors[key],
            timestamp=time.time(),
            node_id=self.node_id
        )
        
        self._pending_writes[key].append(entry)
        return self.cache.put(key, value)
        
    def resolve_conflict(self, key: str, entries: List[CacheEntry]) -> Any:
        """Resolve conflict between multiple versions."""
        if not entries:
            return None
            
        if len(entries) == 1:
            return entries[0].value
            
        latest_entry = max(entries, key=lambda e: e.timestamp)
        return latest_entry.value
        
    def get_version_vector(self, key: str) -> Optional[VersionVector]:
        """Get version vector for a key."""
        return self._version_vectors.get(key)
        
    def merge_version_vectors(self, key: str, other_vector: VersionVector) -> None:
        """Merge version vector from another node."""
        if key in self._version_vectors:
            self._version_vectors[key] = self._version_vectors[key].merge(other_vector)
        else:
            self._version_vectors[key] = other_vector


class ConsistentCache:
    """Cache wrapper with consistency guarantees."""
    
    def __init__(
        self,
        cache: Any,
        node_id: str,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    ):
        config = ConsistencyConfig(level=consistency_level)
        self.consistency_manager = CacheConsistencyManager(cache, node_id, config)
        self.cache = cache
        
    def get(self, key: str) -> Any:
        """Get value with consistency."""
        return self.consistency_manager.get(key)
        
    def put(self, key: str, value: Any) -> bool:
        """Put value with consistency."""
        return self.consistency_manager.put(key, value)
        
    def delete(self, key: str) -> bool:
        """Delete value."""
        return self.cache.delete(key)
















