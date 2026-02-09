"""
Distributed Cache V2
====================

Advanced distributed caching with consistency and replication.
"""

import time
import hashlib
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy."""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    TTL = "ttl"


class ConsistencyLevel(Enum):
    """Consistency level."""
    EVENTUAL = "eventual"
    STRONG = "strong"
    WEAK = "weak"


@dataclass
class CacheNode:
    """Cache node."""
    id: str
    host: str
    port: int
    weight: int = 1
    active: bool = True


@dataclass
class CacheEntry:
    """Cache entry."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None
    version: int = 1
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class DistributedCacheV2:
    """Advanced distributed cache."""
    
    def __init__(
        self,
        nodes: List[CacheNode],
        strategy: CacheStrategy = CacheStrategy.LRU,
        consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL,
        replication_factor: int = 2,
    ):
        """
        Initialize distributed cache.
        
        Args:
            nodes: List of cache nodes
            strategy: Cache strategy
            consistency_level: Consistency level
            replication_factor: Number of replicas
        """
        self.nodes = nodes
        self.strategy = strategy
        self.consistency_level = consistency_level
        self.replication_factor = replication_factor
        
        # Local cache (simplified - in real implementation would use actual distributed cache)
        self.local_cache: Dict[str, CacheEntry] = {}
        self.node_assignments: Dict[str, List[str]] = {}
    
    def _hash_key(self, key: str) -> int:
        """Hash key for node assignment."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def _get_nodes_for_key(self, key: str) -> List[CacheNode]:
        """
        Get nodes for a key (consistent hashing).
        
        Args:
            key: Cache key
            
        Returns:
            List of nodes
        """
        active_nodes = [n for n in self.nodes if n.active]
        if not active_nodes:
            return []
        
        # Simple consistent hashing
        hash_value = self._hash_key(key)
        node_index = hash_value % len(active_nodes)
        
        # Get primary and replica nodes
        selected_nodes = []
        for i in range(min(self.replication_factor, len(active_nodes))):
            node = active_nodes[(node_index + i) % len(active_nodes)]
            selected_nodes.append(node)
        
        return selected_nodes
    
    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        # Check local cache first
        if key in self.local_cache:
            entry = self.local_cache[key]
            
            if entry.is_expired():
                del self.local_cache[key]
                return default
            
            # Update access info
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            return entry.value
        
        # In real implementation, would query distributed nodes
        return default
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
    ) -> None:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional time to live in seconds
        """
        nodes = self._get_nodes_for_key(key)
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            last_accessed=time.time(),
            ttl=ttl,
        )
        
        # Store locally (simplified)
        self.local_cache[key] = entry
        
        # In real implementation, would replicate to nodes
        self.node_assignments[key] = [n.id for n in nodes]
        
        logger.debug(f"Cached key: {key} on {len(nodes)} nodes")
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if deleted
        """
        if key not in self.local_cache:
            return False
        
        del self.local_cache[key]
        self.node_assignments.pop(key, None)
        
        # In real implementation, would delete from all nodes
        return True
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.local_cache.clear()
        self.node_assignments.clear()
        logger.info("Cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        active_nodes = [n for n in self.nodes if n.active]
        
        return {
            "total_entries": len(self.local_cache),
            "active_nodes": len(active_nodes),
            "total_nodes": len(self.nodes),
            "strategy": self.strategy.value,
            "consistency_level": self.consistency_level.value,
            "replication_factor": self.replication_factor,
        }

