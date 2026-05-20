"""
Cache coherence system for KV cache.

This module provides cache coherence protocols for maintaining
consistency across distributed cache instances.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class CoherenceProtocol(Enum):
    """Cache coherence protocols."""
    MESI = "mesi"  # Modified, Exclusive, Shared, Invalid
    MOESI = "moesi"  # Modified, Owned, Exclusive, Shared, Invalid
    MSI = "msi"  # Modified, Shared, Invalid
    WRITE_INVALIDATE = "write_invalidate"
    WRITE_UPDATE = "write_update"


class CacheState(Enum):
    """Cache line states."""
    MODIFIED = "modified"
    OWNED = "owned"
    EXCLUSIVE = "exclusive"
    SHARED = "shared"
    INVALID = "invalid"


@dataclass
class CacheLine:
    """A cache line with coherence information."""
    key: str
    value: Any
    state: CacheState
    owner: Optional[str] = None
    sharers: Set[str] = field(default_factory=set)
    last_modified: float = field(default_factory=time.time)


class CacheCoherenceManager:
    """Manages cache coherence."""
    
    def __init__(
        self,
        cache: Any,
        node_id: str,
        protocol: CoherenceProtocol = CoherenceProtocol.MESI
    ):
        self.cache = cache
        self.node_id = node_id
        self.protocol = protocol
        self._cache_lines: Dict[str, CacheLine] = {}
        self._other_nodes: Set[str] = set()
        self._lock = threading.Lock()
        
    def get(self, key: str) -> Any:
        """Get value with coherence protocol."""
        with self._lock:
            if key in self._cache_lines:
                line = self._cache_lines[key]
                
                # Check state
                if line.state == CacheState.INVALID:
                    # Need to fetch from other node or memory
                    return self._fetch_from_other(key)
                elif line.state in [CacheState.SHARED, CacheState.EXCLUSIVE, CacheState.MODIFIED, CacheState.OWNED]:
                    # Valid state, return value
                    return line.value
            else:
                # Not in cache, fetch
                return self._fetch_from_other(key)
                
    def put(self, key: str, value: Any) -> bool:
        """Put value with coherence protocol."""
        with self._lock:
            if key in self._cache_lines:
                line = self._cache_lines[key]
                
                if self.protocol == CoherenceProtocol.WRITE_INVALIDATE:
                    # Invalidate other copies
                    self._invalidate_others(key)
                    line.state = CacheState.MODIFIED
                elif self.protocol == CoherenceProtocol.WRITE_UPDATE:
                    # Update other copies
                    self._update_others(key, value)
                    line.state = CacheState.MODIFIED
                elif self.protocol == CoherenceProtocol.MESI:
                    if line.state == CacheState.SHARED:
                        # Need exclusive access
                        self._invalidate_others(key)
                        line.state = CacheState.MODIFIED
                    elif line.state == CacheState.EXCLUSIVE:
                        line.state = CacheState.MODIFIED
                    # If already MODIFIED, just update
                    
                line.value = value
                line.last_modified = time.time()
            else:
                # New cache line
                line = CacheLine(
                    key=key,
                    value=value,
                    state=CacheState.EXCLUSIVE if not self._other_nodes else CacheState.MODIFIED,
                    owner=self.node_id
                )
                self._cache_lines[key] = line
                
                # Invalidate others for write
                if self._other_nodes:
                    self._invalidate_others(key)
                    
            return self.cache.put(key, value)
            
    def _fetch_from_other(self, key: str) -> Any:
        """Fetch value from other node."""
        # In real implementation, would make network call
        # For now, try to get from local cache
        value = self.cache.get(key)
        
        if value is not None:
            # Update cache line state
            if key not in self._cache_lines:
                self._cache_lines[key] = CacheLine(
                    key=key,
                    value=value,
                    state=CacheState.SHARED if self._other_nodes else CacheState.EXCLUSIVE
                )
            else:
                self._cache_lines[key].state = CacheState.SHARED
                self._cache_lines[key].value = value
                
        return value
        
    def _invalidate_others(self, key: str) -> None:
        """Invalidate copies in other nodes."""
        # In real implementation, would send invalidation messages
        pass
        
    def _update_others(self, key: str, value: Any) -> None:
        """Update copies in other nodes."""
        # In real implementation, would send update messages
        pass
        
    def invalidate(self, key: str) -> None:
        """Invalidate cache line (received from other node)."""
        with self._lock:
            if key in self._cache_lines:
                self._cache_lines[key].state = CacheState.INVALID
                # Optionally remove from cache
                # self.cache.delete(key)
                
    def add_node(self, node_id: str) -> None:
        """Add another node to coherence protocol."""
        with self._lock:
            self._other_nodes.add(node_id)
            
    def remove_node(self, node_id: str) -> None:
        """Remove node from coherence protocol."""
        with self._lock:
            self._other_nodes.discard(node_id)
            
    def get_cache_state(self, key: str) -> Optional[CacheState]:
        """Get cache line state."""
        with self._lock:
            if key in self._cache_lines:
                return self._cache_lines[key].state
            return None


