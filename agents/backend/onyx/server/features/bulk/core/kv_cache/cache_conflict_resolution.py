"""
Conflict resolution system for KV cache.

This module provides conflict resolution strategies for distributed
cache systems where multiple versions of the same key may exist.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum


class ConflictResolutionStrategy(Enum):
    """Conflict resolution strategies."""
    LAST_WRITE_WINS = "last_write_wins"  # Use most recent timestamp
    FIRST_WRITE_WINS = "first_write_wins"  # Use oldest timestamp
    HIGHEST_VERSION = "highest_version"  # Use highest version number
    MERGE = "merge"  # Merge values if possible
    CUSTOM = "custom"  # Custom resolution function


@dataclass
class ConflictEntry:
    """An entry in conflict."""
    key: str
    value: Any
    version: int
    timestamp: float
    node_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConflictResolution:
    """Result of conflict resolution."""
    resolved_value: Any
    strategy_used: ConflictResolutionStrategy
    resolved_at: float
    conflict_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConflictResolver:
    """Resolves conflicts between cache entries."""
    
    def __init__(
        self,
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS,
        custom_resolver: Optional[Callable[[List[ConflictEntry]], Any]] = None
    ):
        self.strategy = strategy
        self.custom_resolver = custom_resolver
        self._resolution_history: List[ConflictResolution] = []
        self._lock = threading.Lock()
        
    def resolve(self, entries: List[ConflictEntry]) -> ConflictResolution:
        """Resolve conflict between multiple entries."""
        if not entries:
            raise ValueError("No entries to resolve")
            
        if len(entries) == 1:
            return ConflictResolution(
                resolved_value=entries[0].value,
                strategy_used=self.strategy,
                resolved_at=time.time(),
                conflict_count=1
            )
            
        # Resolve based on strategy
        if self.strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            resolved_entry = max(entries, key=lambda e: e.timestamp)
        elif self.strategy == ConflictResolutionStrategy.FIRST_WRITE_WINS:
            resolved_entry = min(entries, key=lambda e: e.timestamp)
        elif self.strategy == ConflictResolutionStrategy.HIGHEST_VERSION:
            resolved_entry = max(entries, key=lambda e: e.version)
        elif self.strategy == ConflictResolutionStrategy.MERGE:
            resolved_entry = self._merge_entries(entries)
        elif self.strategy == ConflictResolutionStrategy.CUSTOM:
            if self.custom_resolver:
                resolved_value = self.custom_resolver(entries)
                resolved_entry = ConflictEntry(
                    key=entries[0].key,
                    value=resolved_value,
                    version=max(e.version for e in entries),
                    timestamp=time.time(),
                    node_id="resolver"
                )
            else:
                raise ValueError("Custom resolver not provided")
        else:
            # Default to last write wins
            resolved_entry = max(entries, key=lambda e: e.timestamp)
            
        resolution = ConflictResolution(
            resolved_value=resolved_entry.value,
            strategy_used=self.strategy,
            resolved_at=time.time(),
            conflict_count=len(entries),
            metadata={
                'resolved_entry': {
                    'version': resolved_entry.version,
                    'timestamp': resolved_entry.timestamp,
                    'node_id': resolved_entry.node_id
                }
            }
        )
        
        with self._lock:
            self._resolution_history.append(resolution)
            
        return resolution
        
    def _merge_entries(self, entries: List[ConflictEntry]) -> ConflictEntry:
        """Merge multiple entries."""
        # Simple merge strategy: combine dictionaries
        merged_value = {}
        max_version = 0
        max_timestamp = 0
        
        for entry in entries:
            if isinstance(entry.value, dict) and isinstance(merged_value, dict):
                merged_value.update(entry.value)
            else:
                # If not mergeable, use last write wins
                merged_value = entry.value
                
            max_version = max(max_version, entry.version)
            max_timestamp = max(max_timestamp, entry.timestamp)
            
        return ConflictEntry(
            key=entries[0].key,
            value=merged_value,
            version=max_version + 1,  # Increment version
            timestamp=max_timestamp,
            node_id="merged"
        )
        
    def get_resolution_history(self) -> List[ConflictResolution]:
        """Get conflict resolution history."""
        return self._resolution_history.copy()


class CacheConflictManager:
    """Manages conflicts in distributed cache."""
    
    def __init__(
        self,
        cache: Any,
        resolver: Optional[ConflictResolver] = None
    ):
        self.cache = cache
        self.resolver = resolver or ConflictResolver()
        self._conflicting_keys: Dict[str, List[ConflictEntry]] = {}
        self._lock = threading.Lock()
        
    def detect_conflict(self, key: str, new_entry: ConflictEntry) -> bool:
        """Detect if there's a conflict for a key."""
        with self._lock:
            if key in self._conflicting_keys:
                existing_entries = self._conflicting_keys[key]
                
                # Check if versions differ
                if any(e.version != new_entry.version for e in existing_entries):
                    return True
                    
            return False
            
    def add_conflict(self, key: str, entry: ConflictEntry) -> None:
        """Add an entry to conflict list."""
        with self._lock:
            if key not in self._conflicting_keys:
                self._conflicting_keys[key] = []
            self._conflicting_keys[key].append(entry)
            
    def resolve_conflict(self, key: str) -> Optional[ConflictResolution]:
        """Resolve conflict for a key."""
        with self._lock:
            if key not in self._conflicting_keys:
                return None
                
            entries = self._conflicting_keys[key]
            if len(entries) <= 1:
                return None
                
            resolution = self.resolver.resolve(entries)
            
            # Update cache with resolved value
            self.cache.put(key, resolution.resolved_value)
            
            # Clear conflict
            del self._conflicting_keys[key]
            
            return resolution
            
    def auto_resolve_conflicts(self) -> List[ConflictResolution]:
        """Automatically resolve all conflicts."""
        resolutions = []
        
        with self._lock:
            keys_to_resolve = list(self._conflicting_keys.keys())
            
        for key in keys_to_resolve:
            resolution = self.resolve_conflict(key)
            if resolution:
                resolutions.append(resolution)
                
        return resolutions
        
    def get_conflicting_keys(self) -> List[str]:
        """Get list of keys with conflicts."""
        return list(self._conflicting_keys.keys())
        
    def get_conflict_count(self) -> int:
        """Get number of conflicting keys."""
        return len(self._conflicting_keys)


class ConflictAwareCache:
    """Cache wrapper with conflict resolution."""
    
    def __init__(
        self,
        cache: Any,
        resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS
    ):
        self.cache = cache
        resolver = ConflictResolver(strategy=resolution_strategy)
        self.conflict_manager = CacheConflictManager(cache, resolver)
        
    def get(self, key: str) -> Any:
        """Get value, resolving conflicts if any."""
        # Check for conflicts first
        if key in self.conflict_manager.get_conflicting_keys():
            self.conflict_manager.resolve_conflict(key)
            
        return self.cache.get(key)
        
    def put(self, key: str, value: Any, version: int = 1, node_id: str = "local") -> bool:
        """Put value, detecting conflicts."""
        entry = ConflictEntry(
            key=key,
            value=value,
            version=version,
            timestamp=time.time(),
            node_id=node_id
        )
        
        # Check for conflict
        if self.conflict_manager.detect_conflict(key, entry):
            self.conflict_manager.add_conflict(key, entry)
            # Auto-resolve
            self.conflict_manager.resolve_conflict(key)
        else:
            # Store entry
            self.conflict_manager.add_conflict(key, entry)
            
        return self.cache.put(key, value)
        
    def resolve_all_conflicts(self) -> List[ConflictResolution]:
        """Resolve all conflicts."""
        return self.conflict_manager.auto_resolve_conflicts()















