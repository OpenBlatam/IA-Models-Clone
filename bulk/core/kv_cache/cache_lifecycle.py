"""
Data lifecycle management for KV cache.

This module provides lifecycle management capabilities including
TTL management, expiration policies, and automatic cleanup.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class LifecycleStage(Enum):
    """Lifecycle stages."""
    CREATED = "created"
    ACTIVE = "active"
    STALE = "stale"
    EXPIRED = "expired"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ExpirationPolicy(Enum):
    """Expiration policies."""
    TTL = "ttl"  # Time To Live
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    ACCESS_COUNT = "access_count"  # Based on access count
    SIZE_BASED = "size_based"  # Based on size
    CUSTOM = "custom"  # Custom policy


@dataclass
class LifecycleEntry:
    """Entry with lifecycle information."""
    key: str
    created_at: float
    last_accessed: float
    access_count: int
    stage: LifecycleStage
    ttl: Optional[float] = None
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at:
            return time.time() > self.expires_at
        return False
        
    def update_access(self) -> None:
        """Update access information."""
        self.last_accessed = time.time()
        self.access_count += 1
        
        if self.stage == LifecycleStage.STALE:
            self.stage = LifecycleStage.ACTIVE


@dataclass
class LifecyclePolicy:
    """Lifecycle policy configuration."""
    expiration_policy: ExpirationPolicy
    default_ttl: Optional[float] = None
    stale_threshold: float = 3600.0  # 1 hour
    max_access_count: Optional[int] = None
    max_size: Optional[int] = None
    cleanup_interval: float = 300.0  # 5 minutes
    enable_auto_cleanup: bool = True


class CacheLifecycleManager:
    """Manages lifecycle of cache entries."""
    
    def __init__(
        self,
        cache: Any,
        policy: LifecyclePolicy
    ):
        self.cache = cache
        self.policy = policy
        self._lifecycle_entries: Dict[str, LifecycleEntry] = {}
        self._lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        
        if policy.enable_auto_cleanup:
            self.start_cleanup()
            
    def start_cleanup(self) -> None:
        """Start automatic cleanup thread."""
        if self._running:
            return
            
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
    def stop_cleanup(self) -> None:
        """Stop automatic cleanup thread."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
            
    def _cleanup_loop(self) -> None:
        """Cleanup loop."""
        while self._running:
            try:
                self.cleanup_expired()
                time.sleep(self.policy.cleanup_interval)
            except Exception as e:
                print(f"Error in cleanup loop: {e}")
                
    def create_entry(
        self,
        key: str,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LifecycleEntry:
        """Create a new lifecycle entry."""
        current_time = time.time()
        entry_ttl = ttl or self.policy.default_ttl
        
        entry = LifecycleEntry(
            key=key,
            created_at=current_time,
            last_accessed=current_time,
            access_count=0,
            stage=LifecycleStage.CREATED,
            ttl=entry_ttl,
            expires_at=current_time + entry_ttl if entry_ttl else None,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._lifecycle_entries[key] = entry
            
        return entry
        
    def record_access(self, key: str) -> None:
        """Record access to a key."""
        with self._lock:
            if key in self._lifecycle_entries:
                entry = self._lifecycle_entries[key]
                entry.update_access()
                
                # Check if entry should become stale
                time_since_access = time.time() - entry.last_accessed
                if time_since_access > self.policy.stale_threshold:
                    entry.stage = LifecycleStage.STALE
                else:
                    entry.stage = LifecycleStage.ACTIVE
                    
    def cleanup_expired(self) -> List[str]:
        """Clean up expired entries."""
        expired_keys = []
        current_time = time.time()
        
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._lifecycle_entries.items():
                if entry.is_expired():
                    expired_keys.append(key)
                    keys_to_remove.append(key)
                elif self.policy.expiration_policy == ExpirationPolicy.ACCESS_COUNT:
                    if self.policy.max_access_count and entry.access_count >= self.policy.max_access_count:
                        expired_keys.append(key)
                        keys_to_remove.append(key)
                        
            # Remove expired entries
            for key in keys_to_remove:
                del self._lifecycle_entries[key]
                self.cache.delete(key)
                
        return expired_keys
        
    def get_entry(self, key: str) -> Optional[LifecycleEntry]:
        """Get lifecycle entry for a key."""
        return self._lifecycle_entries.get(key)
        
    def set_stage(self, key: str, stage: LifecycleStage) -> bool:
        """Set lifecycle stage for a key."""
        with self._lock:
            if key in self._lifecycle_entries:
                self._lifecycle_entries[key].stage = stage
                return True
            return False
            
    def extend_ttl(self, key: str, additional_ttl: float) -> bool:
        """Extend TTL for a key."""
        with self._lock:
            if key in self._lifecycle_entries:
                entry = self._lifecycle_entries[key]
                if entry.expires_at:
                    entry.expires_at += additional_ttl
                elif entry.ttl:
                    entry.expires_at = time.time() + entry.ttl + additional_ttl
                return True
            return False
            
    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get lifecycle statistics."""
        with self._lock:
            stages = defaultdict(int)
            for entry in self._lifecycle_entries.values():
                stages[entry.stage.value] += 1
                
            total_entries = len(self._lifecycle_entries)
            expired_count = sum(1 for e in self._lifecycle_entries.values() if e.is_expired())
            
            return {
                'total_entries': total_entries,
                'expired_count': expired_count,
                'active_count': stages.get(LifecycleStage.ACTIVE.value, 0),
                'stale_count': stages.get(LifecycleStage.STALE.value, 0),
                'stages': dict(stages)
            }
            
    def archive_old_entries(self, age_threshold: float) -> List[str]:
        """Archive entries older than threshold."""
        archived = []
        current_time = time.time()
        
        with self._lock:
            for key, entry in list(self._lifecycle_entries.items()):
                age = current_time - entry.created_at
                if age > age_threshold and entry.stage != LifecycleStage.ARCHIVED:
                    entry.stage = LifecycleStage.ARCHIVED
                    archived.append(key)
                    
        return archived


class LifecycleAwareCache:
    """Cache wrapper with lifecycle management."""
    
    def __init__(
        self,
        cache: Any,
        policy: LifecyclePolicy
    ):
        self.cache = cache
        self.lifecycle_manager = CacheLifecycleManager(cache, policy)
        
    def get(self, key: str) -> Any:
        """Get value and record access."""
        value = self.cache.get(key)
        if value is not None:
            self.lifecycle_manager.record_access(key)
        return value
        
    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Put value with lifecycle tracking."""
        result = self.cache.put(key, value)
        if result:
            self.lifecycle_manager.create_entry(key, ttl, metadata)
        return result
        
    def delete(self, key: str) -> bool:
        """Delete value and lifecycle entry."""
        with self.lifecycle_manager._lock:
            self.lifecycle_manager._lifecycle_entries.pop(key, None)
        return self.cache.delete(key)
        
    def get_lifecycle_entry(self, key: str) -> Optional[LifecycleEntry]:
        """Get lifecycle information for a key."""
        return self.lifecycle_manager.get_entry(key)
        
    def cleanup(self) -> List[str]:
        """Manually trigger cleanup."""
        return self.lifecycle_manager.cleanup_expired()














