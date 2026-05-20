"""
Advanced expiration system for KV cache.

This module provides sophisticated expiration policies and
automatic cleanup mechanisms.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class ExpirationPolicy(Enum):
    """Expiration policies."""
    TTL = "ttl"  # Time To Live
    TTI = "tti"  # Time To Idle
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    CUSTOM = "custom"  # Custom expiration function
    NEVER = "never"  # Never expire


@dataclass
class ExpirationEntry:
    """Expiration entry."""
    key: str
    expires_at: Optional[float] = None
    idle_timeout: Optional[float] = None
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    custom_expiry: Optional[Callable[[], bool]] = None
    policy: ExpirationPolicy = ExpirationPolicy.TTL


class AdvancedExpirationManager:
    """Advanced expiration manager."""
    
    def __init__(self, cache: Any, cleanup_interval: float = 60.0):
        self.cache = cache
        self.cleanup_interval = cleanup_interval
        self._expirations: Dict[str, ExpirationEntry] = {}
        self._lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        
        self.start_cleanup()
        
    def start_cleanup(self) -> None:
        """Start cleanup thread."""
        if self._running:
            return
            
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
    def stop_cleanup(self) -> None:
        """Stop cleanup thread."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
            
    def _cleanup_loop(self) -> None:
        """Cleanup loop."""
        while self._running:
            try:
                self.cleanup_expired()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                print(f"Error in cleanup loop: {e}")
                time.sleep(self.cleanup_interval)
                
    def set_expiration(
        self,
        key: str,
        policy: ExpirationPolicy,
        ttl: Optional[float] = None,
        tti: Optional[float] = None,
        custom_expiry: Optional[Callable[[], bool]] = None
    ) -> None:
        """Set expiration for a key."""
        current_time = time.time()
        
        entry = ExpirationEntry(
            key=key,
            expires_at=current_time + ttl if ttl else None,
            idle_timeout=tti,
            last_accessed=current_time,
            access_count=0,
            custom_expiry=custom_expiry,
            policy=policy
        )
        
        with self._lock:
            self._expirations[key] = entry
            
    def update_access(self, key: str) -> None:
        """Update access information for expiration."""
        with self._lock:
            if key in self._expirations:
                entry = self._expirations[key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                
                # Update TTI expiration
                if entry.idle_timeout:
                    entry.expires_at = time.time() + entry.idle_timeout
                    
    def is_expired(self, key: str) -> bool:
        """Check if key is expired."""
        with self._lock:
            if key not in self._expirations:
                return False
                
            entry = self._expirations[key]
            
            # Check TTL/TTI
            if entry.expires_at and time.time() > entry.expires_at:
                return True
                
            # Check custom expiry
            if entry.custom_expiry and entry.custom_expiry():
                return True
                
            return False
            
    def cleanup_expired(self) -> List[str]:
        """Clean up expired entries."""
        expired_keys = []
        current_time = time.time()
        
        with self._lock:
            keys_to_remove = []
            
            for key, entry in self._expirations.items():
                if self.is_expired(key):
                    expired_keys.append(key)
                    keys_to_remove.append(key)
                    
            # Remove expired entries
            for key in keys_to_remove:
                del self._expirations[key]
                self.cache.delete(key)
                
        return expired_keys
        
    def extend_ttl(self, key: str, additional_ttl: float) -> bool:
        """Extend TTL for a key."""
        with self._lock:
            if key in self._expirations:
                entry = self._expirations[key]
                if entry.expires_at:
                    entry.expires_at += additional_ttl
                else:
                    entry.expires_at = time.time() + additional_ttl
                return True
            return False
            
    def get_expiration_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get expiration information for a key."""
        with self._lock:
            if key in self._expirations:
                entry = self._expirations[key]
                return {
                    'policy': entry.policy.value,
                    'expires_at': entry.expires_at,
                    'time_remaining': entry.expires_at - time.time() if entry.expires_at else None,
                    'last_accessed': entry.last_accessed,
                    'access_count': entry.access_count
                }
            return None


class ExpiringCache:
    """Cache wrapper with advanced expiration."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self.expiration_manager = AdvancedExpirationManager(cache)
        
    def get(self, key: str) -> Any:
        """Get value and check expiration."""
        if self.expiration_manager.is_expired(key):
            self.cache.delete(key)
            return None
            
        value = self.cache.get(key)
        if value is not None:
            self.expiration_manager.update_access(key)
        return value
        
    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tti: Optional[float] = None,
        policy: ExpirationPolicy = ExpirationPolicy.TTL
    ) -> bool:
        """Put value with expiration."""
        result = self.cache.put(key, value)
        if result:
            self.expiration_manager.set_expiration(key, policy, ttl, tti)
        return result
        
    def delete(self, key: str) -> bool:
        """Delete value."""
        with self.expiration_manager._lock:
            self.expiration_manager._expirations.pop(key, None)
        return self.cache.delete(key)


