"""
Distributed locking system for KV cache.

This module provides distributed locking capabilities for coordinating
operations across multiple cache instances.
"""

import time
import threading
import uuid
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class LockType(Enum):
    """Lock types."""
    EXCLUSIVE = "exclusive"  # Only one holder
    SHARED = "shared"  # Multiple readers
    READ_WRITE = "read_write"  # Read-write lock


@dataclass
class Lock:
    """A distributed lock."""
    lock_id: str
    key: str
    holder: str
    lock_type: LockType
    acquired_at: float
    expires_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributedLockManager:
    """Manages distributed locks."""
    
    def __init__(self, cache: Any):
        self.cache = cache
        self._locks: Dict[str, Lock] = {}
        self._lock_requests: Dict[str, List[str]] = defaultdict(list)  # key -> [holder_ids]
        self._lock = threading.Lock()
        
    def acquire(
        self,
        key: str,
        holder: Optional[str] = None,
        lock_type: LockType = LockType.EXCLUSIVE,
        timeout: float = 60.0,
        blocking: bool = True
    ) -> Optional[str]:
        """
        Acquire a lock. Returns lock_id if successful, None otherwise.
        """
        if holder is None:
            holder = str(uuid.uuid4())
            
        lock_id = str(uuid.uuid4())
        current_time = time.time()
        
        with self._lock:
            # Check if lock exists
            if key in self._locks:
                existing_lock = self._locks[key]
                
                # Check if expired
                if current_time > existing_lock.expires_at:
                    # Lock expired, remove it
                    del self._locks[key]
                else:
                    # Lock is held
                    if lock_type == LockType.EXCLUSIVE or existing_lock.lock_type == LockType.EXCLUSIVE:
                        # Exclusive lock conflicts with any other lock
                        if blocking:
                            # Wait for lock to be released
                            return self._wait_for_lock(key, holder, lock_type, timeout)
                        return None
                    elif lock_type == LockType.SHARED and existing_lock.lock_type == LockType.SHARED:
                        # Shared locks can coexist
                        pass
                    else:
                        # Conflict
                        if blocking:
                            return self._wait_for_lock(key, holder, lock_type, timeout)
                        return None
                        
            # Acquire lock
            lock = Lock(
                lock_id=lock_id,
                key=key,
                holder=holder,
                lock_type=lock_type,
                acquired_at=current_time,
                expires_at=current_time + timeout
            )
            
            self._locks[key] = lock
            self._lock_requests[key].append(holder)
            
        return lock_id
        
    def _wait_for_lock(
        self,
        key: str,
        holder: str,
        lock_type: LockType,
        timeout: float
    ) -> Optional[str]:
        """Wait for lock to become available."""
        start_time = time.time()
        check_interval = 0.1  # Check every 100ms
        
        while time.time() - start_time < timeout:
            time.sleep(check_interval)
            
            # Try to acquire again
            result = self.acquire(key, holder, lock_type, timeout - (time.time() - start_time), blocking=False)
            if result:
                return result
                
        return None
        
    def release(self, lock_id: str) -> bool:
        """Release a lock."""
        with self._lock:
            for key, lock in list(self._locks.items()):
                if lock.lock_id == lock_id:
                    del self._locks[key]
                    if key in self._lock_requests:
                        self._lock_requests[key] = [
                            h for h in self._lock_requests[key]
                            if h != lock.holder
                        ]
                    return True
            return False
            
    def release_by_key(self, key: str, holder: Optional[str] = None) -> bool:
        """Release lock by key."""
        with self._lock:
            if key in self._locks:
                lock = self._locks[key]
                if holder is None or lock.holder == holder:
                    del self._locks[key]
                    if key in self._lock_requests:
                        self._lock_requests[key] = [
                            h for h in self._lock_requests[key]
                            if h != lock.holder
                        ]
                    return True
            return False
            
    def extend(self, lock_id: str, additional_time: float) -> bool:
        """Extend lock expiration time."""
        with self._lock:
            for lock in self._locks.values():
                if lock.lock_id == lock_id:
                    lock.expires_at += additional_time
                    return True
            return False
            
    def is_locked(self, key: str) -> bool:
        """Check if a key is locked."""
        with self._lock:
            if key in self._locks:
                lock = self._locks[key]
                if time.time() < lock.expires_at:
                    return True
                else:
                    # Expired, remove it
                    del self._locks[key]
            return False
            
    def get_lock_info(self, key: str) -> Optional[Lock]:
        """Get lock information for a key."""
        with self._lock:
            if key in self._locks:
                lock = self._locks[key]
                if time.time() < lock.expires_at:
                    return lock
                else:
                    del self._locks[key]
            return None


class LockContext:
    """Context manager for distributed locks."""
    
    def __init__(
        self,
        lock_manager: DistributedLockManager,
        key: str,
        holder: Optional[str] = None,
        lock_type: LockType = LockType.EXCLUSIVE,
        timeout: float = 60.0
    ):
        self.lock_manager = lock_manager
        self.key = key
        self.holder = holder
        self.lock_type = lock_type
        self.timeout = timeout
        self.lock_id: Optional[str] = None
        
    def __enter__(self) -> 'LockContext':
        self.lock_id = self.lock_manager.acquire(
            self.key,
            self.holder,
            self.lock_type,
            self.timeout
        )
        if self.lock_id is None:
            raise RuntimeError(f"Failed to acquire lock for key: {self.key}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.lock_id:
            self.lock_manager.release(self.lock_id)



