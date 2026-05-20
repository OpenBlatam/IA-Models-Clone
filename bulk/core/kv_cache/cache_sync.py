"""
Cache synchronization utilities.

Provides synchronization primitives for cache operations.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class SyncStrategy(Enum):
    """Synchronization strategies."""
    LOCK = "lock"
    RW_LOCK = "rw_lock"
    SEMAPHORE = "semaphore"
    CONDITION = "condition"


class CacheSynchronizer:
    """
    Cache synchronizer.
    
    Provides synchronization primitives for cache operations.
    """
    
    def __init__(self, strategy: SyncStrategy = SyncStrategy.RW_LOCK):
        """
        Initialize synchronizer.
        
        Args:
            strategy: Synchronization strategy
        """
        self.strategy = strategy
        
        if strategy == SyncStrategy.LOCK:
            self.lock = threading.Lock()
        elif strategy == SyncStrategy.RW_LOCK:
            self.read_lock = threading.RLock()
            self.write_lock = threading.Lock()
        elif strategy == SyncStrategy.SEMAPHORE:
            self.semaphore = threading.Semaphore()
        elif strategy == SyncStrategy.CONDITION:
            self.condition = threading.Condition()
    
    def acquire_read(self) -> None:
        """Acquire read lock."""
        if self.strategy == SyncStrategy.RW_LOCK:
            self.read_lock.acquire()
        elif self.strategy == SyncStrategy.LOCK:
            self.lock.acquire()
        elif self.strategy == SyncStrategy.SEMAPHORE:
            self.semaphore.acquire()
    
    def release_read(self) -> None:
        """Release read lock."""
        if self.strategy == SyncStrategy.RW_LOCK:
            self.read_lock.release()
        elif self.strategy == SyncStrategy.LOCK:
            self.lock.release()
        elif self.strategy == SyncStrategy.SEMAPHORE:
            self.semaphore.release()
    
    def acquire_write(self) -> None:
        """Acquire write lock."""
        if self.strategy == SyncStrategy.RW_LOCK:
            self.write_lock.acquire()
        elif self.strategy == SyncStrategy.LOCK:
            self.lock.acquire()
        elif self.strategy == SyncStrategy.SEMAPHORE:
            self.semaphore.acquire()
        elif self.strategy == SyncStrategy.CONDITION:
            self.condition.acquire()
    
    def release_write(self) -> None:
        """Release write lock."""
        if self.strategy == SyncStrategy.RW_LOCK:
            self.write_lock.release()
        elif self.strategy == SyncStrategy.LOCK:
            self.lock.release()
        elif self.strategy == SyncStrategy.SEMAPHORE:
            self.semaphore.release()
        elif self.strategy == SyncStrategy.CONDITION:
            self.condition.release()
    
    def __enter__(self):
        """Context manager entry."""
        self.acquire_write()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release_write()


class DistributedLock:
    """
    Distributed lock for cache coordination.
    
    Provides distributed locking capabilities.
    """
    
    def __init__(self, lock_id: str, timeout: float = 30.0):
        """
        Initialize distributed lock.
        
        Args:
            lock_id: Lock identifier
            timeout: Lock timeout in seconds
        """
        self.lock_id = lock_id
        self.timeout = timeout
        self.acquired = False
        self.acquired_at: Optional[float] = None
    
    def acquire(self) -> bool:
        """
        Acquire distributed lock.
        
        Returns:
            True if acquired
        """
        # In production: would use Redis, etcd, or similar
        if not self.acquired:
            self.acquired = True
            self.acquired_at = time.time()
            logger.info(f"Acquired distributed lock: {self.lock_id}")
            return True
        return False
    
    def release(self) -> None:
        """Release distributed lock."""
        if self.acquired:
            self.acquired = False
            self.acquired_at = None
            logger.info(f"Released distributed lock: {self.lock_id}")
    
    def is_acquired(self) -> bool:
        """
        Check if lock is acquired.
        
        Returns:
            True if acquired
        """
        if self.acquired and self.acquired_at:
            elapsed = time.time() - self.acquired_at
            if elapsed > self.timeout:
                self.acquired = False
                self.acquired_at = None
                logger.warning(f"Lock {self.lock_id} expired")
        return self.acquired
    
    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class CacheBarrier:
    """
    Cache barrier for synchronization.
    
    Provides barrier synchronization for cache operations.
    """
    
    def __init__(self, parties: int):
        """
        Initialize barrier.
        
        Args:
            parties: Number of parties
        """
        self.parties = parties
        self.barrier = threading.Barrier(parties)
        self.waiting = 0
    
    def wait(self) -> int:
        """
        Wait at barrier.
        
        Returns:
            Barrier index
        """
        self.waiting += 1
        index = self.barrier.wait()
        self.waiting -= 1
        return index
    
    def reset(self) -> None:
        """Reset barrier."""
        self.barrier.reset()
    
    def abort(self) -> None:
        """Abort barrier."""
        self.barrier.abort()


class CacheMutex:
    """
    Cache mutex.
    
    Provides mutex for cache operations.
    """
    
    def __init__(self):
        """Initialize mutex."""
        self.mutex = threading.Lock()
        self.owner: Optional[threading.Thread] = None
    
    def acquire(self, blocking: bool = True) -> bool:
        """
        Acquire mutex.
        
        Args:
            blocking: Whether to block
            
        Returns:
            True if acquired
        """
        acquired = self.mutex.acquire(blocking=blocking)
        if acquired:
            self.owner = threading.current_thread()
        return acquired
    
    def release(self) -> None:
        """Release mutex."""
        if self.owner == threading.current_thread():
            self.owner = None
            self.mutex.release()
        else:
            raise RuntimeError("Mutex not owned by current thread")
    
    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()

