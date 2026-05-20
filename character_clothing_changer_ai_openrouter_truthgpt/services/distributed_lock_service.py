"""
Distributed Lock Service
========================
Service for distributed locking across multiple instances
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class LockStatus(Enum):
    """Lock status"""
    ACQUIRED = "acquired"
    RELEASED = "released"
    EXPIRED = "expired"
    FAILED = "failed"


@dataclass
class Lock:
    """Distributed lock"""
    lock_id: str
    resource: str
    owner: str
    acquired_at: datetime
    expires_at: datetime
    ttl_seconds: float
    status: LockStatus = LockStatus.ACQUIRED
    
    def is_expired(self) -> bool:
        """Check if lock is expired"""
        return datetime.now() > self.expires_at
    
    def time_remaining(self) -> float:
        """Get remaining time in seconds"""
        if self.is_expired():
            return 0.0
        return (self.expires_at - datetime.now()).total_seconds()


class DistributedLockService:
    """
    Service for distributed locking.
    
    Features:
    - Resource-based locking
    - TTL support
    - Automatic expiration
    - Lock renewal
    - Statistics
    """
    
    def __init__(self):
        """Initialize distributed lock service"""
        self._locks: Dict[str, Lock] = {}  # resource -> lock
        self._lock_owners: Dict[str, str] = {}  # lock_id -> owner
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        self._stats = {
            'acquired': 0,
            'released': 0,
            'expired': 0,
            'failed': 0,
            'renewed': 0
        }
    
    async def acquire(
        self,
        resource: str,
        ttl_seconds: float = 60.0,
        owner: Optional[str] = None,
        wait_timeout: Optional[float] = None
    ) -> Optional[Lock]:
        """
        Acquire a distributed lock.
        
        Args:
            resource: Resource to lock
            ttl_seconds: Time to live in seconds
            owner: Optional owner identifier
            wait_timeout: Optional timeout to wait for lock
        
        Returns:
            Lock object if acquired, None if failed
        """
        if owner is None:
            owner = f"owner_{uuid.uuid4().hex[:12]}"
        
        lock_id = f"lock_{uuid.uuid4().hex[:12]}"
        now = datetime.now()
        expires_at = now + timedelta(seconds=ttl_seconds)
        
        start_time = time.time()
        
        while True:
            # Check if resource is already locked
            existing_lock = self._locks.get(resource)
            
            if existing_lock is None or existing_lock.is_expired():
                # Acquire lock
                lock = Lock(
                    lock_id=lock_id,
                    resource=resource,
                    owner=owner,
                    acquired_at=now,
                    expires_at=expires_at,
                    ttl_seconds=ttl_seconds
                )
                
                self._locks[resource] = lock
                self._lock_owners[lock_id] = owner
                self._stats['acquired'] += 1
                
                logger.info(f"Lock acquired: {resource} (owner: {owner}, TTL: {ttl_seconds}s)")
                return lock
            
            # Check wait timeout
            if wait_timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= wait_timeout:
                    self._stats['failed'] += 1
                    logger.warning(f"Failed to acquire lock: {resource} (timeout)")
                    return None
            
            # Wait before retry
            await asyncio.sleep(0.1)
    
    async def release(self, lock_id: str, owner: Optional[str] = None) -> bool:
        """
        Release a lock.
        
        Args:
            lock_id: Lock ID
            owner: Optional owner (for verification)
        
        Returns:
            True if lock was released
        """
        # Find lock by ID
        lock = None
        resource = None
        
        for res, l in self._locks.items():
            if l.lock_id == lock_id:
                lock = l
                resource = res
                break
        
        if lock is None:
            return False
        
        # Verify owner if provided
        if owner and lock.owner != owner:
            logger.warning(f"Lock {lock_id} release denied: owner mismatch")
            return False
        
        # Release lock
        lock.status = LockStatus.RELEASED
        del self._locks[resource]
        if lock_id in self._lock_owners:
            del self._lock_owners[lock_id]
        
        self._stats['released'] += 1
        logger.info(f"Lock released: {resource} (lock_id: {lock_id})")
        return True
    
    async def renew(self, lock_id: str, ttl_seconds: float, owner: Optional[str] = None) -> bool:
        """
        Renew a lock (extend TTL).
        
        Args:
            lock_id: Lock ID
            ttl_seconds: New TTL in seconds
            owner: Optional owner (for verification)
        
        Returns:
            True if lock was renewed
        """
        # Find lock by ID
        lock = None
        
        for l in self._locks.values():
            if l.lock_id == lock_id:
                lock = l
                break
        
        if lock is None or lock.is_expired():
            return False
        
        # Verify owner if provided
        if owner and lock.owner != owner:
            logger.warning(f"Lock {lock_id} renewal denied: owner mismatch")
            return False
        
        # Renew lock
        lock.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        lock.ttl_seconds = ttl_seconds
        self._stats['renewed'] += 1
        
        logger.info(f"Lock renewed: {lock.resource} (lock_id: {lock_id}, new TTL: {ttl_seconds}s)")
        return True
    
    def get_lock(self, resource: str) -> Optional[Lock]:
        """Get lock for resource"""
        lock = self._locks.get(resource)
        if lock and lock.is_expired():
            # Auto-expire
            lock.status = LockStatus.EXPIRED
            del self._locks[resource]
            if lock.lock_id in self._lock_owners:
                del self._lock_owners[lock.lock_id]
            self._stats['expired'] += 1
            return None
        return lock
    
    async def start_cleanup(self, interval_seconds: float = 10.0):
        """Start automatic cleanup of expired locks"""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(interval_seconds))
        logger.info("Distributed lock cleanup started")
    
    async def stop_cleanup(self):
        """Stop automatic cleanup"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Distributed lock cleanup stopped")
    
    async def _cleanup_loop(self, interval_seconds: float):
        """Cleanup loop for expired locks"""
        while self._running:
            try:
                expired_resources = []
                
                for resource, lock in self._locks.items():
                    if lock.is_expired():
                        expired_resources.append(resource)
                
                for resource in expired_resources:
                    lock = self._locks[resource]
                    lock.status = LockStatus.EXPIRED
                    del self._locks[resource]
                    if lock.lock_id in self._lock_owners:
                        del self._lock_owners[lock.lock_id]
                    self._stats['expired'] += 1
                
                if expired_resources:
                    logger.debug(f"Cleaned up {len(expired_resources)} expired locks")
                
                await asyncio.sleep(interval_seconds)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in lock cleanup loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get lock service statistics"""
        return {
            'active_locks': len(self._locks),
            'acquired': self._stats['acquired'],
            'released': self._stats['released'],
            'expired': self._stats['expired'],
            'failed': self._stats['failed'],
            'renewed': self._stats['renewed'],
            'cleanup_running': self._running
        }


# Global distributed lock service instance
_distributed_lock_service: Optional[DistributedLockService] = None


def get_distributed_lock_service() -> DistributedLockService:
    """Get or create distributed lock service instance"""
    global _distributed_lock_service
    if _distributed_lock_service is None:
        _distributed_lock_service = DistributedLockService()
    return _distributed_lock_service

