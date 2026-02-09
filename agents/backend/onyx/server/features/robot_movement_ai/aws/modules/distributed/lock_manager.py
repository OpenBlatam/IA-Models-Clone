"""
Lock Manager
============

Distributed locking.
"""

import logging
import asyncio
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class Lock:
    """Distributed lock."""
    resource: str
    owner: str
    acquired_at: datetime
    expires_at: datetime
    ttl: float = 30.0


class LockManager:
    """Distributed lock manager."""
    
    def __init__(self):
        self._locks: Dict[str, Lock] = {}
        self._waiters: Dict[str, List[asyncio.Future]] = {}
    
    async def acquire(
        self,
        resource: str,
        ttl: float = 30.0,
        timeout: Optional[float] = None
    ) -> Optional[str]:
        """Acquire distributed lock."""
        lock_id = str(uuid.uuid4())
        
        # Check if resource is locked
        if resource in self._locks:
            lock = self._locks[resource]
            
            # Check if lock expired
            if datetime.now() > lock.expires_at:
                # Lock expired, remove it
                del self._locks[resource]
            else:
                # Lock is held, wait for it
                if timeout:
                    try:
                        await asyncio.wait_for(
                            self._wait_for_lock(resource),
                            timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        return None
                else:
                    await self._wait_for_lock(resource)
        
        # Acquire lock
        lock = Lock(
            resource=resource,
            owner=lock_id,
            acquired_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=ttl),
            ttl=ttl
        )
        
        self._locks[resource] = lock
        
        # Start expiration task
        asyncio.create_task(self._expire_lock(resource, ttl))
        
        logger.info(f"Acquired lock on {resource}: {lock_id}")
        return lock_id
    
    async def _wait_for_lock(self, resource: str):
        """Wait for lock to be released."""
        if resource not in self._waiters:
            self._waiters[resource] = []
        
        future = asyncio.Future()
        self._waiters[resource].append(future)
        await future
    
    async def _expire_lock(self, resource: str, ttl: float):
        """Expire lock after TTL."""
        await asyncio.sleep(ttl)
        
        if resource in self._locks:
            del self._locks[resource]
            
            # Notify waiters
            if resource in self._waiters:
                for waiter in self._waiters[resource]:
                    if not waiter.done():
                        waiter.set_result(None)
                del self._waiters[resource]
            
            logger.info(f"Lock on {resource} expired")
    
    def release(self, resource: str, lock_id: str) -> bool:
        """Release distributed lock."""
        if resource not in self._locks:
            return False
        
        lock = self._locks[resource]
        
        if lock.owner != lock_id:
            return False
        
        del self._locks[resource]
        
        # Notify waiters
        if resource in self._waiters:
            for waiter in self._waiters[resource]:
                if not waiter.done():
                    waiter.set_result(None)
            del self._waiters[resource]
        
        logger.info(f"Released lock on {resource}: {lock_id}")
        return True
    
    def is_locked(self, resource: str) -> bool:
        """Check if resource is locked."""
        if resource not in self._locks:
            return False
        
        lock = self._locks[resource]
        
        if datetime.now() > lock.expires_at:
            del self._locks[resource]
            return False
        
        return True
    
    def get_lock_stats(self) -> Dict[str, Any]:
        """Get lock statistics."""
        return {
            "total_locks": len(self._locks),
            "waiters": sum(len(waiters) for waiters in self._waiters.values())
        }















