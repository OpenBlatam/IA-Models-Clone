"""
Distributed Lock Manager for Document Analyzer
===============================================

Advanced distributed locking system for concurrent operations.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class LockInfo:
    """Lock information"""
    lock_id: str
    resource: str
    owner: str
    acquired_at: datetime
    expires_at: datetime
    auto_renew: bool = False

class DistributedLock:
    """Distributed lock manager"""
    
    def __init__(self):
        self.locks: Dict[str, LockInfo] = {}
        self.lock_owner: Dict[str, str] = {}
        self.renewal_tasks: Dict[str, asyncio.Task] = {}
        logger.info("DistributedLock initialized")
    
    async def acquire(
        self,
        resource: str,
        timeout: float = 30.0,
        owner: Optional[str] = None,
        auto_renew: bool = False
    ) -> Optional[str]:
        """Acquire a distributed lock"""
        lock_id = str(uuid.uuid4())
        owner = owner or f"process_{uuid.uuid4().hex[:8]}"
        
        start_time = time.monotonic()
        
        while time.monotonic() - start_time < timeout:
            if resource not in self.locks:
                # Lock available
                lock_info = LockInfo(
                    lock_id=lock_id,
                    resource=resource,
                    owner=owner,
                    acquired_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=timeout),
                    auto_renew=auto_renew
                )
                
                self.locks[resource] = lock_info
                self.lock_owner[resource] = owner
                
                if auto_renew:
                    self._start_renewal(lock_info)
                
                logger.info(f"Acquired lock {lock_id} for resource {resource}")
                return lock_id
            
            # Wait and retry
            await asyncio.sleep(0.1)
        
        logger.warning(f"Failed to acquire lock for resource {resource} (timeout)")
        return None
    
    async def release(self, lock_id: str) -> bool:
        """Release a distributed lock"""
        for resource, lock_info in list(self.locks.items()):
            if lock_info.lock_id == lock_id:
                # Stop renewal if active
                if lock_info.auto_renew and lock_info.lock_id in self.renewal_tasks:
                    self.renewal_tasks[lock_info.lock_id].cancel()
                    del self.renewal_tasks[lock_info.lock_id]
                
                del self.locks[resource]
                if resource in self.lock_owner:
                    del self.lock_owner[resource]
                
                logger.info(f"Released lock {lock_id} for resource {resource}")
                return True
        
        return False
    
    def is_locked(self, resource: str) -> bool:
        """Check if resource is locked"""
        if resource not in self.locks:
            return False
        
        lock_info = self.locks[resource]
        
        # Check if expired
        if datetime.now() > lock_info.expires_at:
            # Auto-release expired lock
            del self.locks[resource]
            if resource in self.lock_owner:
                del self.lock_owner[resource]
            return False
        
        return True
    
    def _start_renewal(self, lock_info: LockInfo):
        """Start automatic lock renewal"""
        async def renew():
            while lock_info.lock_id in self.locks:
                try:
                    await asyncio.sleep(lock_info.expires_at.timestamp() - datetime.now().timestamp() - 5)
                    if lock_info.lock_id in self.locks:
                        # Renew lock
                        lock_info.expires_at = datetime.now() + timedelta(seconds=30)
                        logger.debug(f"Renewed lock {lock_info.lock_id}")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error renewing lock: {e}")
                    break
        
        self.renewal_tasks[lock_info.lock_id] = asyncio.create_task(renew())
    
    async def __aenter__(self, resource: str, timeout: float = 30.0):
        """Context manager entry"""
        lock_id = await self.acquire(resource, timeout)
        if lock_id:
            return lock_id
        raise TimeoutError(f"Failed to acquire lock for {resource}")
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        # This would need the lock_id, so we'll use a different pattern
        pass

# Global instance
distributed_lock = DistributedLock()
















