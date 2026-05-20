"""
Distributed Lock Manager
========================

Advanced distributed locking for coordination across multiple instances.
"""

import asyncio
import time
import logging
import uuid
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DistributedLock:
    """Distributed lock implementation."""
    
    def __init__(
        self,
        lock_key: str,
        timeout: float = 30.0,
        renewal_interval: float = 10.0,
        redis_client=None
    ):
        self.lock_key = f"lock:{lock_key}"
        self.timeout = timeout
        self.renewal_interval = renewal_interval
        self.redis_client = redis_client
        self.lock_id = str(uuid.uuid4())
        self.is_locked = False
        self.renewal_task = None
    
    async def acquire(self, timeout: Optional[float] = None) -> bool:
        """Acquire lock."""
        if timeout is None:
            timeout = self.timeout
        
        start_time = time.monotonic()
        
        while True:
            if await self._try_acquire():
                self.is_locked = True
                # Start renewal task
                self.renewal_task = asyncio.create_task(self._renewal_loop())
                logger.debug(f"Lock acquired: {self.lock_key}")
                return True
            
            # Check timeout
            if time.monotonic() - start_time > timeout:
                logger.warning(f"Lock acquisition timeout: {self.lock_key}")
                return False
            
            # Wait before retry
            await asyncio.sleep(0.1)
    
    async def _try_acquire(self) -> bool:
        """Try to acquire lock."""
        if not self.redis_client:
            # Fallback to in-memory lock
            return await self._memory_lock_acquire()
        
        try:
            # Try SET with NX (only if not exists) and EX (expiration)
            result = await self.redis_client.set(
                self.lock_key,
                self.lock_id,
                nx=True,
                ex=int(self.timeout)
            )
            return result is True
        except Exception as e:
            logger.error(f"Failed to acquire lock via Redis: {e}")
            return await self._memory_lock_acquire()
    
    async def _memory_lock_acquire(self) -> bool:
        """Fallback in-memory lock."""
        # Simple in-memory lock (not distributed)
        if not hasattr(DistributedLock, '_memory_locks'):
            DistributedLock._memory_locks = {}
        
        if self.lock_key not in DistributedLock._memory_locks:
            DistributedLock._memory_locks[self.lock_key] = {
                'lock_id': self.lock_id,
                'expires_at': datetime.now() + timedelta(seconds=self.timeout)
            }
            return True
        
        # Check if lock expired
        lock_info = DistributedLock._memory_locks[self.lock_key]
        if datetime.now() > lock_info['expires_at']:
            DistributedLock._memory_locks[self.lock_key] = {
                'lock_id': self.lock_id,
                'expires_at': datetime.now() + timedelta(seconds=self.timeout)
            }
            return True
        
        return False
    
    async def release(self):
        """Release lock."""
        if not self.is_locked:
            return
        
        # Stop renewal
        if self.renewal_task:
            self.renewal_task.cancel()
            try:
                await self.renewal_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            try:
                # Lua script to ensure we only delete our own lock
                script = """
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("del", KEYS[1])
                else
                    return 0
                end
                """
                await self.redis_client.eval(script, 1, self.lock_key, self.lock_id)
            except Exception as e:
                logger.error(f"Failed to release lock via Redis: {e}")
        else:
            # Memory lock release
            if hasattr(DistributedLock, '_memory_locks'):
                if self.lock_key in DistributedLock._memory_locks:
                    lock_info = DistributedLock._memory_locks[self.lock_key]
                    if lock_info['lock_id'] == self.lock_id:
                        del DistributedLock._memory_locks[self.lock_key]
        
        self.is_locked = False
        logger.debug(f"Lock released: {self.lock_key}")
    
    async def _renewal_loop(self):
        """Renewal loop to keep lock alive."""
        try:
            while self.is_locked:
                await asyncio.sleep(self.renewal_interval)
                
                if self.is_locked and self.redis_client:
                    try:
                        await self.redis_client.expire(
                            self.lock_key,
                            int(self.timeout)
                        )
                        logger.debug(f"Lock renewed: {self.lock_key}")
                    except Exception as e:
                        logger.error(f"Failed to renew lock: {e}")
                        break
        except asyncio.CancelledError:
            pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.release()
































