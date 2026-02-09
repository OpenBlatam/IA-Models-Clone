"""
Lock Utilities for Piel Mejorador AI SAM3
========================================

Unified lock and synchronization utilities.
"""

import asyncio
import logging
from typing import Optional, Callable, Any, TypeVar, Awaitable
from contextlib import asynccontextmanager
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LockUtils:
    """Unified lock utilities."""
    
    @staticmethod
    @asynccontextmanager
    async def acquire_lock(lock: asyncio.Lock, timeout: Optional[float] = None):
        """
        Acquire lock with optional timeout.
        
        Args:
            lock: AsyncIO lock
            timeout: Optional timeout in seconds
            
        Yields:
            Lock acquired
        """
        if timeout:
            try:
                await asyncio.wait_for(lock.acquire(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Lock acquisition timed out after {timeout}s")
                raise
        else:
            await lock.acquire()
        
        try:
            yield
        finally:
            lock.release()
    
    @staticmethod
    def with_lock(lock: asyncio.Lock, timeout: Optional[float] = None):
        """
        Decorator to execute function with lock.
        
        Args:
            lock: AsyncIO lock
            timeout: Optional timeout in seconds
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                async with LockUtils.acquire_lock(lock, timeout):
                    return await func(*args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def is_locked(lock: asyncio.Lock) -> bool:
        """
        Check if lock is locked.
        
        Args:
            lock: AsyncIO lock
            
        Returns:
            True if locked
        """
        return lock.locked()


class ReadWriteLock:
    """Read-write lock implementation."""
    
    def __init__(self):
        """Initialize read-write lock."""
        self._read_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._readers = 0
        self._readers_lock = asyncio.Lock()
    
    @asynccontextmanager
    async def read(self):
        """
        Acquire read lock.
        
        Yields:
            Read lock acquired
        """
        async with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                await self._write_lock.acquire()
        
        try:
            yield
        finally:
            async with self._readers_lock:
                self._readers -= 1
                if self._readers == 0:
                    self._write_lock.release()
    
    @asynccontextmanager
    async def write(self):
        """
        Acquire write lock.
        
        Yields:
            Write lock acquired
        """
        await self._write_lock.acquire()
        try:
            yield
        finally:
            self._write_lock.release()


class SemaphoreManager:
    """Manager for semaphores."""
    
    def __init__(self, initial: int = 1):
        """
        Initialize semaphore manager.
        
        Args:
            initial: Initial semaphore value
        """
        self._semaphore = asyncio.Semaphore(initial)
    
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire semaphore.
        
        Args:
            timeout: Optional timeout in seconds
            
        Yields:
            Semaphore acquired
        """
        if timeout:
            try:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Semaphore acquisition timed out after {timeout}s")
                raise
        else:
            await self._semaphore.acquire()
        
        try:
            yield
        finally:
            self._semaphore.release()
    
    def available(self) -> int:
        """
        Get available permits.
        
        Returns:
            Number of available permits
        """
        return self._semaphore._value


# Convenience functions
@asynccontextmanager
async def acquire_lock(lock: asyncio.Lock, timeout: Optional[float] = None):
    """Acquire lock with timeout."""
    async with LockUtils.acquire_lock(lock, timeout):
        yield


def with_lock(lock: asyncio.Lock, timeout: Optional[float] = None):
    """Decorator to execute function with lock."""
    return LockUtils.with_lock(lock, timeout)




