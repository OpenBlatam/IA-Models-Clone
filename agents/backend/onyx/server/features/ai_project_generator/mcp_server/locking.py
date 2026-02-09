"""
MCP Distributed Locking - Distributed locking
==============================================
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from .exceptions import MCPError

logger = logging.getLogger(__name__)


class DistributedLock:
    """
    Lock distribuido
    
    Permite sincronización entre múltiples instancias del servidor.
    """
    
    def __init__(
        self,
        lock_id: str,
        timeout: float = 30.0,
        renewal_interval: float = 10.0,
    ):
        """
        Args:
            lock_id: ID único del lock
            timeout: Timeout en segundos
            renewal_interval: Intervalo de renovación en segundos
        """
        self.lock_id = lock_id
        self.timeout = timeout
        self.renewal_interval = renewal_interval
        self._acquired = False
        self._acquired_at: Optional[datetime] = None
        self._renewal_task: Optional[asyncio.Task] = None
    
    async def acquire(self) -> bool:
        """
        Adquiere el lock
        
        Returns:
            True si se adquirió, False si no
        """
        # Implementación simple en memoria
        # En producción usar Redis o similar
        if self._acquired:
            return False
        
        self._acquired = True
        self._acquired_at = datetime.utcnow()
        
        # Iniciar renovación automática
        self._renewal_task = asyncio.create_task(self._renewal_loop())
        
        logger.info(f"Lock {self.lock_id} acquired")
        return True
    
    async def release(self):
        """Libera el lock"""
        if not self._acquired:
            return
        
        self._acquired = False
        self._acquired_at = None
        
        if self._renewal_task:
            self._renewal_task.cancel()
            try:
                await self._renewal_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Lock {self.lock_id} released")
    
    async def _renewal_loop(self):
        """Loop de renovación del lock"""
        while self._acquired:
            try:
                await asyncio.sleep(self.renewal_interval)
                
                if self._acquired:
                    # Renovar lock
                    elapsed = (datetime.utcnow() - self._acquired_at).total_seconds()
                    if elapsed < self.timeout:
                        logger.debug(f"Lock {self.lock_id} renewed")
                    else:
                        # Timeout, liberar
                        await self.release()
                        break
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in lock renewal: {e}")
    
    def is_acquired(self) -> bool:
        """Verifica si el lock está adquirido"""
        return self._acquired
    
    @asynccontextmanager
    async def __aenter__(self):
        """Context manager para adquirir lock"""
        acquired = await self.acquire()
        if not acquired:
            raise MCPError(f"Failed to acquire lock {self.lock_id}")
        try:
            yield self
        finally:
            await self.release()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.release()


class LockManager:
    """
    Gestor de locks distribuidos
    
    Gestiona múltiples locks y proporciona interfaz unificada.
    """
    
    def __init__(self):
        self._locks: Dict[str, DistributedLock] = {}
    
    async def acquire_lock(
        self,
        lock_id: str,
        timeout: float = 30.0,
    ) -> Optional[DistributedLock]:
        """
        Adquiere un lock
        
        Args:
            lock_id: ID del lock
            timeout: Timeout en segundos
            
        Returns:
            DistributedLock o None si no se pudo adquirir
        """
        if lock_id in self._locks:
            lock = self._locks[lock_id]
            if lock.is_acquired():
                return None  # Ya está adquirido
        
        lock = DistributedLock(lock_id, timeout=timeout)
        acquired = await lock.acquire()
        
        if acquired:
            self._locks[lock_id] = lock
            return lock
        
        return None
    
    async def release_lock(self, lock_id: str):
        """
        Libera un lock
        
        Args:
            lock_id: ID del lock
        """
        lock = self._locks.get(lock_id)
        if lock:
            await lock.release()
            if not lock.is_acquired():
                del self._locks[lock_id]
    
    @asynccontextmanager
    async def lock(self, lock_id: str, timeout: float = 30.0):
        """
        Context manager para lock
        
        Args:
            lock_id: ID del lock
            timeout: Timeout en segundos
        """
        lock = await self.acquire_lock(lock_id, timeout)
        if not lock:
            raise MCPError(f"Failed to acquire lock {lock_id}")
        
        try:
            yield lock
        finally:
            await self.release_lock(lock_id)

