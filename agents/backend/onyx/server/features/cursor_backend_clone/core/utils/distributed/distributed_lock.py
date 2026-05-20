"""
Distributed Lock - Sistema de Locks Distribuidos
=================================================

Sistema de locks distribuidos para coordinación entre procesos/nodos.
"""

import asyncio
import logging
import uuid
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class Lock:
    """Lock distribuido"""
    key: str
    owner: str
    acquired_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Verificar si el lock expiró"""
        if self.expires_at:
            return datetime.now() >= self.expires_at
        return False
    
    def is_owned_by(self, owner: str) -> bool:
        """Verificar si el lock pertenece a un owner"""
        return self.owner == owner


class LockBackend(ABC):
    """Backend abstracto para locks"""
    
    @abstractmethod
    async def acquire(
        self,
        key: str,
        owner: str,
        timeout: Optional[float] = None
    ) -> bool:
        """Adquirir lock"""
        pass
    
    @abstractmethod
    async def release(self, key: str, owner: str) -> bool:
        """Liberar lock"""
        pass
    
    @abstractmethod
    async def is_locked(self, key: str) -> bool:
        """Verificar si está bloqueado"""
        pass
    
    @abstractmethod
    async def get_lock_info(self, key: str) -> Optional[Lock]:
        """Obtener información del lock"""
        pass


class MemoryLockBackend(LockBackend):
    """Backend de locks en memoria"""
    
    def __init__(self):
        self._locks: Dict[str, Lock] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def acquire(
        self,
        key: str,
        owner: str,
        timeout: Optional[float] = None
    ) -> bool:
        """Adquirir lock"""
        # Verificar si ya existe
        if key in self._locks:
            existing_lock = self._locks[key]
            
            # Si expiró, removerlo
            if existing_lock.is_expired():
                del self._locks[key]
            else:
                # Si es del mismo owner, renovar
                if existing_lock.is_owned_by(owner):
                    existing_lock.acquired_at = datetime.now()
                    if timeout:
                        existing_lock.expires_at = datetime.now() + timedelta(seconds=timeout)
                    return True
                else:
                    # Ya está bloqueado por otro owner
                    return False
        
        # Crear nuevo lock
        expires_at = None
        if timeout:
            expires_at = datetime.now() + timedelta(seconds=timeout)
        
        lock = Lock(
            key=key,
            owner=owner,
            acquired_at=datetime.now(),
            expires_at=expires_at
        )
        
        self._locks[key] = lock
        
        # Iniciar cleanup si no está corriendo
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_expired())
        
        return True
    
    async def release(self, key: str, owner: str) -> bool:
        """Liberar lock"""
        if key not in self._locks:
            return False
        
        lock = self._locks[key]
        
        if not lock.is_owned_by(owner):
            return False
        
        del self._locks[key]
        return True
    
    async def is_locked(self, key: str) -> bool:
        """Verificar si está bloqueado"""
        if key not in self._locks:
            return False
        
        lock = self._locks[key]
        if lock.is_expired():
            del self._locks[key]
            return False
        
        return True
    
    async def get_lock_info(self, key: str) -> Optional[Lock]:
        """Obtener información del lock"""
        if key not in self._locks:
            return None
        
        lock = self._locks[key]
        if lock.is_expired():
            del self._locks[key]
            return None
        
        return lock
    
    async def _cleanup_expired(self) -> None:
        """Limpiar locks expirados"""
        while True:
            try:
                await asyncio.sleep(60)  # Limpiar cada minuto
                
                expired_keys = [
                    key for key, lock in self._locks.items()
                    if lock.is_expired()
                ]
                
                for key in expired_keys:
                    del self._locks[key]
                
                if expired_keys:
                    logger.debug(f"🧹 Cleaned up {len(expired_keys)} expired locks")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in lock cleanup: {e}")


class DistributedLock:
    """
    Lock distribuido con contexto manager.
    
    Permite adquirir y liberar locks de forma segura.
    """
    
    def __init__(self, backend: LockBackend, key: str, timeout: Optional[float] = None):
        self.backend = backend
        self.key = key
        self.timeout = timeout
        self.owner = str(uuid.uuid4())
        self.acquired = False
    
    async def acquire(self, wait: bool = False, wait_timeout: Optional[float] = None) -> bool:
        """
        Adquirir lock.
        
        Args:
            wait: Si esperar hasta que esté disponible
            wait_timeout: Timeout de espera en segundos
            
        Returns:
            True si se adquirió
        """
        if self.acquired:
            return True
        
        start_time = datetime.now()
        
        while True:
            success = await self.backend.acquire(self.key, self.owner, self.timeout)
            
            if success:
                self.acquired = True
                logger.debug(f"🔒 Lock acquired: {self.key}")
                return True
            
            if not wait:
                return False
            
            # Verificar timeout de espera
            if wait_timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= wait_timeout:
                    return False
            
            # Esperar antes de reintentar
            await asyncio.sleep(0.1)
    
    async def release(self) -> bool:
        """
        Liberar lock.
        
        Returns:
            True si se liberó
        """
        if not self.acquired:
            return False
        
        success = await self.backend.release(self.key, self.owner)
        
        if success:
            self.acquired = False
            logger.debug(f"🔓 Lock released: {self.key}")
        
        return success
    
    async def __aenter__(self):
        """Context manager entry"""
        await self.acquire()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.release()
        return False
    
    def is_acquired(self) -> bool:
        """Verificar si el lock está adquirido"""
        return self.acquired




