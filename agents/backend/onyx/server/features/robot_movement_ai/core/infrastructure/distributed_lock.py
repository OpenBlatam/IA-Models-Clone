"""
Distributed Lock System
=======================

Sistema de locks distribuidos.
"""

import logging
import asyncio
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class LockStatus(Enum):
    """Estado del lock."""
    ACQUIRED = "acquired"
    RELEASED = "released"
    EXPIRED = "expired"


@dataclass
class Lock:
    """Lock."""
    lock_id: str
    resource: str
    owner: str
    ttl: float  # Time to live en segundos
    acquired_at: float
    expires_at: float
    status: LockStatus = LockStatus.ACQUIRED
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributedLockManager:
    """
    Gestor de locks distribuidos.
    
    Gestiona locks para recursos compartidos.
    """
    
    def __init__(self):
        """Inicializar gestor de locks."""
        self.locks: Dict[str, Lock] = {}  # resource -> lock
        self.lock_history: List[Dict[str, Any]] = []
        self.max_history = 10000
        self.cleanup_task: Optional[asyncio.Task] = None
    
    async def acquire(
        self,
        resource: str,
        owner: str,
        ttl: float = 60.0,
        wait_timeout: float = 10.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Lock]:
        """
        Adquirir lock.
        
        Args:
            resource: Recurso a bloquear
            owner: Propietario del lock
            ttl: Tiempo de vida en segundos
            wait_timeout: Timeout de espera en segundos
            metadata: Metadata adicional
            
        Returns:
            Lock adquirido o None si timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < wait_timeout:
            # Verificar si el lock existe y está expirado
            if resource in self.locks:
                existing_lock = self.locks[resource]
                if time.time() >= existing_lock.expires_at:
                    # Lock expirado, liberarlo
                    self.locks[resource].status = LockStatus.EXPIRED
                    del self.locks[resource]
            
            # Intentar adquirir lock
            if resource not in self.locks:
                now = time.time()
                lock = Lock(
                    lock_id=f"lock_{len(self.locks)}",
                    resource=resource,
                    owner=owner,
                    ttl=ttl,
                    acquired_at=now,
                    expires_at=now + ttl,
                    metadata=metadata or {}
                )
                
                self.locks[resource] = lock
                
                # Registrar en historial
                self._record_lock_action("acquired", lock)
                
                logger.info(f"Lock acquired: {resource} by {owner}")
                return lock
            
            # Esperar antes de reintentar
            await asyncio.sleep(0.1)
        
        logger.warning(f"Failed to acquire lock: {resource} (timeout)")
        return None
    
    async def release(
        self,
        resource: str,
        owner: str
    ) -> bool:
        """
        Liberar lock.
        
        Args:
            resource: Recurso
            owner: Propietario
            
        Returns:
            True si se liberó, False si no existe o no es el propietario
        """
        if resource not in self.locks:
            return False
        
        lock = self.locks[resource]
        
        if lock.owner != owner:
            logger.warning(f"Lock release denied: {resource} (owner mismatch)")
            return False
        
        lock.status = LockStatus.RELEASED
        self._record_lock_action("released", lock)
        del self.locks[resource]
        
        logger.info(f"Lock released: {resource} by {owner}")
        return True
    
    def is_locked(self, resource: str) -> bool:
        """Verificar si recurso está bloqueado."""
        if resource not in self.locks:
            return False
        
        lock = self.locks[resource]
        
        # Verificar si está expirado
        if time.time() >= lock.expires_at:
            lock.status = LockStatus.EXPIRED
            del self.locks[resource]
            return False
        
        return True
    
    def get_lock(self, resource: str) -> Optional[Lock]:
        """Obtener lock de recurso."""
        if resource in self.locks:
            lock = self.locks[resource]
            # Verificar expiración
            if time.time() >= lock.expires_at:
                lock.status = LockStatus.EXPIRED
                del self.locks[resource]
                return None
            return lock
        return None
    
    def _record_lock_action(self, action: str, lock: Lock) -> None:
        """Registrar acción de lock."""
        self.lock_history.append({
            "action": action,
            "lock_id": lock.lock_id,
            "resource": lock.resource,
            "owner": lock.owner,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.lock_history) > self.max_history:
            self.lock_history = self.lock_history[-self.max_history:]
    
    async def start_cleanup(self) -> None:
        """Iniciar limpieza de locks expirados."""
        if self.cleanup_task:
            return
        
        async def cleanup_loop():
            while True:
                try:
                    now = time.time()
                    expired_resources = [
                        resource for resource, lock in self.locks.items()
                        if now >= lock.expires_at
                    ]
                    
                    for resource in expired_resources:
                        lock = self.locks[resource]
                        lock.status = LockStatus.EXPIRED
                        self._record_lock_action("expired", lock)
                        del self.locks[resource]
                        logger.info(f"Lock expired: {resource}")
                    
                    await asyncio.sleep(5.0)  # Limpiar cada 5 segundos
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {e}")
                    await asyncio.sleep(5.0)
        
        self.cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Started lock cleanup loop")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de locks."""
        now = time.time()
        active_locks = [
            lock for lock in self.locks.values()
            if now < lock.expires_at
        ]
        
        return {
            "total_locks": len(active_locks),
            "expired_locks": len(self.locks) - len(active_locks),
            "lock_history_size": len(self.lock_history)
        }


# Instancia global
_distributed_lock_manager: Optional[DistributedLockManager] = None


def get_distributed_lock_manager() -> DistributedLockManager:
    """Obtener instancia global del gestor de locks distribuidos."""
    global _distributed_lock_manager
    if _distributed_lock_manager is None:
        _distributed_lock_manager = DistributedLockManager()
    return _distributed_lock_manager






