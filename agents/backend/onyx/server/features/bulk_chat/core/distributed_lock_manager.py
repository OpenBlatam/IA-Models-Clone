"""
Distributed Lock Manager - Gestor de Locks Distribuidos
=========================================================

Sistema de gestión de locks distribuidos con TTL, auto-renovación y detección de deadlocks.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)


class LockStatus(Enum):
    """Estado de lock."""
    ACQUIRED = "acquired"
    WAITING = "waiting"
    EXPIRED = "expired"
    RELEASED = "released"
    STOLEN = "stolen"


@dataclass
class Lock:
    """Lock."""
    lock_id: str
    resource_id: str
    owner_id: str
    ttl_seconds: float
    acquired_at: datetime
    expires_at: datetime
    status: LockStatus = LockStatus.ACQUIRED
    renewals: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LockRequest:
    """Solicitud de lock."""
    request_id: str
    resource_id: str
    owner_id: str
    ttl_seconds: float
    wait_timeout: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)


class DistributedLockManager:
    """Gestor de locks distribuidos."""
    
    def __init__(self, default_ttl: float = 30.0, auto_renew: bool = True):
        self.default_ttl = default_ttl
        self.auto_renew = auto_renew
        
        self.locks: Dict[str, Lock] = {}
        self.resource_locks: Dict[str, str] = {}  # resource_id -> lock_id
        self.waiting_queue: Dict[str, List[LockRequest]] = defaultdict(list)
        self.lock_history: List[Lock] = []
        self.auto_renew_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    async def acquire_lock(
        self,
        resource_id: str,
        owner_id: str,
        ttl_seconds: Optional[float] = None,
        wait_timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Adquirir lock."""
        ttl = ttl_seconds or self.default_ttl
        lock_id = f"lock_{resource_id}_{uuid.uuid4().hex[:8]}"
        
        async with self._lock:
            # Verificar si ya existe lock para este recurso
            existing_lock_id = self.resource_locks.get(resource_id)
            if existing_lock_id:
                existing_lock = self.locks.get(existing_lock_id)
                
                if existing_lock and existing_lock.status == LockStatus.ACQUIRED:
                    # Verificar si expiró
                    if datetime.now() >= existing_lock.expires_at:
                        # Lock expirado, removerlo
                        existing_lock.status = LockStatus.EXPIRED
                        del self.locks[existing_lock_id]
                        del self.resource_locks[resource_id]
                    else:
                        # Lock activo, agregar a cola de espera
                        if wait_timeout:
                            request = LockRequest(
                                request_id=f"req_{uuid.uuid4().hex[:8]}",
                                resource_id=resource_id,
                                owner_id=owner_id,
                                ttl_seconds=ttl,
                                wait_timeout=wait_timeout,
                            )
                            self.waiting_queue[resource_id].append(request)
                            
                            # Esperar con timeout
                            asyncio.create_task(self._wait_for_lock(request, lock_id))
                            return None
                        else:
                            return None  # No esperar
        
            # Crear nuevo lock
            now = datetime.now()
            lock = Lock(
                lock_id=lock_id,
                resource_id=resource_id,
                owner_id=owner_id,
                ttl_seconds=ttl,
                acquired_at=now,
                expires_at=now + timedelta(seconds=ttl),
                metadata=metadata or {},
            )
            
            self.locks[lock_id] = lock
            self.resource_locks[resource_id] = lock_id
            self.lock_history.append(lock)
            
            # Limitar historial
            if len(self.lock_history) > 100000:
                self.lock_history.pop(0)
            
            # Iniciar auto-renovación si está habilitada
            if self.auto_renew:
                self.auto_renew_tasks[lock_id] = asyncio.create_task(
                    self._auto_renew_lock(lock_id)
                )
        
        logger.info(f"Acquired lock {lock_id} for resource {resource_id} by {owner_id}")
        return lock_id
    
    async def _wait_for_lock(self, request: LockRequest, lock_id: str):
        """Esperar por lock."""
        start_time = datetime.now()
        
        while True:
            await asyncio.sleep(0.1)  # Verificar cada 100ms
            
            # Verificar timeout
            if request.wait_timeout:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= request.wait_timeout:
                    # Timeout alcanzado
                    async with self._lock:
                        if request in self.waiting_queue[request.resource_id]:
                            self.waiting_queue[request.resource_id].remove(request)
                    return None
            
            # Verificar si el lock está disponible
            async with self._lock:
                existing_lock_id = self.resource_locks.get(request.resource_id)
                if not existing_lock_id:
                    # Lock disponible, intentar adquirir
                    lock_id = await self.acquire_lock(
                        request.resource_id,
                        request.owner_id,
                        request.ttl_seconds,
                    )
                    if lock_id:
                        # Remover de cola
                        if request in self.waiting_queue[request.resource_id]:
                            self.waiting_queue[request.resource_id].remove(request)
                        return lock_id
    
    async def _auto_renew_lock(self, lock_id: str):
        """Renovar lock automáticamente."""
        while lock_id in self.locks:
            lock = self.locks.get(lock_id)
            if not lock or lock.status != LockStatus.ACQUIRED:
                break
            
            # Esperar hasta que quede 50% del TTL
            wait_time = lock.ttl_seconds * 0.5
            await asyncio.sleep(wait_time)
            
            # Renovar si aún está activo
            async with self._lock:
                lock = self.locks.get(lock_id)
                if lock and lock.status == LockStatus.ACQUIRED:
                    if datetime.now() < lock.expires_at:
                        # Extender TTL
                        lock.expires_at = datetime.now() + timedelta(seconds=lock.ttl_seconds)
                        lock.renewals += 1
                        logger.debug(f"Auto-renewed lock {lock_id}")
                    else:
                        # Lock expirado
                        lock.status = LockStatus.EXPIRED
                        break
                else:
                    break
    
    async def release_lock(self, lock_id: str, owner_id: Optional[str] = None) -> bool:
        """Liberar lock."""
        async with self._lock:
            lock = self.locks.get(lock_id)
            if not lock:
                return False
            
            # Verificar owner si se especifica
            if owner_id and lock.owner_id != owner_id:
                logger.warning(f"Lock {lock_id} owner mismatch: expected {owner_id}, got {lock.owner_id}")
                return False
            
            if lock.status != LockStatus.ACQUIRED:
                return False
            
            lock.status = LockStatus.RELEASED
            
            # Remover de registros
            del self.locks[lock_id]
            if lock.resource_id in self.resource_locks:
                del self.resource_locks[lock.resource_id]
            
            # Cancelar auto-renovación
            if lock_id in self.auto_renew_tasks:
                self.auto_renew_tasks[lock_id].cancel()
                del self.auto_renew_tasks[lock_id]
            
            # Notificar siguiente en cola
            if lock.resource_id in self.waiting_queue and self.waiting_queue[lock.resource_id]:
                next_request = self.waiting_queue[lock.resource_id].pop(0)
                # Intentar adquirir para el siguiente
                asyncio.create_task(
                    self._acquire_for_waiting_request(next_request)
                )
        
        logger.info(f"Released lock {lock_id}")
        return True
    
    async def _acquire_for_waiting_request(self, request: LockRequest):
        """Adquirir lock para solicitud en espera."""
        lock_id = await self.acquire_lock(
            request.resource_id,
            request.owner_id,
            request.ttl_seconds,
        )
        return lock_id
    
    async def renew_lock(self, lock_id: str, owner_id: Optional[str] = None, ttl_seconds: Optional[float] = None) -> bool:
        """Renovar lock manualmente."""
        async with self._lock:
            lock = self.locks.get(lock_id)
            if not lock:
                return False
            
            if owner_id and lock.owner_id != owner_id:
                return False
            
            if lock.status != LockStatus.ACQUIRED:
                return False
            
            # Extender TTL
            new_ttl = ttl_seconds or lock.ttl_seconds
            lock.expires_at = datetime.now() + timedelta(seconds=new_ttl)
            lock.renewals += 1
            lock.ttl_seconds = new_ttl
        
        logger.info(f"Renewed lock {lock_id}")
        return True
    
    def get_lock(self, lock_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de lock."""
        lock = self.locks.get(lock_id)
        if not lock:
            return None
        
        now = datetime.now()
        is_expired = now >= lock.expires_at
        
        return {
            "lock_id": lock.lock_id,
            "resource_id": lock.resource_id,
            "owner_id": lock.owner_id,
            "status": LockStatus.EXPIRED.value if is_expired else lock.status.value,
            "ttl_seconds": lock.ttl_seconds,
            "acquired_at": lock.acquired_at.isoformat(),
            "expires_at": lock.expires_at.isoformat(),
            "time_remaining": (lock.expires_at - now).total_seconds() if not is_expired else 0.0,
            "renewals": lock.renewals,
            "metadata": lock.metadata,
        }
    
    def get_resource_lock(self, resource_id: str) -> Optional[Dict[str, Any]]:
        """Obtener lock de recurso."""
        lock_id = self.resource_locks.get(resource_id)
        if not lock_id:
            return None
        
        return self.get_lock(lock_id)
    
    async def cleanup_expired_locks(self) -> int:
        """Limpiar locks expirados."""
        now = datetime.now()
        expired_count = 0
        
        async with self._lock:
            expired_locks = [
                lock_id for lock_id, lock in self.locks.items()
                if lock.status == LockStatus.ACQUIRED and now >= lock.expires_at
            ]
            
            for lock_id in expired_locks:
                lock = self.locks[lock_id]
                lock.status = LockStatus.EXPIRED
                
                del self.locks[lock_id]
                if lock.resource_id in self.resource_locks:
                    del self.resource_locks[lock.resource_id]
                
                if lock_id in self.auto_renew_tasks:
                    self.auto_renew_tasks[lock_id].cancel()
                    del self.auto_renew_tasks[lock_id]
                
                expired_count += 1
        
        return expired_count
    
    def get_distributed_lock_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        by_status: Dict[str, int] = defaultdict(int)
        
        for lock in self.locks.values():
            by_status[lock.status.value] += 1
        
        return {
            "total_locks": len(self.locks),
            "locks_by_status": dict(by_status),
            "total_resources_locked": len(self.resource_locks),
            "waiting_requests": sum(len(queue) for queue in self.waiting_queue.values()),
            "auto_renew_active": len(self.auto_renew_tasks),
            "total_history": len(self.lock_history),
        }

