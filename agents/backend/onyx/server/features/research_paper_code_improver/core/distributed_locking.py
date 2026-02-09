"""
Distributed Locking - Sistema de locks distribuidos
=====================================================
"""

import logging
import time
import uuid
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class Lock:
    """Lock individual"""
    key: str
    owner: str
    acquired_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    ttl: float = 60.0  # Time to live en segundos
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Verifica si el lock expiró"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def extend(self, additional_seconds: float):
        """Extiende el tiempo de vida del lock"""
        if self.expires_at:
            self.expires_at += timedelta(seconds=additional_seconds)
        else:
            self.expires_at = datetime.now() + timedelta(seconds=additional_seconds)


class DistributedLock:
    """Sistema de locks distribuidos"""
    
    def __init__(self):
        self.locks: Dict[str, Lock] = {}
        self.lock_requests: Dict[str, Dict[str, Any]] = {}  # key -> request info
    
    def acquire(
        self,
        key: str,
        owner: Optional[str] = None,
        ttl: float = 60.0,
        wait_timeout: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Adquiere un lock"""
        if owner is None:
            owner = str(uuid.uuid4())
        
        start_time = time.time()
        
        while True:
            # Verificar si el lock existe y está expirado
            if key in self.locks:
                existing_lock = self.locks[key]
                if existing_lock.is_expired():
                    # Lock expirado, removerlo
                    del self.locks[key]
                elif existing_lock.owner != owner:
                    # Lock activo de otro owner
                    elapsed = time.time() - start_time
                    if wait_timeout > 0 and elapsed < wait_timeout:
                        time.sleep(0.1)  # Esperar un poco
                        continue
                    return False
            
            # Intentar adquirir lock
            if key not in self.locks:
                lock = Lock(
                    key=key,
                    owner=owner,
                    ttl=ttl,
                    expires_at=datetime.now() + timedelta(seconds=ttl),
                    metadata=metadata or {}
                )
                self.locks[key] = lock
                logger.info(f"Lock {key} adquirido por {owner}")
                return True
            
            # Si llegamos aquí, el lock es nuestro
            if self.locks[key].owner == owner:
                # Extender TTL
                self.locks[key].extend(ttl)
                return True
    
    def release(self, key: str, owner: Optional[str] = None) -> bool:
        """Libera un lock"""
        if key not in self.locks:
            return False
        
        lock = self.locks[key]
        
        # Verificar ownership si se especifica
        if owner and lock.owner != owner:
            return False
        
        del self.locks[key]
        logger.info(f"Lock {key} liberado")
        return True
    
    def is_locked(self, key: str) -> bool:
        """Verifica si un key está locked"""
        if key not in self.locks:
            return False
        
        lock = self.locks[key]
        if lock.is_expired():
            del self.locks[key]
            return False
        
        return True
    
    def get_lock_owner(self, key: str) -> Optional[str]:
        """Obtiene el owner de un lock"""
        if key not in self.locks:
            return None
        
        lock = self.locks[key]
        if lock.is_expired():
            del self.locks[key]
            return None
        
        return lock.owner
    
    def extend_lock(self, key: str, owner: str, additional_seconds: float) -> bool:
        """Extiende el tiempo de vida de un lock"""
        if key not in self.locks:
            return False
        
        lock = self.locks[key]
        if lock.owner != owner:
            return False
        
        if lock.is_expired():
            del self.locks[key]
            return False
        
        lock.extend(additional_seconds)
        return True
    
    def cleanup_expired_locks(self):
        """Limpia locks expirados"""
        expired_keys = [
            key for key, lock in self.locks.items()
            if lock.is_expired()
        ]
        
        for key in expired_keys:
            del self.locks[key]
        
        if expired_keys:
            logger.info(f"Limpieza: {len(expired_keys)} locks expirados removidos")
    
    def list_locks(self) -> Dict[str, Dict[str, Any]]:
        """Lista todos los locks activos"""
        self.cleanup_expired_locks()
        
        return {
            key: {
                "owner": lock.owner,
                "acquired_at": lock.acquired_at.isoformat(),
                "expires_at": lock.expires_at.isoformat() if lock.expires_at else None,
                "ttl": lock.ttl,
                "metadata": lock.metadata
            }
            for key, lock in self.locks.items()
        }


class LockManager:
    """Gestor de locks distribuidos"""
    
    def __init__(self):
        self.lock_system = DistributedLock()
        self._cleanup_task = None
    
    def acquire_lock(
        self,
        key: str,
        owner: Optional[str] = None,
        ttl: float = 60.0,
        wait_timeout: float = 0.0
    ) -> bool:
        """Adquiere un lock"""
        return self.lock_system.acquire(key, owner, ttl, wait_timeout)
    
    def release_lock(self, key: str, owner: Optional[str] = None) -> bool:
        """Libera un lock"""
        return self.lock_system.release(key, owner)
    
    def with_lock(
        self,
        key: str,
        ttl: float = 60.0,
        wait_timeout: float = 0.0
    ):
        """Context manager para locks"""
        return LockContext(self.lock_system, key, ttl, wait_timeout)
    
    def get_lock_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un lock"""
        if not self.lock_system.is_locked(key):
            return None
        
        owner = self.lock_system.get_lock_owner(key)
        locks = self.lock_system.list_locks()
        return locks.get(key)


class LockContext:
    """Context manager para locks"""
    
    def __init__(self, lock_system: DistributedLock, key: str, ttl: float, wait_timeout: float):
        self.lock_system = lock_system
        self.key = key
        self.ttl = ttl
        self.wait_timeout = wait_timeout
        self.owner = str(uuid.uuid4())
        self.acquired = False
    
    def __enter__(self):
        self.acquired = self.lock_system.acquire(
            self.key,
            self.owner,
            self.ttl,
            self.wait_timeout
        )
        if not self.acquired:
            raise Exception(f"No se pudo adquirir lock {self.key}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            self.lock_system.release(self.key, self.owner)




