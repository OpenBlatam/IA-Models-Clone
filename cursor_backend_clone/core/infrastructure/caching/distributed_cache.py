"""
Distributed Cache - Caché Distribuido
======================================

Sistema de caché distribuido con soporte para múltiples backends.
"""

import asyncio
import logging
import hashlib
from typing import Any, Optional, Dict, List, Callable
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Backend abstracto para caché"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Obtener valor"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Establecer valor"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Eliminar valor"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Verificar si existe"""
        pass
    
    @abstractmethod
    async def clear(self) -> int:
        """Limpiar todos los valores"""
        pass


class MemoryCacheBackend(CacheBackend):
    """Backend de caché en memoria"""
    
    def __init__(self):
        self._cache: Dict[str, tuple[Any, Optional[datetime]]] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        
        value, expiry = self._cache[key]
        
        if expiry and datetime.now() >= expiry:
            del self._cache[key]
            return None
        
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        expiry = None
        if ttl:
            expiry = datetime.now() + timedelta(seconds=ttl)
        
        self._cache[key] = (value, expiry)
        return True
    
    async def delete(self, key: str) -> bool:
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        if key not in self._cache:
            return False
        
        value, expiry = self._cache[key]
        if expiry and datetime.now() >= expiry:
            del self._cache[key]
            return False
        
        return True
    
    async def clear(self) -> int:
        count = len(self._cache)
        self._cache.clear()
        return count


class DistributedCache:
    """
    Caché distribuido con múltiples backends.
    
    Soporta fallback y replicación entre backends.
    """
    
    def __init__(
        self,
        backends: List[CacheBackend],
        primary_index: int = 0,
        enable_replication: bool = False
    ):
        self.backends = backends
        self.primary_index = primary_index
        self.enable_replication = enable_replication
    
    def _get_primary(self) -> CacheBackend:
        """Obtener backend primario"""
        return self.backends[self.primary_index]
    
    def _get_key_hash(self, key: str) -> int:
        """Obtener hash de key para selección de backend"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def _select_backend(self, key: str) -> CacheBackend:
        """Seleccionar backend para key"""
        if len(self.backends) == 1:
            return self.backends[0]
        
        # Usar hash para distribución
        index = self._get_key_hash(key) % len(self.backends)
        return self.backends[index]
    
    async def get(self, key: str, fallback: bool = True) -> Optional[Any]:
        """
        Obtener valor del caché.
        
        Args:
            key: Clave
            fallback: Si intentar otros backends si falla
            
        Returns:
            Valor o None
        """
        # Intentar backend primario primero
        primary = self._get_primary()
        value = await primary.get(key)
        
        if value is not None:
            return value
        
        # Fallback a otros backends
        if fallback and len(self.backends) > 1:
            for backend in self.backends:
                if backend == primary:
                    continue
                value = await backend.get(key)
                if value is not None:
                    # Replicar al primario
                    if self.enable_replication:
                        await primary.set(key, value)
                    return value
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        replicate: Optional[bool] = None
    ) -> bool:
        """
        Establecer valor en caché.
        
        Args:
            key: Clave
            value: Valor
            ttl: Tiempo de vida en segundos
            replicate: Si replicar a todos los backends
            
        Returns:
            True si se estableció
        """
        replicate = replicate if replicate is not None else self.enable_replication
        
        if replicate:
            # Replicar a todos los backends
            results = await asyncio.gather(
                *[backend.set(key, value, ttl) for backend in self.backends],
                return_exceptions=True
            )
            return all(r is True for r in results if not isinstance(r, Exception))
        else:
            # Solo al backend seleccionado
            backend = self._select_backend(key)
            return await backend.set(key, value, ttl)
    
    async def delete(self, key: str, replicate: Optional[bool] = None) -> bool:
        """
        Eliminar valor del caché.
        
        Args:
            key: Clave
            replicate: Si eliminar de todos los backends
            
        Returns:
            True si se eliminó
        """
        replicate = replicate if replicate is not None else self.enable_replication
        
        if replicate:
            results = await asyncio.gather(
                *[backend.delete(key) for backend in self.backends],
                return_exceptions=True
            )
            return any(r is True for r in results if not isinstance(r, Exception))
        else:
            backend = self._select_backend(key)
            return await backend.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Verificar si existe key"""
        for backend in self.backends:
            if await backend.exists(key):
                return True
        return False
    
    async def clear(self) -> int:
        """Limpiar todos los backends"""
        results = await asyncio.gather(
            *[backend.clear() for backend in self.backends],
            return_exceptions=True
        )
        return sum(r for r in results if isinstance(r, int) and not isinstance(r, Exception))




