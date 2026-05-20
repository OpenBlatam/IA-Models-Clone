"""
Advanced Caching Strategies - Estrategias Avanzadas de Caching
============================================================

Estrategias avanzadas de caching:
- Write-through cache
- Write-behind cache
- Cache-aside pattern
- Read-through cache
- Cache stampede prevention
- TTL strategies
"""

import logging
import asyncio
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStrategy(str, Enum):
    """Estrategias de cache"""
    CACHE_ASIDE = "cache_aside"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"
    READ_THROUGH = "read_through"


class AdvancedCacheStrategy:
    """
    Estrategias avanzadas de caching.
    """
    
    def __init__(
        self,
        cache_service: Any,
        strategy: CacheStrategy = CacheStrategy.CACHE_ASIDE
    ) -> None:
        self.cache_service = cache_service
        self.pattern = pattern
        self.write_queue: asyncio.Queue = asyncio.Queue()
        self.stampede_locks: Dict[str, asyncio.Lock] = {}
    
    async def get(
        self,
        key: str,
        loader: Optional[Callable] = None
    ) -> Optional[Any]:
        """Obtiene valor con estrategia"""
        if self.pattern == CachePattern.CACHE_ASIDE:
            return await self._cache_aside_get(key, loader)
        elif self.pattern == CachePattern.READ_THROUGH:
            return await self._read_through_get(key, loader)
        else:
            # Para write-through y write-behind, leer de cache
            return await self.cache_service.get(key)
    
    async def _cache_aside_get(
        self,
        key: str,
        loader: Optional[Callable]
    ) -> Optional[Any]:
        """Cache-aside pattern"""
        # Intentar leer de cache
        value = await self.cache_service.get(key)
        if value is not None:
            return value
        
        # Cache miss, cargar de fuente
        if loader:
            # Prevenir cache stampede
            if key not in self.stampede_locks:
                self.stampede_locks[key] = asyncio.Lock()
            
            async with self.stampede_locks[key]:
                # Verificar nuevamente (double-check)
                value = await self.cache_service.get(key)
                if value is not None:
                    return value
                
                # Cargar de fuente
                if asyncio.iscoroutinefunction(loader):
                    value = await loader()
                else:
                    value = loader()
                
                # Guardar en cache
                if value is not None:
                    await self.cache_service.set(key, value)
                
                return value
        
        return None
    
    async def _read_through_get(
        self,
        key: str,
        loader: Optional[Callable]
    ) -> Optional[Any]:
        """Read-through pattern"""
        # Leer de cache
        value = await self.cache_service.get(key)
        if value is not None:
            return value
        
        # Cache miss, cargar y guardar automáticamente
        if loader:
            if asyncio.iscoroutinefunction(loader):
                value = await loader()
            else:
                value = loader()
            
            if value is not None:
                await self.cache_service.set(key, value)
            
            return value
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        writer: Optional[Callable] = None
    ) -> None:
        """Establece valor con estrategia"""
        if self.pattern == CachePattern.WRITE_THROUGH:
            await self._write_through_set(key, value, writer)
        elif self.pattern == CachePattern.WRITE_BEHIND:
            await self._write_behind_set(key, value, writer)
        else:
            # Cache-aside y read-through: solo cache
            await self.cache_service.set(key, value)
    
    async def _write_through_set(
        self,
        key: str,
        value: Any,
        writer: Optional[Callable]
    ) -> None:
        """Write-through pattern"""
        # Escribir a fuente primero
        if writer:
            if asyncio.iscoroutinefunction(writer):
                await writer(key, value)
            else:
                writer(key, value)
        
        # Luego escribir a cache
        await self.cache_service.set(key, value)
    
    async def _write_behind_set(
        self,
        key: str,
        value: Any,
        writer: Optional[Callable]
    ) -> None:
        """Write-behind pattern"""
        # Escribir a cache inmediatamente
        await self.cache_service.set(key, value)
        
        # Encolar escritura a fuente (asíncrona)
        if writer:
            await self.write_queue.put((key, value, writer))
    
    async def process_write_queue(self) -> None:
        """Procesa cola de escrituras (write-behind)"""
        while True:
            try:
                key, value, writer = await asyncio.wait_for(
                    self.write_queue.get(),
                    timeout=1.0
                )
                
                try:
                    if asyncio.iscoroutinefunction(writer):
                        await writer(key, value)
                    else:
                        writer(key, value)
                except Exception as e:
                    logger.error(f"Write-behind failed for {key}: {e}")
                    # Re-enqueue para retry
                    await self.write_queue.put((key, value, writer))
                
            except asyncio.TimeoutError:
                continue


def get_advanced_cache_strategy(
    cache_service: Any,
    pattern: CachePattern = CachePattern.CACHE_ASIDE
) -> AdvancedCacheStrategy:
    """Obtiene estrategia avanzada de cache"""
    return AdvancedCacheStrategy(cache_service, pattern)















