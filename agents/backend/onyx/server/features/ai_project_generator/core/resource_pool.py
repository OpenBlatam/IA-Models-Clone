"""
Resource Pool - Pool de Recursos
================================

Pool de recursos:
- Connection pooling
- Resource reuse
- Pool management
- Health checks
- Auto-scaling pools
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, Callable, Generic, TypeVar
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PoolStatus(str, Enum):
    """Estados del pool"""
    ACTIVE = "active"
    DRAINING = "draining"
    CLOSED = "closed"


class ResourcePool(Generic[T]):
    """
    Pool de recursos.
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = 30.0,
        health_check: Optional[Callable[[T], bool]] = None
    ) -> None:
        self.factory = factory
        self.min_size = min_size
        self.max_size = max_size
        self.timeout = timeout
        self.health_check = health_check
        self.pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.active_resources: List[T] = []
        self.status = PoolStatus.ACTIVE
        self.created_count = 0
        self.reused_count = 0
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """Inicializa pool con recursos mínimos"""
        for _ in range(self.min_size):
            try:
                resource = self.factory()
                self.pool.put_nowait(resource)
                self.active_resources.append(resource)
                self.created_count += 1
            except Exception as e:
                logger.error(f"Failed to create initial resource: {e}")
    
    async def acquire(self) -> T:
        """Adquiere recurso del pool"""
        if self.status != PoolStatus.ACTIVE:
            raise RuntimeError(f"Pool is {self.status.value}")
        
        try:
            # Intentar obtener de pool con timeout
            resource = await asyncio.wait_for(self.pool.get(), timeout=self.timeout)
            
            # Health check
            if self.health_check and not self.health_check(resource):
                logger.warning("Resource failed health check, creating new one")
                resource = self._create_new_resource()
            
            self.reused_count += 1
            return resource
            
        except asyncio.TimeoutError:
            # Pool vacío, crear nuevo recurso si no excede max_size
            if len(self.active_resources) < self.max_size:
                resource = self._create_new_resource()
                return resource
            else:
                raise RuntimeError("Pool exhausted and max size reached")
    
    def _create_new_resource(self) -> T:
        """Crea nuevo recurso"""
        resource = self.factory()
        self.active_resources.append(resource)
        self.created_count += 1
        return resource
    
    async def release(self, resource: T) -> None:
        """Libera recurso al pool"""
        if resource not in self.active_resources:
            logger.warning("Attempting to release resource not in pool")
            return
        
        if self.status == PoolStatus.CLOSED:
            await self._destroy_resource(resource)
            return
        
        # Health check antes de devolver al pool
        if self.health_check and not self.health_check(resource):
            logger.warning("Resource failed health check, destroying")
            await self._destroy_resource(resource)
            return
        
        # Devolver al pool si hay espacio
        try:
            self.pool.put_nowait(resource)
        except asyncio.QueueFull:
            # Pool lleno, destruir recurso
            await self._destroy_resource(resource)
    
    async def _destroy_resource(self, resource: T) -> None:
        """Destruye recurso"""
        if resource in self.active_resources:
            self.active_resources.remove(resource)
        
        # Si tiene método close, llamarlo
        if hasattr(resource, 'close'):
            if asyncio.iscoroutinefunction(resource.close):
                await resource.close()
            else:
                resource.close()
    
    async def close(self) -> None:
        """Cierra pool"""
        self.status = PoolStatus.DRAINING
        
        # Esperar a que todos los recursos se liberen
        while not self.pool.empty():
            resource = await self.pool.get()
            await self._destroy_resource(resource)
        
        # Destruir recursos activos
        for resource in list(self.active_resources):
            await self._destroy_resource(resource)
        
        self.status = PoolStatus.CLOSED
        logger.info("Resource pool closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del pool"""
        return {
            "min_size": self.min_size,
            "max_size": self.max_size,
            "current_size": len(self.active_resources),
            "pool_size": self.pool.qsize(),
            "status": self.status.value,
            "created_count": self.created_count,
            "reused_count": self.reused_count,
            "reuse_rate": (
                self.reused_count / (self.created_count + self.reused_count) * 100
                if (self.created_count + self.reused_count) > 0 else 0
            )
        }


def create_resource_pool(
    factory: Callable[[], T],
    min_size: int = 1,
    max_size: int = 10,
    timeout: float = 30.0,
    health_check: Optional[Callable[[T], bool]] = None
) -> ResourcePool[T]:
    """Crea pool de recursos"""
    return ResourcePool(factory, min_size, max_size, timeout, health_check)















