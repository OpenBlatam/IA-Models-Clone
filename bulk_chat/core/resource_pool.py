"""
Resource Pool - Pool de Recursos
==================================

Sistema de pool de recursos con reutilización, límites y estadísticas.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ResourceState(Enum):
    """Estado de recurso."""
    IDLE = "idle"
    IN_USE = "in_use"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class PooledResource(Generic[T]):
    """Recurso del pool."""
    resource_id: str
    resource: T
    state: ResourceState = ResourceState.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    use_count: int = 0
    total_usage_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoolConfig:
    """Configuración del pool."""
    min_size: int = 1
    max_size: int = 10
    idle_timeout: float = 300.0
    max_idle_time: float = 600.0
    acquire_timeout: float = 30.0
    validation_interval: float = 60.0


class ResourcePool(Generic[T]):
    """Pool de recursos."""
    
    def __init__(
        self,
        pool_id: str,
        factory: Callable[[], T],
        validator: Optional[Callable[[T], bool]] = None,
        cleanup: Optional[Callable[[T], None]] = None,
        config: Optional[PoolConfig] = None,
    ):
        self.pool_id = pool_id
        self.factory = factory
        self.validator = validator
        self.cleanup = cleanup
        self.config = config or PoolConfig()
        
        self.resources: Dict[str, PooledResource[T]] = {}
        self.idle_resources: deque = deque()
        self.in_use_resources: Dict[str, PooledResource[T]] = {}
        self.acquisition_queue: deque = deque()
        self.statistics: Dict[str, Any] = {
            "total_created": 0,
            "total_destroyed": 0,
            "total_acquired": 0,
            "total_released": 0,
            "total_errors": 0,
        }
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def acquire(self, timeout: Optional[float] = None) -> T:
        """Adquirir recurso del pool."""
        timeout = timeout or self.config.acquire_timeout
        start_time = time.time()
        
        while True:
            async with self._lock:
                # Buscar recurso idle
                while self.idle_resources:
                    resource_id = self.idle_resources.popleft()
                    resource_obj = self.resources.get(resource_id)
                    
                    if resource_obj and resource_obj.state == ResourceState.IDLE:
                        # Validar recurso si hay validator
                        if self.validator:
                            if not self.validator(resource_obj.resource):
                                # Recurso inválido, destruir
                                await self._destroy_resource(resource_id)
                                continue
                        
                        # Marcar como en uso
                        resource_obj.state = ResourceState.IN_USE
                        resource_obj.last_used = datetime.now()
                        resource_obj.use_count += 1
                        self.in_use_resources[resource_id] = resource_obj
                        
                        self.statistics["total_acquired"] += 1
                        return resource_obj.resource
                
                # No hay recursos disponibles, crear nuevo si es posible
                if len(self.resources) < self.config.max_size:
                    resource_obj = await self._create_resource()
                    if resource_obj:
                        resource_obj.state = ResourceState.IN_USE
                        resource_obj.last_used = datetime.now()
                        resource_obj.use_count += 1
                        self.in_use_resources[resource_obj.resource_id] = resource_obj
                        self.statistics["total_acquired"] += 1
                        return resource_obj.resource
            
            # Esperar en cola
            wait_event = asyncio.Event()
            self.acquisition_queue.append(wait_event)
            
            try:
                elapsed = time.time() - start_time
                remaining_timeout = max(0, timeout - elapsed)
                
                if remaining_timeout <= 0:
                    raise TimeoutError(f"Failed to acquire resource from pool {self.pool_id} within {timeout}s")
                
                await asyncio.wait_for(wait_event.wait(), timeout=remaining_timeout)
                
                # Remover de la cola si fue notificado
                if wait_event in self.acquisition_queue:
                    self.acquisition_queue.remove(wait_event)
            
            except asyncio.TimeoutError:
                # Remover de la cola si timeout
                if wait_event in self.acquisition_queue:
                    self.acquisition_queue.remove(wait_event)
                raise TimeoutError(f"Failed to acquire resource from pool {self.pool_id} within {timeout}s")
    
    async def release(self, resource: T):
        """Liberar recurso al pool."""
        async with self._lock:
            # Buscar recurso
            resource_obj = None
            for rid, robj in self.resources.items():
                if robj.resource is resource:
                    resource_obj = robj
                    break
            
            if not resource_obj:
                logger.warning(f"Resource not found in pool {self.pool_id}")
                return
            
            # Remover de in_use
            if resource_obj.resource_id in self.in_use_resources:
                del self.in_use_resources[resource_obj.resource_id]
            
            # Calcular tiempo de uso
            if resource_obj.last_used:
                usage_time = (datetime.now() - resource_obj.last_used).total_seconds()
                resource_obj.total_usage_time += usage_time
            
            # Validar recurso
            if self.validator and not self.validator(resource):
                # Recurso inválido, destruir
                await self._destroy_resource(resource_obj.resource_id)
            else:
                # Marcar como idle
                resource_obj.state = ResourceState.IDLE
                resource_obj.last_used = datetime.now()
                self.idle_resources.append(resource_obj.resource_id)
            
            self.statistics["total_released"] += 1
            
            # Notificar siguiente en cola
            if self.acquisition_queue:
                wait_event = self.acquisition_queue.popleft()
                wait_event.set()
    
    async def _create_resource(self) -> Optional[PooledResource[T]]:
        """Crear nuevo recurso."""
        try:
            resource = self.factory()
            resource_id = f"res_{self.pool_id}_{datetime.now().timestamp()}"
            
            resource_obj = PooledResource(
                resource_id=resource_id,
                resource=resource,
            )
            
            self.resources[resource_id] = resource_obj
            self.statistics["total_created"] += 1
            
            logger.debug(f"Created resource {resource_id} in pool {self.pool_id}")
            return resource_obj
        
        except Exception as e:
            logger.error(f"Error creating resource in pool {self.pool_id}: {e}")
            self.statistics["total_errors"] += 1
            return None
    
    async def _destroy_resource(self, resource_id: str):
        """Destruir recurso."""
        resource_obj = self.resources.get(resource_id)
        if not resource_obj:
            return
        
        # Cleanup si hay cleanup function
        if self.cleanup:
            try:
                self.cleanup(resource_obj.resource)
            except Exception as e:
                logger.error(f"Error cleaning up resource {resource_id}: {e}")
        
        # Remover de todos los lugares
        if resource_id in self.resources:
            del self.resources[resource_id]
        
        if resource_id in self.in_use_resources:
            del self.in_use_resources[resource_id]
        
        if resource_id in self.idle_resources:
            self.idle_resources.remove(resource_id)
        
        self.statistics["total_destroyed"] += 1
        logger.debug(f"Destroyed resource {resource_id} in pool {self.pool_id}")
    
    async def cleanup_idle_resources(self):
        """Limpiar recursos idle antiguos."""
        now = datetime.now()
        resources_to_destroy = []
        
        async with self._lock:
            for resource_id in list(self.idle_resources):
                resource_obj = self.resources.get(resource_id)
                if resource_obj and resource_obj.last_used:
                    idle_time = (now - resource_obj.last_used).total_seconds()
                    if idle_time > self.config.max_idle_time:
                        resources_to_destroy.append(resource_id)
        
        for resource_id in resources_to_destroy:
            await self._destroy_resource(resource_id)
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Obtener estado del pool."""
        return {
            "pool_id": self.pool_id,
            "total_resources": len(self.resources),
            "idle_resources": len(self.idle_resources),
            "in_use_resources": len(self.in_use_resources),
            "waiting_acquires": len(self.acquisition_queue),
            "config": {
                "min_size": self.config.min_size,
                "max_size": self.config.max_size,
            },
            "statistics": self.statistics.copy(),
        }
    
    def get_resource_pool_summary(self) -> Dict[str, Any]:
        """Obtener resumen del pool."""
        return {
            "pool_id": self.pool_id,
            "total_resources": len(self.resources),
            "idle_resources": len(self.idle_resources),
            "in_use_resources": len(self.in_use_resources),
            "statistics": self.statistics.copy(),
        }

