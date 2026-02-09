"""
Resource Pool System
====================

Sistema de pool de recursos para gestión eficiente.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Generic, TypeVar, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ResourceStatus(Enum):
    """Estado del recurso."""
    IDLE = "idle"
    IN_USE = "in_use"
    MAINTENANCE = "maintenance"
    ERROR = "error"


@dataclass
class Resource(Generic[T]):
    """Recurso."""
    resource_id: str
    resource: T
    status: ResourceStatus = ResourceStatus.IDLE
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_used_at: Optional[str] = None
    use_count: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourcePool(Generic[T]):
    """
    Pool de recursos.
    
    Gestiona pool de recursos reutilizables.
    """
    
    def __init__(
        self,
        name: str,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 2,
        max_idle_time: float = 300.0,
        health_check: Optional[Callable[[T], bool]] = None
    ):
        """
        Inicializar pool de recursos.
        
        Args:
            name: Nombre del pool
            factory: Función para crear recursos
            max_size: Tamaño máximo del pool
            min_size: Tamaño mínimo del pool
            max_idle_time: Tiempo máximo de inactividad (segundos)
            health_check: Función de verificación de salud
        """
        self.name = name
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self.health_check = health_check
        
        self.resources: Dict[str, Resource[T]] = {}
        self.available_resources: asyncio.Queue = asyncio.Queue()
        self.lock = asyncio.Lock()
        
        # Inicializar pool mínimo
        asyncio.create_task(self._initialize_pool())
    
    async def _initialize_pool(self) -> None:
        """Inicializar pool con recursos mínimos."""
        for _ in range(self.min_size):
            await self._create_resource()
    
    async def _create_resource(self) -> Optional[Resource[T]]:
        """Crear nuevo recurso."""
        if len(self.resources) >= self.max_size:
            return None
        
        try:
            resource_id = f"res_{len(self.resources)}"
            resource = self.factory()
            
            resource_obj = Resource(
                resource_id=resource_id,
                resource=resource,
                status=ResourceStatus.IDLE
            )
            
            self.resources[resource_id] = resource_obj
            await self.available_resources.put(resource_id)
            
            logger.info(f"Created resource in pool {self.name}: {resource_id}")
            
            return resource_obj
        except Exception as e:
            logger.error(f"Error creating resource in pool {self.name}: {e}")
            return None
    
    async def acquire(self, timeout: float = 10.0) -> Optional[Resource[T]]:
        """
        Adquirir recurso del pool.
        
        Args:
            timeout: Timeout en segundos
            
        Returns:
            Recurso o None si timeout
        """
        try:
            # Intentar obtener recurso disponible
            resource_id = await asyncio.wait_for(
                self.available_resources.get(),
                timeout=timeout
            )
            
            resource = self.resources.get(resource_id)
            if resource and resource.status == ResourceStatus.IDLE:
                resource.status = ResourceStatus.IN_USE
                resource.last_used_at = datetime.now().isoformat()
                resource.use_count += 1
                return resource
            
            # Si no hay recurso disponible, crear nuevo
            resource = await self._create_resource()
            if resource:
                resource.status = ResourceStatus.IN_USE
                resource.last_used_at = datetime.now().isoformat()
                resource.use_count += 1
                return resource
            
            return None
        except asyncio.TimeoutError:
            logger.warning(f"Timeout acquiring resource from pool {self.name}")
            return None
    
    async def release(self, resource: Resource[T], healthy: bool = True) -> None:
        """
        Liberar recurso al pool.
        
        Args:
            resource: Recurso a liberar
            healthy: Si el recurso está saludable
        """
        if resource.resource_id not in self.resources:
            return
        
        if not healthy:
            resource.status = ResourceStatus.ERROR
            resource.error_count += 1
            
            # Si tiene muchos errores, removerlo
            if resource.error_count >= 3:
                await self._remove_resource(resource)
                return
        
        # Verificar salud si hay health check
        if self.health_check:
            try:
                is_healthy = self.health_check(resource.resource)
                if not is_healthy:
                    resource.status = ResourceStatus.MAINTENANCE
                    await self._remove_resource(resource)
                    return
            except Exception as e:
                logger.warning(f"Health check failed for resource {resource.resource_id}: {e}")
                resource.status = ResourceStatus.ERROR
                await self._remove_resource(resource)
                return
        
        resource.status = ResourceStatus.IDLE
        await self.available_resources.put(resource.resource_id)
    
    async def _remove_resource(self, resource: Resource[T]) -> None:
        """Remover recurso del pool."""
        if resource.resource_id in self.resources:
            del self.resources[resource.resource_id]
            logger.info(f"Removed resource from pool {self.name}: {resource.resource_id}")
            
            # Crear nuevo recurso si es necesario
            if len(self.resources) < self.min_size:
                await self._create_resource()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas del pool."""
        total = len(self.resources)
        idle = sum(1 for r in self.resources.values() if r.status == ResourceStatus.IDLE)
        in_use = sum(1 for r in self.resources.values() if r.status == ResourceStatus.IN_USE)
        
        return {
            "name": self.name,
            "total_resources": total,
            "idle_resources": idle,
            "in_use_resources": in_use,
            "available_resources": self.available_resources.qsize(),
            "max_size": self.max_size,
            "min_size": self.min_size
        }


# Instancia global de pools
_resource_pools: Dict[str, ResourcePool] = {}


def create_resource_pool(
    name: str,
    factory: Callable[[], T],
    max_size: int = 10,
    min_size: int = 2,
    health_check: Optional[Callable[[T], bool]] = None
) -> ResourcePool[T]:
    """
    Crear pool de recursos.
    
    Args:
        name: Nombre del pool
        factory: Función para crear recursos
        max_size: Tamaño máximo
        min_size: Tamaño mínimo
        health_check: Función de verificación de salud
        
    Returns:
        Pool de recursos
    """
    pool = ResourcePool(name, factory, max_size, min_size, health_check=health_check)
    _resource_pools[name] = pool
    return pool


def get_resource_pool(name: str) -> Optional[ResourcePool]:
    """Obtener pool de recursos por nombre."""
    return _resource_pools.get(name)






