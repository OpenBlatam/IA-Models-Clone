"""
Service Discovery
=================
Utilidades para service discovery.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio
from .logger import get_logger

logger = get_logger(__name__)


class ServiceStatus(str, Enum):
    """Estados de servicio."""
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class Service:
    """Representación de un servicio."""
    
    def __init__(
        self,
        name: str,
        url: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar servicio.
        
        Args:
            name: Nombre del servicio
            url: URL del servicio
            version: Versión del servicio
            metadata: Metadatos adicionales
        """
        self.name = name
        self.url = url
        self.version = version
        self.metadata = metadata or {}
        self.status = ServiceStatus.UNKNOWN
        self.last_health_check: Optional[datetime] = None
        self.registered_at = datetime.now()
    
    def update_status(self, status: ServiceStatus):
        """Actualizar estado."""
        self.status = status
        self.last_health_check = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "name": self.name,
            "url": self.url,
            "version": self.version,
            "status": self.status.value,
            "metadata": self.metadata,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "registered_at": self.registered_at.isoformat()
        }


class ServiceRegistry:
    """Registry para servicios."""
    
    def __init__(self, health_check_interval: float = 30.0):
        """
        Inicializar registry.
        
        Args:
            health_check_interval: Intervalo de health checks en segundos
        """
        self.services: Dict[str, Service] = {}
        self.health_check_interval = health_check_interval
        self._lock = asyncio.Lock()
    
    async def register(self, service: Service):
        """Registrar servicio."""
        async with self._lock:
            self.services[service.name] = service
            logger.info(f"Service registered: {service.name}")
    
    async def unregister(self, service_name: str):
        """Desregistrar servicio."""
        async with self._lock:
            if service_name in self.services:
                del self.services[service_name]
                logger.info(f"Service unregistered: {service_name}")
    
    async def get_service(self, name: str) -> Optional[Service]:
        """Obtener servicio por nombre."""
        return self.services.get(name)
    
    async def list_services(
        self,
        status: Optional[ServiceStatus] = None
    ) -> List[Service]:
        """
        Listar servicios.
        
        Args:
            status: Filtrar por estado (opcional)
            
        Returns:
            Lista de servicios
        """
        services = list(self.services.values())
        
        if status:
            services = [s for s in services if s.status == status]
        
        return services
    
    async def discover_service(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[Service]:
        """
        Descubrir servicio.
        
        Args:
            name: Nombre del servicio
            version: Versión específica (opcional)
            
        Returns:
            Servicio encontrado
        """
        service = await self.get_service(name)
        
        if not service:
            return None
        
        if version and service.version != version:
            return None
        
        if service.status != ServiceStatus.UP:
            return None
        
        return service
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del registry."""
        status_counts = {}
        for service in self.services.values():
            status_counts[service.status.value] = status_counts.get(service.status.value, 0) + 1
        
        return {
            "total_services": len(self.services),
            "services_by_status": status_counts,
            "services": [s.to_dict() for s in self.services.values()]
        }


# Instancia global
_service_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Obtener instancia global del registry."""
    global _service_registry
    if _service_registry is None:
        _service_registry = ServiceRegistry()
    return _service_registry

