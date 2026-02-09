"""
Service Discovery - Service Discovery
====================================

Service discovery para microservicios:
- Service registration
- Service lookup
- Health-based filtering
- Load balancing integration
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Estados de servicio"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceInstance:
    """Instancia de servicio"""
    
    def __init__(
        self,
        service_name: str,
        instance_id: str,
        host: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self.service_name = service_name
        self.instance_id = instance_id
        self.host = host
        self.port = port
        self.metadata = metadata or {}
        self.status = ServiceStatus.UNKNOWN
        self.registered_at = datetime.now()
        self.last_health_check: Optional[datetime] = None
    
    @property
    def url(self) -> str:
        """URL del servicio"""
        return f"http://{self.host}:{self.port}"
    
    def is_healthy(self) -> bool:
        """Verifica si está saludable"""
        if self.status != ServiceStatus.HEALTHY:
            return False
        
        # Verificar si el health check es reciente (últimos 30 segundos)
        if self.last_health_check:
            elapsed = (datetime.now() - self.last_health_check).total_seconds()
            return elapsed < 30
        
        return False


class ServiceRegistry:
    """
    Registry de servicios.
    """
    
    def __init__(self) -> None:
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.instances: Dict[str, ServiceInstance] = {}
    
    def register(
        self,
        service_name: str,
        instance_id: str,
        host: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ServiceInstance:
        """Registra instancia de servicio"""
        instance = ServiceInstance(service_name, instance_id, host, port, metadata)
        
        if service_name not in self.services:
            self.services[service_name] = []
        
        self.services[service_name].append(instance)
        self.instances[instance_id] = instance
        
        logger.info(f"Service instance registered: {service_name}@{instance_id}")
        return instance
    
    def deregister(self, instance_id: str) -> bool:
        """Desregistra instancia"""
        instance = self.instances.get(instance_id)
        if not instance:
            return False
        
        service_name = instance.service_name
        if service_name in self.services:
            self.services[service_name] = [
                i for i in self.services[service_name]
                if i.instance_id != instance_id
            ]
        
        del self.instances[instance_id]
        logger.info(f"Service instance deregistered: {instance_id}")
        return True
    
    def discover(
        self,
        service_name: str,
        healthy_only: bool = True
    ) -> List[ServiceInstance]:
        """Descubre instancias de un servicio"""
        instances = self.services.get(service_name, [])
        
        if healthy_only:
            instances = [i for i in instances if i.is_healthy()]
        
        return instances
    
    def get_instance(self, instance_id: str) -> Optional[ServiceInstance]:
        """Obtiene instancia por ID"""
        return self.instances.get(instance_id)
    
    def update_health(
        self,
        instance_id: str,
        status: ServiceStatus
    ) -> None:
        """Actualiza estado de salud"""
        instance = self.instances.get(instance_id)
        if instance:
            instance.status = status
            instance.last_health_check = datetime.now()
    
    def get_all_services(self) -> List[str]:
        """Obtiene lista de todos los servicios"""
        return list(self.services.keys())


def get_service_registry() -> ServiceRegistry:
    """Obtiene registry de servicios"""
    return ServiceRegistry()















