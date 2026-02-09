"""
Service Discovery Service - Descubrimiento de servicios
========================================================

Sistema de descubrimiento y registro de servicios.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Estados de servicio"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceInstance:
    """Instancia de servicio"""
    service_id: str
    service_name: str
    host: str
    port: int
    protocol: str = "http"
    status: ServiceStatus = ServiceStatus.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)


@dataclass
class Service:
    """Servicio"""
    name: str
    instances: List[ServiceInstance]
    load_balancer: str = "round_robin"  # round_robin, least_connections, random


class ServiceDiscoveryService:
    """Servicio de descubrimiento de servicios"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.services: Dict[str, Service] = {}
        self.instances: Dict[str, ServiceInstance] = {}  # service_id -> instance
        logger.info("ServiceDiscoveryService initialized")
    
    def register_service(
        self,
        service_name: str,
        host: str,
        port: int,
        protocol: str = "http",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ServiceInstance:
        """Registrar instancia de servicio"""
        service_id = f"{service_name}_{host}_{port}"
        
        instance = ServiceInstance(
            service_id=service_id,
            service_name=service_name,
            host=host,
            port=port,
            protocol=protocol,
            status=ServiceStatus.HEALTHY,
            metadata=metadata or {},
        )
        
        self.instances[service_id] = instance
        
        if service_name not in self.services:
            self.services[service_name] = Service(
                name=service_name,
                instances=[],
            )
        
        # Actualizar o agregar instancia
        existing = next((i for i in self.services[service_name].instances if i.service_id == service_id), None)
        if existing:
            existing.status = instance.status
            existing.last_heartbeat = datetime.now()
        else:
            self.services[service_name].instances.append(instance)
        
        logger.info(f"Service registered: {service_id}")
        return instance
    
    def heartbeat(self, service_id: str) -> bool:
        """Enviar heartbeat"""
        instance = self.instances.get(service_id)
        if not instance:
            return False
        
        instance.last_heartbeat = datetime.now()
        instance.status = ServiceStatus.HEALTHY
        
        return True
    
    def get_service_instance(
        self,
        service_name: str,
        strategy: str = "round_robin"
    ) -> Optional[ServiceInstance]:
        """Obtener instancia de servicio"""
        service = self.services.get(service_name)
        if not service:
            return None
        
        healthy_instances = [
            i for i in service.instances
            if i.status == ServiceStatus.HEALTHY
        ]
        
        if not healthy_instances:
            return None
        
        if strategy == "round_robin":
            # Rotación simple (en producción usaría contador persistente)
            return healthy_instances[0]
        elif strategy == "random":
            import random
            return random.choice(healthy_instances)
        elif strategy == "least_connections":
            # En producción, esto usaría métricas reales
            return healthy_instances[0]
        else:
            return healthy_instances[0]
    
    def get_service_url(
        self,
        service_name: str,
        path: str = "",
        strategy: str = "round_robin"
    ) -> Optional[str]:
        """Obtener URL de servicio"""
        instance = self.get_service_instance(service_name, strategy)
        if not instance:
            return None
        
        base_url = f"{instance.protocol}://{instance.host}:{instance.port}"
        return f"{base_url}{path}" if path else base_url
    
    def mark_unhealthy(self, service_id: str):
        """Marcar servicio como no saludable"""
        instance = self.instances.get(service_id)
        if instance:
            instance.status = ServiceStatus.UNHEALTHY
    
    def cleanup_stale_instances(self, timeout_seconds: int = 60):
        """Limpiar instancias sin heartbeat"""
        cutoff_time = datetime.now() - timedelta(seconds=timeout_seconds)
        
        stale_instances = [
            sid for sid, instance in self.instances.items()
            if instance.last_heartbeat < cutoff_time
        ]
        
        for service_id in stale_instances:
            self.mark_unhealthy(service_id)
            logger.warning(f"Service instance {service_id} marked as unhealthy (stale)")
        
        return len(stale_instances)
    
    def get_all_services(self) -> List[Dict[str, Any]]:
        """Obtener todos los servicios"""
        return [
            {
                "name": s.name,
                "instances_count": len(s.instances),
                "healthy_instances": sum(1 for i in s.instances if i.status == ServiceStatus.HEALTHY),
            }
            for s in self.services.values()
        ]




