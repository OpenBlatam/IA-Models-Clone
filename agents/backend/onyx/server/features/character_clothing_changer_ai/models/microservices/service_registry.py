"""
Microservices Service Registry
================================
Sistema de registro y descubrimiento de servicios
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ServiceStatus(Enum):
    """Estados de servicio"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


@dataclass
class ServiceEndpoint:
    """Endpoint de servicio"""
    protocol: str  # http, https, grpc, etc
    host: str
    port: int
    path: Optional[str] = None


@dataclass
class Service:
    """Servicio registrado"""
    id: str
    name: str
    version: str
    endpoints: List[ServiceEndpoint]
    status: ServiceStatus
    metadata: Dict[str, Any]
    registered_at: float
    last_heartbeat: float
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ServiceRegistry:
    """
    Registro de servicios para arquitectura de microservicios
    """
    
    def __init__(self):
        self.services: Dict[str, Service] = {}
        self.service_instances: Dict[str, List[str]] = {}  # service_name -> [instance_ids]
        self.heartbeat_timeout = 30  # segundos
    
    def register_service(
        self,
        name: str,
        version: str,
        endpoints: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Service:
        """
        Registrar servicio
        
        Args:
            name: Nombre del servicio
            version: Versión del servicio
            endpoints: Lista de endpoints
            metadata: Metadata adicional
            tags: Tags del servicio
        """
        service_id = f"{name}_{version}_{int(time.time())}"
        
        service_endpoints = [
            ServiceEndpoint(
                protocol=ep.get('protocol', 'http'),
                host=ep.get('host', 'localhost'),
                port=ep.get('port', 8000),
                path=ep.get('path')
            )
            for ep in endpoints
        ]
        
        service = Service(
            id=service_id,
            name=name,
            version=version,
            endpoints=service_endpoints,
            status=ServiceStatus.STARTING,
            metadata=metadata or {},
            registered_at=time.time(),
            last_heartbeat=time.time(),
            tags=tags or []
        )
        
        self.services[service_id] = service
        
        if name not in self.service_instances:
            self.service_instances[name] = []
        self.service_instances[name].append(service_id)
        
        return service
    
    def update_service_status(self, service_id: str, status: ServiceStatus):
        """Actualizar estado de servicio"""
        if service_id in self.services:
            self.services[service_id].status = status
    
    def heartbeat(self, service_id: str):
        """Registrar heartbeat de servicio"""
        if service_id in self.services:
            self.services[service_id].last_heartbeat = time.time()
            if self.services[service_id].status == ServiceStatus.STARTING:
                self.services[service_id].status = ServiceStatus.HEALTHY
    
    def get_service(self, service_id: str) -> Optional[Service]:
        """Obtener servicio por ID"""
        return self.services.get(service_id)
    
    def find_services(
        self,
        name: Optional[str] = None,
        version: Optional[str] = None,
        status: Optional[ServiceStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[Service]:
        """
        Buscar servicios
        
        Args:
            name: Filtrar por nombre
            version: Filtrar por versión
            status: Filtrar por estado
            tags: Filtrar por tags
        """
        results = []
        
        for service in self.services.values():
            if name and service.name != name:
                continue
            if version and service.version != version:
                continue
            if status and service.status != status:
                continue
            if tags and not any(tag in service.tags for tag in tags):
                continue
            
            results.append(service)
        
        return results
    
    def get_healthy_instances(self, service_name: str) -> List[Service]:
        """Obtener instancias saludables de un servicio"""
        if service_name not in self.service_instances:
            return []
        
        healthy = []
        current_time = time.time()
        
        for instance_id in self.service_instances[service_name]:
            service = self.services.get(instance_id)
            if not service:
                continue
            
            # Verificar heartbeat
            if current_time - service.last_heartbeat > self.heartbeat_timeout:
                service.status = ServiceStatus.UNHEALTHY
                continue
            
            if service.status == ServiceStatus.HEALTHY:
                healthy.append(service)
        
        return healthy
    
    def deregister_service(self, service_id: str) -> bool:
        """Desregistrar servicio"""
        if service_id not in self.services:
            return False
        
        service = self.services[service_id]
        service.status = ServiceStatus.STOPPING
        
        # Remover de instancias
        if service.name in self.service_instances:
            if service_id in self.service_instances[service.name]:
                self.service_instances[service.name].remove(service_id)
        
        del self.services[service_id]
        return True
    
    def cleanup_stale_services(self):
        """Limpiar servicios sin heartbeat"""
        current_time = time.time()
        stale_services = []
        
        for service_id, service in self.services.items():
            if current_time - service.last_heartbeat > self.heartbeat_timeout:
                if service.status == ServiceStatus.HEALTHY:
                    service.status = ServiceStatus.UNHEALTHY
                    stale_services.append(service_id)
        
        return stale_services
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """Obtener estadísticas de servicios"""
        status_counts = {}
        for service in self.services.values():
            status = service.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_services': len(self.services),
            'total_service_types': len(self.service_instances),
            'status_counts': status_counts,
            'healthy_services': len([s for s in self.services.values() if s.status == ServiceStatus.HEALTHY]),
            'unhealthy_services': len([s for s in self.services.values() if s.status == ServiceStatus.UNHEALTHY])
        }


# Instancia global
service_registry = ServiceRegistry()

