"""
Service Discovery - Sistema de descubrimiento de servicios
==========================================================
"""

import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Estados de servicio"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DRAINING = "draining"


@dataclass
class ServiceInstance:
    """Instancia de servicio"""
    id: str
    service_name: str
    host: str
    port: int
    protocol: str = "http"
    status: ServiceStatus = ServiceStatus.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    health_check_url: Optional[str] = None
    weight: int = 1  # Para load balancing
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "id": self.id,
            "service_name": self.service_name,
            "host": self.host,
            "port": self.port,
            "protocol": self.protocol,
            "status": self.status.value,
            "metadata": self.metadata,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "health_check_url": self.health_check_url,
            "weight": self.weight,
            "url": f"{self.protocol}://{self.host}:{self.port}"
        }


class ServiceDiscovery:
    """Sistema de descubrimiento de servicios"""
    
    def __init__(self, heartbeat_timeout: float = 30.0):
        self.services: Dict[str, List[ServiceInstance]] = {}  # service_name -> instances
        self.instances: Dict[str, ServiceInstance] = {}  # instance_id -> instance
        self.heartbeat_timeout = heartbeat_timeout
        self._health_check_interval = 10.0
    
    def register_service(
        self,
        service_name: str,
        host: str,
        port: int,
        instance_id: Optional[str] = None,
        protocol: str = "http",
        metadata: Optional[Dict[str, Any]] = None,
        health_check_url: Optional[str] = None,
        weight: int = 1
    ) -> ServiceInstance:
        """Registra un servicio"""
        if instance_id is None:
            import uuid
            instance_id = str(uuid.uuid4())
        
        instance = ServiceInstance(
            id=instance_id,
            service_name=service_name,
            host=host,
            port=port,
            protocol=protocol,
            metadata=metadata or {},
            health_check_url=health_check_url,
            weight=weight
        )
        
        if service_name not in self.services:
            self.services[service_name] = []
        
        self.services[service_name].append(instance)
        self.instances[instance_id] = instance
        
        logger.info(f"Servicio {service_name} registrado: {instance.id}")
        return instance
    
    def deregister_service(self, instance_id: str) -> bool:
        """Desregistra un servicio"""
        if instance_id not in self.instances:
            return False
        
        instance = self.instances[instance_id]
        service_name = instance.service_name
        
        if service_name in self.services:
            self.services[service_name] = [
                inst for inst in self.services[service_name]
                if inst.id != instance_id
            ]
            if not self.services[service_name]:
                del self.services[service_name]
        
        del self.instances[instance_id]
        logger.info(f"Servicio {instance_id} desregistrado")
        return True
    
    def heartbeat(self, instance_id: str) -> bool:
        """Registra un heartbeat"""
        if instance_id not in self.instances:
            return False
        
        instance = self.instances[instance_id]
        instance.last_heartbeat = datetime.now()
        
        # Marcar como healthy si estaba unknown
        if instance.status == ServiceStatus.UNKNOWN:
            instance.status = ServiceStatus.HEALTHY
        
        return True
    
    def get_instances(
        self,
        service_name: str,
        healthy_only: bool = True
    ) -> List[ServiceInstance]:
        """Obtiene instancias de un servicio"""
        if service_name not in self.services:
            return []
        
        instances = self.services[service_name]
        
        # Filtrar instancias no saludables
        if healthy_only:
            instances = [
                inst for inst in instances
                if inst.status == ServiceStatus.HEALTHY
            ]
        
        # Filtrar instancias con heartbeat expirado
        now = datetime.now()
        instances = [
            inst for inst in instances
            if (now - inst.last_heartbeat).total_seconds() < self.heartbeat_timeout
        ]
        
        return instances
    
    def get_instance(self, instance_id: str) -> Optional[ServiceInstance]:
        """Obtiene una instancia específica"""
        return self.instances.get(instance_id)
    
    def update_instance_status(
        self,
        instance_id: str,
        status: ServiceStatus
    ) -> bool:
        """Actualiza el estado de una instancia"""
        if instance_id not in self.instances:
            return False
        
        instance = self.instances[instance_id]
        instance.status = status
        return True
    
    def mark_draining(self, instance_id: str) -> bool:
        """Marca una instancia como draining"""
        return self.update_instance_status(instance_id, ServiceStatus.DRAINING)
    
    def cleanup_stale_instances(self):
        """Limpia instancias con heartbeat expirado"""
        now = datetime.now()
        stale_instances = []
        
        for instance_id, instance in self.instances.items():
            elapsed = (now - instance.last_heartbeat).total_seconds()
            if elapsed > self.heartbeat_timeout:
                stale_instances.append(instance_id)
                instance.status = ServiceStatus.UNHEALTHY
        
        for instance_id in stale_instances:
            logger.warning(f"Instancia {instance_id} marcada como unhealthy (heartbeat expirado)")
    
    def list_services(self) -> List[str]:
        """Lista todos los servicios registrados"""
        return list(self.services.keys())
    
    def get_service_info(self, service_name: str) -> Dict[str, Any]:
        """Obtiene información de un servicio"""
        instances = self.get_instances(service_name, healthy_only=False)
        
        return {
            "service_name": service_name,
            "total_instances": len(instances),
            "healthy_instances": len([i for i in instances if i.status == ServiceStatus.HEALTHY]),
            "unhealthy_instances": len([i for i in instances if i.status == ServiceStatus.UNHEALTHY]),
            "instances": [i.to_dict() for i in instances]
        }
    
    async def health_check(self, instance: ServiceInstance) -> bool:
        """Realiza health check de una instancia"""
        if not instance.health_check_url:
            return True
        
        try:
            import httpx
            url = instance.health_check_url
            if not url.startswith("http"):
                url = f"{instance.protocol}://{instance.host}:{instance.port}{url}"
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)
                is_healthy = response.status_code == 200
                
                if is_healthy:
                    instance.status = ServiceStatus.HEALTHY
                else:
                    instance.status = ServiceStatus.UNHEALTHY
                
                return is_healthy
        except Exception as e:
            logger.error(f"Error en health check de {instance.id}: {e}")
            instance.status = ServiceStatus.UNHEALTHY
            return False




