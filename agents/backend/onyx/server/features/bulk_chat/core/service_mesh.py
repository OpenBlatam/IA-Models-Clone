"""
Service Mesh - Malla de Servicios
===================================

Sistema de malla de servicios con discovery, load balancing y observabilidad.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Estado de servicio."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class LoadBalancingStrategy(Enum):
    """Estrategia de load balancing."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    CONSISTENT_HASH = "consistent_hash"
    RANDOM = "random"
    WEIGHTED = "weighted"


@dataclass
class ServiceInstance:
    """Instancia de servicio."""
    instance_id: str
    service_name: str
    address: str
    port: int
    status: ServiceStatus = ServiceStatus.UNKNOWN
    weight: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.now)
    last_health_check: Optional[datetime] = None


@dataclass
class Service:
    """Servicio."""
    service_name: str
    instances: List[ServiceInstance] = field(default_factory=list)
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check_interval: float = 30.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ServiceMesh:
    """Malla de servicios."""
    
    def __init__(self):
        self.services: Dict[str, Service] = {}
        self.instance_usage: Dict[str, int] = defaultdict(int)
        self.round_robin_index: Dict[str, int] = defaultdict(int)
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
    
    def register_service(
        self,
        service_name: str,
        load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
        health_check_interval: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar servicio."""
        service = Service(
            service_name=service_name,
            load_balancing_strategy=load_balancing_strategy,
            health_check_interval=health_check_interval,
            metadata=metadata or {},
        )
        
        async def save_service():
            async with self._lock:
                self.services[service_name] = service
        
        asyncio.create_task(save_service())
        
        logger.info(f"Registered service: {service_name}")
        return service_name
    
    def register_instance(
        self,
        instance_id: str,
        service_name: str,
        address: str,
        port: int,
        weight: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Registrar instancia de servicio."""
        instance = ServiceInstance(
            instance_id=instance_id,
            service_name=service_name,
            address=address,
            port=port,
            weight=weight,
            metadata=metadata or {},
        )
        
        async def save_instance():
            async with self._lock:
                service = self.services.get(service_name)
                if not service:
                    # Crear servicio automáticamente
                    service = Service(service_name=service_name)
                    self.services[service_name] = service
                
                # Verificar si ya existe
                existing = next(
                    (inst for inst in service.instances if inst.instance_id == instance_id),
                    None
                )
                if existing:
                    # Actualizar
                    existing.address = address
                    existing.port = port
                    existing.weight = weight
                    existing.metadata = metadata or {}
                else:
                    service.instances.append(instance)
        
        asyncio.create_task(save_instance())
        
        logger.info(f"Registered instance {instance_id} for service {service_name}")
        return instance_id
    
    def get_instance(
        self,
        service_name: str,
        client_id: Optional[str] = None,
    ) -> Optional[ServiceInstance]:
        """Obtener instancia de servicio."""
        service = self.services.get(service_name)
        if not service:
            return None
        
        # Filtrar instancias saludables
        healthy_instances = [
            inst for inst in service.instances
            if inst.status == ServiceStatus.HEALTHY
        ]
        
        if not healthy_instances:
            return None
        
        # Aplicar estrategia de load balancing
        if service.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            index = self.round_robin_index[service_name] % len(healthy_instances)
            instance = healthy_instances[index]
            self.round_robin_index[service_name] += 1
            return instance
        
        elif service.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            instance = min(
                healthy_instances,
                key=lambda inst: self.instance_usage.get(inst.instance_id, 0)
            )
            self.instance_usage[instance.instance_id] += 1
            return instance
        
        elif service.load_balancing_strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            if client_id:
                hash_value = int(hashlib.md5(f"{service_name}_{client_id}".encode()).hexdigest(), 16)
                index = hash_value % len(healthy_instances)
                return healthy_instances[index]
            return healthy_instances[0]
        
        elif service.load_balancing_strategy == LoadBalancingStrategy.RANDOM:
            import random
            return random.choice(healthy_instances)
        
        elif service.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED:
            # Selección basada en pesos
            total_weight = sum(inst.weight for inst in healthy_instances)
            if total_weight == 0:
                return healthy_instances[0]
            
            import random
            rand = random.uniform(0, total_weight)
            cumulative = 0
            for inst in healthy_instances:
                cumulative += inst.weight
                if rand <= cumulative:
                    return inst
            
            return healthy_instances[0]
        
        return healthy_instances[0]
    
    def update_instance_status(
        self,
        instance_id: str,
        status: ServiceStatus,
    ):
        """Actualizar estado de instancia."""
        async def update():
            async with self._lock:
                for service in self.services.values():
                    for instance in service.instances:
                        if instance.instance_id == instance_id:
                            instance.status = status
                            instance.last_health_check = datetime.now()
                            return
        
        asyncio.create_task(update())
    
    def get_service_instances(self, service_name: str) -> List[Dict[str, Any]]:
        """Obtener instancias de servicio."""
        service = self.services.get(service_name)
        if not service:
            return []
        
        return [
            {
                "instance_id": inst.instance_id,
                "address": inst.address,
                "port": inst.port,
                "status": inst.status.value,
                "weight": inst.weight,
                "registered_at": inst.registered_at.isoformat(),
            }
            for inst in service.instances
        ]
    
    def get_service_mesh_summary(self) -> Dict[str, Any]:
        """Obtener resumen de la malla."""
        by_status: Dict[str, int] = defaultdict(int)
        total_instances = 0
        
        for service in self.services.values():
            for instance in service.instances:
                by_status[instance.status.value] += 1
                total_instances += 1
        
        return {
            "total_services": len(self.services),
            "total_instances": total_instances,
            "instances_by_status": dict(by_status),
        }


