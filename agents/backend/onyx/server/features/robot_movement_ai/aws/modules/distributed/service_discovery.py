"""
Service Discovery
=================

Advanced service discovery.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ServiceInstance:
    """Service instance."""
    id: str
    service_name: str
    endpoint: str
    port: int
    health: bool = True
    registered_at: datetime = None
    last_heartbeat: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.registered_at is None:
            self.registered_at = datetime.now()
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class ServiceDiscovery:
    """Service discovery manager."""
    
    def __init__(self, heartbeat_timeout: float = 30.0):
        self.heartbeat_timeout = heartbeat_timeout
        self._services: Dict[str, List[ServiceInstance]] = {}  # service_name -> instances
        self._instances: Dict[str, ServiceInstance] = {}  # instance_id -> instance
    
    def register_service(
        self,
        instance_id: str,
        service_name: str,
        endpoint: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ServiceInstance:
        """Register service instance."""
        instance = ServiceInstance(
            id=instance_id,
            service_name=service_name,
            endpoint=endpoint,
            port=port,
            metadata=metadata or {}
        )
        
        if service_name not in self._services:
            self._services[service_name] = []
        
        self._services[service_name].append(instance)
        self._instances[instance_id] = instance
        
        logger.info(f"Registered service instance: {instance_id} for {service_name}")
        return instance
    
    def deregister_service(self, instance_id: str):
        """Deregister service instance."""
        if instance_id not in self._instances:
            return
        
        instance = self._instances[instance_id]
        service_name = instance.service_name
        
        if service_name in self._services:
            self._services[service_name] = [
                i for i in self._services[service_name]
                if i.id != instance_id
            ]
        
        del self._instances[instance_id]
        logger.info(f"Deregistered service instance: {instance_id}")
    
    def heartbeat(self, instance_id: str):
        """Update service heartbeat."""
        if instance_id in self._instances:
            self._instances[instance_id].last_heartbeat = datetime.now()
            self._instances[instance_id].health = True
    
    def discover_services(
        self,
        service_name: str,
        healthy_only: bool = True
    ) -> List[ServiceInstance]:
        """Discover service instances."""
        if service_name not in self._services:
            return []
        
        instances = self._services[service_name]
        
        # Filter unhealthy instances
        now = datetime.now()
        healthy_instances = [
            i for i in instances
            if i.health and (now - i.last_heartbeat).total_seconds() < self.heartbeat_timeout
        ]
        
        if healthy_only:
            return healthy_instances
        
        return instances
    
    def get_service_instance(self, instance_id: str) -> Optional[ServiceInstance]:
        """Get service instance by ID."""
        return self._instances.get(instance_id)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service discovery statistics."""
        now = datetime.now()
        healthy_count = sum(
            1 for i in self._instances.values()
            if i.health and (now - i.last_heartbeat).total_seconds() < self.heartbeat_timeout
        )
        
        return {
            "total_services": len(self._services),
            "total_instances": len(self._instances),
            "healthy_instances": healthy_count,
            "unhealthy_instances": len(self._instances) - healthy_count,
            "by_service": {
                service_name: len(instances)
                for service_name, instances in self._services.items()
            }
        }















