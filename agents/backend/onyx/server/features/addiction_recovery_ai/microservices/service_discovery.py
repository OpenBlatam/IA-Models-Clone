"""
Service Discovery
Dynamic service discovery for microservices architecture
"""

import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class ServiceInstance:
    """Service instance information"""
    service_name: str
    instance_id: str
    host: str
    port: int
    protocol: str = "http"
    metadata: Dict = None
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_heartbeat: datetime = None
    version: str = "1.0.0"
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.utcnow()
    
    @property
    def url(self) -> str:
        """Get service URL"""
        return f"{self.protocol}://{self.host}:{self.port}"
    
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        if self.status != ServiceStatus.HEALTHY:
            return False
        
        # Check if heartbeat is recent (within 30 seconds)
        if self.last_heartbeat:
            age = datetime.utcnow() - self.last_heartbeat
            return age.total_seconds() < 30
        
        return False


class ServiceRegistry:
    """
    Service registry for service discovery
    
    Features:
    - Service registration
    - Health checking
    - Load balancing
    - Service discovery
    """
    
    def __init__(self):
        self._services: Dict[str, List[ServiceInstance]] = {}
        self._health_checkers: Dict[str, Callable] = {}
        self._heartbeat_timeout = 60  # seconds
    
    def register(
        self,
        service_name: str,
        instance_id: str,
        host: str,
        port: int,
        protocol: str = "http",
        metadata: Optional[Dict] = None,
        version: str = "1.0.0"
    ) -> ServiceInstance:
        """Register a service instance"""
        instance = ServiceInstance(
            service_name=service_name,
            instance_id=instance_id,
            host=host,
            port=port,
            protocol=protocol,
            metadata=metadata or {},
            version=version
        )
        
        if service_name not in self._services:
            self._services[service_name] = []
        
        # Check if instance already exists
        existing = next(
            (i for i in self._services[service_name] if i.instance_id == instance_id),
            None
        )
        
        if existing:
            # Update existing
            existing.host = host
            existing.port = port
            existing.metadata = metadata or {}
            existing.last_heartbeat = datetime.utcnow()
            existing.status = ServiceStatus.HEALTHY
            logger.info(f"Updated service instance: {service_name}/{instance_id}")
            return existing
        else:
            # Add new instance
            self._services[service_name].append(instance)
            logger.info(f"Registered service instance: {service_name}/{instance_id}")
            return instance
    
    def deregister(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance"""
        if service_name not in self._services:
            return False
        
        service_list = self._services[service_name]
        initial_count = len(service_list)
        self._services[service_name] = [
            i for i in service_list if i.instance_id != instance_id
        ]
        
        removed = len(service_list) != len(self._services[service_name])
        if removed:
            logger.info(f"Deregistered service instance: {service_name}/{instance_id}")
        
        return removed
    
    def discover(self, service_name: str, healthy_only: bool = True) -> List[ServiceInstance]:
        """Discover service instances"""
        if service_name not in self._services:
            return []
        
        instances = self._services[service_name]
        
        if healthy_only:
            instances = [i for i in instances if i.is_healthy()]
        
        return instances
    
    def get_instance(
        self,
        service_name: str,
        strategy: str = "round_robin"
    ) -> Optional[ServiceInstance]:
        """Get a service instance using load balancing strategy"""
        instances = self.discover(service_name, healthy_only=True)
        
        if not instances:
            return None
        
        if strategy == "round_robin":
            # Simple round-robin (in production, use consistent hashing)
            return instances[0]  # Simplified
        elif strategy == "random":
            import random
            return random.choice(instances)
        elif strategy == "least_connections":
            # Would need connection tracking
            return instances[0]
        else:
            return instances[0]
    
    def heartbeat(self, service_name: str, instance_id: str) -> bool:
        """Update service heartbeat"""
        if service_name not in self._services:
            return False
        
        instance = next(
            (i for i in self._services[service_name] if i.instance_id == instance_id),
            None
        )
        
        if instance:
            instance.last_heartbeat = datetime.utcnow()
            instance.status = ServiceStatus.HEALTHY
            return True
        
        return False
    
    def check_health(self, service_name: str, instance_id: str) -> ServiceStatus:
        """Check service health"""
        if service_name not in self._services:
            return ServiceStatus.UNKNOWN
        
        instance = next(
            (i for i in self._services[service_name] if i.instance_id == instance_id),
            None
        )
        
        if not instance:
            return ServiceStatus.UNKNOWN
        
        # Check heartbeat timeout
        if instance.last_heartbeat:
            age = datetime.utcnow() - instance.last_heartbeat
            if age.total_seconds() > self._heartbeat_timeout:
                instance.status = ServiceStatus.UNHEALTHY
                return ServiceStatus.UNHEALTHY
        
        # Use custom health checker if available
        if service_name in self._health_checkers:
            try:
                is_healthy = self._health_checkers[service_name](instance)
                instance.status = ServiceStatus.HEALTHY if is_healthy else ServiceStatus.UNHEALTHY
                return instance.status
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {str(e)}")
                instance.status = ServiceStatus.UNHEALTHY
                return ServiceStatus.UNHEALTHY
        
        return instance.status
    
    def register_health_checker(self, service_name: str, checker: Callable) -> None:
        """Register custom health checker"""
        self._health_checkers[service_name] = checker
    
    def list_services(self) -> List[str]:
        """List all registered services"""
        return list(self._services.keys())
    
    def get_service_info(self, service_name: str) -> Dict:
        """Get service information"""
        if service_name not in self._services:
            return {}
        
        instances = self._services[service_name]
        healthy = [i for i in instances if i.is_healthy()]
        
        return {
            "service_name": service_name,
            "total_instances": len(instances),
            "healthy_instances": len(healthy),
            "instances": [
                {
                    "instance_id": i.instance_id,
                    "url": i.url,
                    "status": i.status.value,
                    "version": i.version
                }
                for i in instances
            ]
        }


# Global registry instance
_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get global service registry"""
    global _registry
    if _registry is None:
        _registry = ServiceRegistry()
    return _registry















