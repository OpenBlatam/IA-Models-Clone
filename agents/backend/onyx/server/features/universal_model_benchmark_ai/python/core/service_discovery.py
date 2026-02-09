"""
Service Discovery Module - Service discovery and registration.

Provides:
- Service registration
- Service discovery
- Health checking
- Load balancing
- Service metadata
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DRAINING = "draining"


@dataclass
class Service:
    """Service definition."""
    id: str
    name: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())
    health_check_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "metadata": self.metadata,
            "tags": self.tags,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "health_check_url": self.health_check_url,
        }


class ServiceRegistry:
    """Service registry."""
    
    def __init__(self, heartbeat_timeout: float = 30.0):
        """
        Initialize service registry.
        
        Args:
            heartbeat_timeout: Heartbeat timeout in seconds
        """
        self.services: Dict[str, Service] = {}
        self.lock = Lock()
        self.heartbeat_timeout = heartbeat_timeout
    
    def register(
        self,
        name: str,
        host: str,
        port: int,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        health_check_url: Optional[str] = None,
    ) -> Service:
        """
        Register a service.
        
        Args:
            name: Service name
            host: Service host
            port: Service port
            metadata: Optional metadata
            tags: Optional tags
            health_check_url: Optional health check URL
            
        Returns:
            Registered service
        """
        service_id = f"{name}_{host}_{port}_{int(time.time())}"
        
        service = Service(
            id=service_id,
            name=name,
            host=host,
            port=port,
            metadata=metadata or {},
            tags=tags or [],
            health_check_url=health_check_url,
        )
        
        with self.lock:
            self.services[service_id] = service
        
        logger.info(f"Registered service: {name} at {host}:{port}")
        return service
    
    def deregister(self, service_id: str) -> None:
        """
        Deregister a service.
        
        Args:
            service_id: Service ID
        """
        with self.lock:
            if service_id in self.services:
                service = self.services.pop(service_id)
                logger.info(f"Deregistered service: {service.name}")
    
    def update_heartbeat(self, service_id: str) -> None:
        """
        Update service heartbeat.
        
        Args:
            service_id: Service ID
        """
        with self.lock:
            if service_id in self.services:
                self.services[service_id].last_heartbeat = datetime.now().isoformat()
                self.services[service_id].status = ServiceStatus.HEALTHY
    
    def discover(self, name: str, tags: Optional[List[str]] = None) -> List[Service]:
        """
        Discover services by name.
        
        Args:
            name: Service name
            tags: Optional tags filter
            
        Returns:
            List of matching services
        """
        with self.lock:
            services = [
                s for s in self.services.values()
                if s.name == name and s.status == ServiceStatus.HEALTHY
            ]
            
            if tags:
                services = [
                    s for s in services
                    if all(tag in s.tags for tag in tags)
                ]
            
            return services
    
    def get_service(self, service_id: str) -> Optional[Service]:
        """Get service by ID."""
        with self.lock:
            return self.services.get(service_id)
    
    def list_services(self, name: Optional[str] = None) -> List[Service]:
        """
        List all services.
        
        Args:
            name: Optional name filter
            
        Returns:
            List of services
        """
        with self.lock:
            services = list(self.services.values())
            if name:
                services = [s for s in services if s.name == name]
            return services
    
    def check_health(self) -> None:
        """Check health of all services."""
        with self.lock:
            now = datetime.now()
            for service in self.services.values():
                last_heartbeat = datetime.fromisoformat(service.last_heartbeat)
                elapsed = (now - last_heartbeat).total_seconds()
                
                if elapsed > self.heartbeat_timeout:
                    service.status = ServiceStatus.UNHEALTHY
                    logger.warning(f"Service {service.name} is unhealthy (no heartbeat)")


class LoadBalancer:
    """Load balancer."""
    
    class Strategy(str, Enum):
        """Load balancing strategies."""
        ROUND_ROBIN = "round_robin"
        LEAST_CONNECTIONS = "least_connections"
        RANDOM = "random"
        WEIGHTED = "weighted"
    
    def __init__(self, strategy: Strategy = Strategy.ROUND_ROBIN):
        """
        Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
        """
        self.strategy = strategy
        self.current_index = 0
        self.connection_counts: Dict[str, int] = {}
    
    def select_service(self, services: List[Service]) -> Optional[Service]:
        """
        Select a service using load balancing strategy.
        
        Args:
            services: List of available services
            
        Returns:
            Selected service
        """
        if not services:
            return None
        
        if self.strategy == self.Strategy.ROUND_ROBIN:
            service = services[self.current_index % len(services)]
            self.current_index += 1
            return service
        
        elif self.strategy == self.Strategy.RANDOM:
            import random
            return random.choice(services)
        
        elif self.strategy == self.Strategy.LEAST_CONNECTIONS:
            # Select service with least connections
            min_connections = min(
                self.connection_counts.get(s.id, 0) for s in services
            )
            candidates = [
                s for s in services
                if self.connection_counts.get(s.id, 0) == min_connections
            ]
            return candidates[0] if candidates else services[0]
        
        else:
            return services[0]
    
    def record_connection(self, service_id: str) -> None:
        """Record a connection to service."""
        self.connection_counts[service_id] = self.connection_counts.get(service_id, 0) + 1
    
    def release_connection(self, service_id: str) -> None:
        """Release a connection from service."""
        if service_id in self.connection_counts:
            self.connection_counts[service_id] = max(0, self.connection_counts[service_id] - 1)












