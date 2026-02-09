"""
Service Registry
================

Service discovery and registration for microservices.
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import httpx

logger = logging.getLogger(__name__)


@dataclass
class ServiceInstance:
    """Service instance information."""
    service_name: str
    instance_id: str
    host: str
    port: int
    health_check_url: str
    metadata: Dict = field(default_factory=dict)
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    status: str = "healthy"  # healthy, unhealthy, unknown
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "service_name": self.service_name,
            "instance_id": self.instance_id,
            "host": self.host,
            "port": self.port,
            "health_check_url": self.health_check_url,
            "metadata": self.metadata,
            "status": self.status,
        }


class ServiceRegistry:
    """Registry for service discovery."""
    
    def __init__(self, heartbeat_interval: int = 30, heartbeat_timeout: int = 60):
        self._services: Dict[str, List[ServiceInstance]] = {}
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    def register(self, instance: ServiceInstance) -> bool:
        """Register a service instance."""
        if instance.service_name not in self._services:
            self._services[instance.service_name] = []
        
        # Check if instance already exists
        existing = next(
            (i for i in self._services[instance.service_name] 
             if i.instance_id == instance.instance_id),
            None
        )
        
        if existing:
            existing.last_heartbeat = datetime.utcnow()
            existing.status = "healthy"
            logger.debug(f"Updated heartbeat for {instance.service_name}:{instance.instance_id}")
        else:
            self._services[instance.service_name].append(instance)
            logger.info(f"Registered service: {instance.service_name}:{instance.instance_id}")
        
        return True
    
    def deregister(self, service_name: str, instance_id: str) -> bool:
        """Deregister a service instance."""
        if service_name not in self._services:
            return False
        
        self._services[service_name] = [
            i for i in self._services[service_name]
            if i.instance_id != instance_id
        ]
        
        logger.info(f"Deregistered service: {service_name}:{instance_id}")
        return True
    
    def get_instances(self, service_name: str, healthy_only: bool = True) -> List[ServiceInstance]:
        """Get instances of a service."""
        if service_name not in self._services:
            return []
        
        instances = self._services[service_name]
        
        if healthy_only:
            instances = [i for i in instances if i.status == "healthy"]
        
        return instances
    
    def get_instance(self, service_name: str, strategy: str = "round_robin") -> Optional[ServiceInstance]:
        """Get a single instance using load balancing strategy."""
        instances = self.get_instances(service_name, healthy_only=True)
        
        if not instances:
            return None
        
        if strategy == "round_robin":
            # Simple round-robin (in production, use consistent hashing)
            return instances[0]
        elif strategy == "random":
            import random
            return random.choice(instances)
        else:
            return instances[0]
    
    def list_services(self) -> List[str]:
        """List all registered service names."""
        return list(self._services.keys())
    
    def start_heartbeat_checker(self):
        """Start background heartbeat checker."""
        if self._running:
            return
        
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Service registry heartbeat checker started")
    
    def stop_heartbeat_checker(self):
        """Stop background heartbeat checker."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        logger.info("Service registry heartbeat checker stopped")
    
    async def _heartbeat_loop(self):
        """Background loop to check service health."""
        while self._running:
            try:
                await asyncio.sleep(self._heartbeat_interval)
                await self._check_heartbeats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def _check_heartbeats(self):
        """Check health of all registered services."""
        for service_name, instances in self._services.items():
            for instance in instances:
                try:
                    timeout = httpx.Timeout(5.0)
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.get(instance.health_check_url)
                        
                        if response.status_code == 200:
                            instance.status = "healthy"
                            instance.last_heartbeat = datetime.utcnow()
                        else:
                            instance.status = "unhealthy"
                except Exception as e:
                    logger.warning(f"Health check failed for {instance.service_name}:{instance.instance_id}: {e}")
                    instance.status = "unhealthy"
                
                # Remove stale instances
                if instance.status == "unhealthy":
                    time_since_heartbeat = (datetime.utcnow() - instance.last_heartbeat).total_seconds()
                    if time_since_heartbeat > self._heartbeat_timeout:
                        logger.warning(f"Removing stale instance: {instance.service_name}:{instance.instance_id}")
                        self.deregister(service_name, instance.instance_id)


# Global registry instance
_global_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get global service registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
    return _global_registry















