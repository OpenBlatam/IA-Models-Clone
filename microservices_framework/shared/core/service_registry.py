"""
Service Registry for Microservices Architecture
Implements service discovery, health checks, and load balancing
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import redis.asyncio as redis
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"

class ServiceType(Enum):
    """Service type enumeration"""
    API = "api"
    WORKER = "worker"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_BROKER = "message_broker"

@dataclass
class ServiceInstance:
    """Represents a service instance"""
    service_id: str
    service_name: str
    service_type: ServiceType
    host: str
    port: int
    version: str
    status: ServiceStatus
    health_check_url: str
    metadata: Dict[str, Any]
    last_heartbeat: float
    registered_at: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['service_type'] = self.service_type.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceInstance':
        """Create from dictionary"""
        data['service_type'] = ServiceType(data['service_type'])
        data['status'] = ServiceStatus(data['status'])
        return cls(**data)

class ServiceRegistry:
    """
    Service Registry for microservices architecture
    
    Features:
    - Service registration and discovery
    - Health checks and monitoring
    - Load balancing
    - Service metadata management
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.health_check_interval = 30  # seconds
        self.service_ttl = 60  # seconds
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the service registry"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Service registry started successfully")
            
            # Start health check task
            self._running = True
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
        except Exception as e:
            logger.error(f"Failed to start service registry: {e}")
            raise
    
    async def stop(self):
        """Stop the service registry"""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Service registry stopped")
    
    async def register_service(self, service: ServiceInstance) -> bool:
        """
        Register a service instance
        
        Args:
            service: Service instance to register
            
        Returns:
            bool: True if registration successful
        """
        try:
            # Update registration time
            service.registered_at = time.time()
            service.last_heartbeat = time.time()
            
            # Store in Redis
            service_key = f"service:{service.service_name}:{service.service_id}"
            await self.redis_client.setex(
                service_key,
                self.service_ttl,
                json.dumps(service.to_dict())
            )
            
            # Add to service list
            service_list_key = f"services:{service.service_name}"
            await self.redis_client.sadd(service_list_key, service.service_id)
            
            # Update local cache
            if service.service_name not in self.services:
                self.services[service.service_name] = []
            
            # Remove existing instance with same ID
            self.services[service.service_name] = [
                s for s in self.services[service.service_name] 
                if s.service_id != service.service_id
            ]
            
            # Add new instance
            self.services[service.service_name].append(service)
            
            logger.info(f"Registered service: {service.service_name}:{service.service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service.service_name}:{service.service_id}: {e}")
            return False
    
    async def unregister_service(self, service_name: str, service_id: str) -> bool:
        """
        Unregister a service instance
        
        Args:
            service_name: Name of the service
            service_id: ID of the service instance
            
        Returns:
            bool: True if unregistration successful
        """
        try:
            # Remove from Redis
            service_key = f"service:{service_name}:{service_id}"
            await self.redis_client.delete(service_key)
            
            # Remove from service list
            service_list_key = f"services:{service_name}"
            await self.redis_client.srem(service_list_key, service_id)
            
            # Update local cache
            if service_name in self.services:
                self.services[service_name] = [
                    s for s in self.services[service_name] 
                    if s.service_id != service_id
                ]
            
            logger.info(f"Unregistered service: {service_name}:{service_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister service {service_name}:{service_id}: {e}")
            return False
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """
        Discover healthy service instances
        
        Args:
            service_name: Name of the service to discover
            
        Returns:
            List of healthy service instances
        """
        try:
            # Get from local cache first
            if service_name in self.services:
                healthy_services = [
                    s for s in self.services[service_name]
                    if s.status == ServiceStatus.HEALTHY
                ]
                if healthy_services:
                    return healthy_services
            
            # Fallback to Redis
            service_list_key = f"services:{service_name}"
            service_ids = await self.redis_client.smembers(service_list_key)
            
            services = []
            for service_id in service_ids:
                service_key = f"service:{service_name}:{service_id.decode()}"
                service_data = await self.redis_client.get(service_key)
                
                if service_data:
                    service_dict = json.loads(service_data)
                    service = ServiceInstance.from_dict(service_dict)
                    if service.status == ServiceStatus.HEALTHY:
                        services.append(service)
            
            # Update local cache
            self.services[service_name] = services
            
            return services
            
        except Exception as e:
            logger.error(f"Failed to discover services for {service_name}: {e}")
            return []
    
    async def get_service_instance(self, service_name: str, load_balance: bool = True) -> Optional[ServiceInstance]:
        """
        Get a service instance with optional load balancing
        
        Args:
            service_name: Name of the service
            load_balance: Whether to use load balancing
            
        Returns:
            Service instance or None
        """
        services = await self.discover_services(service_name)
        
        if not services:
            return None
        
        if not load_balance:
            return services[0]
        
        # Simple round-robin load balancing
        # In production, you might want to use more sophisticated algorithms
        return services[0]  # For now, return first healthy instance
    
    async def update_service_heartbeat(self, service_name: str, service_id: str) -> bool:
        """
        Update service heartbeat
        
        Args:
            service_name: Name of the service
            service_id: ID of the service instance
            
        Returns:
            bool: True if update successful
        """
        try:
            service_key = f"service:{service_name}:{service_id}"
            service_data = await self.redis_client.get(service_key)
            
            if service_data:
                service_dict = json.loads(service_data)
                service = ServiceInstance.from_dict(service_dict)
                service.last_heartbeat = time.time()
                
                # Update in Redis
                await self.redis_client.setex(
                    service_key,
                    self.service_ttl,
                    json.dumps(service.to_dict())
                )
                
                # Update local cache
                if service_name in self.services:
                    for i, s in enumerate(self.services[service_name]):
                        if s.service_id == service_id:
                            self.services[service_name][i] = service
                            break
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to update heartbeat for {service_name}:{service_id}: {e}")
            return False
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _perform_health_checks(self):
        """Perform health checks on all registered services"""
        async with aiohttp.ClientSession() as session:
            for service_name, services in self.services.items():
                for service in services:
                    try:
                        # Check if service is still alive
                        if time.time() - service.last_heartbeat > self.service_ttl:
                            service.status = ServiceStatus.UNHEALTHY
                            continue
                        
                        # Perform HTTP health check
                        async with session.get(
                            service.health_check_url,
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                service.status = ServiceStatus.HEALTHY
                            else:
                                service.status = ServiceStatus.UNHEALTHY
                                
                    except Exception as e:
                        logger.warning(f"Health check failed for {service_name}:{service.service_id}: {e}")
                        service.status = ServiceStatus.UNHEALTHY
    
    async def get_service_metrics(self) -> Dict[str, Any]:
        """Get service registry metrics"""
        total_services = sum(len(services) for services in self.services.values())
        healthy_services = sum(
            len([s for s in services if s.status == ServiceStatus.HEALTHY])
            for services in self.services.values()
        )
        
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "unhealthy_services": total_services - healthy_services,
            "service_types": list(set(
                service.service_type.value
                for services in self.services.values()
                for service in services
            )),
            "services_by_type": {
                service_type.value: len([
                    service for services in self.services.values()
                    for service in services
                    if service.service_type == service_type
                ])
                for service_type in ServiceType
            }
        }

# Global service registry instance
service_registry = ServiceRegistry()






























