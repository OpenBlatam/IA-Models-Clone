"""
Service Discovery for Microservices Architecture
Supports Consul, Eureka, and Kubernetes service discovery
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DiscoveryType(str, Enum):
    """Service discovery types"""
    CONSUL = "consul"
    EUREKA = "eureka"
    KUBERNETES = "kubernetes"
    STATIC = "static"
    NONE = "none"


@dataclass
class ServiceInstance:
    """Service instance information"""
    service_name: str
    instance_id: str
    host: str
    port: int
    health_check_url: Optional[str] = None
    metadata: Dict[str, Any] = None
    last_heartbeat: Optional[datetime] = None
    status: str = "UP"


class ServiceRegistry:
    """
    Service registry for service discovery.
    Supports multiple discovery backends.
    """
    
    def __init__(self, discovery_type: DiscoveryType = DiscoveryType.NONE):
        self.discovery_type = discovery_type
        self.services: Dict[str, List[ServiceInstance]] = {}
        self.registered = False
    
    async def register(
        self,
        service_name: str,
        host: str,
        port: int,
        instance_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register service instance
        
        Args:
            service_name: Name of the service
            host: Host address
            port: Port number
            instance_id: Optional instance ID
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        instance_id = instance_id or f"{service_name}-{host}-{port}"
        
        instance = ServiceInstance(
            service_name=service_name,
            instance_id=instance_id,
            host=host,
            port=port,
            health_check_url=f"http://{host}:{port}/health",
            metadata=metadata or {},
            last_heartbeat=datetime.utcnow(),
        )
        
        if service_name not in self.services:
            self.services[service_name] = []
        
        self.services[service_name].append(instance)
        self.registered = True
        
        logger.info(f"✅ Registered service instance: {instance_id} at {host}:{port}")
        return True
    
    async def deregister(self, service_name: str, instance_id: str) -> bool:
        """Deregister service instance"""
        if service_name in self.services:
            self.services[service_name] = [
                inst for inst in self.services[service_name]
                if inst.instance_id != instance_id
            ]
            logger.info(f"Deregistered service instance: {instance_id}")
            return True
        return False
    
    async def discover(self, service_name: str) -> List[ServiceInstance]:
        """
        Discover service instances
        
        Args:
            service_name: Name of the service to discover
            
        Returns:
            List of service instances
        """
        if self.discovery_type == DiscoveryType.CONSUL:
            return await self._discover_consul(service_name)
        elif self.discovery_type == DiscoveryType.EUREKA:
            return await self._discover_eureka(service_name)
        elif self.discovery_type == DiscoveryType.KUBERNETES:
            return await self._discover_kubernetes(service_name)
        else:
            # Static/local discovery
            return self.services.get(service_name, [])
    
    async def _discover_consul(self, service_name: str) -> List[ServiceInstance]:
        """Discover services via Consul"""
        try:
            import aiohttp
            
            consul_url = os.getenv("CONSUL_URL", "http://localhost:8500")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{consul_url}/v1/health/service/{service_name}?passing=true"
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        instances = []
                        
                        for item in data:
                            service = item.get("Service", {})
                            instances.append(ServiceInstance(
                                service_name=service_name,
                                instance_id=service.get("ID", ""),
                                host=service.get("Address", ""),
                                port=service.get("Port", 0),
                                metadata=service.get("Meta", {}),
                                status="UP"
                            ))
                        
                        return instances
                    else:
                        logger.warning(f"Consul discovery failed: {resp.status}")
                        return []
        except ImportError:
            logger.warning("aiohttp not installed for Consul integration")
            return []
        except Exception as e:
            logger.error(f"Consul discovery error: {e}")
            return []
    
    async def _discover_eureka(self, service_name: str) -> List[ServiceInstance]:
        """Discover services via Eureka"""
        try:
            import aiohttp
            
            eureka_url = os.getenv("EUREKA_URL", "http://localhost:8761")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{eureka_url}/eureka/apps/{service_name.upper()}"
                ) as resp:
                    if resp.status == 200:
                        # Parse Eureka XML response (simplified)
                        # In production, use proper XML parser
                        logger.info(f"Eureka discovery for {service_name}")
                        return []
                    else:
                        logger.warning(f"Eureka discovery failed: {resp.status}")
                        return []
        except Exception as e:
            logger.error(f"Eureka discovery error: {e}")
            return []
    
    async def _discover_kubernetes(self, service_name: str) -> List[ServiceInstance]:
        """Discover services via Kubernetes DNS"""
        # Kubernetes services are discoverable via DNS
        # Format: <service-name>.<namespace>.svc.cluster.local
        namespace = os.getenv("KUBERNETES_NAMESPACE", "default")
        service_dns = f"{service_name}.{namespace}.svc.cluster.local"
        
        # In production, use Kubernetes client to get endpoints
        logger.info(f"Kubernetes service discovery: {service_dns}")
        
        # Return placeholder - in production, query Kubernetes API
        return [
            ServiceInstance(
                service_name=service_name,
                instance_id=f"{service_name}-k8s",
                host=service_dns,
                port=80,
                status="UP"
            )
        ]
    
    async def health_check(self, instance: ServiceInstance) -> bool:
        """Perform health check on service instance"""
        if not instance.health_check_url:
            return True
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    instance.health_check_url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False
    
    async def start_heartbeat(
        self,
        service_name: str,
        instance_id: str,
        interval_seconds: int = 30
    ):
        """Start heartbeat to keep service registered"""
        while self.registered:
            try:
                # Update last heartbeat
                if service_name in self.services:
                    for instance in self.services[service_name]:
                        if instance.instance_id == instance_id:
                            instance.last_heartbeat = datetime.utcnow()
                            break
                
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break


def get_service_registry() -> ServiceRegistry:
    """Get service registry from environment"""
    discovery_type = os.getenv("SERVICE_DISCOVERY_TYPE", "none").lower()
    
    return ServiceRegistry(discovery_type=DiscoveryType(discovery_type))















