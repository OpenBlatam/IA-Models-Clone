from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
from typing import Any, List, Dict, Optional
"""
Service Discovery Implementation
===============================

Advanced service discovery with multiple backend support.
"""


logger = logging.getLogger(__name__)

@dataclass
class ServiceInstance:
    """Service instance metadata."""
    id: str
    name: str
    host: str
    port: int
    scheme: str = "http"
    health_check_url: Optional[str] = None
    metadata: Dict[str, str] = None
    tags: List[str] = None
    last_seen: datetime = None
    
    @property
    def url(self) -> str:
        """Get service URL."""
        return f"{self.scheme}://{self.host}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is considered healthy."""
        if not self.last_seen:
            return False
        return (datetime.utcnow() - self.last_seen).seconds < 30


class IServiceDiscovery(ABC):
    """Abstract interface for service discovery."""
    
    @abstractmethod
    async def register_service(self, instance: ServiceInstance) -> bool:
        """Register a service instance."""
        pass
    
    @abstractmethod
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover instances of a service."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if discovery service is healthy."""
        pass


class ConsulServiceDiscovery(IServiceDiscovery):
    """Consul-based service discovery."""
    
    def __init__(self, consul_url: str = "http://localhost:8500"):
        
    """__init__ function."""
self.consul_url = consul_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def register_service(self, instance: ServiceInstance) -> bool:
        """Register service with Consul."""
        try:
            session = await self._get_session()
            
            service_def = {
                "ID": instance.id,
                "Name": instance.name,
                "Address": instance.host,
                "Port": instance.port,
                "Tags": instance.tags or [],
                "Meta": instance.metadata or {},
                "Check": {
                    "HTTP": instance.health_check_url or f"{instance.url}/health",
                    "Interval": "10s",
                    "Timeout": "5s"
                }
            }
            
            async with session.put(
                f"{self.consul_url}/v1/agent/service/register",
                json=service_def
            ) as response:
                success = response.status == 200
                if success:
                    logger.info(f"Registered service {instance.name} with Consul")
                return success
                
        except Exception as e:
            logger.error(f"Error registering service with Consul: {e}")
            return False
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover services from Consul."""
        try:
            session = await self._get_session()
            
            async with session.get(
                f"{self.consul_url}/v1/health/service/{service_name}"
            ) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                instances = []
                
                for entry in data:
                    service = entry["Service"]
                    checks = entry["Checks"]
                    
                    # Check if service is healthy
                    is_healthy = all(
                        check["Status"] in ["passing", "warning"] 
                        for check in checks
                    )
                    
                    if is_healthy:
                        instance = ServiceInstance(
                            id=service["ID"],
                            name=service["Service"],
                            host=service["Address"],
                            port=service["Port"],
                            metadata=service.get("Meta", {}),
                            tags=service.get("Tags", []),
                            last_seen=datetime.utcnow()
                        )
                        instances.append(instance)
                
                return instances
                
        except Exception as e:
            logger.error(f"Error discovering services from Consul: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check Consul health."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.consul_url}/v1/status/leader") as response:
                return response.status == 200
        except:
            return False


class ServiceDiscoveryManager:
    """Manager for multiple service discovery backends."""
    
    def __init__(self) -> Any:
        self.discoveries: Dict[str, IServiceDiscovery] = {}
        self.primary_discovery: Optional[str] = None
        
    def add_discovery(self, name: str, discovery: IServiceDiscovery, is_primary: bool = False):
        """Add a service discovery backend."""
        self.discoveries[name] = discovery
        if is_primary:
            self.primary_discovery = name
        logger.info(f"Added service discovery backend: {name}")
    
    async def register_service(self, instance: ServiceInstance) -> Dict[str, bool]:
        """Register service with all discovery backends."""
        results = {}
        
        for name, discovery in self.discoveries.items():
            try:
                result = await discovery.register_service(instance)
                results[name] = result
            except Exception as e:
                logger.error(f"Error registering with {name}: {e}")
                results[name] = False
        
        return results
    
    async def discover_services(self, service_name: str) -> List[ServiceInstance]:
        """Discover services from primary backend."""
        if self.primary_discovery and self.primary_discovery in self.discoveries:
            return await self.discoveries[self.primary_discovery].discover_services(service_name)
        
        # Fallback to first available
        if self.discoveries:
            discovery = next(iter(self.discoveries.values()))
            return await discovery.discover_services(service_name)
        
        return [] 