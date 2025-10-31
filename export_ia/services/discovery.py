"""
Service discovery and registration system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class ServiceEndpoint:
    """Service endpoint information."""
    name: str
    host: str
    port: int
    health_url: str
    api_url: str
    metadata: Dict[str, Any]
    last_seen: datetime
    status: str = "unknown"


class ServiceDiscovery:
    """Service discovery and health monitoring."""
    
    def __init__(self, heartbeat_interval: float = 30.0, timeout: float = 60.0):
        self.services: Dict[str, ServiceEndpoint] = {}
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def start(self) -> None:
        """Start service discovery."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_services())
        logger.info("Service discovery started")
    
    async def stop(self) -> None:
        """Stop service discovery."""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Service discovery stopped")
    
    async def register_service(
        self, 
        name: str, 
        host: str, 
        port: int,
        health_url: str = "/health",
        api_url: str = "/api",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a service."""
        async with self._lock:
            service_id = f"{name}:{host}:{port}"
            
            self.services[service_id] = ServiceEndpoint(
                name=name,
                host=host,
                port=port,
                health_url=health_url,
                api_url=api_url,
                metadata=metadata or {},
                last_seen=datetime.now()
            )
            
            logger.info(f"Service registered: {service_id}")
    
    async def unregister_service(self, name: str, host: str, port: int) -> None:
        """Unregister a service."""
        async with self._lock:
            service_id = f"{name}:{host}:{port}"
            
            if service_id in self.services:
                del self.services[service_id]
                logger.info(f"Service unregistered: {service_id}")
    
    async def discover_services(self, name: Optional[str] = None) -> List[ServiceEndpoint]:
        """Discover services by name."""
        async with self._lock:
            if name:
                return [
                    service for service in self.services.values()
                    if service.name == name and service.status == "healthy"
                ]
            else:
                return [
                    service for service in self.services.values()
                    if service.status == "healthy"
                ]
    
    async def get_service_url(self, name: str) -> Optional[str]:
        """Get the URL for a service."""
        services = await self.discover_services(name)
        if services:
            service = services[0]  # Return first healthy service
            return f"http://{service.host}:{service.port}{service.api_url}"
        return None
    
    async def _monitor_services(self) -> None:
        """Monitor service health."""
        while self._running:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in service monitoring: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _check_all_services(self) -> None:
        """Check health of all registered services."""
        async with self._lock:
            for service_id, service in self.services.items():
                try:
                    await self._check_service_health(service)
                except Exception as e:
                    logger.error(f"Error checking health of {service_id}: {e}")
                    service.status = "unhealthy"
    
    async def _check_service_health(self, service: ServiceEndpoint) -> None:
        """Check health of a specific service."""
        try:
            import aiohttp
            
            url = f"http://{service.host}:{service.port}{service.health_url}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5.0) as response:
                    if response.status == 200:
                        service.status = "healthy"
                        service.last_seen = datetime.now()
                    else:
                        service.status = "unhealthy"
        except Exception as e:
            logger.warning(f"Health check failed for {service.name}: {e}")
            service.status = "unhealthy"
    
    async def get_healthy_services(self) -> Dict[str, List[ServiceEndpoint]]:
        """Get all healthy services grouped by name."""
        async with self._lock:
            healthy_services = {}
            
            for service in self.services.values():
                if service.status == "healthy":
                    if service.name not in healthy_services:
                        healthy_services[service.name] = []
                    healthy_services[service.name].append(service)
            
            return healthy_services
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """Get service discovery statistics."""
        async with self._lock:
            total_services = len(self.services)
            healthy_services = sum(1 for s in self.services.values() if s.status == "healthy")
            unhealthy_services = total_services - healthy_services
            
            service_counts = {}
            for service in self.services.values():
                service_counts[service.name] = service_counts.get(service.name, 0) + 1
            
            return {
                "total_services": total_services,
                "healthy_services": healthy_services,
                "unhealthy_services": unhealthy_services,
                "service_counts": service_counts,
                "last_updated": datetime.now().isoformat()
            }


# Global service discovery instance
_service_discovery: Optional[ServiceDiscovery] = None


def get_service_discovery() -> ServiceDiscovery:
    """Get the global service discovery instance."""
    global _service_discovery
    if _service_discovery is None:
        _service_discovery = ServiceDiscovery()
    return _service_discovery




