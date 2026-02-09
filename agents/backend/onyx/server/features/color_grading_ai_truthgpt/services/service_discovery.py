"""
Service Discovery for Color Grading AI
=======================================

Automatic service discovery and registration system.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DEGRADED = "degraded"


@dataclass
class ServiceInfo:
    """Service information."""
    name: str
    service_type: str
    version: str
    endpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_heartbeat: Optional[datetime] = None
    registered_at: datetime = field(default_factory=datetime.now)


class ServiceDiscovery:
    """
    Service discovery system.
    
    Features:
    - Automatic service registration
    - Health checking
    - Service lookup
    - Service metadata
    - Heartbeat monitoring
    - Service filtering
    """
    
    def __init__(self, heartbeat_interval: float = 30.0, timeout: float = 60.0):
        """
        Initialize service discovery.
        
        Args:
            heartbeat_interval: Heartbeat interval in seconds
            timeout: Service timeout in seconds
        """
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self._services: Dict[str, ServiceInfo] = {}
        self._lock = asyncio.Lock()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def register(
        self,
        name: str,
        service_type: str,
        version: str = "1.0.0",
        endpoint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ServiceInfo:
        """
        Register a service.
        
        Args:
            name: Service name
            service_type: Service type
            version: Service version
            endpoint: Optional endpoint URL
            metadata: Optional metadata
            
        Returns:
            Service info
        """
        async with self._lock:
            service = ServiceInfo(
                name=name,
                service_type=service_type,
                version=version,
                endpoint=endpoint,
                metadata=metadata or {},
                status=ServiceStatus.HEALTHY,
                last_heartbeat=datetime.now()
            )
            
            self._services[name] = service
            logger.info(f"Registered service: {name} ({service_type})")
            
            if not self._running:
                await self._start_heartbeat_monitor()
            
            return service
    
    async def unregister(self, name: str) -> bool:
        """
        Unregister a service.
        
        Args:
            name: Service name
            
        Returns:
            True if unregistered
        """
        async with self._lock:
            if name in self._services:
                del self._services[name]
                logger.info(f"Unregistered service: {name}")
                return True
            return False
    
    async def heartbeat(self, name: str) -> bool:
        """
        Send heartbeat for a service.
        
        Args:
            name: Service name
            
        Returns:
            True if service exists
        """
        async with self._lock:
            if name in self._services:
                self._services[name].last_heartbeat = datetime.now()
                self._services[name].status = ServiceStatus.HEALTHY
                return True
            return False
    
    async def discover(
        self,
        service_type: Optional[str] = None,
        status: Optional[ServiceStatus] = None,
        filter_func: Optional[Callable[[ServiceInfo], bool]] = None
    ) -> List[ServiceInfo]:
        """
        Discover services.
        
        Args:
            service_type: Optional service type filter
            status: Optional status filter
            filter_func: Optional custom filter function
            
        Returns:
            List of matching services
        """
        async with self._lock:
            services = list(self._services.values())
            
            # Apply filters
            if service_type:
                services = [s for s in services if s.service_type == service_type]
            
            if status:
                services = [s for s in services if s.status == status]
            
            if filter_func:
                services = [s for s in services if filter_func(s)]
            
            return services
    
    async def get_service(self, name: str) -> Optional[ServiceInfo]:
        """
        Get service by name.
        
        Args:
            name: Service name
            
        Returns:
            Service info or None
        """
        async with self._lock:
            return self._services.get(name)
    
    async def update_status(self, name: str, status: ServiceStatus) -> bool:
        """
        Update service status.
        
        Args:
            name: Service name
            status: New status
            
        Returns:
            True if updated
        """
        async with self._lock:
            if name in self._services:
                self._services[name].status = status
                return True
            return False
    
    async def _start_heartbeat_monitor(self):
        """Start heartbeat monitoring."""
        if self._running:
            return
        
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Started heartbeat monitor")
    
    async def _heartbeat_loop(self):
        """Heartbeat monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self._check_heartbeats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
    
    async def _check_heartbeats(self):
        """Check service heartbeats."""
        async with self._lock:
            now = datetime.now()
            for name, service in self._services.items():
                if service.last_heartbeat:
                    elapsed = (now - service.last_heartbeat).total_seconds()
                    if elapsed > self.timeout:
                        if service.status == ServiceStatus.HEALTHY:
                            service.status = ServiceStatus.UNHEALTHY
                            logger.warning(f"Service {name} missed heartbeat (elapsed: {elapsed:.1f}s)")
    
    async def stop(self):
        """Stop service discovery."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped service discovery")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        status_counts = {}
        for service in self._services.values():
            status_counts[service.status.value] = status_counts.get(service.status.value, 0) + 1
        
        return {
            "total_services": len(self._services),
            "status_counts": status_counts,
            "heartbeat_interval": self.heartbeat_interval,
            "timeout": self.timeout,
        }


