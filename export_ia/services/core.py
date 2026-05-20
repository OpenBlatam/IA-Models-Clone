"""
Core microservices infrastructure.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ServiceInfo:
    """Information about a service."""
    name: str
    version: str
    host: str
    port: int
    status: ServiceStatus = ServiceStatus.STOPPED
    health_endpoint: str = "/health"
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)


class ServiceRegistry:
    """Registry for managing service instances."""
    
    def __init__(self):
        self.services: Dict[str, ServiceInfo] = {}
        self._lock = asyncio.Lock()
    
    async def register_service(self, service_info: ServiceInfo) -> bool:
        """Register a service in the registry."""
        async with self._lock:
            service_id = f"{service_info.name}:{service_info.host}:{service_info.port}"
            
            if service_id in self.services:
                logger.warning(f"Service {service_id} already registered")
                return False
            
            self.services[service_id] = service_info
            logger.info(f"Service registered: {service_id}")
            return True
    
    async def unregister_service(self, service_id: str) -> bool:
        """Unregister a service from the registry."""
        async with self._lock:
            if service_id not in self.services:
                logger.warning(f"Service {service_id} not found")
                return False
            
            del self.services[service_id]
            logger.info(f"Service unregistered: {service_id}")
            return True
    
    async def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """Get service information by ID."""
        async with self._lock:
            return self.services.get(service_id)
    
    async def get_services_by_name(self, name: str) -> List[ServiceInfo]:
        """Get all services with a specific name."""
        async with self._lock:
            return [
                service for service in self.services.values()
                if service.name == name
            ]
    
    async def get_healthy_services(self, name: str) -> List[ServiceInfo]:
        """Get healthy services by name."""
        services = await self.get_services_by_name(name)
        return [s for s in services if s.status == ServiceStatus.RUNNING]
    
    async def update_service_status(self, service_id: str, status: ServiceStatus) -> bool:
        """Update service status."""
        async with self._lock:
            if service_id not in self.services:
                return False
            
            self.services[service_id].status = status
            self.services[service_id].last_heartbeat = datetime.now()
            return True
    
    async def list_services(self) -> Dict[str, ServiceInfo]:
        """List all registered services."""
        async with self._lock:
            return self.services.copy()


class BaseService:
    """Base class for all microservices."""
    
    def __init__(self, name: str, version: str, host: str = "localhost", port: int = 8000):
        self.name = name
        self.version = version
        self.host = host
        self.port = port
        self.service_id = f"{name}:{host}:{port}"
        self.status = ServiceStatus.STOPPED
        self.logger = logging.getLogger(f"service.{name}")
        
        # Service info
        self.info = ServiceInfo(
            name=name,
            version=version,
            host=host,
            port=port,
            status=self.status
        )
        
        # Dependencies
        self.dependencies: List[str] = []
        self.dependency_services: Dict[str, Any] = {}
        
        # Lifecycle hooks
        self._startup_hooks: List[Callable] = []
        self._shutdown_hooks: List[Callable] = []
    
    async def start(self) -> None:
        """Start the service."""
        self.logger.info(f"Starting service {self.name}")
        self.status = ServiceStatus.STARTING
        
        try:
            # Run startup hooks
            for hook in self._startup_hooks:
                await self._run_hook(hook, "startup")
            
            # Start service-specific logic
            await self._start()
            
            self.status = ServiceStatus.RUNNING
            self.info.status = self.status
            self.logger.info(f"Service {self.name} started successfully")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.info.status = self.status
            self.logger.error(f"Failed to start service {self.name}: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the service."""
        self.logger.info(f"Stopping service {self.name}")
        self.status = ServiceStatus.STOPPING
        
        try:
            # Run shutdown hooks
            for hook in self._shutdown_hooks:
                await self._run_hook(hook, "shutdown")
            
            # Stop service-specific logic
            await self._stop()
            
            self.status = ServiceStatus.STOPPED
            self.info.status = self.status
            self.logger.info(f"Service {self.name} stopped successfully")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.info.status = self.status
            self.logger.error(f"Error stopping service {self.name}: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": self.status.value,
            "name": self.name,
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "dependencies": self._check_dependencies()
        }
    
    def add_startup_hook(self, hook: Callable) -> None:
        """Add a startup hook."""
        self._startup_hooks.append(hook)
    
    def add_shutdown_hook(self, hook: Callable) -> None:
        """Add a shutdown hook."""
        self._shutdown_hooks.append(hook)
    
    def add_dependency(self, service_name: str) -> None:
        """Add a service dependency."""
        if service_name not in self.dependencies:
            self.dependencies.append(service_name)
            self.info.dependencies = self.dependencies.copy()
    
    async def _start(self) -> None:
        """Service-specific startup logic. Override in subclasses."""
        pass
    
    async def _stop(self) -> None:
        """Service-specific shutdown logic. Override in subclasses."""
        pass
    
    async def _run_hook(self, hook: Callable, hook_type: str) -> None:
        """Run a lifecycle hook."""
        try:
            if asyncio.iscoroutinefunction(hook):
                await hook()
            else:
                hook()
        except Exception as e:
            self.logger.error(f"Error in {hook_type} hook: {e}")
    
    def _check_dependencies(self) -> Dict[str, str]:
        """Check dependency health."""
        dependency_status = {}
        for dep_name in self.dependencies:
            if dep_name in self.dependency_services:
                dep_service = self.dependency_services[dep_name]
                dependency_status[dep_name] = dep_service.status.value
            else:
                dependency_status[dep_name] = "not_connected"
        return dependency_status


class ServiceManager:
    """Manages multiple services and their lifecycle."""
    
    def __init__(self):
        self.services: Dict[str, BaseService] = {}
        self.registry = ServiceRegistry()
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start the service manager."""
        if self._running:
            return
        
        self._running = True
        logger.info("Service manager started")
    
    async def stop(self) -> None:
        """Stop the service manager and all services."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop all services
        for service in self.services.values():
            try:
                await service.stop()
            except Exception as e:
                logger.error(f"Error stopping service {service.name}: {e}")
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        logger.info("Service manager stopped")
    
    def register_service(self, service: BaseService) -> None:
        """Register a service with the manager."""
        self.services[service.service_id] = service
        logger.info(f"Service registered with manager: {service.name}")
    
    async def start_service(self, service_id: str) -> bool:
        """Start a specific service."""
        if service_id not in self.services:
            logger.error(f"Service {service_id} not found")
            return False
        
        service = self.services[service_id]
        try:
            await service.start()
            await self.registry.register_service(service.info)
            return True
        except Exception as e:
            logger.error(f"Failed to start service {service_id}: {e}")
            return False
    
    async def stop_service(self, service_id: str) -> bool:
        """Stop a specific service."""
        if service_id not in self.services:
            logger.error(f"Service {service_id} not found")
            return False
        
        service = self.services[service_id]
        try:
            await service.stop()
            await self.registry.unregister_service(service_id)
            return True
        except Exception as e:
            logger.error(f"Failed to stop service {service_id}: {e}")
            return False
    
    async def get_service_status(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific service."""
        if service_id not in self.services:
            return None
        
        service = self.services[service_id]
        return await service.health_check()
    
    async def list_services(self) -> Dict[str, Dict[str, Any]]:
        """List all services and their status."""
        services_status = {}
        for service_id, service in self.services.items():
            services_status[service_id] = await service.health_check()
        return services_status
    
    async def start_all_services(self) -> None:
        """Start all registered services."""
        for service_id in self.services.keys():
            await self.start_service(service_id)
    
    async def stop_all_services(self) -> None:
        """Stop all registered services."""
        for service_id in self.services.keys():
            await self.stop_service(service_id)


# Global service manager instance
_service_manager: Optional[ServiceManager] = None


def get_service_manager() -> ServiceManager:
    """Get the global service manager instance."""
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager




