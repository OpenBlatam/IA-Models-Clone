"""
🔧 BLATAM AI SERVICES MODULE v6.0.0
====================================

Service layer for the Blatam AI system:
- 🎯 Service lifecycle management
- 🔧 Dependency injection
- ⚡ Async service execution
- 📊 Service health monitoring
- 🚀 Service discovery and registration
- 🧹 Resource cleanup
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
import uuid
import weakref

from ..core import BlatamComponent, ComponentStatus, ComponentType

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# 🎯 SERVICE TYPES AND STATUS
# =============================================================================

class ServiceType(Enum):
    """Service types for categorization."""
    CORE = "core"
    ENGINE = "engine"
    UTILITY = "utility"
    INTERFACE = "interface"
    EXTERNAL = "external"

class ServiceStatus(Enum):
    """Service lifecycle status."""
    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

# =============================================================================
# 🎯 SERVICE CONFIGURATION
# =============================================================================

@dataclass
class ServiceConfig:
    """Configuration for services."""
    name: str
    service_type: ServiceType
    enabled: bool = True
    auto_start: bool = True
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    retry_attempts: int = 3
    health_check_interval: float = 60.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'service_type': self.service_type.value,
            'enabled': self.enabled,
            'auto_start': self.auto_start,
            'dependencies': self.dependencies,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts,
            'health_check_interval': self.health_check_interval,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceConfig':
        """Create config from dictionary."""
        if 'service_type' in data and isinstance(data['service_type'], str):
            data['service_type'] = ServiceType(data['service_type'])
        return cls(**data)

# =============================================================================
# 🎯 SERVICE INTERFACES
# =============================================================================

T = TypeVar('T', bound='BlatamService')

class BlatamService(BlatamComponent):
    """Base service interface."""
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.service_config = config
        self.service_type = config.service_type
        self.dependencies = config.dependencies
        self.dependent_services: List[str] = []
        self._health_check_task: Optional[asyncio.Task] = None
        self._last_health_check = 0.0
        
    @abstractmethod
    async def start(self) -> bool:
        """Start the service."""
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """Stop the service."""
        pass
    
    @abstractmethod
    async def pause(self) -> bool:
        """Pause the service."""
        pass
    
    @abstractmethod
    async def resume(self) -> bool:
        """Resume the service."""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        current_time = time.time()
        if current_time - self._last_health_check < self.service_config.health_check_interval:
            return {'status': 'cached', 'last_check': self._last_health_check}
        
        try:
            health_result = await self._perform_health_check()
            self._last_health_check = current_time
            return health_result
        except Exception as e:
            logger.error(f"❌ Health check failed for service '{self.service_config.name}': {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform actual health check - override in subclasses."""
        return {
            'status': self.status.value,
            'service_type': self.service_type.value,
            'dependencies': self.dependencies,
            'dependent_services': self.dependent_services
        }
    
    def add_dependent_service(self, service_name: str) -> None:
        """Add a service that depends on this one."""
        if service_name not in self.dependent_services:
            self.dependent_services.append(service_name)
    
    def remove_dependent_service(self, service_name: str) -> None:
        """Remove a dependent service."""
        if service_name in self.dependent_services:
            self.dependent_services.remove(service_name)
    
    async def initialize(self) -> bool:
        """Initialize the service."""
        try:
            self.status = ComponentStatus.INITIALIZING
            logger.info(f"🚀 Initializing service: {self.service_config.name}")
            
            # Check dependencies
            if not await self._check_dependencies():
                logger.error(f"❌ Dependencies not met for service: {self.service_config.name}")
                return False
            
            # Start health monitoring if enabled
            if self.service_config.health_check_interval > 0:
                self._start_health_monitoring()
            
            self.status = ComponentStatus.READY
            logger.info(f"✅ Service initialized: {self.service_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize service '{self.service_config.name}': {e}")
            self.status = ComponentStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the service."""
        try:
            self.status = ComponentStatus.SHUTTING_DOWN
            logger.info(f"🔄 Shutting down service: {self.service_config.name}")
            
            # Stop health monitoring
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Stop the service
            await self.stop()
            
            self.status = ComponentStatus.SHUTDOWN
            logger.info(f"✅ Service shutdown: {self.service_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to shutdown service '{self.service_config.name}': {e}")
            return False
    
    async def _check_dependencies(self) -> bool:
        """Check if all dependencies are available."""
        # This should be implemented by the service registry
        return True
    
    def _start_health_monitoring(self) -> None:
        """Start periodic health monitoring."""
        if self._health_check_task:
            return
        
        async def health_monitor():
            while True:
                try:
                    await asyncio.sleep(self.service_config.health_check_interval)
                    await self.health_check()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ Health monitoring error for service '{self.service_config.name}': {e}")
        
        self._health_check_task = asyncio.create_task(health_monitor())

# =============================================================================
# 🎯 SERVICE REGISTRY
# =============================================================================

class ServiceRegistry:
    """Centralized service registry."""
    
    def __init__(self):
        self._services: Dict[str, BlatamService] = {}
        self._service_configs: Dict[str, ServiceConfig] = {}
        self._service_metadata: Dict[str, Dict[str, Any]] = {}
        self._dependency_graph: Dict[str, List[str]] = {}
        self._reverse_dependencies: Dict[str, List[str]] = {}
        self._registry_lock = asyncio.Lock()
    
    async def register_service(self, service: BlatamService, config: ServiceConfig) -> bool:
        """Register a service."""
        async with self._registry_lock:
            try:
                service_name = config.name
                
                # Check if service already exists
                if service_name in self._services:
                    logger.warning(f"⚠️ Service '{service_name}' already registered, replacing")
                    await self._unregister_service_internal(service_name)
                
                # Register service
                self._services[service_name] = service
                self._service_configs[service_name] = config
                self._service_metadata[service_name] = config.metadata
                
                # Update dependency graph
                self._update_dependency_graph(service_name, config.dependencies)
                
                logger.info(f"🔧 Service registered: {service_name}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to register service '{config.name}': {e}")
                return False
    
    async def unregister_service(self, service_name: str) -> bool:
        """Unregister a service."""
        async with self._registry_lock:
            return await self._unregister_service_internal(service_name)
    
    async def _unregister_service_internal(self, service_name: str) -> bool:
        """Internal service unregistration."""
        try:
            if service_name not in self._services:
                return True
            
            # Get service instance
            service = self._services[service_name]
            
            # Shutdown service if running
            if service.status in [ComponentStatus.RUNNING, ComponentStatus.READY]:
                await service.shutdown()
            
            # Remove from registry
            del self._services[service_name]
            del self._service_configs[service_name]
            if service_name in self._service_metadata:
                del self._service_metadata[service_name]
            
            # Update dependency graph
            self._remove_from_dependency_graph(service_name)
            
            logger.info(f"🗑️ Service unregistered: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unregister service '{service_name}': {e}")
            return False
    
    def get_service(self, service_name: str) -> Optional[BlatamService]:
        """Get a service by name."""
        return self._services.get(service_name)
    
    def get_service_config(self, service_name: str) -> Optional[ServiceConfig]:
        """Get service configuration."""
        return self._service_configs.get(service_name)
    
    def list_services(self, service_type: Optional[ServiceType] = None) -> List[str]:
        """List services, optionally filtered by type."""
        if service_type is None:
            return list(self._services.keys())
        
        return [
            name for name, service in self._services.items()
            if service.service_type == service_type
        ]
    
    def get_service_dependencies(self, service_name: str) -> List[str]:
        """Get dependencies for a service."""
        return self._dependency_graph.get(service_name, [])
    
    def get_dependent_services(self, service_name: str) -> List[str]:
        """Get services that depend on this one."""
        return self._reverse_dependencies.get(service_name, [])
    
    def _update_dependency_graph(self, service_name: str, dependencies: List[str]) -> None:
        """Update dependency graph when registering a service."""
        self._dependency_graph[service_name] = dependencies
        
        # Update reverse dependencies
        for dep in dependencies:
            if dep not in self._reverse_dependencies:
                self._reverse_dependencies[dep] = []
            if service_name not in self._reverse_dependencies[dep]:
                self._reverse_dependencies[dep].append(service_name)
    
    def _remove_from_dependency_graph(self, service_name: str) -> None:
        """Remove service from dependency graph."""
        # Remove from forward dependencies
        if service_name in self._dependency_graph:
            del self._dependency_graph[service_name]
        
        # Remove from reverse dependencies
        if service_name in self._reverse_dependencies:
            del self._reverse_dependencies[service_name]
        
        # Remove from other services' reverse dependencies
        for deps in self._reverse_dependencies.values():
            if service_name in deps:
                deps.remove(service_name)
    
    async def resolve_dependencies(self, service_names: List[str]) -> List[str]:
        """Resolve dependency order for services."""
        resolved = []
        pending = set(service_names)
        
        while pending:
            ready = []
            for service_name in pending:
                dependencies = self.get_service_dependencies(service_name)
                if all(dep in resolved for dep in dependencies):
                    ready.append(service_name)
            
            if not ready:
                raise ValueError(f"Circular dependency detected in services: {pending}")
            
            for service_name in ready:
                resolved.append(service_name)
                pending.remove(service_name)
        
        return resolved

# =============================================================================
# 🎯 SERVICE MANAGER
# =============================================================================

class ServiceManager:
    """Manages service lifecycle and operations."""
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self._startup_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
    
    async def start_all_services(self, service_names: Optional[List[str]] = None) -> bool:
        """Start all services or specified services."""
        async with self._startup_lock:
            try:
                if service_names is None:
                    service_names = self.registry.list_services()
                
                # Resolve dependencies
                ordered_services = await self.registry.resolve_dependencies(service_names)
                
                logger.info(f"🚀 Starting {len(ordered_services)} services...")
                
                for service_name in ordered_services:
                    service = self.registry.get_service(service_name)
                    if service and service.service_config.enabled:
                        success = await self._start_service(service)
                        if not success:
                            logger.error(f"❌ Failed to start service: {service_name}")
                            return False
                
                logger.info("✅ All services started successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to start services: {e}")
                return False
    
    async def stop_all_services(self, service_names: Optional[List[str]] = None) -> bool:
        """Stop all services or specified services."""
        async with self._shutdown_lock:
            try:
                if service_names is None:
                    service_names = self.registry.list_services()
                
                # Stop in reverse dependency order
                ordered_services = list(reversed(service_names))
                
                logger.info(f"🔄 Stopping {len(ordered_services)} services...")
                
                for service_name in ordered_services:
                    service = self.registry.get_service(service_name)
                    if service:
                        success = await self._stop_service(service)
                        if not success:
                            logger.warning(f"⚠️ Failed to stop service: {service_name}")
                
                logger.info("✅ All services stopped")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to stop services: {e}")
                return False
    
    async def _start_service(self, service: BlatamService) -> bool:
        """Start a single service."""
        try:
            if service.status == ComponentStatus.RUNNING:
                return True
            
            if service.status not in [ComponentStatus.READY, ComponentStatus.STOPPED]:
                logger.warning(f"⚠️ Service '{service.service_config.name}' not in ready state: {service.status}")
                return False
            
            success = await service.start()
            if success:
                service.status = ComponentStatus.RUNNING
                logger.info(f"✅ Service started: {service.service_config.name}")
            else:
                service.status = ComponentStatus.ERROR
                logger.error(f"❌ Service start failed: {service.service_config.name}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error starting service '{service.service_config.name}': {e}")
            service.status = ComponentStatus.ERROR
            return False
    
    async def _stop_service(self, service: BlatamService) -> bool:
        """Stop a single service."""
        try:
            if service.status == ComponentStatus.STOPPED:
                return True
            
            success = await service.stop()
            if success:
                service.status = ComponentStatus.STOPPED
                logger.info(f"✅ Service stopped: {service.service_config.name}")
            else:
                logger.warning(f"⚠️ Service stop failed: {service.service_config.name}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error stopping service '{service.service_config.name}': {e}")
            return False
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all services."""
        health_results = {}
        
        for service_name in self.registry.list_services():
            service = self.registry.get_service(service_name)
            if service:
                try:
                    health_results[service_name] = await service.health_check()
                except Exception as e:
                    health_results[service_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        return health_results
    
    def get_service_status(self) -> Dict[str, str]:
        """Get status of all services."""
        return {
            name: service.status.value
            for name, service in self.registry._services.items()
        }

# =============================================================================
# 🚀 FACTORY FUNCTIONS
# =============================================================================

def create_service_registry() -> ServiceRegistry:
    """Create a new service registry."""
    return ServiceRegistry()

def create_service_manager(registry: Optional[ServiceRegistry] = None) -> ServiceManager:
    """Create a new service manager."""
    if registry is None:
        registry = create_service_registry()
    return ServiceManager(registry)

def create_default_service_configs() -> Dict[str, ServiceConfig]:
    """Create default service configurations."""
    return {
        'core': ServiceConfig(
            name='core',
            service_type=ServiceType.CORE,
            auto_start=True,
            dependencies=[]
        ),
        'engine_manager': ServiceConfig(
            name='engine_manager',
            service_type=ServiceType.CORE,
            auto_start=True,
            dependencies=['core']
        ),
        'performance_monitor': ServiceConfig(
            name='performance_monitor',
            service_type=ServiceType.UTILITY,
            auto_start=True,
            dependencies=['core']
        )
    }

# =============================================================================
# 🌟 EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ServiceType",
    "ServiceStatus",
    
    # Configuration
    "ServiceConfig",
    
    # Interfaces
    "BlatamService",
    
    # Registry and Management
    "ServiceRegistry",
    "ServiceManager",
    
    # Factory functions
    "create_service_registry",
    "create_service_manager",
    "create_default_service_configs"
] 