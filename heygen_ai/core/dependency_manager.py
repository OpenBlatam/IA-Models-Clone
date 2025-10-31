"""
Centralized Dependency Management System
======================================

Manages all system dependencies and their lifecycle:
- Service registration and discovery
- Dependency injection
- Lifecycle management
- Health monitoring integration
- Configuration integration
"""

import asyncio
import time
from typing import Dict, Any, Optional, Callable, List, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import weakref

from .config_manager import get_config
from .logger_manager import get_logger

logger = get_logger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ServicePriority(Enum):
    """Service priority enumeration"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class ServiceInfo:
    """Service information"""
    name: str
    service_type: str
    priority: ServicePriority
    status: ServiceStatus
    start_time: Optional[float] = None
    stop_time: Optional[float] = None
    error_count: int = 0
    last_error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ServiceLifecycle:
    """Service lifecycle management"""
    
    def __init__(self, name: str, service_type: str, priority: ServicePriority = ServicePriority.NORMAL):
        self.name = name
        self.service_type = service_type
        self.priority = priority
        self.status = ServiceStatus.UNKNOWN
        self.start_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.dependencies: List[str] = []
        self.metadata: Dict[str, Any] = {}
        
        # Lifecycle hooks
        self._on_start: Optional[Callable] = None
        self._on_stop: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        
        logger.debug(f"Created service lifecycle for {name}")
    
    def add_dependency(self, service_name: str):
        """Add a dependency"""
        if service_name not in self.dependencies:
            self.dependencies.append(service_name)
            logger.debug(f"Added dependency {service_name} to {self.name}")
    
    def remove_dependency(self, service_name: str):
        """Remove a dependency"""
        if service_name in self.dependencies:
            self.dependencies.remove(service_name)
            logger.debug(f"Removed dependency {service_name} from {self.name}")
    
    def on_start(self, callback: Callable):
        """Set start callback"""
        self._on_start = callback
    
    def on_stop(self, callback: Callable):
        """Set stop callback"""
        self._on_stop = callback
    
    def on_error(self, callback: Callable):
        """Set error callback"""
        self._on_error = callback
    
    async def start(self):
        """Start the service"""
        try:
            logger.info(f"Starting service {self.name}")
            self.status = ServiceStatus.STARTING
            self.start_time = time.time()
            
            if self._on_start:
                await self._on_start()
            
            self.status = ServiceStatus.RUNNING
            logger.info(f"Service {self.name} started successfully")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Failed to start service {self.name}: {e}")
            
            if self._on_error:
                await self._on_error(e)
            
            raise
    
    async def stop(self):
        """Stop the service"""
        try:
            logger.info(f"Stopping service {self.name}")
            self.status = ServiceStatus.STOPPING
            self.stop_time = time.time()
            
            if self._on_stop:
                await self._on_stop()
            
            self.status = ServiceStatus.STOPPED
            logger.info(f"Service {self.name} stopped successfully")
            
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.error_count += 1
            self.last_error = str(e)
            logger.error(f"Failed to stop service {self.name}: {e}")
            
            if self._on_error:
                await self._on_error(e)
            
            raise
    
    def to_info(self) -> ServiceInfo:
        """Convert to ServiceInfo"""
        return ServiceInfo(
            name=self.name,
            service_type=self.service_type,
            priority=self.priority,
            status=self.status,
            start_time=self.start_time,
            stop_time=self.stop_time,
            error_count=self.error_count,
            last_error=self.last_error,
            dependencies=self.dependencies.copy(),
            metadata=self.metadata.copy()
        )


class DependencyManager:
    """Centralized dependency manager"""
    
    def __init__(self):
        self.config = get_config()
        self.services: Dict[str, ServiceLifecycle] = {}
        self.service_instances: Dict[str, Any] = {}
        self.service_factories: Dict[str, Callable] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.is_running = False
        
        logger.info("Dependency manager initialized")
    
    def register_service(
        self,
        name: str,
        service_type: str,
        factory: Callable,
        priority: ServicePriority = ServicePriority.NORMAL,
        dependencies: Optional[List[str]] = None
    ):
        """Register a service"""
        if name in self.services:
            logger.warning(f"Service {name} already registered, overwriting")
        
        # Create lifecycle
        lifecycle = ServiceLifecycle(name, service_type, priority)
        
        # Add dependencies
        if dependencies:
            for dep in dependencies:
                lifecycle.add_dependency(dep)
        
        # Store
        self.services[name] = lifecycle
        self.service_factories[name] = factory
        self.dependency_graph[name] = dependencies or []
        
        logger.info(f"Registered service {name} ({service_type}) with priority {priority.value}")
    
    def unregister_service(self, name: str):
        """Unregister a service"""
        if name not in self.services:
            logger.warning(f"Service {name} not registered")
            return
        
        # Stop if running
        if self.services[name].status == ServiceStatus.RUNNING:
            asyncio.create_task(self.services[name].stop())
        
        # Remove
        del self.services[name]
        if name in self.service_instances:
            del self.service_instances[name]
        if name in self.service_factories:
            del self.service_factories[name]
        if name in self.dependency_graph:
            del self.dependency_graph[name]
        
        logger.info(f"Unregistered service {name}")
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get a service instance"""
        return self.service_instances.get(name)
    
    def has_service(self, name: str) -> bool:
        """Check if service exists"""
        return name in self.services
    
    def get_service_status(self, name: str) -> Optional[ServiceStatus]:
        """Get service status"""
        if name in self.services:
            return self.services[name].status
        return None
    
    def get_service_info(self, name: str) -> Optional[ServiceInfo]:
        """Get service information"""
        if name in self.services:
            return self.services[name].to_info()
        return None
    
    def get_all_services(self) -> List[ServiceInfo]:
        """Get information about all services"""
        return [service.to_info() for service in self.services.values()]
    
    def get_services_by_status(self, status: ServiceStatus) -> List[ServiceInfo]:
        """Get services by status"""
        return [
            service.to_info() 
            for service in self.services.values() 
            if service.status == status
        ]
    
    def get_services_by_priority(self, priority: ServicePriority) -> List[ServiceInfo]:
        """Get services by priority"""
        return [
            service.to_info() 
            for service in self.services.values() 
            if service.priority == priority
        ]
    
    def check_dependencies(self, service_name: str) -> bool:
        """Check if service dependencies are satisfied"""
        if service_name not in self.dependency_graph:
            return True
        
        dependencies = self.dependency_graph[service_name]
        for dep in dependencies:
            if dep not in self.services:
                logger.warning(f"Service {service_name} depends on {dep} which is not registered")
                return False
            
            if self.services[dep].status != ServiceStatus.RUNNING:
                logger.warning(f"Service {service_name} depends on {dep} which is not running")
                return False
        
        return True
    
    def get_startup_order(self) -> List[str]:
        """Get the order in which services should be started"""
        # Simple topological sort for dependencies
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(service_name: str):
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected: {service_name}")
            if service_name in visited:
                return
            
            temp_visited.add(service_name)
            
            # Visit dependencies first
            if service_name in self.dependency_graph:
                for dep in self.dependency_graph[service_name]:
                    visit(dep)
            
            temp_visited.remove(service_name)
            visited.add(service_name)
            order.append(service_name)
        
        # Visit all services
        for service_name in self.services.keys():
            if service_name not in visited:
                visit(service_name)
        
        # Sort by priority within dependency order
        def sort_key(service_name: str):
            return (
                order.index(service_name),
                self.services[service_name].priority.value
            )
        
        return sorted(order, key=sort_key)
    
    async def start_all_services(self):
        """Start all services in dependency order"""
        if self.is_running:
            logger.warning("Services already running")
            return
        
        self.is_running = True
        startup_order = self.get_startup_order()
        
        logger.info(f"Starting {len(startup_order)} services in order: {startup_order}")
        
        for service_name in startup_order:
            try:
                # Check dependencies
                if not self.check_dependencies(service_name):
                    logger.error(f"Cannot start {service_name}: dependencies not satisfied")
                    continue
                
                # Create instance
                if service_name not in self.service_instances:
                    factory = self.service_factories[service_name]
                    self.service_instances[service_name] = factory()
                
                # Start service
                await self.services[service_name].start()
                
                logger.info(f"Started service {service_name}")
                
            except Exception as e:
                logger.error(f"Failed to start service {service_name}: {e}")
                self.services[service_name].status = ServiceStatus.ERROR
                self.services[service_name].last_error = str(e)
        
        logger.info("All services started")
    
    async def stop_all_services(self):
        """Stop all services in reverse dependency order"""
        if not self.is_running:
            logger.warning("Services not running")
            return
        
        self.is_running = False
        startup_order = self.get_startup_order()
        shutdown_order = list(reversed(startup_order))
        
        logger.info(f"Stopping {len(shutdown_order)} services in order: {shutdown_order}")
        
        for service_name in shutdown_order:
            try:
                if self.services[service_name].status == ServiceStatus.RUNNING:
                    await self.services[service_name].stop()
                    logger.info(f"Stopped service {service_name}")
                
            except Exception as e:
                logger.error(f"Failed to stop service {service_name}: {e}")
        
        logger.info("All services stopped")
    
    async def restart_service(self, service_name: str):
        """Restart a specific service"""
        if service_name not in self.services:
            logger.warning(f"Service {service_name} not registered")
            return
        
        try:
            logger.info(f"Restarting service {service_name}")
            
            # Stop if running
            if self.services[service_name].status == ServiceStatus.RUNNING:
                await self.services[service_name].stop()
            
            # Start again
            await self.services[service_name].start()
            
            logger.info(f"Service {service_name} restarted successfully")
            
        except Exception as e:
            logger.error(f"Failed to restart service {service_name}: {e}")
            raise
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary of all services"""
        total_services = len(self.services)
        running_services = len(self.get_services_by_status(ServiceStatus.RUNNING))
        error_services = len(self.get_services_by_status(ServiceStatus.ERROR))
        stopped_services = len(self.get_services_by_status(ServiceStatus.STOPPED))
        
        return {
            'total_services': total_services,
            'running_services': running_services,
            'error_services': error_services,
            'stopped_services': stopped_services,
            'health_percentage': (running_services / total_services * 100) if total_services > 0 else 0,
            'services': self.get_all_services(),
            'timestamp': time.time()
        }
    
    @asynccontextmanager
    async def managed_services(self):
        """Context manager for service lifecycle"""
        try:
            await self.start_all_services()
            yield self
        finally:
            await self.stop_all_services()


# Global dependency manager instance
dependency_manager = DependencyManager()


def get_dependency_manager() -> DependencyManager:
    """Get global dependency manager instance"""
    return dependency_manager


def register_service(
    name: str,
    service_type: str,
    factory: Callable,
    priority: ServicePriority = ServicePriority.NORMAL,
    dependencies: Optional[List[str]] = None
):
    """Register a service using global manager"""
    dependency_manager.register_service(name, service_type, factory, priority, dependencies)


def get_service(name: str) -> Optional[Any]:
    """Get a service using global manager"""
    return dependency_manager.get_service(name)


def has_service(name: str) -> bool:
    """Check if service exists using global manager"""
    return dependency_manager.has_service(name)


async def start_all_services():
    """Start all services using global manager"""
    await dependency_manager.start_all_services()


async def stop_all_services():
    """Stop all services using global manager"""
    await dependency_manager.stop_all_services()


if __name__ == "__main__":
    # Example usage
    async def main():
        # Register some services
        register_service("config", "configuration", lambda: get_config())
        register_service("logger", "logging", lambda: get_logger("main"))
        
        # Start services
        async with dependency_manager.managed_services():
            # Services are running
            config_service = get_service("config")
            logger_service = get_service("logger")
            
            print(f"Config service: {config_service}")
            print(f"Logger service: {logger_service}")
            
            # Get health summary
            health = dependency_manager.get_health_summary()
            print(f"Health: {health}")
    
    asyncio.run(main())
