"""
Blaze AI Enhanced Services v7.0.0

Service layer with advanced lifecycle management, dependency resolution,
and integration with quantum optimization and neural turbo engines.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type, Union
import threading
import time
from pathlib import Path

from ..core import BlazeComponent, ComponentConfig, ComponentType, ComponentStatus
from ..engines import EngineManager, EngineConfig, OptimizationLevel

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ServiceType(Enum):
    """Service types for different AI operations."""
    AI_PROCESSING = "ai_processing"
    DATA_MANAGEMENT = "data_management"
    MODEL_TRAINING = "model_training"
    INFERENCE = "inference"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"
    INTEGRATION = "integration"
    CUSTOM = "custom"

class ServiceStatus(Enum):
    """Service operational status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class ServicePriority(Enum):
    """Service priority levels."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class ServiceConfig(ComponentConfig):
    """Configuration for services."""
    service_type: ServiceType = ServiceType.AI_PROCESSING
    priority: ServicePriority = ServicePriority.MEDIUM
    auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_delay: float = 5.0
    dependencies: List[str] = field(default_factory=list)
    health_check_interval: float = 30.0
    timeout_seconds: float = 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "service_type": self.service_type.value,
            "priority": self.priority.value,
            "auto_restart": self.auto_restart,
            "max_restart_attempts": self.max_restart_attempts,
            "restart_delay": self.auto_restart,
            "dependencies": self.dependencies,
            "health_check_interval": self.health_check_interval
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceConfig':
        """Create from dictionary."""
        base_config = super().from_dict(data)
        return cls(
            **base_config.__dict__,
            service_type=ServiceType(data.get("service_type", "ai_processing")),
            priority=ServicePriority(data.get("priority", 3)),
            auto_restart=data.get("auto_restart", True),
            max_restart_attempts=data.get("max_restart_attempts", 3),
            restart_delay=data.get("restart_delay", 5.0),
            dependencies=data.get("dependencies", []),
            health_check_interval=data.get("health_check_interval", 30.0)
        )

# ============================================================================
# BASE SERVICE CLASSES
# ============================================================================

class BlazeService(BlazeComponent):
    """Base service class for Blaze AI."""
    
    def __init__(self, config: ServiceConfig):
        super().__init__(config)
        self.service_config = config
        self.service_status = ServiceStatus.STOPPED
        self.restart_count = 0
        self.last_health_check = 0
        self.health_check_cache: Optional[Dict[str, Any]] = None
        self.health_check_cache_ttl = 5.0  # 5 seconds cache
        self._service_lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
    
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
    
    async def initialize(self) -> bool:
        """Initialize the service."""
        try:
            if self.service_config.auto_start:
                await self.start()
            else:
                self.service_status = ServiceStatus.STOPPED
            
            self.status = ComponentStatus.READY
            self.start_time = time.time()
            
            # Start health check monitoring
            if self.service_config.health_check_interval > 0:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info(f"Service initialized: {self.service_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize service {self.service_config.name}: {e}")
            self._record_error(e)
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the service."""
        try:
            # Stop health check monitoring
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # Stop the service
            if self.service_status == ServiceStatus.RUNNING:
                await self.stop()
            
            self.status = ComponentStatus.SHUTDOWN
            logger.info(f"Service shutdown: {self.service_config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error during service shutdown: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health with caching."""
        current_time = time.time()
        
        # Return cached result if still valid
        if (self.health_check_cache and 
            current_time - self.last_health_check < self.health_check_cache_ttl):
            return self.health_check_cache
        
        # Perform actual health check
        try:
            health_data = await self._perform_health_check()
            health_data.update({
                "service_status": self.service_status.value,
                "restart_count": self.restart_count,
                "last_health_check": current_time
            })
            
            # Cache the result
            self.health_check_cache = health_data
            self.last_health_check = current_time
            
            return health_data
            
        except Exception as e:
            error_data = {
                "status": "error",
                "error": str(e),
                "service_status": self.service_status.value,
                "restart_count": self.restart_count,
                "last_health_check": current_time
            }
            self.health_check_cache = error_data
            self.last_health_check = current_time
            return error_data
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform the actual health check."""
        # Base health check - override in subclasses
        return {
            "status": "healthy",
            "uptime": time.time() - self.start_time if self.start_time else 0,
            "error_count": self.error_count
        }
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.service_config.health_check_interval)
                await self.health_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5.0)
    
    async def restart(self) -> bool:
        """Restart the service."""
        try:
            logger.info(f"Restarting service: {self.service_config.name}")
            
            if self.service_status == ServiceStatus.RUNNING:
                await self.stop()
            
            await asyncio.sleep(self.service_config.restart_delay)
            
            if await self.start():
                self.restart_count += 1
                logger.info(f"Service restarted successfully: {self.service_config.name}")
                return True
            else:
                logger.error(f"Failed to restart service: {self.service_config.name}")
                return False
                
        except Exception as e:
            logger.error(f"Error during service restart: {e}")
            return False

# ============================================================================
# SERVICE REGISTRY
# ============================================================================

class ServiceRegistry:
    """Registry for managing available services."""
    
    def __init__(self):
        self.services: Dict[str, Type[BlazeService]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.dependencies: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()
    
    def register_service(self, name: str, service_class: Type[BlazeService], 
                        metadata: Optional[Dict[str, Any]] = None,
                        dependencies: Optional[List[str]] = None):
        """Register a service class."""
        with self._lock:
            self.services[name] = service_class
            self.metadata[name] = metadata or {}
            self.dependencies[name] = set(dependencies or [])
            logger.info(f"Service registered: {name}")
    
    def get_service_class(self, name: str) -> Optional[Type[BlazeService]]:
        """Get service class by name."""
        return self.services.get(name)
    
    def list_services(self) -> List[str]:
        """List all registered services."""
        return list(self.services.keys())
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get service metadata."""
        return self.metadata.get(name, {})
    
    def get_dependencies(self, name: str) -> Set[str]:
        """Get service dependencies."""
        return self.dependencies.get(name, set())
    
    def resolve_dependencies(self, service_name: str) -> List[str]:
        """Resolve service dependencies in order."""
        resolved = []
        visited = set()
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            
            for dep in self.dependencies.get(name, []):
                visit(dep)
            
            resolved.append(name)
        
        visit(service_name)
        return resolved
    
    def check_circular_dependencies(self) -> List[List[str]]:
        """Check for circular dependencies."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(name: str, path: List[str]):
            if name in rec_stack:
                cycle = path[path.index(name):]
                cycles.append(cycle)
                return
            
            if name in visited:
                return
            
            visited.add(name)
            rec_stack.add(name)
            path.append(name)
            
            for dep in self.dependencies.get(name, []):
                dfs(dep, path)
            
            rec_stack.remove(name)
            path.pop()
        
        for service_name in self.services:
            if service_name not in visited:
                dfs(service_name, [])
        
        return cycles

# ============================================================================
# SERVICE MANAGER
# ============================================================================

class ServiceManager:
    """Manager for creating and managing service instances."""
    
    def __init__(self, registry: ServiceRegistry, engine_manager: Optional[EngineManager] = None):
        self.registry = registry
        self.engine_manager = engine_manager
        self.active_services: Dict[str, BlazeService] = {}
        self.service_configs: Dict[str, ServiceConfig] = {}
        self.service_status: Dict[str, ServiceStatus] = {}
        self._lock = threading.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def create_service(self, name: str, config: ServiceConfig) -> Optional[BlazeService]:
        """Create and initialize a service instance."""
        try:
            service_class = self.registry.get_service_class(name)
            if not service_class:
                logger.error(f"Unknown service: {name}")
                return None
            
            # Check dependencies
            if not await self._check_dependencies(config.dependencies):
                logger.error(f"Service dependencies not met: {name}")
                return None
            
            service = service_class(config)
            if await service.initialize():
                with self._lock:
                    self.active_services[name] = service
                    self.service_configs[name] = config
                    self.service_status[name] = service.service_status
                logger.info(f"Service created successfully: {name}")
                return service
            else:
                logger.error(f"Failed to initialize service: {name}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating service {name}: {e}")
            return None
    
    async def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if service dependencies are met."""
        for dep in dependencies:
            if dep not in self.active_services:
                return False
            if self.service_status.get(dep) != ServiceStatus.RUNNING:
                return False
        return True
    
    async def start_service(self, name: str) -> bool:
        """Start a service."""
        try:
            service = self.active_services.get(name)
            if not service:
                logger.error(f"Service not found: {name}")
                return False
            
            if await service.start():
                with self._lock:
                    self.service_status[name] = service.service_status
                logger.info(f"Service started: {name}")
                return True
            else:
                logger.error(f"Failed to start service: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting service {name}: {e}")
            return False
    
    async def stop_service(self, name: str) -> bool:
        """Stop a service."""
        try:
            service = self.active_services.get(name)
            if not service:
                logger.error(f"Service not found: {name}")
                return False
            
            if await service.stop():
                with self._lock:
                    self.service_status[name] = service.service_status
                logger.info(f"Service stopped: {name}")
                return True
            else:
                logger.error(f"Failed to stop service: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping service {name}: {e}")
            return False
    
    async def restart_service(self, name: str) -> bool:
        """Restart a service."""
        try:
            service = self.active_services.get(name)
            if not service:
                logger.error(f"Service not found: {name}")
                return False
            
            if await service.restart():
                with self._lock:
                    self.service_status[name] = service.service_status
                logger.info(f"Service restarted: {name}")
                return True
            else:
                logger.error(f"Failed to restart service: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error restarting service {name}: {e}")
            return False
    
    async def get_service(self, name: str) -> Optional[BlazeService]:
        """Get an active service instance."""
        return self.active_services.get(name)
    
    async def shutdown_service(self, name: str) -> bool:
        """Shutdown and remove a service."""
        try:
            service = self.active_services.get(name)
            if service:
                await service.shutdown()
                with self._lock:
                    del self.active_services[name]
                    del self.service_configs[name]
                    del self.service_status[name]
                logger.info(f"Service shutdown: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error shutting down service {name}: {e}")
            return False
    
    async def shutdown_all(self):
        """Shutdown all active services."""
        for name in list(self.active_services.keys()):
            await self.shutdown_service(name)
    
    async def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active services."""
        status = {}
        for name, service in self.active_services.items():
            try:
                status[name] = await service.health_check()
            except Exception as e:
                status[name] = {"status": "error", "error": str(e)}
        return status
    
    async def start_monitoring(self):
        """Start service monitoring."""
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Service monitoring started")
    
    async def stop_monitoring(self):
        """Stop service monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Service monitoring stopped")
    
    async def _monitoring_loop(self):
        """Service monitoring loop."""
        while True:
            try:
                await asyncio.sleep(30.0)  # Check every 30 seconds
                
                # Check for services that need restart
                for name, service in self.active_services.items():
                    config = self.service_configs[name]
                    if (config.auto_restart and 
                        service.service_status == ServiceStatus.ERROR and
                        service.restart_count < config.max_restart_attempts):
                        
                        logger.info(f"Auto-restarting service: {name}")
                        await service.restart()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in service monitoring loop: {e}")
                await asyncio.sleep(5.0)

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_service_registry() -> ServiceRegistry:
    """Create a default service registry."""
    registry = ServiceRegistry()
    
    # Register built-in services
    registry.register_service("ai_processor", BlazeService, {
        "description": "AI processing service",
        "version": "7.0.0",
        "type": ServiceType.AI_PROCESSING
    })
    
    registry.register_service("data_manager", BlazeService, {
        "description": "Data management service",
        "version": "7.0.0",
        "type": ServiceType.DATA_MANAGEMENT
    })
    
    registry.register_service("model_trainer", BlazeService, {
        "description": "Model training service",
        "version": "7.0.0",
        "type": ServiceType.MODEL_TRAINING
    })
    
    return registry

def create_service_manager(registry: Optional[ServiceRegistry] = None,
                          engine_manager: Optional[EngineManager] = None) -> ServiceManager:
    """Create a service manager."""
    if registry is None:
        registry = create_service_registry()
    return ServiceManager(registry, engine_manager)

def create_default_service_configs() -> Dict[str, ServiceConfig]:
    """Create default service configurations."""
    configs = {}
    
    configs["ai_processor"] = ServiceConfig(
        name="ai_processor",
        component_type=ComponentType.SERVICE,
        service_type=ServiceType.AI_PROCESSING,
        priority=ServicePriority.HIGH,
        performance_level=OptimizationLevel.ADVANCED,
        auto_restart=True,
        max_restart_attempts=3
    )
    
    configs["data_manager"] = ServiceConfig(
        name="data_manager",
        component_type=ComponentType.SERVICE,
        service_type=ServiceType.DATA_MANAGEMENT,
        priority=ServicePriority.MEDIUM,
        performance_level=OptimizationLevel.STANDARD,
        auto_restart=True,
        max_restart_attempts=3
    )
    
    configs["model_trainer"] = ServiceConfig(
        name="model_trainer",
        component_type=ComponentType.SERVICE,
        service_type=ServiceType.MODEL_TRAINING,
        priority=ServicePriority.MEDIUM,
        performance_level=OptimizationLevel.ADVANCED,
        auto_restart=True,
        max_restart_attempts=3
    )
    
    return configs

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "ServiceType",
    "ServiceStatus",
    "ServicePriority",
    
    # Configuration
    "ServiceConfig",
    
    # Base Classes
    "BlazeService",
    
    # Management
    "ServiceRegistry",
    "ServiceManager",
    
    # Factory Functions
    "create_service_registry",
    "create_service_manager",
    "create_default_service_configs"
]

# Version info
__version__ = "7.0.0"


