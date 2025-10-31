"""
Blaze AI Enhanced Factories v7.0.0

Factory system for creating and managing components with advanced
optimization, quantum processing, and neural turbo acceleration.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Generic, Union
import threading
import time
from pathlib import Path
import weakref

from ..core import BlazeComponent, ComponentConfig, ComponentType, ComponentStatus
from ..engines import EngineManager, EngineConfig, OptimizationLevel
from ..services import ServiceManager, ServiceConfig

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class FactoryType(Enum):
    """Factory types for different component categories."""
    ENGINE = "engine"
    SERVICE = "service"
    UTILITY = "utility"
    CORE = "core"
    HYBRID = "hybrid"

class FactoryStatus(Enum):
    """Factory operational status."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PRODUCING = "producing"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class ProductionMode(Enum):
    """Component production modes."""
    SINGLETON = "singleton"
    PROTOTYPE = "prototype"
    BATCH = "batch"
    STREAMING = "streaming"
    ADAPTIVE = "adaptive"

# Generic type for components
T = TypeVar('T', bound=BlazeComponent)

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class FactoryConfig(ComponentConfig):
    """Configuration for component factories."""
    factory_type: FactoryType = FactoryType.CORE
    production_mode: ProductionMode = ProductionMode.SINGLETON
    max_instances: int = 100
    instance_timeout: float = 300.0  # 5 minutes
    enable_caching: bool = True
    enable_monitoring: bool = True
    enable_auto_cleanup: bool = True
    cleanup_interval: float = 60.0  # 1 minute
    performance_tracking: bool = True
    custom_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "factory_type": self.factory_type.value,
            "production_mode": self.production_mode.value,
            "max_instances": self.max_instances,
            "instance_timeout": self.instance_timeout,
            "enable_caching": self.enable_caching,
            "enable_monitoring": self.enable_monitoring,
            "enable_auto_cleanup": self.enable_auto_cleanup,
            "cleanup_interval": self.cleanup_interval,
            "performance_tracking": self.performance_tracking
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactoryConfig':
        """Create from dictionary."""
        base_config = super().from_dict(data)
        return cls(
            **base_config.__dict__,
            factory_type=FactoryType(data.get("factory_type", "core")),
            production_mode=ProductionMode(data.get("production_mode", "singleton")),
            max_instances=data.get("max_instances", 100),
            instance_timeout=data.get("instance_timeout", 300.0),
            enable_caching=data.get("enable_caching", True),
            enable_monitoring=data.get("enable_monitoring", True),
            enable_auto_cleanup=data.get("enable_auto_cleanup", True),
            cleanup_interval=data.get("cleanup_interval", 60.0),
            performance_tracking=data.get("performance_tracking", True)
        )

@dataclass
class ProductionStats:
    """Statistics for component production."""
    total_created: int = 0
    total_destroyed: int = 0
    active_instances: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_creation_time: float = 0.0
    total_creation_time: float = 0.0
    error_count: int = 0
    last_production_time: Optional[float] = None
    
    def record_creation(self, creation_time: float):
        """Record a successful component creation."""
        self.total_created += 1
        self.active_instances += 1
        self.total_creation_time += creation_time
        self.average_creation_time = self.total_creation_time / self.total_created
        self.last_production_time = time.time()
    
    def record_destruction(self):
        """Record a component destruction."""
        self.total_destroyed += 1
        self.active_instances = max(0, self.active_instances - 1)
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1
    
    def record_error(self):
        """Record a production error."""
        self.error_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_created": self.total_created,
            "total_destroyed": self.total_destroyed,
            "active_instances": self.active_instances,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "average_creation_time": self.average_creation_time,
            "total_creation_time": self.total_creation_time,
            "error_count": self.error_count,
            "last_production_time": self.last_production_time,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0
        }

# ============================================================================
# BASE FACTORY CLASSES
# ============================================================================

class BlazeFactory(ABC, Generic[T]):
    """Base abstract factory class for Blaze AI."""
    
    def __init__(self, config: FactoryConfig):
        self.config = config
        self.status = FactoryStatus.UNINITIALIZED
        self.instances: Dict[str, T] = {}
        self.instance_metadata: Dict[str, Dict[str, Any]] = {}
        self.production_stats = ProductionStats()
        self._lock = threading.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
    
    @abstractmethod
    async def create_component(self, config: ComponentConfig) -> T:
        """Create a component instance."""
        pass
    
    @abstractmethod
    async def destroy_component(self, component: T) -> bool:
        """Destroy a component instance."""
        pass
    
    async def initialize(self) -> bool:
        """Initialize the factory."""
        try:
            # Start auto-cleanup if enabled
            if self.config.enable_auto_cleanup:
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            # Start monitoring if enabled
            if self.config.enable_monitoring:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.status = FactoryStatus.READY
            self.start_time = time.time()
            logger.info(f"Factory initialized: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize factory {self.config.name}: {e}")
            self._record_error(e)
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the factory."""
        try:
            # Stop cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Stop monitoring task
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            # Destroy all instances
            for instance_id in list(self.instances.keys()):
                await self._destroy_instance(instance_id)
            
            self.status = FactoryStatus.SHUTDOWN
            logger.info(f"Factory shutdown: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error during factory shutdown: {e}")
            return False
    
    async def create_component_with_monitoring(self, config: ComponentConfig) -> T:
        """Create a component with performance monitoring."""
        start_time = time.time()
        
        try:
            # Check instance limit
            if len(self.instances) >= self.config.max_instances:
                # Try to clean up expired instances first
                await self._cleanup_expired_instances()
                if len(self.instances) >= self.config.max_instances:
                    raise RuntimeError(f"Factory instance limit reached: {self.config.max_instances}")
            
            # Create component
            component = await self.create_component(config)
            
            # Record creation
            creation_time = time.time() - start_time
            self.production_stats.record_creation(creation_time)
            
            # Store instance
            instance_id = self._generate_instance_id(component)
            with self._lock:
                self.instances[instance_id] = component
                self.instance_metadata[instance_id] = {
                    "created_at": time.time(),
                    "config": config,
                    "creation_time": creation_time
                }
            
            logger.info(f"Component created: {instance_id} in {creation_time:.3f}s")
            return component
            
        except Exception as e:
            self.production_stats.record_error()
            logger.error(f"Failed to create component: {e}")
            raise
    
    async def _destroy_instance(self, instance_id: str) -> bool:
        """Destroy a specific instance."""
        try:
            component = self.instances.get(instance_id)
            if component:
                await self.destroy_component(component)
                with self._lock:
                    del self.instances[instance_id]
                    del self.instance_metadata[instance_id]
                self.production_stats.record_destruction()
                logger.info(f"Instance destroyed: {instance_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error destroying instance {instance_id}: {e}")
            return False
    
    def _generate_instance_id(self, component: T) -> str:
        """Generate a unique instance ID."""
        return f"{self.config.name}_{id(component)}_{int(time.time())}"
    
    async def _cleanup_expired_instances(self):
        """Clean up expired component instances."""
        current_time = time.time()
        expired_instances = []
        
        with self._lock:
            for instance_id, metadata in self.instance_metadata.items():
                if current_time - metadata["created_at"] > self.config.instance_timeout:
                    expired_instances.append(instance_id)
        
        for instance_id in expired_instances:
            await self._destroy_instance(instance_id)
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired_instances()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _monitoring_loop(self):
        """Background monitoring loop."""
        while True:
            try:
                await asyncio.sleep(30.0)  # Monitor every 30 seconds
                
                # Update performance metrics
                self._update_metrics("active_instances", len(self.instances))
                self._update_metrics("total_instances_created", self.production_stats.total_created)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get factory statistics."""
        return {
            "name": self.config.name,
            "type": self.config.factory_type.value,
            "status": self.status.value,
            "production_stats": self.production_stats.to_dict(),
            "active_instances": len(self.instances),
            "max_instances": self.config.max_instances,
            "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }
    
    def _update_metrics(self, metric_name: str, value: Any):
        """Update performance metrics."""
        # This would update component-level metrics
        pass
    
    def _record_error(self, error: Exception):
        """Record an error occurrence."""
        self.production_stats.record_error()
        self.status = FactoryStatus.ERROR
        logger.error(f"Factory {self.config.name} error: {error}")

# ============================================================================
# SPECIALIZED FACTORIES
# ============================================================================

class EngineFactory(BlazeFactory):
    """Factory for creating AI engine components."""
    
    def __init__(self, config: FactoryConfig, engine_manager: Optional[EngineManager] = None):
        super().__init__(config)
        self.engine_manager = engine_manager
    
    async def create_component(self, config: ComponentConfig) -> Any:
        """Create an engine component."""
        # This would create specific engine types based on config
        # For now, return a placeholder
        return None
    
    async def destroy_component(self, component: Any) -> bool:
        """Destroy an engine component."""
        # This would properly destroy engine components
        return True

class ServiceFactory(BlazeFactory):
    """Factory for creating service components."""
    
    def __init__(self, config: FactoryConfig, service_manager: Optional[ServiceManager] = None):
        super().__init__(config)
        self.service_manager = service_manager
    
    async def create_component(self, config: ComponentConfig) -> Any:
        """Create a service component."""
        # This would create specific service types based on config
        # For now, return a placeholder
        return None
    
    async def destroy_component(self, component: Any) -> bool:
        """Destroy a service component."""
        # This would properly destroy service components
        return True

class HybridFactory(BlazeFactory):
    """Factory for creating hybrid components that combine multiple types."""
    
    def __init__(self, config: FactoryConfig, 
                 engine_manager: Optional[EngineManager] = None,
                 service_manager: Optional[ServiceManager] = None):
        super().__init__(config)
        self.engine_manager = engine_manager
        self.service_manager = service_manager
    
    async def create_component(self, config: ComponentConfig) -> Any:
        """Create a hybrid component."""
        # This would create components that combine engines and services
        # For now, return a placeholder
        return None
    
    async def destroy_component(self, component: Any) -> bool:
        """Destroy a hybrid component."""
        # This would properly destroy hybrid components
        return True

# ============================================================================
# FACTORY REGISTRY
# ============================================================================

class FactoryRegistry:
    """Registry for managing component factories."""
    
    def __init__(self):
        self.factories: Dict[str, Type[BlazeFactory]] = {}
        self.factory_instances: Dict[str, BlazeFactory] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def register_factory(self, name: str, factory_class: Type[BlazeFactory], 
                        metadata: Optional[Dict[str, Any]] = None):
        """Register a factory class."""
        with self._lock:
            self.factories[name] = factory_class
            self.metadata[name] = metadata or {}
            logger.info(f"Factory registered: {name}")
    
    def get_factory_class(self, name: str) -> Optional[Type[BlazeFactory]]:
        """Get factory class by name."""
        return self.factories.get(name)
    
    def create_factory_instance(self, name: str, config: FactoryConfig, **kwargs) -> Optional[BlazeFactory]:
        """Create a factory instance."""
        factory_class = self.get_factory_class(name)
        if not factory_class:
            logger.error(f"Unknown factory: {name}")
            return None
        
        try:
            factory = factory_class(config, **kwargs)
            self.factory_instances[name] = factory
            logger.info(f"Factory instance created: {name}")
            return factory
        except Exception as e:
            logger.error(f"Failed to create factory instance {name}: {e}")
            return None
    
    def get_factory_instance(self, name: str) -> Optional[BlazeFactory]:
        """Get a factory instance."""
        return self.factory_instances.get(name)
    
    def list_factories(self) -> List[str]:
        """List all registered factories."""
        return list(self.factories.keys())
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get factory metadata."""
        return self.metadata.get(name, {})

# ============================================================================
# FACTORY MANAGER
# ============================================================================

class FactoryManager:
    """Manager for creating and managing factory instances."""
    
    def __init__(self, registry: FactoryRegistry):
        self.registry = registry
        self.active_factories: Dict[str, BlazeFactory] = {}
        self.factory_configs: Dict[str, FactoryConfig] = {}
        self._lock = threading.Lock()
    
    async def create_factory(self, name: str, config: FactoryConfig, **kwargs) -> Optional[BlazeFactory]:
        """Create and initialize a factory."""
        try:
            factory = self.registry.create_factory_instance(name, config, **kwargs)
            if factory and await factory.initialize():
                with self._lock:
                    self.active_factories[name] = factory
                    self.factory_configs[name] = config
                logger.info(f"Factory created successfully: {name}")
                return factory
            else:
                logger.error(f"Failed to initialize factory: {name}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating factory {name}: {e}")
            return None
    
    async def get_factory(self, name: str) -> Optional[BlazeFactory]:
        """Get an active factory instance."""
        return self.active_factories.get(name)
    
    async def shutdown_factory(self, name: str) -> bool:
        """Shutdown and remove a factory."""
        try:
            factory = self.active_factories.get(name)
            if factory:
                await factory.shutdown()
                with self._lock:
                    del self.active_factories[name]
                    del self.factory_configs[name]
                logger.info(f"Factory shutdown: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error shutting down factory {name}: {e}")
            return False
    
    async def shutdown_all(self):
        """Shutdown all active factories."""
        for name in list(self.active_factories.keys()):
            await self.shutdown_factory(name)
    
    def get_all_factories_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all active factories."""
        stats = {}
        for name, factory in self.active_factories.items():
            try:
                stats[name] = factory.get_stats()
            except Exception as e:
                stats[name] = {"status": "error", "error": str(e)}
        return stats

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_factory_registry() -> FactoryRegistry:
    """Create a default factory registry."""
    registry = FactoryRegistry()
    
    # Register built-in factories
    registry.register_factory("engine", EngineFactory, {
        "description": "AI engine factory",
        "version": "7.0.0",
        "type": FactoryType.ENGINE
    })
    
    registry.register_factory("service", ServiceFactory, {
        "description": "Service factory",
        "version": "7.0.0",
        "type": FactoryType.SERVICE
    })
    
    registry.register_factory("hybrid", HybridFactory, {
        "description": "Hybrid component factory",
        "version": "7.0.0",
        "type": FactoryType.HYBRID
    })
    
    return registry

def create_factory_manager(registry: Optional[FactoryRegistry] = None) -> FactoryManager:
    """Create a factory manager."""
    if registry is None:
        registry = create_factory_registry()
    return FactoryManager(registry)

def create_default_factory_configs() -> Dict[str, FactoryConfig]:
    """Create default factory configurations."""
    configs = {}
    
    configs["engine_factory"] = FactoryConfig(
        name="engine_factory",
        component_type=ComponentType.UTILITY,
        factory_type=FactoryType.ENGINE,
        production_mode=ProductionMode.SINGLETON,
        max_instances=50,
        performance_level=OptimizationLevel.ADVANCED
    )
    
    configs["service_factory"] = FactoryConfig(
        name="service_factory",
        component_type=ComponentType.UTILITY,
        factory_type=FactoryType.SERVICE,
        production_mode=ProductionMode.SINGLETON,
        max_instances=100,
        performance_level=OptimizationLevel.STANDARD
    )
    
    configs["hybrid_factory"] = FactoryConfig(
        name="hybrid_factory",
        component_type=ComponentType.UTILITY,
        factory_type=FactoryType.HYBRID,
        production_mode=ProductionMode.PROTOTYPE,
        max_instances=25,
        performance_level=OptimizationLevel.NEURAL_TURBO
    )
    
    return configs

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "FactoryType",
    "FactoryStatus",
    "ProductionMode",
    
    # Configuration
    "FactoryConfig",
    "ProductionStats",
    
    # Base Classes
    "BlazeFactory",
    
    # Specialized Factories
    "EngineFactory",
    "ServiceFactory",
    "HybridFactory",
    
    # Management
    "FactoryRegistry",
    "FactoryManager",
    
    # Factory Functions
    "create_factory_registry",
    "create_factory_manager",
    "create_default_factory_configs"
]

# Version info
__version__ = "7.0.0"
