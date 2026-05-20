"""
🏭 BLATAM AI FACTORIES MODULE v6.0.0
=====================================

Factory patterns for the Blatam AI system:
- 🏭 Component factories with dependency injection
- 🔧 Configuration-driven component creation
- ⚡ Async factory operations
- 📊 Factory performance monitoring
- 🚀 Component lifecycle management
- 🧹 Resource cleanup and management
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic, ClassVar
import uuid
import inspect

from ..core import BlatamComponent, ComponentConfig, ComponentType, ServiceContainer

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# 🎯 FACTORY TYPES AND STATUS
# =============================================================================

class FactoryType(Enum):
    """Factory types for categorization."""
    COMPONENT = "component"
    ENGINE = "engine"
    SERVICE = "service"
    UTILITY = "utility"
    INTERFACE = "interface"

class FactoryStatus(Enum):
    """Factory lifecycle status."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    CREATING = "creating"
    ERROR = "error"
    SHUTDOWN = "shutdown"

# =============================================================================
# 🎯 FACTORY CONFIGURATION
# =============================================================================

@dataclass
class FactoryConfig:
    """Configuration for factories."""
    name: str
    factory_type: FactoryType
    enabled: bool = True
    max_instances: int = 100
    instance_timeout: float = 300.0
    enable_caching: bool = True
    cache_size: int = 1000
    enable_monitoring: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'factory_type': self.factory_type.value,
            'enabled': self.enabled,
            'max_instances': self.max_instances,
            'instance_timeout': self.instance_timeout,
            'enable_caching': self.enable_caching,
            'cache_size': self.cache_size,
            'enable_monitoring': self.enable_monitoring,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactoryConfig':
        """Create config from dictionary."""
        if 'factory_type' in data and isinstance(data['factory_type'], str):
            data['factory_type'] = FactoryType(data['factory_type'])
        return cls(**data)

# =============================================================================
# 🎯 BASE FACTORY INTERFACES
# =============================================================================

T = TypeVar('T', bound=BlatamComponent)

class BlatamFactory(ABC, Generic[T]):
    """Base factory interface for creating Blatam components."""
    
    def __init__(self, config: FactoryConfig, service_container: ServiceContainer):
        self.config = config
        self.service_container = service_container
        self.status = FactoryStatus.UNINITIALIZED
        self.factory_id = str(uuid.uuid4())
        self.created_instances: List[str] = []
        self.active_instances: Dict[str, T] = {}
        self.instance_metadata: Dict[str, Dict[str, Any]] = {}
        self._creation_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self._creation_count = 0
        self._total_creation_time = 0.0
        self._error_count = 0
        
        logger.debug(f"🏭 Factory '{self.__class__.__name__}' created with ID: {self.factory_id}")
    
    @abstractmethod
    async def create_component(self, config: ComponentConfig, **kwargs) -> T:
        """Create a component instance."""
        pass
    
    @abstractmethod
    def get_component_type(self) -> ComponentType:
        """Get the type of component this factory creates."""
        pass
    
    @abstractmethod
    def get_required_config_fields(self) -> List[str]:
        """Get required configuration fields."""
        pass
    
    async def initialize(self) -> bool:
        """Initialize the factory."""
        try:
            self.status = FactoryStatus.INITIALIZING
            logger.info(f"🚀 Initializing factory: {self.config.name}")
            
            # Start cleanup task if caching is enabled
            if self.config.enable_caching:
                self._start_cleanup_task()
            
            self.status = FactoryStatus.READY
            logger.info(f"✅ Factory initialized: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize factory '{self.config.name}': {e}")
            self.status = FactoryStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the factory."""
        try:
            self.status = FactoryStatus.SHUTDOWN
            logger.info(f"🔄 Shutting down factory: {self.config.name}")
            
            # Stop cleanup task
            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup active instances
            await self._cleanup_all_instances()
            
            logger.info(f"✅ Factory shutdown: {self.config.name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to shutdown factory '{self.config.name}': {e}")
            return False
    
    async def create_component_with_monitoring(self, config: ComponentConfig, **kwargs) -> T:
        """Create component with performance monitoring."""
        if not self.config.enable_monitoring:
            return await self.create_component(config, **kwargs)
        
        start_time = time.time()
        try:
            component = await self.create_component(config, **kwargs)
            creation_time = time.time() - start_time
            
            # Record performance metrics
            self._creation_count += 1
            self._total_creation_time += creation_time
            
            # Track instance
            instance_id = str(uuid.uuid4())
            self.created_instances.append(instance_id)
            self.active_instances[instance_id] = component
            self.instance_metadata[instance_id] = {
                'created_at': time.time(),
                'config': config.to_dict() if hasattr(config, 'to_dict') else str(config),
                'creation_time': creation_time,
                'kwargs': kwargs
            }
            
            logger.debug(f"🏭 Component created by factory '{self.config.name}' in {creation_time:.3f}s")
            return component
            
        except Exception as e:
            self._error_count += 1
            creation_time = time.time() - start_time
            logger.error(f"❌ Component creation failed in factory '{self.config.name}' after {creation_time:.3f}s: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get factory statistics."""
        return {
            'factory_id': self.factory_id,
            'name': self.config.name,
            'status': self.status.value,
            'creation_count': self._creation_count,
            'total_creation_time': self._total_creation_time,
            'avg_creation_time': (
                self._total_creation_time / self._creation_count 
                if self._creation_count > 0 else 0.0
            ),
            'error_count': self._error_count,
            'active_instances': len(self.active_instances),
            'total_instances_created': len(self.created_instances)
        }
    
    def _start_cleanup_task(self) -> None:
        """Start periodic cleanup task."""
        if self._cleanup_task:
            return
        
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.instance_timeout / 2)
                    await self._cleanup_expired_instances()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"❌ Cleanup error in factory '{self.config.name}': {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_expired_instances(self) -> None:
        """Cleanup expired instances."""
        current_time = time.time()
        expired_instances = []
        
        for instance_id, metadata in self.instance_metadata.items():
            if current_time - metadata['created_at'] > self.config.instance_timeout:
                expired_instances.append(instance_id)
        
        for instance_id in expired_instances:
            await self._cleanup_instance(instance_id)
    
    async def _cleanup_instance(self, instance_id: str) -> None:
        """Cleanup a specific instance."""
        try:
            if instance_id in self.active_instances:
                instance = self.active_instances[instance_id]
                if hasattr(instance, 'shutdown'):
                    await instance.shutdown()
                del self.active_instances[instance_id]
            
            if instance_id in self.instance_metadata:
                del self.instance_metadata[instance_id]
            
            logger.debug(f"🧹 Cleaned up instance {instance_id} in factory '{self.config.name}'")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup instance {instance_id} in factory '{self.config.name}': {e}")
    
    async def _cleanup_all_instances(self) -> None:
        """Cleanup all active instances."""
        instance_ids = list(self.active_instances.keys())
        for instance_id in instance_ids:
            await self._cleanup_instance(instance_id)

# =============================================================================
# 🎯 FACTORY REGISTRY
# =============================================================================

class FactoryRegistry:
    """Centralized registry for all factories."""
    
    def __init__(self):
        self._factories: Dict[str, BlatamFactory] = {}
        self._factory_configs: Dict[str, FactoryConfig] = {}
        self._factory_metadata: Dict[str, Dict[str, Any]] = {}
        self._type_to_factories: Dict[ComponentType, List[str]] = {}
        self._registry_lock = asyncio.Lock()
    
    async def register_factory(self, factory: BlatamFactory, config: FactoryConfig) -> bool:
        """Register a factory."""
        async with self._registry_lock:
            try:
                factory_name = config.name
                
                # Check if factory already exists
                if factory_name in self._factories:
                    logger.warning(f"⚠️ Factory '{factory_name}' already registered, replacing")
                    await self._unregister_factory_internal(factory_name)
                
                # Register factory
                self._factories[factory_name] = factory
                self._factory_configs[factory_name] = config
                self._factory_metadata[factory_name] = config.metadata
                
                # Update type mapping
                component_type = factory.get_component_type()
                if component_type not in self._type_to_factories:
                    self._type_to_factories[component_type] = []
                self._type_to_factories[component_type].append(factory_name)
                
                # Initialize factory
                success = await factory.initialize()
                if not success:
                    logger.error(f"❌ Failed to initialize factory '{factory_name}'")
                    return False
                
                logger.info(f"🏭 Factory registered: {factory_name}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to register factory '{config.name}': {e}")
                return False
    
    async def unregister_factory(self, factory_name: str) -> bool:
        """Unregister a factory."""
        async with self._registry_lock:
            return await self._unregister_factory_internal(factory_name)
    
    async def _unregister_factory_internal(self, factory_name: str) -> bool:
        """Internal factory unregistration."""
        try:
            if factory_name not in self._factories:
                return True
            
            # Get factory instance
            factory = self._factories[factory_name]
            
            # Shutdown factory if running
            if factory.status != FactoryStatus.SHUTDOWN:
                await factory.shutdown()
            
            # Remove from registry
            del self._factories[factory_name]
            del self._factory_configs[factory_name]
            if factory_name in self._factory_metadata:
                del self._factory_metadata[factory_name]
            
            # Update type mapping
            component_type = factory.get_component_type()
            if component_type in self._type_to_factories:
                if factory_name in self._type_to_factories[component_type]:
                    self._type_to_factories[component_type].remove(factory_name)
                if not self._type_to_factories[component_type]:
                    del self._type_to_factories[component_type]
            
            logger.info(f"🗑️ Factory unregistered: {factory_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to unregister factory '{factory_name}': {e}")
            return False
    
    def get_factory(self, factory_name: str) -> Optional[BlatamFactory]:
        """Get a factory by name."""
        return self._factories.get(factory_name)
    
    def get_factories_by_type(self, component_type: ComponentType) -> List[BlatamFactory]:
        """Get factories that create a specific component type."""
        factory_names = self._type_to_factories.get(component_type, [])
        return [self._factories[name] for name in factory_names if name in self._factories]
    
    def get_factory_config(self, factory_name: str) -> Optional[FactoryConfig]:
        """Get factory configuration."""
        return self._factory_configs.get(factory_name)
    
    def list_factories(self, factory_type: Optional[FactoryType] = None) -> List[str]:
        """List factories, optionally filtered by type."""
        if factory_type is None:
            return list(self._factories.keys())
        
        return [
            name for name, factory in self._factories.items()
            if factory.config.factory_type == factory_type
        ]
    
    async def create_component(self, component_type: ComponentType, config: ComponentConfig, **kwargs) -> Optional[BlatamComponent]:
        """Create a component using an appropriate factory."""
        factories = self.get_factories_by_type(component_type)
        if not factories:
            logger.error(f"❌ No factory found for component type: {component_type.value}")
            return None
        
        # Use the first available factory
        factory = factories[0]
        try:
            return await factory.create_component_with_monitoring(config, **kwargs)
        except Exception as e:
            logger.error(f"❌ Failed to create component using factory '{factory.config.name}': {e}")
            return None

# =============================================================================
# 🎯 FACTORY MANAGER
# =============================================================================

class FactoryManager:
    """Manages factory lifecycle and operations."""
    
    def __init__(self, registry: FactoryRegistry):
        self.registry = registry
        self._initialization_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()
    
    async def initialize_all_factories(self) -> bool:
        """Initialize all registered factories."""
        async with self._initialization_lock:
            try:
                factories = self.registry.list_factories()
                logger.info(f"🚀 Initializing {len(factories)} factories...")
                
                for factory_name in factories:
                    factory = self.registry.get_factory(factory_name)
                    if factory and factory.config.enabled:
                        if factory.status == FactoryStatus.UNINITIALIZED:
                            success = await factory.initialize()
                            if not success:
                                logger.error(f"❌ Failed to initialize factory: {factory_name}")
                                return False
                
                logger.info("✅ All factories initialized successfully")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize factories: {e}")
                return False
    
    async def shutdown_all_factories(self) -> bool:
        """Shutdown all registered factories."""
        async with self._shutdown_lock:
            try:
                factories = self.registry.list_factories()
                logger.info(f"🔄 Shutting down {len(factories)} factories...")
                
                for factory_name in factories:
                    factory = self.registry.get_factory(factory_name)
                    if factory:
                        success = await factory.shutdown()
                        if not success:
                            logger.warning(f"⚠️ Failed to shutdown factory: {factory_name}")
                
                logger.info("✅ All factories shutdown")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to shutdown factories: {e}")
                return False
    
    def get_factory_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all factories."""
        stats = {}
        
        for factory_name in self.registry.list_factories():
            factory = self.registry.get_factory(factory_name)
            if factory:
                stats[factory_name] = factory.get_stats()
        
        return stats
    
    def get_factory_status(self) -> Dict[str, str]:
        """Get status of all factories."""
        return {
            name: factory.status.value
            for name, factory in self.registry._factories.items()
        }

# =============================================================================
# 🚀 FACTORY FUNCTIONS
# =============================================================================

def create_factory_registry() -> FactoryRegistry:
    """Create a new factory registry."""
    return FactoryRegistry()

def create_factory_manager(registry: Optional[FactoryRegistry] = None) -> FactoryManager:
    """Create a new factory manager."""
    if registry is None:
        registry = create_factory_registry()
    return FactoryManager(registry)

def create_default_factory_configs() -> Dict[str, FactoryConfig]:
    """Create default factory configurations."""
    return {
        'component': FactoryConfig(
            name='component',
            factory_type=FactoryType.COMPONENT,
            max_instances=100,
            enable_caching=True
        ),
        'engine': FactoryConfig(
            name='engine',
            factory_type=FactoryType.ENGINE,
            max_instances=50,
            enable_caching=True
        ),
        'service': FactoryConfig(
            name='service',
            factory_type=FactoryType.SERVICE,
            max_instances=200,
            enable_caching=False
        )
    }

# =============================================================================
# 🌟 EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "FactoryType",
    "FactoryStatus",
    
    # Configuration
    "FactoryConfig",
    
    # Interfaces
    "BlatamFactory",
    
    # Registry and Management
    "FactoryRegistry",
    "FactoryManager",
    
    # Factory functions
    "create_factory_registry",
    "create_factory_manager",
    "create_default_factory_configs"
] 