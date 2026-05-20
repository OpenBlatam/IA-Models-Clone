"""
Module System
=============

System for organizing and managing application modules.
"""

import logging
from typing import Dict, Any, Optional, List, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .lifecycle import LifecycleComponent
from .dependency_injection import DependencyContainer

logger = logging.getLogger(__name__)


@dataclass
class ModuleConfig:
    """Module configuration."""
    name: str
    enabled: bool = True
    priority: int = 0  # Lower priority loads first
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class Module(ABC, LifecycleComponent):
    """Base class for application modules."""
    
    def __init__(
        self,
        name: str,
        config: Optional[ModuleConfig] = None,
        container: Optional[DependencyContainer] = None
    ):
        """
        Initialize module.
        
        Args:
            name: Module name
            config: Module configuration
            container: Dependency injection container
        """
        super().__init__(name)
        self.config = config or ModuleConfig(name=name)
        self.container = container
        self._initialized = False
    
    @abstractmethod
    async def _do_initialize(self):
        """Module-specific initialization."""
        pass
    
    @abstractmethod
    async def _do_start(self):
        """Module-specific start logic."""
        pass
    
    @abstractmethod
    async def _do_stop(self):
        """Module-specific stop logic."""
        pass
    
    @abstractmethod
    async def _do_shutdown(self):
        """Module-specific shutdown logic."""
        pass
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get module configuration value.
        
        Args:
            key: Config key
            default: Default value
            
        Returns:
            Config value
        """
        return self.config.config.get(key, default)
    
    def register_service(
        self,
        service_type: Type,
        instance: Any,
        alias: Optional[str] = None
    ):
        """
        Register service in DI container.
        
        Args:
            service_type: Service type
            instance: Service instance
            alias: Optional alias
        """
        if self.container:
            self.container.register(service_type, instance=instance, alias=alias)


class ModuleRegistry:
    """Registry for application modules."""
    
    def __init__(self, container: Optional[DependencyContainer] = None):
        """
        Initialize module registry.
        
        Args:
            container: Dependency injection container
        """
        self.container = container or DependencyContainer()
        self.modules: Dict[str, Module] = {}
        self.module_configs: Dict[str, ModuleConfig] = {}
    
    def register(
        self,
        module: Module,
        config: Optional[ModuleConfig] = None
    ):
        """
        Register a module.
        
        Args:
            module: Module instance
            config: Optional module configuration
        """
        if config:
            module.config = config
        
        self.modules[module.config.name] = module
        self.module_configs[module.config.name] = module.config
        
        # Set container if not set
        if not module.container:
            module.container = self.container
        
        logger.info(f"Registered module: {module.config.name}")
    
    def get(self, name: str) -> Optional[Module]:
        """
        Get module by name.
        
        Args:
            name: Module name
            
        Returns:
            Module or None
        """
        return self.modules.get(name)
    
    def get_all(self) -> Dict[str, Module]:
        """Get all modules."""
        return self.modules.copy()
    
    def get_enabled_modules(self) -> List[Module]:
        """Get all enabled modules sorted by priority."""
        enabled = [
            module for module in self.modules.values()
            if module.config.enabled
        ]
        return sorted(enabled, key=lambda m: m.config.priority)
    
    async def initialize_all(self):
        """Initialize all enabled modules in dependency order."""
        modules = self._resolve_dependencies()
        
        for module in modules:
            if module.config.enabled:
                try:
                    await module.initialize()
                    logger.info(f"Initialized module: {module.config.name}")
                except Exception as e:
                    logger.error(f"Error initializing module {module.config.name}: {e}")
                    raise
    
    async def start_all(self):
        """Start all enabled modules."""
        modules = self.get_enabled_modules()
        
        for module in modules:
            try:
                await module.start()
                logger.info(f"Started module: {module.config.name}")
            except Exception as e:
                logger.error(f"Error starting module {module.config.name}: {e}")
                raise
    
    async def stop_all(self):
        """Stop all modules."""
        modules = reversed(self.get_enabled_modules())
        
        for module in modules:
            try:
                await module.stop()
                logger.info(f"Stopped module: {module.config.name}")
            except Exception as e:
                logger.error(f"Error stopping module {module.config.name}: {e}")
    
    async def shutdown_all(self):
        """Shutdown all modules."""
        modules = reversed(self.get_enabled_modules())
        
        for module in modules:
            try:
                await module.shutdown()
                logger.info(f"Shutdown module: {module.config.name}")
            except Exception as e:
                logger.error(f"Error shutting down module {module.config.name}: {e}")
    
    def _resolve_dependencies(self) -> List[Module]:
        """Resolve module dependencies and return sorted list."""
        enabled = self.get_enabled_modules()
        resolved = []
        remaining = enabled.copy()
        
        while remaining:
            progress = False
            
            for module in remaining[:]:
                # Check if all dependencies are satisfied
                deps_satisfied = all(
                    dep in [m.config.name for m in resolved] or
                    any(m.config.name == dep for m in enabled if not m.config.enabled)
                    for dep in module.config.dependencies
                )
                
                if deps_satisfied:
                    resolved.append(module)
                    remaining.remove(module)
                    progress = True
            
            if not progress:
                # Circular dependency or missing dependency
                unresolved = [m.config.name for m in remaining]
                logger.warning(f"Could not resolve dependencies for: {unresolved}")
                resolved.extend(remaining)
                break
        
        return resolved




