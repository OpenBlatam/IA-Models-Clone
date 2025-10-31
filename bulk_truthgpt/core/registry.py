"""
Component Registry System
========================

Ultra-modular component registry with advanced patterns.
"""

import logging
import threading
from typing import Dict, Any, Optional, Type, List, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import inspect
import weakref

logger = logging.getLogger(__name__)

class ComponentType(str, Enum):
    """Component types."""
    SERVICE = "service"
    ENGINE = "engine"
    MANAGER = "manager"
    MIDDLEWARE = "middleware"
    VALIDATOR = "validator"
    SERIALIZER = "serializer"
    CACHE = "cache"
    LOGGER = "logger"
    METRIC = "metric"
    SECURITY = "security"
    DATABASE = "database"
    QUEUE = "queue"
    SCHEDULER = "scheduler"
    MONITOR = "monitor"
    CONFIG = "config"
    PLUGIN = "plugin"
    API = "api"
    TEST = "test"
    DEPLOYMENT = "deployment"

@dataclass
class ComponentInfo:
    """Component information."""
    name: str
    component_type: ComponentType
    instance: Any
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: __import__('time').time())
    initialized: bool = False
    healthy: bool = True

class BaseRegistry(ABC):
    """Base registry class."""
    
    def __init__(self):
        self._components: Dict[str, ComponentInfo] = {}
        self._lock = threading.RLock()
        self._dependencies: Dict[str, List[str]] = {}
        self._dependents: Dict[str, List[str]] = {}
    
    @abstractmethod
    def register(self, name: str, component: Any, **kwargs) -> None:
        """Register a component."""
        pass
    
    @abstractmethod
    def unregister(self, name: str) -> None:
        """Unregister a component."""
        pass
    
    @abstractmethod
    def get(self, name: str) -> Any:
        """Get a component."""
        pass
    
    @abstractmethod
    def list(self) -> List[str]:
        """List all components."""
        pass
    
    def _validate_component(self, component: Any) -> bool:
        """Validate component."""
        try:
            # Check if component has required methods
            required_methods = ['initialize', 'cleanup']
            for method in required_methods:
                if not hasattr(component, method):
                    logger.warning(f"Component missing method: {method}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Component validation error: {str(e)}")
            return False
    
    def _resolve_dependencies(self, name: str) -> List[str]:
        """Resolve component dependencies."""
        try:
            if name not in self._dependencies:
                return []
            
            resolved = []
            to_resolve = self._dependencies[name].copy()
            
            while to_resolve:
                dep = to_resolve.pop(0)
                if dep in resolved:
                    continue
                
                if dep not in self._components:
                    logger.error(f"Dependency not found: {dep}")
                    return []
                
                resolved.append(dep)
                
                # Add dependencies of this dependency
                if dep in self._dependencies:
                    to_resolve.extend(self._dependencies[dep])
            
            return resolved
        except Exception as e:
            logger.error(f"Dependency resolution error: {str(e)}")
            return []
    
    def _check_health(self, name: str) -> bool:
        """Check component health."""
        try:
            if name not in self._components:
                return False
            
            component = self._components[name].instance
            if hasattr(component, 'health_check'):
                return component.health_check()
            
            return True
        except Exception as e:
            logger.error(f"Health check error for {name}: {str(e)}")
            return False

class ComponentRegistry(BaseRegistry):
    """Component registry."""
    
    def __init__(self):
        super().__init__()
        self._component_types: Dict[str, ComponentType] = {}
    
    def register(self, name: str, component: Any, component_type: ComponentType = ComponentType.SERVICE, 
                 dependencies: List[str] = None, metadata: Dict[str, Any] = None) -> None:
        """Register a component."""
        try:
            with self._lock:
                if name in self._components:
                    logger.warning(f"Component {name} already registered")
                    return
                
                if not self._validate_component(component):
                    logger.error(f"Component {name} validation failed")
                    return
                
                # Store dependencies
                if dependencies:
                    self._dependencies[name] = dependencies
                    
                    # Update dependents
                    for dep in dependencies:
                        if dep not in self._dependents:
                            self._dependents[dep] = []
                        self._dependents[dep].append(name)
                
                # Create component info
                info = ComponentInfo(
                    name=name,
                    component_type=component_type,
                    instance=component,
                    dependencies=dependencies or [],
                    metadata=metadata or {}
                )
                
                self._components[name] = info
                self._component_types[name] = component_type
                
                logger.info(f"Component {name} registered successfully")
                
        except Exception as e:
            logger.error(f"Failed to register component {name}: {str(e)}")
            raise
    
    def unregister(self, name: str) -> None:
        """Unregister a component."""
        try:
            with self._lock:
                if name not in self._components:
                    logger.warning(f"Component {name} not found")
                    return
                
                # Check if other components depend on this one
                if name in self._dependents and self._dependents[name]:
                    logger.error(f"Cannot unregister {name}: has dependents")
                    return
                
                # Cleanup component
                component = self._components[name].instance
                if hasattr(component, 'cleanup'):
                    component.cleanup()
                
                # Remove from registries
                del self._components[name]
                if name in self._component_types:
                    del self._component_types[name]
                
                # Clean up dependencies
                if name in self._dependencies:
                    del self._dependencies[name]
                
                if name in self._dependents:
                    del self._dependents[name]
                
                logger.info(f"Component {name} unregistered successfully")
                
        except Exception as e:
            logger.error(f"Failed to unregister component {name}: {str(e)}")
            raise
    
    def get(self, name: str) -> Any:
        """Get a component."""
        try:
            with self._lock:
                if name not in self._components:
                    logger.error(f"Component {name} not found")
                    return None
                
                return self._components[name].instance
                
        except Exception as e:
            logger.error(f"Failed to get component {name}: {str(e)}")
            return None
    
    def list(self) -> List[str]:
        """List all components."""
        try:
            with self._lock:
                return list(self._components.keys())
        except Exception as e:
            logger.error(f"Failed to list components: {str(e)}")
            return []
    
    def get_by_type(self, component_type: ComponentType) -> List[str]:
        """Get components by type."""
        try:
            with self._lock:
                return [name for name, info in self._components.items() 
                       if info.component_type == component_type]
        except Exception as e:
            logger.error(f"Failed to get components by type: {str(e)}")
            return []
    
    def initialize_all(self) -> None:
        """Initialize all components."""
        try:
            with self._lock:
                # Resolve initialization order
                init_order = self._resolve_initialization_order()
                
                for name in init_order:
                    if name in self._components:
                        component = self._components[name].instance
                        if hasattr(component, 'initialize'):
                            component.initialize()
                            self._components[name].initialized = True
                            logger.info(f"Component {name} initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def cleanup_all(self) -> None:
        """Cleanup all components."""
        try:
            with self._lock:
                # Reverse initialization order
                init_order = self._resolve_initialization_order()
                init_order.reverse()
                
                for name in init_order:
                    if name in self._components:
                        component = self._components[name].instance
                        if hasattr(component, 'cleanup'):
                            component.cleanup()
                            self._components[name].initialized = False
                            logger.info(f"Component {name} cleaned up")
                
        except Exception as e:
            logger.error(f"Failed to cleanup components: {str(e)}")
            raise
    
    def health_check_all(self) -> Dict[str, bool]:
        """Check health of all components."""
        try:
            with self._lock:
                health_status = {}
                for name in self._components:
                    health_status[name] = self._check_health(name)
                    self._components[name].healthy = health_status[name]
                
                return health_status
        except Exception as e:
            logger.error(f"Failed to check component health: {str(e)}")
            return {}
    
    def _resolve_initialization_order(self) -> List[str]:
        """Resolve component initialization order."""
        try:
            # Topological sort
            visited = set()
            temp_visited = set()
            result = []
            
            def visit(name):
                if name in temp_visited:
                    raise ValueError(f"Circular dependency detected: {name}")
                if name in visited:
                    return
                
                temp_visited.add(name)
                
                # Visit dependencies first
                if name in self._dependencies:
                    for dep in self._dependencies[name]:
                        visit(dep)
                
                temp_visited.remove(name)
                visited.add(name)
                result.append(name)
            
            for name in self._components:
                if name not in visited:
                    visit(name)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to resolve initialization order: {str(e)}")
            return list(self._components.keys())

class ServiceRegistry(ComponentRegistry):
    """Service registry."""
    
    def __init__(self):
        super().__init__()
        self._service_interfaces: Dict[str, Type] = {}
    
    def register_service(self, name: str, service: Any, interface: Type = None, 
                        dependencies: List[str] = None, metadata: Dict[str, Any] = None) -> None:
        """Register a service."""
        try:
            self.register(name, service, ComponentType.SERVICE, dependencies, metadata)
            
            if interface:
                self._service_interfaces[name] = interface
                
        except Exception as e:
            logger.error(f"Failed to register service {name}: {str(e)}")
            raise
    
    def get_service(self, name: str) -> Any:
        """Get a service."""
        return self.get(name)
    
    def get_services_by_interface(self, interface: Type) -> List[Any]:
        """Get services by interface."""
        try:
            services = []
            for name, service_interface in self._service_interfaces.items():
                if issubclass(service_interface, interface):
                    services.append(self.get(name))
            return services
        except Exception as e:
            logger.error(f"Failed to get services by interface: {str(e)}")
            return []

class EngineRegistry(ComponentRegistry):
    """Engine registry."""
    
    def __init__(self):
        super().__init__()
        self._engine_types: Dict[str, str] = {}
    
    def register_engine(self, name: str, engine: Any, engine_type: str = None,
                       dependencies: List[str] = None, metadata: Dict[str, Any] = None) -> None:
        """Register an engine."""
        try:
            self.register(name, engine, ComponentType.ENGINE, dependencies, metadata)
            
            if engine_type:
                self._engine_types[name] = engine_type
                
        except Exception as e:
            logger.error(f"Failed to register engine {name}: {str(e)}")
            raise
    
    def get_engine(self, name: str) -> Any:
        """Get an engine."""
        return self.get(name)
    
    def get_engines_by_type(self, engine_type: str) -> List[Any]:
        """Get engines by type."""
        try:
            engines = []
            for name, et in self._engine_types.items():
                if et == engine_type:
                    engines.append(self.get(name))
            return engines
        except Exception as e:
            logger.error(f"Failed to get engines by type: {str(e)}")
            return []

# Global registries
component_registry = ComponentRegistry()
service_registry = ServiceRegistry()
engine_registry = EngineRegistry()

# Registry decorators
def register_component(name: str, component_type: ComponentType = ComponentType.SERVICE, 
                      dependencies: List[str] = None, metadata: Dict[str, Any] = None):
    """Decorator to register a component."""
    def decorator(cls):
        component_registry.register(name, cls(), component_type, dependencies, metadata)
        return cls
    return decorator

def register_service(name: str, interface: Type = None, dependencies: List[str] = None, 
                    metadata: Dict[str, Any] = None):
    """Decorator to register a service."""
    def decorator(cls):
        service_registry.register_service(name, cls(), interface, dependencies, metadata)
        return cls
    return decorator

def register_engine(name: str, engine_type: str = None, dependencies: List[str] = None, 
                   metadata: Dict[str, Any] = None):
    """Decorator to register an engine."""
    def decorator(cls):
        engine_registry.register_engine(name, cls(), engine_type, dependencies, metadata)
        return cls
    return decorator









