"""
Service Initializer for Color Grading AI
==========================================

Unified service initialization with dependency injection and lifecycle management.
"""

import logging
from typing import Dict, Any, Optional, Type, Callable, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class InitializationPhase(Enum):
    """Initialization phases."""
    PRE_INIT = "pre_init"
    INIT = "init"
    POST_INIT = "post_init"


@dataclass
class ServiceDependency:
    """Service dependency definition."""
    service_name: str
    required: bool = True
    phase: InitializationPhase = InitializationPhase.INIT


@dataclass
class ServiceDefinition:
    """Service definition."""
    name: str
    service_class: Type
    dependencies: List[ServiceDependency] = None
    init_params: Dict[str, Any] = None
    category: str = "support"
    singleton: bool = True


class ServiceInitializer:
    """
    Unified service initializer.
    
    Features:
    - Dependency injection
    - Lifecycle management
    - Initialization phases
    - Error handling
    - Circular dependency detection
    """
    
    def __init__(self):
        """Initialize service initializer."""
        self._definitions: Dict[str, ServiceDefinition] = {}
        self._instances: Dict[str, Any] = {}
        self._initialized: Dict[str, bool] = {}
        self._init_order: List[str] = []
    
    def register_service(
        self,
        name: str,
        service_class: Type,
        dependencies: Optional[List[ServiceDependency]] = None,
        init_params: Optional[Dict[str, Any]] = None,
        category: str = "support",
        singleton: bool = True
    ):
        """
        Register service definition.
        
        Args:
            name: Service name
            service_class: Service class
            dependencies: Service dependencies
            init_params: Initialization parameters
            category: Service category
            singleton: Whether service is singleton
        """
        definition = ServiceDefinition(
            name=name,
            service_class=service_class,
            dependencies=dependencies or [],
            init_params=init_params or {},
            category=category,
            singleton=singleton
        )
        
        self._definitions[name] = definition
        logger.info(f"Registered service: {name} (category: {category})")
    
    def initialize_service(self, name: str, services: Dict[str, Any]) -> Any:
        """
        Initialize single service with dependencies.
        
        Args:
            name: Service name
            services: Available services dictionary
            
        Returns:
            Service instance
        """
        if name in self._instances and self._definitions[name].singleton:
            return self._instances[name]
        
        definition = self._definitions.get(name)
        if not definition:
            raise ValueError(f"Service not registered: {name}")
        
        # Resolve dependencies
        resolved_deps = {}
        for dep in definition.dependencies:
            if dep.service_name not in services:
                if dep.required:
                    raise ValueError(f"Required dependency not available: {dep.service_name}")
                continue
            
            resolved_deps[dep.service_name] = services[dep.service_name]
        
        # Merge init params with resolved dependencies
        init_kwargs = {**definition.init_params, **resolved_deps}
        
        # Create instance
        try:
            instance = definition.service_class(**init_kwargs)
            
            # Store if singleton
            if definition.singleton:
                self._instances[name] = instance
            
            self._initialized[name] = True
            logger.info(f"Initialized service: {name}")
            
            return instance
        
        except Exception as e:
            logger.error(f"Failed to initialize service {name}: {e}")
            raise
    
    def initialize_all(self, services: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize all registered services in dependency order.
        
        Args:
            services: Available services dictionary
            
        Returns:
            Dictionary of initialized services
        """
        # Topological sort for dependency order
        initialized = set()
        self._init_order = []
        
        while len(initialized) < len(self._definitions):
            progress = False
            
            for name, definition in self._definitions.items():
                if name in initialized:
                    continue
                
                # Check if dependencies are initialized
                deps_ready = all(
                    dep.service_name in initialized or dep.service_name in services
                    for dep in definition.dependencies
                    if dep.required
                )
                
                if deps_ready:
                    try:
                        instance = self.initialize_service(name, services)
                        initialized.add(name)
                        self._init_order.append(name)
                        progress = True
                    except Exception as e:
                        logger.error(f"Error initializing {name}: {e}")
                        progress = False
            
            if not progress:
                # Circular dependency or missing dependency
                uninitialized = [n for n in self._definitions.keys() if n not in initialized]
                raise ValueError(f"Cannot resolve dependencies for: {uninitialized}")
        
        return {name: self._instances.get(name) for name in self._init_order}
    
    def get_initialization_order(self) -> List[str]:
        """Get service initialization order."""
        return self._init_order.copy()
    
    def get_service(self, name: str) -> Optional[Any]:
        """Get initialized service."""
        return self._instances.get(name)
    
    def is_initialized(self, name: str) -> bool:
        """Check if service is initialized."""
        return self._initialized.get(name, False)




