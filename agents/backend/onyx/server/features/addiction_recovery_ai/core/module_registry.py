"""
Module Registry System
Dynamic module loading and registration for microservices architecture
"""

import logging
import importlib
from typing import Dict, Any, Optional, Type, List, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class IModule(ABC):
    """Interface for modules"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Module name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Module version"""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize module"""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown module"""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Get module dependencies"""
        pass


class ModuleRegistry:
    """
    Registry for managing modules
    
    Features:
    - Dynamic module loading
    - Dependency resolution
    - Lifecycle management
    - Module discovery
    """
    
    def __init__(self):
        self._modules: Dict[str, IModule] = {}
        self._initialized: set = set()
        self._load_order: List[str] = []
    
    def register(self, module: IModule) -> None:
        """Register a module"""
        if module.name in self._modules:
            logger.warning(f"Module {module.name} already registered, overwriting")
        
        self._modules[module.name] = module
        logger.info(f"Module registered: {module.name} v{module.version}")
    
    def load_module(self, module_path: str) -> Optional[IModule]:
        """Dynamically load module from path"""
        try:
            module_parts = module_path.split(".")
            module_name = module_parts[-1]
            package_path = ".".join(module_parts[:-1])
            
            module = importlib.import_module(module_path)
            
            # Look for module class
            module_class = getattr(module, module_name, None)
            if module_class and issubclass(module_class, IModule):
                instance = module_class()
                self.register(instance)
                return instance
            
            logger.warning(f"Module class not found in {module_path}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load module {module_path}: {str(e)}")
            return None
    
    def discover_modules(self, package: str) -> List[IModule]:
        """Discover and load modules from package"""
        discovered = []
        
        try:
            import pkgutil
            import importlib
            
            package_module = importlib.import_module(package)
            package_path = package_module.__path__
            
            for importer, modname, ispkg in pkgutil.iter_modules(package_path):
                if ispkg:
                    module_path = f"{package}.{modname}"
                    module = self.load_module(module_path)
                    if module:
                        discovered.append(module)
            
            return discovered
            
        except Exception as e:
            logger.error(f"Failed to discover modules in {package}: {str(e)}")
            return []
    
    def initialize_all(self) -> None:
        """Initialize all modules in dependency order"""
        # Resolve dependencies
        self._resolve_dependencies()
        
        # Initialize in order
        for module_name in self._load_order:
            if module_name not in self._initialized:
                module = self._modules[module_name]
                try:
                    logger.info(f"Initializing module: {module_name}")
                    module.initialize()
                    self._initialized.add(module_name)
                except Exception as e:
                    logger.error(f"Failed to initialize module {module_name}: {str(e)}")
    
    def shutdown_all(self) -> None:
        """Shutdown all modules in reverse order"""
        for module_name in reversed(self._load_order):
            if module_name in self._initialized:
                module = self._modules[module_name]
                try:
                    logger.info(f"Shutting down module: {module_name}")
                    module.shutdown()
                    self._initialized.remove(module_name)
                except Exception as e:
                    logger.error(f"Failed to shutdown module {module_name}: {str(e)}")
    
    def _resolve_dependencies(self) -> None:
        """Resolve module dependencies and determine load order"""
        # Topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(module_name: str):
            if module_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {module_name}")
            
            if module_name in visited:
                return
            
            temp_visited.add(module_name)
            module = self._modules[module_name]
            
            # Visit dependencies first
            for dep in module.get_dependencies():
                if dep not in self._modules:
                    raise ValueError(f"Missing dependency: {dep} for module {module_name}")
                visit(dep)
            
            temp_visited.remove(module_name)
            visited.add(module_name)
            order.append(module_name)
        
        # Visit all modules
        for module_name in self._modules.keys():
            if module_name not in visited:
                visit(module_name)
        
        self._load_order = order
    
    def get_module(self, name: str) -> Optional[IModule]:
        """Get module by name"""
        return self._modules.get(name)
    
    def list_modules(self) -> List[str]:
        """List all registered modules"""
        return list(self._modules.keys())
    
    def is_initialized(self, name: str) -> bool:
        """Check if module is initialized"""
        return name in self._initialized


# Global registry instance
_registry: Optional[ModuleRegistry] = None


def get_registry() -> ModuleRegistry:
    """Get global module registry"""
    global _registry
    if _registry is None:
        _registry = ModuleRegistry()
    return _registry















