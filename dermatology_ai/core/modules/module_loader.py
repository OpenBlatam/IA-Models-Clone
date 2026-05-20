"""
Advanced Module Loader
Dynamic module discovery and loading
"""

import importlib
import inspect
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
import logging

from .module import Module, ModuleMetadata
from .module_registry import ModuleRegistry

logger = logging.getLogger(__name__)


class ModuleLoader:
    """
    Advanced module loader with discovery and dynamic loading.
    Supports loading modules from directories, packages, and registries.
    """
    
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self.loaded_modules: Dict[str, Module] = {}
    
    def discover_modules(self, directory: str) -> List[Type[Module]]:
        """
        Discover modules in directory
        
        Args:
            directory: Directory path to search
            
        Returns:
            List of module classes
        """
        modules = []
        path = Path(directory)
        
        if not path.exists():
            logger.warning(f"Directory {directory} does not exist")
            return modules
        
        # Find all Python files
        for file_path in path.rglob("*.py"):
            if file_path.name.startswith("_"):
                continue
            
            try:
                # Import module
                module_path = str(file_path.relative_to(path.parent).with_suffix("")).replace("/", ".")
                module = importlib.import_module(module_path)
                
                # Find Module subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, Module) and 
                        obj != Module and 
                        obj.__module__ == module_path):
                        modules.append(obj)
                        logger.debug(f"Discovered module class: {name}")
            
            except Exception as e:
                logger.warning(f"Failed to load module from {file_path}: {e}")
        
        return modules
    
    def load_from_directory(self, directory: str, config: Optional[Dict[str, Any]] = None):
        """
        Load all modules from directory
        
        Args:
            directory: Directory path
            config: Configuration for modules
        """
        module_classes = self.discover_modules(directory)
        
        for module_class in module_classes:
            try:
                # Create module instance
                # Module should have metadata as class attribute or constructor
                if hasattr(module_class, 'metadata'):
                    metadata = module_class.metadata
                else:
                    # Try to get from constructor
                    metadata = ModuleMetadata(
                        name=module_class.__name__.lower(),
                        version="1.0.0",
                        description=module_class.__doc__ or ""
                    )
                
                module = module_class(metadata)
                
                # Apply config if provided
                if config and metadata.name in config:
                    module.config = config[metadata.name]
                
                # Register module
                self.registry.register(module)
                self.loaded_modules[metadata.name] = module
                
                logger.info(f"Loaded module: {metadata.name}")
            
            except Exception as e:
                logger.error(f"Failed to load module {module_class}: {e}", exc_info=True)
    
    def load_from_package(self, package_name: str):
        """
        Load modules from Python package
        
        Args:
            package_name: Package name
        """
        try:
            package = importlib.import_module(package_name)
            package_path = Path(package.__file__).parent
            
            self.load_from_directory(str(package_path))
        
        except Exception as e:
            logger.error(f"Failed to load package {package_name}: {e}", exc_info=True)
    
    async def initialize_all(self, config: Optional[Dict[str, Any]] = None):
        """Initialize all registered modules in dependency order"""
        load_order = self.registry.get_load_order()
        
        logger.info(f"Initializing {len(load_order)} modules in order: {load_order}")
        
        for module_name in load_order:
            module_config = config.get(module_name) if config else None
            await self.registry.initialize_module(module_name, module_config)
    
    async def start_all(self):
        """Start all modules in dependency order"""
        load_order = self.registry.get_load_order()
        
        logger.info(f"Starting {len(load_order)} modules")
        
        for module_name in load_order:
            await self.registry.start_module(module_name)
    
    async def stop_all(self):
        """Stop all modules in reverse dependency order"""
        load_order = self.registry.get_load_order()
        reverse_order = list(reversed(load_order))
        
        logger.info(f"Stopping {len(reverse_order)} modules")
        
        for module_name in reverse_order:
            await self.registry.stop_module(module_name)
    
    def get_module(self, name: str) -> Optional[Module]:
        """Get loaded module"""
        return self.loaded_modules.get(name) or self.registry.get_module(name)


# Global loader
_module_loader: Optional[ModuleLoader] = None


def get_module_loader(registry: Optional[ModuleRegistry] = None) -> ModuleLoader:
    """Get or create global module loader"""
    global _module_loader
    if _module_loader is None:
        if registry is None:
            from .module_registry import get_module_registry
            registry = get_module_registry()
        _module_loader = ModuleLoader(registry)
    return _module_loader















