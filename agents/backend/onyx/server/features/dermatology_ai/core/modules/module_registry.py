"""
Module Registry
Manages module registration, dependencies, and lifecycle
"""

from typing import Dict, List, Optional, Set
from collections import defaultdict
import logging

from .module import Module, ModuleMetadata, ModuleState

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """
    Registry for managing modules.
    Handles dependency resolution, loading order, and lifecycle.
    """
    
    def __init__(self):
        self.modules: Dict[str, Module] = {}
        self.metadata: Dict[str, ModuleMetadata] = {}
        self.service_providers: Dict[str, List[str]] = defaultdict(list)  # service -> [modules]
        self.load_order: List[str] = []
    
    def register(self, module: Module):
        """
        Register module
        
        Args:
            module: Module instance
        """
        name = module.metadata.name
        
        if name in self.modules:
            logger.warning(f"Module {name} already registered, replacing")
        
        self.modules[name] = module
        self.metadata[name] = module.metadata
        
        # Track service providers
        for service in module.metadata.provides:
            self.service_providers[service].append(name)
        
        logger.info(f"Registered module: {name} v{module.metadata.version}")
    
    def get_module(self, name: str) -> Optional[Module]:
        """Get module by name"""
        return self.modules.get(name)
    
    def get_modules_by_tag(self, tag: str) -> List[Module]:
        """Get modules by tag"""
        return [
            module for module in self.modules.values()
            if tag in module.metadata.tags
        ]
    
    def get_modules_providing_service(self, service_name: str) -> List[Module]:
        """Get modules that provide a service"""
        module_names = self.service_providers.get(service_name, [])
        return [self.modules[name] for name in module_names if name in self.modules]
    
    def resolve_dependencies(self, module_name: str) -> List[str]:
        """
        Resolve module dependencies (topological sort)
        
        Args:
            module_name: Name of module
            
        Returns:
            List of module names in dependency order
        """
        if module_name not in self.modules:
            raise ValueError(f"Module {module_name} not found")
        
        visited: Set[str] = set()
        temp_visited: Set[str] = set()
        result: List[str] = []
        
        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return
            
            temp_visited.add(name)
            
            if name in self.modules:
                module = self.modules[name]
                for dep in module.metadata.dependencies:
                    visit(dep)
            
            temp_visited.remove(name)
            visited.add(name)
            result.append(name)
        
        visit(module_name)
        return result
    
    def get_load_order(self, module_names: Optional[List[str]] = None) -> List[str]:
        """
        Get load order for modules (topological sort)
        
        Args:
            module_names: List of module names (None = all modules)
            
        Returns:
            List of module names in load order
        """
        if module_names is None:
            module_names = list(self.modules.keys())
        
        visited: Set[str] = set()
        temp_visited: Set[str] = set()
        result: List[str] = []
        
        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {name}")
            if name in visited:
                return
            
            if name not in self.modules:
                return
            
            temp_visited.add(name)
            
            module = self.modules[name]
            for dep in module.metadata.dependencies:
                if dep in module_names:
                    visit(dep)
            
            temp_visited.remove(name)
            visited.add(name)
            result.append(name)
        
        for name in module_names:
            if name not in visited:
                visit(name)
        
        return result
    
    async def load_module(self, module_name: str) -> bool:
        """Load a module"""
        if module_name not in self.modules:
            logger.error(f"Module {module_name} not found")
            return False
        
        module = self.modules[module_name]
        
        # Resolve and load dependencies first
        deps = self.resolve_dependencies(module_name)
        for dep_name in deps[:-1]:  # All except the module itself
            if dep_name != module_name:
                dep_module = self.modules[dep_name]
                if dep_module.state == ModuleState.UNLOADED:
                    await self.load_module(dep_name)
        
        # Load dependencies into module
        for dep_name in module.metadata.dependencies:
            if dep_name in self.modules:
                dep_module = self.modules[dep_name]
                module.set_dependency(dep_name, dep_module)
        
        # Load module
        try:
            module.state = ModuleState.LOADING
            success = await module.load()
            if success:
                module.state = ModuleState.LOADED
                logger.info(f"Module {module_name} loaded")
            else:
                module.state = ModuleState.ERROR
                logger.error(f"Module {module_name} failed to load")
            return success
        except Exception as e:
            module.state = ModuleState.ERROR
            module.error = str(e)
            logger.error(f"Module {module_name} load error: {e}", exc_info=True)
            return False
    
    async def initialize_module(self, module_name: str, config: Optional[Dict] = None) -> bool:
        """Initialize a module"""
        if module_name not in self.modules:
            return False
        
        module = self.modules[module_name]
        
        # Initialize dependencies first
        for dep_name in module.metadata.dependencies:
            if dep_name in self.modules:
                dep_module = self.modules[dep_name]
                if dep_module.state == ModuleState.LOADED:
                    await self.initialize_module(dep_name)
        
        try:
            module.state = ModuleState.INITIALIZING
            success = await module.initialize(config)
            if success:
                module.state = ModuleState.INITIALIZED
                logger.info(f"Module {module_name} initialized")
            else:
                module.state = ModuleState.ERROR
            return success
        except Exception as e:
            module.state = ModuleState.ERROR
            module.error = str(e)
            logger.error(f"Module {module_name} initialization error: {e}", exc_info=True)
            return False
    
    async def start_module(self, module_name: str) -> bool:
        """Start a module"""
        if module_name not in self.modules:
            return False
        
        module = self.modules[module_name]
        
        # Start dependencies first
        for dep_name in module.metadata.dependencies:
            if dep_name in self.modules:
                dep_module = self.modules[dep_name]
                if dep_module.state == ModuleState.INITIALIZED:
                    await self.start_module(dep_name)
        
        try:
            module.state = ModuleState.STARTING
            success = await module.start()
            if success:
                module.state = ModuleState.RUNNING
                logger.info(f"Module {module_name} started")
            else:
                module.state = ModuleState.ERROR
            return success
        except Exception as e:
            module.state = ModuleState.ERROR
            module.error = str(e)
            logger.error(f"Module {module_name} start error: {e}", exc_info=True)
            return False
    
    async def stop_module(self, module_name: str) -> bool:
        """Stop a module (reverse dependency order)"""
        if module_name not in self.modules:
            return False
        
        module = self.modules[module_name]
        
        # Stop dependents first
        dependents = [
            name for name, mod in self.modules.items()
            if module_name in mod.metadata.dependencies
        ]
        
        for dep_name in dependents:
            await self.stop_module(dep_name)
        
        try:
            module.state = ModuleState.STOPPING
            success = await module.stop()
            if success:
                module.state = ModuleState.STOPPED
                logger.info(f"Module {module_name} stopped")
            return success
        except Exception as e:
            module.state = ModuleState.ERROR
            module.error = str(e)
            logger.error(f"Module {module_name} stop error: {e}", exc_info=True)
            return False
    
    async def unload_module(self, module_name: str) -> bool:
        """Unload a module"""
        if module_name not in self.modules:
            return False
        
        module = self.modules[module_name]
        
        try:
            success = await module.unload()
            if success:
                module.state = ModuleState.UNLOADED
                logger.info(f"Module {module_name} unloaded")
            return success
        except Exception as e:
            module.error = str(e)
            logger.error(f"Module {module_name} unload error: {e}", exc_info=True)
            return False
    
    def list_modules(self) -> List[Dict[str, Any]]:
        """List all modules with their state"""
        return [module.get_state() for module in self.modules.values()]
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Health check for all modules"""
        results = {}
        for name, module in self.modules.items():
            results[name] = await module.health_check()
        return results


# Global registry
_module_registry: Optional[ModuleRegistry] = None


def get_module_registry() -> ModuleRegistry:
    """Get or create global module registry"""
    global _module_registry
    if _module_registry is None:
        _module_registry = ModuleRegistry()
    return _module_registry















