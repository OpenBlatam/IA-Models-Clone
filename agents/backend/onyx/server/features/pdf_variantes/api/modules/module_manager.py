"""
Module Manager
Complete module management system
"""

from typing import Dict, Any, Optional, List
from pathlib import Path

from .module_registry import ModuleRegistry, get_registry
from .module_loader import ModuleLoader, ModuleInitializer
from .module_factory import ModuleFactory
from .module_router import ModuleRouter
from .config import ModulesConfig, DEFAULT_MODULES_CONFIG
from .shared.interfaces import IModule


class ModuleManager:
    """Complete module management system"""
    
    def __init__(
        self,
        registry: Optional[ModuleRegistry] = None,
        config: Optional[ModulesConfig] = None
    ):
        self.registry = registry or get_registry()
        self.config = config or ModulesConfig.from_dict(DEFAULT_MODULES_CONFIG)
        self.loader = ModuleLoader(self.registry)
        self.initializer = ModuleInitializer(self.loader)
        self.factory = ModuleFactory(self.registry)
        self.router = ModuleRouter(self.registry)
        self.initialized_modules: Dict[str, Any] = {}
    
    def discover_and_register_modules(self, modules_path: Path) -> List[str]:
        """Discover and register all modules automatically"""
        modules = self.loader.discover_modules(modules_path)
        
        for module_name in modules:
            if self.config.is_module_enabled(module_name):
                self.register_module(module_name)
        
        return modules
    
    def register_module(self, module_name: str) -> bool:
        """Register a single module"""
        module_path = f"api.modules.{module_name}"
        
        components = self.loader.load_module_from_path(module_name, module_path)
        
        if components:
            self.registry.register_module(module_name, {
                "path": module_path,
                "components": components
            })
            return True
        
        return False
    
    def initialize_all_modules(
        self,
        dependencies: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Initialize all registered modules"""
        modules = self.registry.list_modules()
        dependencies = dependencies or {}
        
        for module_name in modules:
            if not self.config.is_module_enabled(module_name):
                continue
            
            # Check dependencies
            module_deps = self.config.get_module_dependencies(module_name)
            if not self._check_dependencies(module_deps):
                print(f"Module {module_name} dependencies not satisfied")
                continue
            
            # Initialize module
            module_instance = self.initializer.initialize_module(
                module_name,
                dependencies.get(module_name, {})
            )
            
            if module_instance:
                self.initialized_modules[module_name] = module_instance
        
        return self.initialized_modules
    
    def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if all dependencies are available"""
        for dep in dependencies:
            if dep not in self.initialized_modules:
                return False
        return True
    
    def get_module(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get initialized module"""
        return self.initialized_modules.get(module_name)
    
    def get_module_controller(self, module_name: str) -> Optional[Any]:
        """Get module controller"""
        module = self.get_module(module_name)
        return module.get("controller") if module else None
    
    def create_routers(self) -> List[Any]:
        """Create FastAPI routers for all modules"""
        routers = []
        modules = self.registry.list_modules()
        
        for module_name in modules:
            if not self.config.is_module_enabled(module_name):
                continue
            
            router = self.router.create_router(module_name)
            if router:
                routers.append(router)
        
        return routers
    
    def register_routers(self, app) -> None:
        """Register all module routers to FastAPI app"""
        routers = self.create_routers()
        
        for router in routers:
            app.include_router(router)
    
    def get_module_info(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Get complete module information"""
        module = self.get_module(module_name)
        config = self.config.get_module_config(module_name)
        
        if not module or not config:
            return None
        
        return {
            "name": module_name,
            "enabled": config.enabled,
            "version": config.version,
            "dependencies": config.dependencies,
            "settings": config.settings,
            "components": {
                "repository": module.get("repository") is not None,
                "use_cases": list(module.get("use_cases", {}).keys()),
                "controller": module.get("controller") is not None
            }
        }
    
    def list_all_modules(self) -> List[Dict[str, Any]]:
        """List all modules with their info"""
        modules = []
        for module_name in self.registry.list_modules():
            info = self.get_module_info(module_name)
            if info:
                modules.append(info)
        return modules






