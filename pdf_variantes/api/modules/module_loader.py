"""
Module Loader
Dynamic module loading and initialization
"""

import importlib
import inspect
from typing import Dict, Any, Optional, List, Type
from pathlib import Path

from .module_registry import ModuleRegistry, get_registry
from .shared.interfaces import IModule


class ModuleLoader:
    """Load and initialize modules dynamically"""
    
    def __init__(self, registry: Optional[ModuleRegistry] = None):
        self.registry = registry or get_registry()
        self.loaded_modules: Dict[str, Any] = {}
    
    def discover_modules(self, modules_path: Path) -> List[str]:
        """Discover available modules in path"""
        modules = []
        
        for item in modules_path.iterdir():
            if item.is_dir() and not item.name.startswith('_'):
                # Check if it's a valid module
                init_file = item / "__init__.py"
                if init_file.exists():
                    modules.append(item.name)
        
        return modules
    
    def load_module_from_path(
        self,
        module_name: str,
        module_path: str
    ) -> Optional[Dict[str, Any]]:
        """Load module from file path"""
        try:
            module = importlib.import_module(module_path)
            
            # Extract module components
            components = {
                "module": module,
                "controllers": {},
                "use_cases": {},
                "repositories": {},
                "entities": {},
                "config": {}
            }
            
            # Find controllers
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.endswith("Controller"):
                    components["controllers"][name] = obj
            
            # Find use cases
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and ("UseCase" in name or "Query" in name or "Command" in name):
                    components["use_cases"][name] = obj
            
            # Find repositories
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.endswith("Repository"):
                    components["repositories"][name] = obj
            
            # Find entities
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and name.endswith("Entity"):
                    components["entities"][name] = obj
            
            self.loaded_modules[module_name] = components
            return components
        
        except Exception as e:
            print(f"Error loading module {module_name}: {e}")
            return None
    
    def auto_register_modules(self, base_path: Path) -> None:
        """Automatically discover and register all modules"""
        modules = self.discover_modules(base_path)
        
        for module_name in modules:
            module_path = f"api.modules.{module_name}"
            components = self.load_module_from_path(module_name, module_path)
            
            if components:
                self.registry.register_module(module_name, {
                    "path": module_path,
                    "components": components
                })
    
    def get_module_controller(self, module_name: str) -> Optional[Type]:
        """Get controller class from module"""
        if module_name not in self.loaded_modules:
            return None
        
        controllers = self.loaded_modules[module_name].get("controllers", {})
        
        # Return first controller or specific one
        if controllers:
            return list(controllers.values())[0]
        
        return None
    
    def get_module_use_cases(self, module_name: str) -> Dict[str, Type]:
        """Get use case classes from module"""
        if module_name not in self.loaded_modules:
            return {}
        
        return self.loaded_modules[module_name].get("use_cases", {})
    
    def get_module_repository(self, module_name: str) -> Optional[Type]:
        """Get repository class from module"""
        if module_name not in self.loaded_modules:
            return None
        
        repositories = self.loaded_modules[module_name].get("repositories", {})
        
        if repositories:
            return list(repositories.values())[0]
        
        return None


class ModuleInitializer:
    """Initialize modules with dependencies"""
    
    def __init__(self, loader: ModuleLoader, container: Any = None):
        self.loader = loader
        self.container = container
    
    def initialize_module(
        self,
        module_name: str,
        dependencies: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Initialize module with dependencies"""
        components = self.loader.loaded_modules.get(module_name)
        
        if not components:
            return None
        
        dependencies = dependencies or {}
        
        # Initialize repository
        repo_class = self.loader.get_module_repository(module_name)
        repository = None
        if repo_class:
            # Initialize with dependencies if available
            repository = repo_class(**dependencies.get("repository", {}))
        
        # Initialize use cases with repository
        use_cases = {}
        for use_case_name, use_case_class in components.get("use_cases", {}).items():
            if inspect.isabstract(use_case_class):
                continue
            
            # Create use case instance with dependencies
            use_cases[use_case_name] = use_case_class(
                repository=repository,
                **dependencies.get("use_case", {})
            )
        
        # Initialize controller with use cases
        controller_class = self.loader.get_module_controller(module_name)
        controller = None
        if controller_class:
            controller = controller_class(**use_cases, **dependencies.get("controller", {}))
        
        return {
            "repository": repository,
            "use_cases": use_cases,
            "controller": controller,
            "components": components
        }






