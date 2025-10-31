"""
Module Registry
Central registry for managing and loading modules
"""

from typing import Dict, Any, Type, Optional, List
import importlib
import inspect
from abc import ABC


class ModuleRegistry:
    """Registry for managing modules"""
    
    def __init__(self):
        self._modules: Dict[str, Dict[str, Any]] = {}
        self._controllers: Dict[str, Any] = {}
        self._use_cases: Dict[str, Any] = {}
        self._repositories: Dict[str, Any] = {}
    
    def register_module(
        self,
        name: str,
        module_config: Dict[str, Any]
    ) -> None:
        """Register a module"""
        self._modules[name] = {
            "config": module_config,
            "loaded": False,
            "controllers": {},
            "use_cases": {},
            "repositories": {}
        }
    
    def load_module(self, name: str) -> bool:
        """Load a module dynamically"""
        if name not in self._modules:
            return False
        
        module_info = self._modules[name]
        if module_info["loaded"]:
            return True
        
        try:
            # Import module
            module_path = module_info["config"]["path"]
            module = importlib.import_module(module_path)
            
            # Extract components
            if hasattr(module, "controllers"):
                module_info["controllers"] = module.controllers
            
            if hasattr(module, "use_cases"):
                module_info["use_cases"] = module.use_cases
            
            if hasattr(module, "repositories"):
                module_info["repositories"] = module.repositories
            
            module_info["loaded"] = True
            return True
        
        except Exception as e:
            print(f"Error loading module {name}: {e}")
            return False
    
    def get_controller(self, module_name: str, controller_name: str) -> Optional[Any]:
        """Get controller from module"""
        if not self._modules.get(module_name, {}).get("loaded"):
            self.load_module(module_name)
        
        module = self._modules.get(module_name, {})
        return module.get("controllers", {}).get(controller_name)
    
    def get_use_case(self, module_name: str, use_case_name: str) -> Optional[Any]:
        """Get use case from module"""
        if not self._modules.get(module_name, {}).get("loaded"):
            self.load_module(module_name)
        
        module = self._modules.get(module_name, {})
        return module.get("use_cases", {}).get(use_case_name)
    
    def list_modules(self) -> List[str]:
        """List all registered modules"""
        return list(self._modules.keys())
    
    def get_module_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get module information"""
        return self._modules.get(name)


# Global module registry
_module_registry = ModuleRegistry()


def get_registry() -> ModuleRegistry:
    """Get global module registry"""
    return _module_registry


def register_module(name: str, config: Dict[str, Any]) -> None:
    """Register module in global registry"""
    _module_registry.register_module(name, config)


def load_module(name: str) -> bool:
    """Load module from global registry"""
    return _module_registry.load_module(name)






