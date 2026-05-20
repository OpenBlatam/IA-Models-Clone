"""
Module Integration
Integration helpers for modules
"""

from typing import Dict, Any, Optional, List, Callable
from fastapi import FastAPI, APIRouter

from .module_manager import ModuleManager
from .config import ModulesConfig, DEFAULT_MODULES_CONFIG


def setup_modules(
    app: FastAPI,
    modules_path: Optional[str] = None,
    config: Optional[ModulesConfig] = None
) -> ModuleManager:
    """Setup and register all modules in FastAPI app"""
    from pathlib import Path
    
    # Create module manager
    manager = ModuleManager(config=config or ModulesConfig.from_dict(DEFAULT_MODULES_CONFIG))
    
    # Discover modules
    if modules_path:
        modules_path_obj = Path(modules_path)
    else:
        # Default to modules directory
        modules_path_obj = Path(__file__).parent
    
    # Discover and register
    manager.discover_and_register_modules(modules_path_obj)
    
    # Initialize all modules
    manager.initialize_all_modules()
    
    # Register routers
    manager.register_routers(app)
    
    return manager


def create_module_endpoint(
    module_name: str,
    endpoint_name: str,
    method: str = "GET",
    path: str = None
) -> Callable:
    """Create endpoint for module dynamically"""
    def decorator(func: Callable):
        # This would create a FastAPI endpoint
        # Implementation depends on specific needs
        return func
    return decorator


class ModuleAPI:
    """API wrapper for modules"""
    
    def __init__(self, manager: ModuleManager):
        self.manager = manager
    
    def get_router(self, module_name: str) -> Optional[APIRouter]:
        """Get router for module"""
        return self.manager.router.create_router(module_name)
    
    def execute_use_case(
        self,
        module_name: str,
        use_case_name: str,
        *args,
        **kwargs
    ):
        """Execute use case from module"""
        module = self.manager.get_module(module_name)
        if not module:
            return None
        
        use_cases = module.get("use_cases", {})
        use_case = use_cases.get(use_case_name)
        
        if not use_case:
            return None
        
        # This would execute the use case
        # Would need to be async if use case is async
        return use_case
    
    def call_controller(
        self,
        module_name: str,
        method_name: str,
        *args,
        **kwargs
    ):
        """Call controller method"""
        controller = self.manager.get_module_controller(module_name)
        if not controller:
            return None
        
        method = getattr(controller, method_name, None)
        if not method:
            return None
        
        return method(*args, **kwargs)






