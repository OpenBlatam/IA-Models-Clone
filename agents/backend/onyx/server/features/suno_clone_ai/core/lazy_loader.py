"""
Lazy Loader
Carga perezosa agresiva de módulos y recursos
"""

import logging
import sys
import importlib
from typing import Any, Optional, Callable, Dict
from functools import wraps

logger = logging.getLogger(__name__)


class LazyLoader:
    """Cargador perezoso de módulos"""
    
    def __init__(self):
        self._loaded_modules: Dict[str, Any] = {}
        self._loaders: Dict[str, Callable] = {}
    
    def register_lazy_module(self, name: str, loader: Callable[[], Any]):
        """
        Registra un módulo para carga perezosa
        
        Args:
            name: Nombre del módulo
            loader: Función que carga el módulo
        """
        self._loaders[name] = loader
        logger.debug(f"Registered lazy module: {name}")
    
    def get_module(self, name: str) -> Any:
        """
        Obtiene un módulo (carga si es necesario)
        
        Args:
            name: Nombre del módulo
            
        Returns:
            Módulo cargado
        """
        if name in self._loaded_modules:
            return self._loaded_modules[name]
        
        if name in self._loaders:
            logger.debug(f"Lazy loading module: {name}")
            module = self._loaders[name]()
            self._loaded_modules[name] = module
            return module
        
        # Intentar import normal
        try:
            module = importlib.import_module(name)
            self._loaded_modules[name] = module
            return module
        except ImportError as e:
            logger.error(f"Failed to load module {name}: {e}")
            raise
    
    def lazy_import(self, module_name: str):
        """
        Decorator para lazy import
        
        Usage:
            @lazy_import("torch")
            def use_torch():
                import torch
                return torch.tensor([1, 2, 3])
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Cargar módulo solo cuando se usa
                if module_name not in self._loaded_modules:
                    try:
                        module = importlib.import_module(module_name)
                        self._loaded_modules[module_name] = module
                    except ImportError:
                        logger.warning(f"Could not lazy load {module_name}")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator


# Instancia global
_lazy_loader: Optional[LazyLoader] = None


def get_lazy_loader() -> LazyLoader:
    """Obtiene el cargador perezoso"""
    global _lazy_loader
    if _lazy_loader is None:
        _lazy_loader = LazyLoader()
    return _lazy_loader


def lazy_import(module_name: str):
    """Decorator para lazy import"""
    loader = get_lazy_loader()
    return loader.lazy_import(module_name)















