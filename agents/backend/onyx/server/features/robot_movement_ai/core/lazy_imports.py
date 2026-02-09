"""
Lazy Import System
==================

Sistema de imports diferidos para mejorar tiempos de carga y organización.
"""

from typing import Any, Optional, Callable, Dict
import importlib
import sys
from functools import wraps


class LazyImport:
    """Wrapper para imports diferidos."""
    
    def __init__(self, module_path: str, attribute: Optional[str] = None):
        """
        Inicializar lazy import.
        
        Args:
            module_path: Ruta del módulo (ej: 'core.robot.movement_engine')
            attribute: Atributo específico a importar (opcional)
        """
        self.module_path = module_path
        self.attribute = attribute
        self._module: Optional[Any] = None
        self._value: Optional[Any] = None
    
    def __getattr__(self, name: str) -> Any:
        """Obtener atributo del módulo o valor."""
        if self._value is None:
            self._load()
        return getattr(self._value, name)
    
    def __call__(self, *args, **kwargs) -> Any:
        """Llamar al valor si es callable."""
        if self._value is None:
            self._load()
        return self._value(*args, **kwargs)
    
    def _load(self) -> None:
        """Cargar el módulo o atributo."""
        if self._module is None:
            self._module = importlib.import_module(self.module_path)
        
        if self.attribute:
            self._value = getattr(self._module, self.attribute)
        else:
            self._value = self._module


class LazyModule:
    """Módulo con imports diferidos."""
    
    def __init__(self, name: str):
        """Inicializar módulo lazy."""
        self._name = name
        self._module: Optional[Any] = None
    
    def __getattr__(self, name: str) -> Any:
        """Obtener atributo del módulo."""
        if self._module is None:
            self._module = importlib.import_module(self._name)
        return getattr(self._module, name)


def lazy_import(module_path: str, attribute: Optional[str] = None) -> Any:
    """
    Crear un lazy import.
    
    Args:
        module_path: Ruta del módulo
        attribute: Atributo específico (opcional)
    
    Returns:
        LazyImport wrapper
    """
    return LazyImport(module_path, attribute)


def lazy_module(name: str) -> LazyModule:
    """
    Crear un módulo lazy.
    
    Args:
        name: Nombre del módulo
    
    Returns:
        LazyModule wrapper
    """
    return LazyModule(name)


class ImportRegistry:
    """Registro centralizado de imports con fallback."""
    
    def __init__(self):
        """Inicializar registro."""
        self._imports: Dict[str, Callable[[], Any]] = {}
        self._cache: Dict[str, Any] = {}
        self._failed: set = set()
    
    def register(
        self,
        name: str,
        import_func: Callable[[], Any],
        fallback: Optional[Any] = None
    ) -> None:
        """
        Registrar función de import.
        
        Args:
            name: Nombre del import
            import_func: Función que realiza el import
            fallback: Valor por defecto si falla
        """
        def _import_with_fallback():
            if name in self._cache:
                return self._cache[name]
            
            if name in self._failed:
                return fallback
            
            try:
                value = import_func()
                self._cache[name] = value
                return value
            except (ImportError, AttributeError) as e:
                self._failed.add(name)
                if fallback is not None:
                    return fallback
                raise
        
        self._imports[name] = _import_with_fallback
    
    def get(self, name: str) -> Any:
        """
        Obtener import registrado.
        
        Args:
            name: Nombre del import
        
        Returns:
            Valor importado o fallback
        """
        if name not in self._imports:
            raise KeyError(f"Import '{name}' not registered")
        return self._imports[name]()
    
    def clear_cache(self) -> None:
        """Limpiar caché de imports."""
        self._cache.clear()
        self._failed.clear()


# Instancia global del registro
_import_registry = ImportRegistry()


def get_import_registry() -> ImportRegistry:
    """Obtener registro global de imports."""
    return _import_registry

