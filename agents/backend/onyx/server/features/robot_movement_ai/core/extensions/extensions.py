"""
Extension Utilities
===================

Utilidades de extensión y plugins para el sistema.
"""

from typing import Dict, List, Callable, Any, Optional, Type
from abc import ABC, abstractmethod
import importlib
import inspect
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Extension(ABC):
    """Clase base para extensiones del sistema."""
    
    def __init__(self, name: str):
        """
        Inicializar extensión.
        
        Args:
            name: Nombre de la extensión
        """
        self.name = name
        self.enabled = True
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Inicializar extensión.
        
        Returns:
            True si fue exitoso
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Cerrar extensión."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Obtener información de la extensión.
        
        Returns:
            Diccionario con información
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "class": self.__class__.__name__
        }


class ExtensionManager:
    """
    Gestor de extensiones del sistema.
    
    Permite registrar y gestionar extensiones dinámicamente.
    """
    
    def __init__(self):
        """Inicializar gestor de extensiones."""
        self.extensions: Dict[str, Extension] = {}
        self.hooks: Dict[str, List[Callable]] = {}
    
    def register(self, extension: Extension) -> bool:
        """
        Registrar extensión.
        
        Args:
            extension: Extensión a registrar
            
        Returns:
            True si fue exitoso
        """
        if extension.name in self.extensions:
            logger.warning(f"Extension {extension.name} already registered, overwriting")
        
        try:
            if extension.initialize():
                self.extensions[extension.name] = extension
                logger.info(f"Extension {extension.name} registered successfully")
                return True
            else:
                logger.error(f"Failed to initialize extension {extension.name}")
                return False
        except Exception as e:
            logger.error(f"Error registering extension {extension.name}: {e}")
            return False
    
    def unregister(self, name: str) -> bool:
        """
        Desregistrar extensión.
        
        Args:
            name: Nombre de la extensión
            
        Returns:
            True si fue exitoso
        """
        if name not in self.extensions:
            logger.warning(f"Extension {name} not found")
            return False
        
        extension = self.extensions[name]
        try:
            extension.shutdown()
            del self.extensions[name]
            logger.info(f"Extension {name} unregistered")
            return True
        except Exception as e:
            logger.error(f"Error unregistering extension {name}: {e}")
            return False
    
    def get_extension(self, name: str) -> Optional[Extension]:
        """
        Obtener extensión por nombre.
        
        Args:
            name: Nombre de la extensión
            
        Returns:
            Extensión o None
        """
        return self.extensions.get(name)
    
    def get_all_extensions(self) -> List[Extension]:
        """Obtener todas las extensiones."""
        return list(self.extensions.values())
    
    def register_hook(self, hook_name: str, callback: Callable):
        """
        Registrar hook (callback).
        
        Args:
            hook_name: Nombre del hook
            callback: Función callback
        """
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
        logger.debug(f"Registered hook {hook_name}")
    
    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Llamar a todos los callbacks de un hook.
        
        Args:
            hook_name: Nombre del hook
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Lista de resultados de callbacks
        """
        if hook_name not in self.hooks:
            return []
        
        results = []
        for callback in self.hooks[hook_name]:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in hook {hook_name} callback: {e}")
        
        return results
    
    def load_from_module(self, module_path: str) -> bool:
        """
        Cargar extensiones desde módulo.
        
        Args:
            module_path: Ruta al módulo
            
        Returns:
            True si fue exitoso
        """
        try:
            module = importlib.import_module(module_path)
            
            # Buscar clases que hereden de Extension
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Extension) and obj != Extension:
                    extension = obj(name)
                    self.register(extension)
            
            return True
        except Exception as e:
            logger.error(f"Error loading extensions from {module_path}: {e}")
            return False
    
    def shutdown_all(self):
        """Cerrar todas las extensiones."""
        for extension in list(self.extensions.values()):
            try:
                extension.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down extension {extension.name}: {e}")
        
        self.extensions.clear()
        self.hooks.clear()


# Instancia global
_extension_manager: Optional[ExtensionManager] = None


def get_extension_manager() -> ExtensionManager:
    """Obtener instancia global del gestor de extensiones."""
    global _extension_manager
    if _extension_manager is None:
        _extension_manager = ExtensionManager()
    return _extension_manager






