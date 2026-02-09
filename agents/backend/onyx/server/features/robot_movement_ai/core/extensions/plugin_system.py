"""
Plugin System
=============

Sistema de plugins para extender funcionalidad.
"""

from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
import importlib
import importlib.util
import inspect
from pathlib import Path
import logging

from .extensions import Extension, ExtensionManager

logger = logging.getLogger(__name__)


class Plugin(Extension):
    """
    Clase base para plugins.
    
    Los plugins son extensiones con funcionalidad específica.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Inicializar plugin.
        
        Args:
            name: Nombre del plugin
            version: Versión del plugin
        """
        super().__init__(name)
        self.version = version
        self.dependencies: List[str] = []
        self.config: Dict[str, Any] = {}
    
    @abstractmethod
    def initialize(self) -> bool:
        """Inicializar plugin."""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Ejecutar funcionalidad del plugin.
        
        Args:
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Resultado de la ejecución
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Obtener información del plugin."""
        info = super().get_info()
        info.update({
            "version": self.version,
            "dependencies": self.dependencies,
            "config": self.config
        })
        return info


class PluginManager:
    """
    Gestor de plugins del sistema.
    
    Permite cargar, registrar y gestionar plugins dinámicamente.
    """
    
    def __init__(self):
        """Inicializar gestor de plugins."""
        self.plugins: Dict[str, Plugin] = {}
        self.loaded_modules: Dict[str, Any] = {}
        self.extension_manager = ExtensionManager()
    
    def register_plugin(self, plugin: Plugin) -> bool:
        """
        Registrar plugin.
        
        Args:
            plugin: Plugin a registrar
            
        Returns:
            True si fue exitoso
        """
        # Verificar dependencias
        for dep in plugin.dependencies:
            if dep not in self.plugins:
                logger.warning(
                    f"Plugin {plugin.name} requires {dep} but it's not loaded"
                )
        
        # Registrar como extensión también
        if self.extension_manager.register(plugin):
            self.plugins[plugin.name] = plugin
            logger.info(f"Plugin {plugin.name} v{plugin.version} registered")
            return True
        return False
    
    def load_plugin_from_module(self, module_path: str) -> bool:
        """
        Cargar plugin desde módulo.
        
        Args:
            module_path: Ruta al módulo
            
        Returns:
            True si fue exitoso
        """
        try:
            module = importlib.import_module(module_path)
            self.loaded_modules[module_path] = module
            
            # Buscar clases que hereden de Plugin
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Plugin) and obj != Plugin:
                    plugin = obj()
                    self.register_plugin(plugin)
            
            return True
        except Exception as e:
            logger.error(f"Error loading plugin from {module_path}: {e}")
            return False
    
    def load_plugins_from_directory(self, directory: str) -> int:
        """
        Cargar plugins desde directorio.
        
        Args:
            directory: Directorio con plugins
            
        Returns:
            Número de plugins cargados
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning(f"Plugin directory {directory} does not exist")
            return 0
        
        loaded = 0
        for file_path in dir_path.glob("*.py"):
            if file_path.name == "__init__.py":
                continue
            
            module_name = file_path.stem
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Buscar plugins
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, Plugin) and obj != Plugin:
                            plugin = obj()
                            if self.register_plugin(plugin):
                                loaded += 1
            except Exception as e:
                logger.error(f"Error loading plugin from {file_path}: {e}")
        
        return loaded
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """Obtener plugin por nombre."""
        return self.plugins.get(name)
    
    def execute_plugin(self, name: str, *args, **kwargs) -> Any:
        """
        Ejecutar plugin.
        
        Args:
            name: Nombre del plugin
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados
            
        Returns:
            Resultado de la ejecución
        """
        plugin = self.get_plugin(name)
        if plugin is None:
            raise ValueError(f"Plugin {name} not found")
        
        if not plugin.enabled:
            raise RuntimeError(f"Plugin {name} is disabled")
        
        return plugin.execute(*args, **kwargs)
    
    def get_all_plugins(self) -> List[Plugin]:
        """Obtener todos los plugins."""
        return list(self.plugins.values())
    
    def unregister_plugin(self, name: str) -> bool:
        """Desregistrar plugin."""
        if name not in self.plugins:
            return False
        
        plugin = self.plugins[name]
        self.extension_manager.unregister(name)
        del self.plugins[name]
        logger.info(f"Plugin {name} unregistered")
        return True
    
    def shutdown_all(self):
        """Cerrar todos los plugins."""
        for plugin in list(self.plugins.values()):
            try:
                plugin.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down plugin {plugin.name}: {e}")
        
        self.plugins.clear()
        self.extension_manager.shutdown_all()


# Instancia global
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Obtener instancia global del gestor de plugins."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager

