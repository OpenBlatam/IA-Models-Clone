"""
Plugin System
=============

Sistema de plugins para extensibilidad.
"""

import logging
import importlib
from typing import Dict, Any, List, Optional, Type
from abc import ABC, abstractmethod

from .interfaces import IRouteStrategy, IRouteModel

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """Clase base para plugins."""
    
    @abstractmethod
    def get_name(self) -> str:
        """Obtener nombre del plugin."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Obtener versión del plugin."""
        pass
    
    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None):
        """Inicializar plugin."""
        pass
    
    def cleanup(self):
        """Limpiar recursos (opcional)."""
        pass


class RouteStrategyPlugin(Plugin):
    """Plugin para estrategias de routing."""
    
    @abstractmethod
    def create_strategy(self) -> IRouteStrategy:
        """Crear instancia de estrategia."""
        pass


class ModelPlugin(Plugin):
    """Plugin para modelos."""
    
    @abstractmethod
    def create_model(self, config: Optional[Dict[str, Any]] = None) -> IRouteModel:
        """Crear instancia de modelo."""
        pass


class PluginManager:
    """
    Gestor de plugins.
    """
    
    def __init__(self):
        """Inicializar gestor."""
        self._plugins: Dict[str, Plugin] = {}
        self._initialized: Dict[str, bool] = {}
    
    def register_plugin(self, plugin: Plugin):
        """
        Registrar plugin.
        
        Args:
            plugin: Instancia de plugin
        """
        name = plugin.get_name()
        self._plugins[name] = plugin
        self._initialized[name] = False
        logger.info(f"Plugin registrado: {name} v{plugin.get_version()}")
    
    def load_plugin_from_module(
        self,
        module_path: str,
        plugin_class_name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Cargar plugin desde módulo.
        
        Args:
            module_path: Ruta del módulo (ej: "plugins.my_plugin")
            plugin_class_name: Nombre de la clase del plugin
            config: Configuración (opcional)
        """
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, plugin_class_name)
            plugin = plugin_class()
            plugin.initialize(config)
            self.register_plugin(plugin)
        except Exception as e:
            logger.error(f"Error cargando plugin desde {module_path}: {e}")
            raise
    
    def initialize_plugin(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar plugin.
        
        Args:
            name: Nombre del plugin
            config: Configuración (opcional)
        """
        plugin = self._plugins.get(name)
        if not plugin:
            raise ValueError(f"Plugin '{name}' no encontrado")
        
        if not self._initialized[name]:
            plugin.initialize(config)
            self._initialized[name] = True
            logger.info(f"Plugin inicializado: {name}")
    
    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Obtener plugin.
        
        Args:
            name: Nombre del plugin
            
        Returns:
            Plugin o None
        """
        return self._plugins.get(name)
    
    def get_strategy_plugin(self, name: str) -> Optional[RouteStrategyPlugin]:
        """Obtener plugin de estrategia."""
        plugin = self.get_plugin(name)
        if isinstance(plugin, RouteStrategyPlugin):
            return plugin
        return None
    
    def get_model_plugin(self, name: str) -> Optional[ModelPlugin]:
        """Obtener plugin de modelo."""
        plugin = self.get_plugin(name)
        if isinstance(plugin, ModelPlugin):
            return plugin
        return None
    
    def list_plugins(self) -> List[str]:
        """Listar plugins registrados."""
        return list(self._plugins.keys())
    
    def unregister_plugin(self, name: str):
        """
        Desregistrar plugin.
        
        Args:
            name: Nombre del plugin
        """
        plugin = self._plugins.pop(name, None)
        if plugin:
            if self._initialized.get(name):
                plugin.cleanup()
            del self._initialized[name]
            logger.info(f"Plugin desregistrado: {name}")
    
    def cleanup_all(self):
        """Limpiar todos los plugins."""
        for name, plugin in self._plugins.items():
            if self._initialized.get(name):
                try:
                    plugin.cleanup()
                except Exception as e:
                    logger.error(f"Error limpiando plugin '{name}': {e}")
        
        self._plugins.clear()
        self._initialized.clear()

