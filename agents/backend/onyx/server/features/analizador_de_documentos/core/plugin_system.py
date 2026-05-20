"""
Sistema de Plugins
==================

Sistema extensible para plugins personalizados.
"""

import logging
import importlib
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """Clase base para plugins"""
    
    @abstractmethod
    def name(self) -> str:
        """Nombre del plugin"""
        pass
    
    @abstractmethod
    def version(self) -> str:
        """Versión del plugin"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Inicializar plugin"""
        pass
    
    @abstractmethod
    def execute(self, data: Any, **kwargs) -> Any:
        """Ejecutar plugin"""
        pass
    
    def cleanup(self):
        """Limpiar recursos"""
        pass


@dataclass
class PluginInfo:
    """Información del plugin"""
    name: str
    version: str
    description: str
    author: str
    enabled: bool = True


class PluginManager:
    """
    Gestor de plugins
    
    Permite:
    - Cargar plugins dinámicamente
    - Ejecutar plugins en cadena
    - Gestionar ciclo de vida
    """
    
    def __init__(self):
        """Inicializar gestor"""
        self.plugins: Dict[str, Plugin] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        self.plugin_order: List[str] = []
        logger.info("PluginManager inicializado")
    
    def register_plugin(
        self,
        plugin: Plugin,
        info: PluginInfo,
        order: Optional[int] = None
    ):
        """
        Registrar plugin
        
        Args:
            plugin: Instancia del plugin
            info: Información del plugin
            order: Orden de ejecución
        """
        if not info.enabled:
            logger.info(f"Plugin {info.name} deshabilitado, saltando")
            return
        
        self.plugins[info.name] = plugin
        self.plugin_info[info.name] = info
        
        if order is not None:
            self.plugin_order.insert(order, info.name)
        else:
            self.plugin_order.append(info.name)
        
        # Inicializar plugin
        try:
            plugin.initialize({})
            logger.info(f"Plugin registrado: {info.name} v{info.version}")
        except Exception as e:
            logger.error(f"Error inicializando plugin {info.name}: {e}")
    
    def load_plugin_from_module(
        self,
        module_path: str,
        class_name: str,
        info: PluginInfo
    ) -> bool:
        """
        Cargar plugin desde módulo
        
        Args:
            module_path: Ruta del módulo
            class_name: Nombre de la clase
            info: Información del plugin
        
        Returns:
            True si se cargó correctamente
        """
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, class_name)
            plugin_instance = plugin_class()
            
            self.register_plugin(plugin_instance, info)
            return True
        except Exception as e:
            logger.error(f"Error cargando plugin {info.name}: {e}")
            return False
    
    def execute_plugin(
        self,
        plugin_name: str,
        data: Any,
        **kwargs
    ) -> Any:
        """
        Ejecutar plugin específico
        
        Args:
            plugin_name: Nombre del plugin
            data: Datos de entrada
            **kwargs: Argumentos adicionales
        
        Returns:
            Resultado del plugin
        """
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin no encontrado: {plugin_name}")
        
        plugin = self.plugins[plugin_name]
        return plugin.execute(data, **kwargs)
    
    def execute_pipeline(
        self,
        data: Any,
        plugin_names: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        """
        Ejecutar pipeline de plugins
        
        Args:
            data: Datos de entrada
            plugin_names: Lista de plugins (None = todos en orden)
            **kwargs: Argumentos adicionales
        
        Returns:
            Resultado final
        """
        plugins_to_run = plugin_names or self.plugin_order
        
        result = data
        for plugin_name in plugins_to_run:
            if plugin_name not in self.plugins:
                continue
            
            if not self.plugin_info[plugin_name].enabled:
                continue
            
            try:
                result = self.execute_plugin(plugin_name, result, **kwargs)
            except Exception as e:
                logger.error(f"Error ejecutando plugin {plugin_name}: {e}")
                # Continuar con siguiente plugin
                continue
        
        return result
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """Listar todos los plugins"""
        return [
            {
                "name": info.name,
                "version": info.version,
                "description": info.description,
                "author": info.author,
                "enabled": info.enabled
            }
            for info in self.plugin_info.values()
        ]
    
    def enable_plugin(self, plugin_name: str):
        """Habilitar plugin"""
        if plugin_name in self.plugin_info:
            self.plugin_info[plugin_name].enabled = True
    
    def disable_plugin(self, plugin_name: str):
        """Deshabilitar plugin"""
        if plugin_name in self.plugin_info:
            self.plugin_info[plugin_name].enabled = False
    
    def cleanup_all(self):
        """Limpiar todos los plugins"""
        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error limpiando plugin: {e}")


# Instancia global
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Obtener instancia global del gestor"""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager
















