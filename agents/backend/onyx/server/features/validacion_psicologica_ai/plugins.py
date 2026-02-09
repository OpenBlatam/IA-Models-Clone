"""
Sistema de Plugins para Validación Psicológica AI
===================================================
Sistema extensible de plugins para funcionalidades adicionales
"""

from typing import Dict, Any, List, Optional, Callable, Type
from abc import ABC, abstractmethod
from uuid import UUID
import structlog
import importlib
from pathlib import Path

from .models import PsychologicalValidation, PsychologicalProfile

logger = structlog.get_logger()


class BasePlugin(ABC):
    """Plugin base"""
    
    def __init__(self, name: str, version: str):
        """
        Inicializar plugin
        
        Args:
            name: Nombre del plugin
            version: Versión del plugin
        """
        self.name = name
        self.version = version
        self.enabled = True
    
    @abstractmethod
    def on_validation_completed(
        self,
        validation: PsychologicalValidation
    ) -> None:
        """
        Callback cuando se completa una validación
        
        Args:
            validation: Validación completada
        """
        pass
    
    @abstractmethod
    def on_profile_generated(
        self,
        profile: PsychologicalProfile
    ) -> None:
        """
        Callback cuando se genera un perfil
        
        Args:
            profile: Perfil generado
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Obtener información del plugin"""
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled
        }


class PluginManager:
    """Gestor de plugins"""
    
    def __init__(self):
        """Inicializar gestor"""
        self._plugins: Dict[str, BasePlugin] = {}
        logger.info("PluginManager initialized")
    
    def register_plugin(self, plugin: BasePlugin) -> bool:
        """
        Registrar plugin
        
        Args:
            plugin: Plugin a registrar
            
        Returns:
            True si se registró exitosamente
        """
        if plugin.name in self._plugins:
            logger.warning("Plugin already registered", name=plugin.name)
            return False
        
        self._plugins[plugin.name] = plugin
        logger.info("Plugin registered", name=plugin.name, version=plugin.version)
        return True
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Desregistrar plugin
        
        Args:
            plugin_name: Nombre del plugin
            
        Returns:
            True si se desregistró exitosamente
        """
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
            logger.info("Plugin unregistered", name=plugin_name)
            return True
        return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        Obtener plugin
        
        Args:
            plugin_name: Nombre del plugin
            
        Returns:
            Plugin o None
        """
        return self._plugins.get(plugin_name)
    
    def get_all_plugins(self) -> List[BasePlugin]:
        """
        Obtener todos los plugins
        
        Returns:
            Lista de plugins
        """
        return list(self._plugins.values())
    
    def get_enabled_plugins(self) -> List[BasePlugin]:
        """
        Obtener plugins habilitados
        
        Returns:
            Lista de plugins habilitados
        """
        return [p for p in self._plugins.values() if p.enabled]
    
    async def notify_validation_completed(
        self,
        validation: PsychologicalValidation
    ) -> None:
        """
        Notificar a plugins sobre validación completada
        
        Args:
            validation: Validación completada
        """
        for plugin in self.get_enabled_plugins():
            try:
                plugin.on_validation_completed(validation)
            except Exception as e:
                logger.error(
                    "Error in plugin validation callback",
                    plugin=plugin.name,
                    error=str(e)
                )
    
    async def notify_profile_generated(
        self,
        profile: PsychologicalProfile
    ) -> None:
        """
        Notificar a plugins sobre perfil generado
        
        Args:
            profile: Perfil generado
        """
        for plugin in self.get_enabled_plugins():
            try:
                plugin.on_profile_generated(profile)
            except Exception as e:
                logger.error(
                    "Error in plugin profile callback",
                    plugin=plugin.name,
                    error=str(e)
                )
    
    def load_plugin_from_module(
        self,
        module_path: str,
        plugin_class_name: str
    ) -> bool:
        """
        Cargar plugin desde módulo
        
        Args:
            module_path: Ruta del módulo
            plugin_class_name: Nombre de la clase del plugin
            
        Returns:
            True si se cargó exitosamente
        """
        try:
            module = importlib.import_module(module_path)
            plugin_class = getattr(module, plugin_class_name)
            plugin = plugin_class()
            return self.register_plugin(plugin)
        except Exception as e:
            logger.error(
                "Error loading plugin",
                module_path=module_path,
                error=str(e)
            )
            return False


# Instancia global del gestor de plugins
plugin_manager = PluginManager()




