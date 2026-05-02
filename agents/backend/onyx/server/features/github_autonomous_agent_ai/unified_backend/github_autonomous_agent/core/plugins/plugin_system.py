"""
Sistema de Plugins y Extensiones.
"""

import importlib
import inspect
from typing import Dict, Any, Optional, List, Callable, Type
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


@dataclass
class PluginInfo:
    """Información de un plugin."""
    name: str
    version: str
    description: str
    author: str
    enabled: bool = True
    loaded_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BasePlugin(ABC):
    """Clase base para plugins."""
    
    def __init__(self):
        """Inicializar plugin."""
        self.info: Optional[PluginInfo] = None
        self.enabled = True
    
    @abstractmethod
    def get_info(self) -> PluginInfo:
        """
        Obtener información del plugin.
        
        Returns:
            PluginInfo
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Inicializar plugin."""
        pass
    
    def on_enable(self) -> None:
        """Llamado cuando el plugin se habilita."""
        pass
    
    def on_disable(self) -> None:
        """Llamado cuando el plugin se deshabilita."""
        pass
    
    def cleanup(self) -> None:
        """Limpieza al desactivar plugin."""
        pass


class TaskPlugin(BasePlugin):
    """Plugin para extender procesamiento de tareas."""
    
    @abstractmethod
    async def before_task_process(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Llamado antes de procesar tarea.
        
        Args:
            task: Tarea a procesar
            
        Returns:
            Tarea modificada o None para cancelar
        """
        pass
    
    @abstractmethod
    async def after_task_process(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Llamado después de procesar tarea.
        
        Args:
            task: Tarea procesada
            result: Resultado del procesamiento
        """
        pass


class EventPlugin(BasePlugin):
    """Plugin para manejar eventos."""
    
    @abstractmethod
    async def handle_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Manejar evento.
        
        Args:
            event_type: Tipo de evento
            event_data: Datos del evento
        """
        pass


class PluginManager:
    """Manager de plugins."""
    
    def __init__(self, plugins_dir: Optional[Path] = None):
        """
        Inicializar manager de plugins.
        
        Args:
            plugins_dir: Directorio de plugins
        """
        if plugins_dir is None:
            plugins_dir = Path(settings.STORAGE_PATH) / "plugins"
        self.plugins_dir = plugins_dir
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self.plugins: Dict[str, BasePlugin] = {}
        self.task_plugins: List[TaskPlugin] = []
        self.event_plugins: List[EventPlugin] = []
    
    def register_plugin(self, plugin: BasePlugin) -> None:
        """
        Registrar plugin.
        
        Args:
            plugin: Plugin a registrar
        """
        plugin.initialize()
        info = plugin.get_info()
        plugin.info = info
        
        self.plugins[info.name] = plugin
        
        # Categorizar plugin
        if isinstance(plugin, TaskPlugin):
            self.task_plugins.append(plugin)
        if isinstance(plugin, EventPlugin):
            self.event_plugins.append(plugin)
        
        logger.info(f"Plugin registrado: {info.name} v{info.version}")
    
    def load_plugin_from_module(self, module_path: str) -> Optional[BasePlugin]:
        """
        Cargar plugin desde módulo.
        
        Args:
            module_path: Ruta al módulo
            
        Returns:
            Plugin cargado o None si falla
        """
        try:
            module = importlib.import_module(module_path)
            
            # Buscar clase que herede de BasePlugin
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    plugin = obj()
                    self.register_plugin(plugin)
                    return plugin
        except Exception as e:
            logger.error(f"Error cargando plugin desde {module_path}: {e}", exc_info=True)
        return None
    
    def enable_plugin(self, name: str) -> bool:
        """
        Habilitar plugin.
        
        Args:
            name: Nombre del plugin
            
        Returns:
            True si se habilitó exitosamente
        """
        if name not in self.plugins:
            logger.warning(f"Plugin no encontrado: {name}")
            return False
        
        plugin = self.plugins[name]
        if plugin.enabled:
            return True
        
        try:
            plugin.on_enable()
            plugin.enabled = True
            if plugin.info:
                plugin.info.enabled = True
            logger.info(f"Plugin habilitado: {name}")
            return True
        except Exception as e:
            logger.error(f"Error habilitando plugin {name}: {e}", exc_info=True)
            return False
    
    def disable_plugin(self, name: str) -> bool:
        """
        Deshabilitar plugin.
        
        Args:
            name: Nombre del plugin
            
        Returns:
            True si se deshabilitó exitosamente
        """
        if name not in self.plugins:
            logger.warning(f"Plugin no encontrado: {name}")
            return False
        
        plugin = self.plugins[name]
        if not plugin.enabled:
            return True
        
        try:
            plugin.on_disable()
            plugin.cleanup()
            plugin.enabled = False
            if plugin.info:
                plugin.info.enabled = False
            logger.info(f"Plugin deshabilitado: {name}")
            return True
        except Exception as e:
            logger.error(f"Error deshabilitando plugin {name}: {e}", exc_info=True)
            return False
    
    async def process_task_plugins_before(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Ejecutar plugins de tarea antes de procesar.
        
        Args:
            task: Tarea a procesar
            
        Returns:
            Tarea modificada o None para cancelar
        """
        for plugin in self.task_plugins:
            if not plugin.enabled:
                continue
            try:
                result = await plugin.before_task_process(task)
                if result is None:
                    logger.info(f"Tarea cancelada por plugin: {plugin.info.name if plugin.info else 'unknown'}")
                    return None
                task = result
            except Exception as e:
                logger.error(f"Error en plugin {plugin.info.name if plugin.info else 'unknown'}: {e}", exc_info=True)
        return task
    
    async def process_task_plugins_after(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Ejecutar plugins de tarea después de procesar.
        
        Args:
            task: Tarea procesada
            result: Resultado del procesamiento
        """
        for plugin in self.task_plugins:
            if not plugin.enabled:
                continue
            try:
                await plugin.after_task_process(task, result)
            except Exception as e:
                logger.error(f"Error en plugin {plugin.info.name if plugin.info else 'unknown'}: {e}", exc_info=True)
    
    async def broadcast_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Broadcast evento a plugins.
        
        Args:
            event_type: Tipo de evento
            event_data: Datos del evento
        """
        for plugin in self.event_plugins:
            if not plugin.enabled:
                continue
            try:
                await plugin.handle_event(event_type, event_data)
            except Exception as e:
                logger.error(f"Error en plugin {plugin.info.name if plugin.info else 'unknown'}: {e}", exc_info=True)
    
    def get_plugins(self) -> List[Dict[str, Any]]:
        """Obtener lista de plugins."""
        return [
            {
                "name": info.name,
                "version": info.version,
                "description": info.description,
                "author": info.author,
                "enabled": info.enabled,
                "loaded_at": info.loaded_at.isoformat() if info.loaded_at else None,
                "metadata": info.metadata
            }
            for plugin in self.plugins.values()
            if plugin.info
        ]
    
    def get_plugin(self, name: str) -> Optional[Dict[str, Any]]:
        """Obtener información de un plugin."""
        plugin = self.plugins.get(name)
        if plugin and plugin.info:
            return {
                "name": plugin.info.name,
                "version": plugin.info.version,
                "description": plugin.info.description,
                "author": plugin.info.author,
                "enabled": plugin.info.enabled,
                "loaded_at": plugin.info.loaded_at.isoformat() if plugin.info.loaded_at else None,
                "metadata": plugin.info.metadata
            }
        return None


# Instancia global
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Obtener manager de plugins."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager



