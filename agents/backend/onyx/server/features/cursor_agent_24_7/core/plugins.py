"""
Plugins - Sistema de plugins
============================

Sistema extensible de plugins para agregar funcionalidades.
"""

import asyncio
import logging
import importlib
import importlib.util
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BasePlugin(ABC):
    """Clase base para plugins"""
    
    def __init__(self, agent):
        self.agent = agent
        self.name = self.__class__.__name__
        self.enabled = True
    
    @abstractmethod
    async def on_start(self):
        """Llamado cuando el agente inicia"""
        pass
    
    @abstractmethod
    async def on_stop(self):
        """Llamado cuando el agente se detiene"""
        pass
    
    async def on_task_added(self, task_id: str, command: str):
        """Llamado cuando se agrega una tarea"""
        pass
    
    async def on_task_completed(self, task_id: str, result: str):
        """Llamado cuando se completa una tarea"""
        pass
    
    async def on_task_failed(self, task_id: str, error: str):
        """Llamado cuando falla una tarea"""
        pass


class PluginManager:
    """Gestor de plugins"""
    
    def __init__(self, agent):
        self.agent = agent
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_dir = Path("./plugins")
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
    
    def register_plugin(self, plugin: BasePlugin):
        """Registrar un plugin"""
        self.plugins[plugin.name] = plugin
        logger.info(f"🔌 Plugin registered: {plugin.name}")
    
    def unregister_plugin(self, plugin_name: str):
        """Desregistrar un plugin"""
        if plugin_name in self.plugins:
            del self.plugins[plugin_name]
            logger.info(f"🔌 Plugin unregistered: {plugin_name}")
    
    async def load_plugin_from_file(self, file_path: str) -> bool:
        """Cargar plugin desde archivo"""
        try:
            plugin_path = Path(file_path)
            if not plugin_path.exists():
                logger.error(f"Plugin file not found: {file_path}")
                return False
            
            # Cargar módulo
            spec = importlib.util.spec_from_file_location(
                plugin_path.stem,
                plugin_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Buscar clase de plugin
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and
                    issubclass(attr, BasePlugin) and
                    attr != BasePlugin):
                    plugin = attr(self.agent)
                    self.register_plugin(plugin)
                    return True
            
            logger.error(f"No plugin class found in {file_path}")
            return False
            
        except Exception as e:
            logger.error(f"Error loading plugin from {file_path}: {e}")
            return False
    
    async def load_plugins_from_dir(self, directory: Optional[str] = None):
        """Cargar plugins desde directorio"""
        plugin_dir = Path(directory) if directory else self.plugin_dir
        
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return
        
        plugin_files = list(plugin_dir.glob("*.py"))
        
        for plugin_file in plugin_files:
            if plugin_file.name.startswith("_"):
                continue
            
            await self.load_plugin_from_file(str(plugin_file))
    
    async def start_all(self):
        """Iniciar todos los plugins"""
        for plugin in self.plugins.values():
            if plugin.enabled:
                try:
                    await plugin.on_start()
                except Exception as e:
                    logger.error(f"Error starting plugin {plugin.name}: {e}")
    
    async def stop_all(self):
        """Detener todos los plugins"""
        for plugin in self.plugins.values():
            if plugin.enabled:
                try:
                    await plugin.on_stop()
                except Exception as e:
                    logger.error(f"Error stopping plugin {plugin.name}: {e}")
    
    async def notify_task_added(self, task_id: str, command: str):
        """Notificar a plugins sobre tarea agregada"""
        for plugin in self.plugins.values():
            if plugin.enabled:
                try:
                    await plugin.on_task_added(task_id, command)
                except Exception as e:
                    logger.error(f"Error in plugin {plugin.name}.on_task_added: {e}")
    
    async def notify_task_completed(self, task_id: str, result: str):
        """Notificar a plugins sobre tarea completada"""
        for plugin in self.plugins.values():
            if plugin.enabled:
                try:
                    await plugin.on_task_completed(task_id, result)
                except Exception as e:
                    logger.error(f"Error in plugin {plugin.name}.on_task_completed: {e}")
    
    async def notify_task_failed(self, task_id: str, error: str):
        """Notificar a plugins sobre tarea fallida"""
        for plugin in self.plugins.values():
            if plugin.enabled:
                try:
                    await plugin.on_task_failed(task_id, error)
                except Exception as e:
                    logger.error(f"Error in plugin {plugin.name}.on_task_failed: {e}")
    
    def get_plugins(self) -> List[Dict]:
        """Obtener lista de plugins"""
        return [
            {
                "name": plugin.name,
                "enabled": plugin.enabled,
                "class": plugin.__class__.__name__
            }
            for plugin in self.plugins.values()
        ]


# Plugin de ejemplo
class LoggingPlugin(BasePlugin):
    """Plugin de ejemplo que registra eventos"""
    
    async def on_start(self):
        logger.info(f"📝 LoggingPlugin: Agent started")
    
    async def on_stop(self):
        logger.info(f"📝 LoggingPlugin: Agent stopped")
    
    async def on_task_added(self, task_id: str, command: str):
        logger.info(f"📝 LoggingPlugin: Task added - {task_id[:8]}...")
    
    async def on_task_completed(self, task_id: str, result: str):
        logger.info(f"📝 LoggingPlugin: Task completed - {task_id[:8]}...")
    
    async def on_task_failed(self, task_id: str, error: str):
        logger.warning(f"📝 LoggingPlugin: Task failed - {task_id[:8]}... - {error[:50]}")

