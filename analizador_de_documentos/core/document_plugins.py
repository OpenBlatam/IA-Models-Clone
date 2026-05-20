"""
Document Plugins - Sistema de Plugins
======================================

Sistema extensible de plugins para agregar funcionalidades personalizadas.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Type
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import importlib
import inspect

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Información de plugin."""
    name: str
    version: str
    author: Optional[str] = None
    description: Optional[str] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentPlugin(ABC):
    """Base class para plugins."""
    
    def __init__(self, analyzer):
        """Inicializar plugin."""
        self.analyzer = analyzer
        self.info = self.get_info()
    
    @abstractmethod
    def get_info(self) -> PluginInfo:
        """Obtener información del plugin."""
        pass
    
    @abstractmethod
    async def initialize(self):
        """Inicializar plugin."""
        pass
    
    async def cleanup(self):
        """Limpiar recursos."""
        pass


class PluginManager:
    """Gestor de plugins."""
    
    def __init__(self, analyzer):
        """Inicializar gestor."""
        self.analyzer = analyzer
        self.plugins: Dict[str, DocumentPlugin] = {}
        self.hooks: Dict[str, List[Callable]] = {}
    
    def register_plugin(self, plugin: DocumentPlugin):
        """Registrar plugin."""
        if plugin.info.name in self.plugins:
            logger.warning(f"Plugin {plugin.info.name} ya está registrado, reemplazando...")
        
        self.plugins[plugin.info.name] = plugin
        logger.info(f"Plugin registrado: {plugin.info.name} v{plugin.info.version}")
    
    async def initialize_plugin(self, plugin_name: str):
        """Inicializar plugin."""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin {plugin_name} no encontrado")
        
        plugin = self.plugins[plugin_name]
        if plugin.info.enabled:
            await plugin.initialize()
            logger.info(f"Plugin {plugin_name} inicializado")
    
    async def initialize_all(self):
        """Inicializar todos los plugins."""
        for plugin_name, plugin in self.plugins.items():
            if plugin.info.enabled:
                try:
                    await self.initialize_plugin(plugin_name)
                except Exception as e:
                    logger.error(f"Error inicializando plugin {plugin_name}: {e}")
    
    def register_hook(self, hook_name: str, callback: Callable):
        """Registrar hook."""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(callback)
    
    async def trigger_hook(self, hook_name: str, *args, **kwargs):
        """Disparar hook."""
        if hook_name in self.hooks:
            for callback in self.hooks[hook_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error en hook {hook_name}: {e}")
    
    def get_plugin(self, plugin_name: str) -> Optional[DocumentPlugin]:
        """Obtener plugin."""
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[PluginInfo]:
        """Listar plugins."""
        return [plugin.info for plugin in self.plugins.values()]
    
    async def unload_plugin(self, plugin_name: str):
        """Des cargar plugin."""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            await plugin.cleanup()
            del self.plugins[plugin_name]
            logger.info(f"Plugin {plugin_name} des cargado")


# Plugin de ejemplo
class ExampleAnalysisPlugin(DocumentPlugin):
    """Plugin de ejemplo para análisis adicional."""
    
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="example_analysis",
            version="1.0.0",
            author="System",
            description="Plugin de ejemplo para análisis adicional"
        )
    
    async def initialize(self):
        """Inicializar plugin."""
        logger.info("Example plugin inicializado")
    
    async def analyze_additional(self, content: str) -> Dict[str, Any]:
        """Análisis adicional."""
        return {
            "plugin_name": self.info.name,
            "additional_analysis": "Ejemplo de análisis adicional",
            "timestamp": datetime.now().isoformat()
        }


__all__ = [
    "PluginManager",
    "DocumentPlugin",
    "PluginInfo",
    "ExampleAnalysisPlugin"
]
















