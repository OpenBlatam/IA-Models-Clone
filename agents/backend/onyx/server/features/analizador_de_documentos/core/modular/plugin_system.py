"""
Plugin System - Sistema de Plugins
===================================

Sistema de plugins modular y extensible.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import importlib
import inspect

logger = logging.getLogger(__name__)


@runtime_checkable
class PluginInterface(Protocol):
    """Interfaz de plugin."""
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """Obtener información del plugin."""
        ...
    
    async def initialize(self, context: Dict[str, Any]) -> None:
        """Inicializar plugin."""
        ...
    
    async def execute(self, *args, **kwargs) -> Any:
        """Ejecutar plugin."""
        ...
    
    async def cleanup(self) -> None:
        """Limpiar recursos del plugin."""
        ...


@dataclass
class PluginInfo:
    """Información de plugin."""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    plugin_type: str
    enabled: bool = True
    instance: Optional[PluginInterface] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PluginSystem:
    """Sistema de plugins."""
    
    def __init__(self):
        """Inicializar sistema."""
        self.plugins: Dict[str, PluginInfo] = {}
        self.initialized_plugins: Dict[str, PluginInterface] = {}
        self.context: Dict[str, Any] = {}
    
    def register_plugin(
        self,
        plugin_id: str,
        name: str,
        version: str,
        description: str,
        author: str,
        plugin_type: str,
        plugin_class: Type[PluginInterface],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Registrar plugin."""
        if plugin_id in self.plugins:
            logger.warning(f"Plugin {plugin_id} ya registrado")
        
        self.plugins[plugin_id] = PluginInfo(
            plugin_id=plugin_id,
            name=name,
            version=version,
            description=description,
            author=author,
            plugin_type=plugin_type,
            enabled=True,
            metadata=metadata or {}
        )
        
        # Crear instancia
        try:
            instance = plugin_class()
            self.plugins[plugin_id].instance = instance
            self.initialized_plugins[plugin_id] = instance
            logger.info(f"Plugin registrado: {plugin_id} ({name})")
        except Exception as e:
            logger.error(f"Error creando instancia de plugin {plugin_id}: {e}")
    
    async def initialize_plugin(self, plugin_id: str, context: Optional[Dict[str, Any]] = None):
        """Inicializar plugin."""
        if plugin_id not in self.plugins:
            raise ValueError(f"Plugin {plugin_id} no registrado")
        
        plugin_info = self.plugins[plugin_id]
        if not plugin_info.enabled:
            logger.warning(f"Plugin {plugin_id} está deshabilitado")
            return
        
        if plugin_id in self.initialized_plugins:
            instance = self.initialized_plugins[plugin_id]
        else:
            instance = plugin_info.instance
            if not instance:
                raise ValueError(f"Instancia de plugin {plugin_id} no disponible")
        
        init_context = context or self.context
        
        try:
            await instance.initialize(init_context)
            logger.info(f"Plugin inicializado: {plugin_id}")
        except Exception as e:
            logger.error(f"Error inicializando plugin {plugin_id}: {e}")
            raise
    
    async def initialize_all_plugins(self, context: Optional[Dict[str, Any]] = None):
        """Inicializar todos los plugins."""
        for plugin_id in self.plugins:
            if self.plugins[plugin_id].enabled:
                try:
                    await self.initialize_plugin(plugin_id, context)
                except Exception as e:
                    logger.error(f"Error inicializando {plugin_id}: {e}")
    
    async def execute_plugin(self, plugin_id: str, *args, **kwargs) -> Any:
        """Ejecutar plugin."""
        if plugin_id not in self.initialized_plugins:
            raise ValueError(f"Plugin {plugin_id} no inicializado")
        
        instance = self.initialized_plugins[plugin_id]
        return await instance.execute(*args, **kwargs)
    
    def get_plugin(self, plugin_id: str) -> Optional[PluginInterface]:
        """Obtener plugin."""
        return self.initialized_plugins.get(plugin_id)
    
    def get_plugins_by_type(self, plugin_type: str) -> List[PluginInterface]:
        """Obtener plugins por tipo."""
        return [
            self.initialized_plugins[pid]
            for pid, info in self.plugins.items()
            if info.plugin_type == plugin_type and pid in self.initialized_plugins
        ]
    
    def enable_plugin(self, plugin_id: str):
        """Habilitar plugin."""
        if plugin_id in self.plugins:
            self.plugins[plugin_id].enabled = True
    
    def disable_plugin(self, plugin_id: str):
        """Deshabilitar plugin."""
        if plugin_id in self.plugins:
            self.plugins[plugin_id].enabled = False
    
    async def unload_plugin(self, plugin_id: str):
        """Descargar plugin."""
        if plugin_id in self.initialized_plugins:
            instance = self.initialized_plugins[plugin_id]
            try:
                await instance.cleanup()
            except Exception as e:
                logger.error(f"Error limpiando plugin {plugin_id}: {e}")
            
            del self.initialized_plugins[plugin_id]
            logger.info(f"Plugin descargado: {plugin_id}")
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """Listar todos los plugins."""
        return [
            {
                "id": info.plugin_id,
                "name": info.name,
                "version": info.version,
                "type": info.plugin_type,
                "enabled": info.enabled,
                "initialized": info.plugin_id in self.initialized_plugins
            }
            for info in self.plugins.values()
        ]


__all__ = [
    "PluginSystem",
    "PluginInterface",
    "PluginInfo"
]


