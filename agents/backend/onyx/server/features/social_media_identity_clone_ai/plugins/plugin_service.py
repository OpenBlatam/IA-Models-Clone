"""
Sistema de plugins y extensions
"""

import logging
import importlib
import inspect
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Plugin:
    """Plugin del sistema"""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    enabled: bool = True
    hooks: List[str] = None  # Hooks que el plugin escucha
    functions: Dict[str, Callable] = None  # Funciones expuestas


class PluginService:
    """Servicio de gestión de plugins"""
    
    def __init__(self):
        self.settings = get_settings()
        self.plugins_dir = Path(self.settings.storage_path) / "plugins"
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.plugins: Dict[str, Plugin] = {}
        self.hooks: Dict[str, List[Callable]] = {}  # hook_name -> [callbacks]
    
    def register_plugin(self, plugin: Plugin):
        """Registra un plugin"""
        self.plugins[plugin.plugin_id] = plugin
        
        # Registrar hooks
        if plugin.hooks:
            for hook in plugin.hooks:
                if hook not in self.hooks:
                    self.hooks[hook] = []
                # Agregar funciones del plugin que escuchan este hook
                if plugin.functions:
                    for func_name, func in plugin.functions.items():
                        if hasattr(func, '__hook__') and func.__hook__ == hook:
                            self.hooks[hook].append(func)
        
        logger.info(f"Plugin registrado: {plugin.name} ({plugin.plugin_id})")
    
    def unregister_plugin(self, plugin_id: str):
        """Desregistra un plugin"""
        plugin = self.plugins.pop(plugin_id, None)
        if plugin:
            # Remover hooks
            for hook_name, callbacks in self.hooks.items():
                if plugin.functions:
                    self.hooks[hook_name] = [
                        cb for cb in callbacks
                        if cb not in plugin.functions.values()
                    ]
            logger.info(f"Plugin desregistrado: {plugin_id}")
    
    def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Llama a todos los callbacks registrados para un hook
        
        Args:
            hook_name: Nombre del hook
            *args, **kwargs: Argumentos para los callbacks
            
        Returns:
            Lista de resultados de los callbacks
        """
        results = []
        callbacks = self.hooks.get(hook_name, [])
        
        for callback in callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    import asyncio
                    result = asyncio.run(callback(*args, **kwargs))
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error en hook {hook_name}: {e}", exc_info=True)
        
        return results
    
    def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Obtiene un plugin"""
        return self.plugins.get(plugin_id)
    
    def list_plugins(self, enabled_only: bool = False) -> List[Plugin]:
        """Lista plugins"""
        plugins = list(self.plugins.values())
        if enabled_only:
            plugins = [p for p in plugins if p.enabled]
        return plugins


def hook(hook_name: str):
    """Decorator para marcar función como hook"""
    def decorator(func: Callable) -> Callable:
        func.__hook__ = hook_name
        return func
    return decorator


# Singleton global
_plugin_service: Optional[PluginService] = None


def get_plugin_service() -> PluginService:
    """Obtiene instancia singleton del servicio de plugins"""
    global _plugin_service
    if _plugin_service is None:
        _plugin_service = PluginService()
    return _plugin_service




