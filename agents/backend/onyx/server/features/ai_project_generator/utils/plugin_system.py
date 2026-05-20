"""
Plugin System - Sistema de Plugins
===================================

Sistema extensible de plugins para agregar funcionalidades.
"""

import logging
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class PluginSystem:
    """Sistema de plugins"""

    def __init__(self, plugins_dir: Path = None):
        """
        Inicializa el sistema de plugins.

        Args:
            plugins_dir: Directorio donde se almacenan los plugins
        """
        if plugins_dir is None:
            plugins_dir = Path(__file__).parent.parent / "plugins"
        self.plugins_dir = Path(plugins_dir)
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self.plugins: Dict[str, Dict[str, Any]] = {}
        self.hooks: Dict[str, List[Callable]] = {}

    def register_plugin(
        self,
        plugin_name: str,
        plugin_module: str,
        hooks: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Registra un plugin.

        Args:
            plugin_name: Nombre del plugin
            plugin_module: Módulo del plugin
            hooks: Hooks del plugin

        Returns:
            True si se registró exitosamente
        """
        try:
            # Intentar cargar el módulo
            module = importlib.import_module(plugin_module)
            
            plugin_info = {
                "name": plugin_name,
                "module": plugin_module,
                "hooks": hooks or {},
                "registered_at": datetime.now().isoformat(),
                "active": True,
            }

            self.plugins[plugin_name] = plugin_info

            # Registrar hooks
            if hooks:
                for hook_name, hook_func in hooks.items():
                    if hook_name not in self.hooks:
                        self.hooks[hook_name] = []
                    self.hooks[hook_name].append(hook_func)

            logger.info(f"Plugin registrado: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error registrando plugin {plugin_name}: {e}")
            return False

    async def trigger_hook(
        self,
        hook_name: str,
        *args,
        **kwargs,
    ) -> List[Any]:
        """
        Dispara un hook.

        Args:
            hook_name: Nombre del hook
            *args: Argumentos posicionales
            **kwargs: Argumentos nombrados

        Returns:
            Lista de resultados de los hooks
        """
        results = []
        
        if hook_name in self.hooks:
            for hook_func in self.hooks[hook_name]:
                try:
                    if callable(hook_func):
                        result = await hook_func(*args, **kwargs) if hasattr(hook_func, '__call__') else hook_func(*args, **kwargs)
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error ejecutando hook {hook_name}: {e}")

        return results

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Lista todos los plugins registrados"""
        return [
            {
                "name": plugin["name"],
                "module": plugin["module"],
                "active": plugin.get("active", True),
                "registered_at": plugin.get("registered_at"),
            }
            for plugin in self.plugins.values()
        ]

    def disable_plugin(self, plugin_name: str) -> bool:
        """Desactiva un plugin"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name]["active"] = False
            logger.info(f"Plugin desactivado: {plugin_name}")
            return True
        return False

    def enable_plugin(self, plugin_name: str) -> bool:
        """Activa un plugin"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name]["active"] = True
            logger.info(f"Plugin activado: {plugin_name}")
            return True
        return False


