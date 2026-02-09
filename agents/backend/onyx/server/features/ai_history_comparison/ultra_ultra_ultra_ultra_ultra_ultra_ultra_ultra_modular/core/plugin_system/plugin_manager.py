"""
Plugin Manager - Gestor de Plugins
=================================

Sistema de gestión de plugins dinámicos con carga/descarga en tiempo de ejecución.
"""

import asyncio
import importlib
import inspect
import json
import os
import sys
from typing import Dict, List, Any, Optional, Type, Callable
from datetime import datetime
import logging
from pathlib import Path

from ..interfaces.base_interfaces import IPlugin, IComponent, IRegistry

logger = logging.getLogger(__name__)


class PluginInfo:
    """Información de un plugin."""
    
    def __init__(self, name: str, version: str, description: str = "", 
                 author: str = "", dependencies: List[str] = None):
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.dependencies = dependencies or []
        self.installed_at = datetime.utcnow()
        self.status = "uninstalled"
        self.loaded_module = None
        self.plugin_instance = None


class PluginRegistry(IRegistry):
    """Registro de plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugin_instances: Dict[str, IPlugin] = {}
        self._plugin_hooks: Dict[str, List[Callable]] = {}
        self._lock = asyncio.Lock()
    
    async def register(self, name: str, component: Any) -> bool:
        """Registrar plugin."""
        try:
            async with self._lock:
                if isinstance(component, IPlugin):
                    self._plugin_instances[name] = component
                    return True
                return False
        except Exception as e:
            logger.error(f"Error registering plugin {name}: {e}")
            return False
    
    async def unregister(self, name: str) -> bool:
        """Desregistrar plugin."""
        try:
            async with self._lock:
                if name in self._plugin_instances:
                    plugin = self._plugin_instances[name]
                    await plugin.deactivate()
                    await plugin.uninstall()
                    del self._plugin_instances[name]
                    return True
                return False
        except Exception as e:
            logger.error(f"Error unregistering plugin {name}: {e}")
            return False
    
    async def get(self, name: str) -> Optional[Any]:
        """Obtener plugin."""
        return self._plugin_instances.get(name)
    
    async def list_all(self) -> List[str]:
        """Listar todos los plugins."""
        return list(self._plugin_instances.keys())
    
    async def register_plugin_info(self, plugin_info: PluginInfo) -> bool:
        """Registrar información de plugin."""
        try:
            async with self._lock:
                self._plugins[plugin_info.name] = plugin_info
                return True
        except Exception as e:
            logger.error(f"Error registering plugin info {plugin_info.name}: {e}")
            return False
    
    async def get_plugin_info(self, name: str) -> Optional[PluginInfo]:
        """Obtener información de plugin."""
        return self._plugins.get(name)
    
    async def add_hook(self, hook_name: str, callback: Callable) -> None:
        """Agregar hook."""
        if hook_name not in self._plugin_hooks:
            self._plugin_hooks[hook_name] = []
        self._plugin_hooks[hook_name].append(callback)
    
    async def execute_hooks(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Ejecutar hooks."""
        results = []
        if hook_name in self._plugin_hooks:
            for callback in self._plugin_hooks[hook_name]:
                try:
                    result = await callback(*args, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error executing hook {hook_name}: {e}")
        return results


class PluginLoader:
    """Cargador de plugins."""
    
    def __init__(self, plugin_directory: str = "plugins"):
        self.plugin_directory = Path(plugin_directory)
        self.plugin_directory.mkdir(exist_ok=True)
        self._loaded_modules: Dict[str, Any] = {}
    
    async def discover_plugins(self) -> List[PluginInfo]:
        """Descubrir plugins disponibles."""
        plugins = []
        
        for plugin_path in self.plugin_directory.iterdir():
            if plugin_path.is_dir() and (plugin_path / "plugin.json").exists():
                try:
                    plugin_info = await self._load_plugin_info(plugin_path)
                    if plugin_info:
                        plugins.append(plugin_info)
                except Exception as e:
                    logger.error(f"Error discovering plugin {plugin_path}: {e}")
        
        return plugins
    
    async def _load_plugin_info(self, plugin_path: Path) -> Optional[PluginInfo]:
        """Cargar información de plugin."""
        try:
            with open(plugin_path / "plugin.json", "r") as f:
                info_data = json.load(f)
            
            return PluginInfo(
                name=info_data.get("name", ""),
                version=info_data.get("version", "1.0.0"),
                description=info_data.get("description", ""),
                author=info_data.get("author", ""),
                dependencies=info_data.get("dependencies", [])
            )
        except Exception as e:
            logger.error(f"Error loading plugin info from {plugin_path}: {e}")
            return None
    
    async def load_plugin(self, plugin_name: str) -> Optional[IPlugin]:
        """Cargar plugin."""
        try:
            plugin_path = self.plugin_directory / plugin_name
            
            if not plugin_path.exists():
                logger.error(f"Plugin directory {plugin_path} does not exist")
                return None
            
            # Agregar al path de Python
            if str(plugin_path) not in sys.path:
                sys.path.insert(0, str(plugin_path))
            
            # Cargar módulo principal
            main_module_name = f"{plugin_name}.main"
            module = importlib.import_module(main_module_name)
            
            # Buscar clase de plugin
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, IPlugin) and 
                    obj != IPlugin):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                logger.error(f"No plugin class found in {plugin_name}")
                return None
            
            # Crear instancia
            plugin_instance = plugin_class()
            self._loaded_modules[plugin_name] = module
            
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return None
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Descargar plugin."""
        try:
            if plugin_name in self._loaded_modules:
                module = self._loaded_modules[plugin_name]
                
                # Remover del path de Python
                plugin_path = self.plugin_directory / plugin_name
                if str(plugin_path) in sys.path:
                    sys.path.remove(str(plugin_path))
                
                # Remover módulo de la caché
                main_module_name = f"{plugin_name}.main"
                if main_module_name in sys.modules:
                    del sys.modules[main_module_name]
                
                del self._loaded_modules[plugin_name]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False


class PluginManager(IComponent):
    """Gestor principal de plugins."""
    
    def __init__(self, plugin_directory: str = "plugins"):
        self.plugin_directory = plugin_directory
        self.registry = PluginRegistry()
        self.loader = PluginLoader(plugin_directory)
        self._initialized = False
    
    async def initialize(self) -> None:
        """Inicializar gestor de plugins."""
        try:
            # Descubrir plugins disponibles
            available_plugins = await self.loader.discover_plugins()
            
            # Registrar información de plugins
            for plugin_info in available_plugins:
                await self.registry.register_plugin_info(plugin_info)
            
            self._initialized = True
            logger.info(f"Plugin manager initialized with {len(available_plugins)} plugins")
            
        except Exception as e:
            logger.error(f"Error initializing plugin manager: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Cerrar gestor de plugins."""
        try:
            # Desactivar todos los plugins activos
            active_plugins = await self.registry.list_all()
            for plugin_name in active_plugins:
                await self.deactivate_plugin(plugin_name)
            
            self._initialized = False
            logger.info("Plugin manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error shutting down plugin manager: {e}")
    
    async def health_check(self) -> bool:
        """Verificar salud del gestor."""
        return self._initialized
    
    @property
    def name(self) -> str:
        return "PluginManager"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def install_plugin(self, plugin_name: str) -> bool:
        """Instalar plugin."""
        try:
            plugin_info = await self.registry.get_plugin_info(plugin_name)
            if not plugin_info:
                logger.error(f"Plugin {plugin_name} not found")
                return False
            
            # Verificar dependencias
            for dependency in plugin_info.dependencies:
                if not await self.is_plugin_installed(dependency):
                    logger.error(f"Dependency {dependency} not installed for plugin {plugin_name}")
                    return False
            
            # Cargar plugin
            plugin_instance = await self.loader.load_plugin(plugin_name)
            if not plugin_instance:
                logger.error(f"Failed to load plugin {plugin_name}")
                return False
            
            # Instalar plugin
            if await plugin_instance.install():
                plugin_info.status = "installed"
                plugin_info.plugin_instance = plugin_instance
                await self.registry.register(plugin_name, plugin_instance)
                logger.info(f"Plugin {plugin_name} installed successfully")
                return True
            else:
                logger.error(f"Failed to install plugin {plugin_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error installing plugin {plugin_name}: {e}")
            return False
    
    async def uninstall_plugin(self, plugin_name: str) -> bool:
        """Desinstalar plugin."""
        try:
            plugin_info = await self.registry.get_plugin_info(plugin_name)
            if not plugin_info:
                logger.error(f"Plugin {plugin_name} not found")
                return False
            
            # Desactivar si está activo
            if plugin_info.status == "active":
                await self.deactivate_plugin(plugin_name)
            
            # Desinstalar
            plugin_instance = await self.registry.get(plugin_name)
            if plugin_instance:
                if await plugin_instance.uninstall():
                    plugin_info.status = "uninstalled"
                    await self.registry.unregister(plugin_name)
                    await self.loader.unload_plugin(plugin_name)
                    logger.info(f"Plugin {plugin_name} uninstalled successfully")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error uninstalling plugin {plugin_name}: {e}")
            return False
    
    async def activate_plugin(self, plugin_name: str) -> bool:
        """Activar plugin."""
        try:
            plugin_info = await self.registry.get_plugin_info(plugin_name)
            if not plugin_info:
                logger.error(f"Plugin {plugin_name} not found")
                return False
            
            if plugin_info.status != "installed":
                logger.error(f"Plugin {plugin_name} is not installed")
                return False
            
            plugin_instance = await self.registry.get(plugin_name)
            if not plugin_instance:
                logger.error(f"Plugin instance {plugin_name} not found")
                return False
            
            if await plugin_instance.activate():
                plugin_info.status = "active"
                logger.info(f"Plugin {plugin_name} activated successfully")
                return True
            else:
                logger.error(f"Failed to activate plugin {plugin_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error activating plugin {plugin_name}: {e}")
            return False
    
    async def deactivate_plugin(self, plugin_name: str) -> bool:
        """Desactivar plugin."""
        try:
            plugin_info = await self.registry.get_plugin_info(plugin_name)
            if not plugin_info:
                logger.error(f"Plugin {plugin_name} not found")
                return False
            
            if plugin_info.status != "active":
                logger.error(f"Plugin {plugin_name} is not active")
                return False
            
            plugin_instance = await self.registry.get(plugin_name)
            if not plugin_instance:
                logger.error(f"Plugin instance {plugin_name} not found")
                return False
            
            if await plugin_instance.deactivate():
                plugin_info.status = "installed"
                logger.info(f"Plugin {plugin_name} deactivated successfully")
                return True
            else:
                logger.error(f"Failed to deactivate plugin {plugin_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error deactivating plugin {plugin_name}: {e}")
            return False
    
    async def is_plugin_installed(self, plugin_name: str) -> bool:
        """Verificar si plugin está instalado."""
        plugin_info = await self.registry.get_plugin_info(plugin_name)
        return plugin_info and plugin_info.status in ["installed", "active"]
    
    async def is_plugin_active(self, plugin_name: str) -> bool:
        """Verificar si plugin está activo."""
        plugin_info = await self.registry.get_plugin_info(plugin_name)
        return plugin_info and plugin_info.status == "active"
    
    async def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """Obtener información de plugin."""
        return await self.registry.get_plugin_info(plugin_name)
    
    async def list_available_plugins(self) -> List[PluginInfo]:
        """Listar plugins disponibles."""
        return list(self.registry._plugins.values())
    
    async def list_installed_plugins(self) -> List[PluginInfo]:
        """Listar plugins instalados."""
        return [info for info in self.registry._plugins.values() 
                if info.status in ["installed", "active"]]
    
    async def list_active_plugins(self) -> List[PluginInfo]:
        """Listar plugins activos."""
        return [info for info in self.registry._plugins.values() 
                if info.status == "active"]
    
    async def execute_plugin_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Ejecutar hook de plugins."""
        return await self.registry.execute_hooks(hook_name, *args, **kwargs)
    
    async def add_plugin_hook(self, hook_name: str, callback: Callable) -> None:
        """Agregar hook de plugin."""
        await self.registry.add_hook(hook_name, callback)
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Recargar plugin."""
        try:
            # Desactivar y desinstalar
            if await self.is_plugin_active(plugin_name):
                await self.deactivate_plugin(plugin_name)
            
            if await self.is_plugin_installed(plugin_name):
                await self.uninstall_plugin(plugin_name)
            
            # Reinstalar y activar
            if await self.install_plugin(plugin_name):
                return await self.activate_plugin(plugin_name)
            
            return False
            
        except Exception as e:
            logger.error(f"Error reloading plugin {plugin_name}: {e}")
            return False
    
    async def get_plugin_dependencies(self, plugin_name: str) -> List[str]:
        """Obtener dependencias de plugin."""
        plugin_info = await self.registry.get_plugin_info(plugin_name)
        return plugin_info.dependencies if plugin_info else []
    
    async def check_dependencies(self, plugin_name: str) -> Dict[str, bool]:
        """Verificar dependencias de plugin."""
        dependencies = await self.get_plugin_dependencies(plugin_name)
        return {dep: await self.is_plugin_installed(dep) for dep in dependencies}
    
    async def auto_install_dependencies(self, plugin_name: str) -> bool:
        """Instalar dependencias automáticamente."""
        try:
            dependencies = await self.get_plugin_dependencies(plugin_name)
            
            for dependency in dependencies:
                if not await self.is_plugin_installed(dependency):
                    logger.info(f"Auto-installing dependency {dependency}")
                    if not await self.install_plugin(dependency):
                        logger.error(f"Failed to auto-install dependency {dependency}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error auto-installing dependencies for {plugin_name}: {e}")
            return False




