"""
Plugin service with functional approach.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

from ..core.exceptions import PluginError, NotFoundError
from ..models.schemas import PluginInfo, PluginStatus
from ..core.logging import get_logger

logger = get_logger(__name__)


# Plugin registry (in-memory for simplicity)
_plugin_registry: Dict[str, PluginInfo] = {}
_plugin_instances: Dict[str, Any] = {}
_plugin_hooks: Dict[str, List[Callable]] = {}


async def discover_plugins(plugin_directory: str = "plugins") -> List[PluginInfo]:
    """Discover available plugins in the plugin directory."""
    try:
        plugins = []
        plugin_path = Path(plugin_directory)
        
        if not plugin_path.exists():
            logger.warning(f"Plugin directory {plugin_directory} does not exist")
            return plugins
        
        for plugin_dir in plugin_path.iterdir():
            if not plugin_dir.is_dir():
                continue
                
            plugin_json = plugin_dir / "plugin.json"
            if not plugin_json.exists():
                continue
            
            try:
                with open(plugin_json, "r") as f:
                    plugin_data = json.load(f)
                
                plugin_info = PluginInfo(
                    name=plugin_data.get("name", plugin_dir.name),
                    version=plugin_data.get("version", "1.0.0"),
                    description=plugin_data.get("description", ""),
                    author=plugin_data.get("author", ""),
                    dependencies=plugin_data.get("dependencies", []),
                    status=PluginStatus.UNINSTALLED
                )
                
                plugins.append(plugin_info)
                _plugin_registry[plugin_info.name] = plugin_info
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing plugin {plugin_dir.name}: {e}")
                continue
        
        logger.info(f"Discovered {len(plugins)} plugins")
        return plugins
        
    except Exception as e:
        logger.error(f"Error discovering plugins: {e}")
        raise PluginError(f"Failed to discover plugins: {e}")


async def install_plugin(plugin_name: str, auto_install_dependencies: bool = True) -> bool:
    """Install a plugin."""
    if not plugin_name:
        raise ValidationError("Plugin name cannot be empty")
    
    plugin_info = _plugin_registry.get(plugin_name)
    if not plugin_info:
        raise NotFoundError(f"Plugin {plugin_name} not found")
    
    if plugin_info.status == PluginStatus.INSTALLED:
        logger.info(f"Plugin {plugin_name} is already installed")
        return True
    
    try:
        # Check dependencies
        if auto_install_dependencies:
            for dependency in plugin_info.dependencies:
                if not await is_plugin_installed(dependency):
                    logger.info(f"Auto-installing dependency {dependency}")
                    await install_plugin(dependency, auto_install_dependencies)
        
        # Simulate plugin installation
        await _simulate_plugin_installation(plugin_name)
        
        # Update plugin status
        plugin_info.status = PluginStatus.INSTALLED
        plugin_info.installed_at = datetime.utcnow()
        
        logger.info(f"Plugin {plugin_name} installed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error installing plugin {plugin_name}: {e}")
        plugin_info.status = PluginStatus.ERROR
        raise PluginError(f"Failed to install plugin {plugin_name}: {e}", plugin_name)


async def activate_plugin(plugin_name: str) -> bool:
    """Activate a plugin."""
    if not plugin_name:
        raise ValidationError("Plugin name cannot be empty")
    
    plugin_info = _plugin_registry.get(plugin_name)
    if not plugin_info:
        raise NotFoundError(f"Plugin {plugin_name} not found")
    
    if plugin_info.status != PluginStatus.INSTALLED:
        raise PluginError(f"Plugin {plugin_name} is not installed", plugin_name)
    
    try:
        # Simulate plugin activation
        await _simulate_plugin_activation(plugin_name)
        
        # Update plugin status
        plugin_info.status = PluginStatus.ACTIVE
        
        logger.info(f"Plugin {plugin_name} activated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error activating plugin {plugin_name}: {e}")
        plugin_info.status = PluginStatus.ERROR
        raise PluginError(f"Failed to activate plugin {plugin_name}: {e}", plugin_name)


async def deactivate_plugin(plugin_name: str) -> bool:
    """Deactivate a plugin."""
    if not plugin_name:
        raise ValidationError("Plugin name cannot be empty")
    
    plugin_info = _plugin_registry.get(plugin_name)
    if not plugin_info:
        raise NotFoundError(f"Plugin {plugin_name} not found")
    
    if plugin_info.status != PluginStatus.ACTIVE:
        logger.info(f"Plugin {plugin_name} is not active")
        return True
    
    try:
        # Simulate plugin deactivation
        await _simulate_plugin_deactivation(plugin_name)
        
        # Update plugin status
        plugin_info.status = PluginStatus.INSTALLED
        
        logger.info(f"Plugin {plugin_name} deactivated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error deactivating plugin {plugin_name}: {e}")
        raise PluginError(f"Failed to deactivate plugin {plugin_name}: {e}", plugin_name)


async def uninstall_plugin(plugin_name: str) -> bool:
    """Uninstall a plugin."""
    if not plugin_name:
        raise ValidationError("Plugin name cannot be empty")
    
    plugin_info = _plugin_registry.get(plugin_name)
    if not plugin_info:
        raise NotFoundError(f"Plugin {plugin_name} not found")
    
    try:
        # Deactivate if active
        if plugin_info.status == PluginStatus.ACTIVE:
            await deactivate_plugin(plugin_name)
        
        # Simulate plugin uninstallation
        await _simulate_plugin_uninstallation(plugin_name)
        
        # Update plugin status
        plugin_info.status = PluginStatus.UNINSTALLED
        plugin_info.installed_at = None
        
        # Remove from instances
        if plugin_name in _plugin_instances:
            del _plugin_instances[plugin_name]
        
        logger.info(f"Plugin {plugin_name} uninstalled successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error uninstalling plugin {plugin_name}: {e}")
        raise PluginError(f"Failed to uninstall plugin {plugin_name}: {e}", plugin_name)


async def is_plugin_installed(plugin_name: str) -> bool:
    """Check if a plugin is installed."""
    plugin_info = _plugin_registry.get(plugin_name)
    return plugin_info and plugin_info.status in [PluginStatus.INSTALLED, PluginStatus.ACTIVE]


async def is_plugin_active(plugin_name: str) -> bool:
    """Check if a plugin is active."""
    plugin_info = _plugin_registry.get(plugin_name)
    return plugin_info and plugin_info.status == PluginStatus.ACTIVE


async def get_plugin_info(plugin_name: str) -> Optional[PluginInfo]:
    """Get plugin information."""
    return _plugin_registry.get(plugin_name)


async def list_plugins() -> List[PluginInfo]:
    """List all plugins."""
    return list(_plugin_registry.values())


async def list_installed_plugins() -> List[PluginInfo]:
    """List installed plugins."""
    return [
        plugin for plugin in _plugin_registry.values()
        if plugin.status in [PluginStatus.INSTALLED, PluginStatus.ACTIVE]
    ]


async def list_active_plugins() -> List[PluginInfo]:
    """List active plugins."""
    return [
        plugin for plugin in _plugin_registry.values()
        if plugin.status == PluginStatus.ACTIVE
    ]


async def add_plugin_hook(hook_name: str, callback: Callable) -> None:
    """Add a plugin hook."""
    if hook_name not in _plugin_hooks:
        _plugin_hooks[hook_name] = []
    _plugin_hooks[hook_name].append(callback)


async def execute_plugin_hook(hook_name: str, *args, **kwargs) -> List[Any]:
    """Execute plugin hooks."""
    results = []
    
    if hook_name not in _plugin_hooks:
        return results
    
    for callback in _plugin_hooks[hook_name]:
        try:
            if asyncio.iscoroutinefunction(callback):
                result = await callback(*args, **kwargs)
            else:
                result = callback(*args, **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"Error executing hook {hook_name}: {e}")
    
    return results


async def get_plugin_stats() -> Dict[str, Any]:
    """Get plugin statistics."""
    total_plugins = len(_plugin_registry)
    installed_plugins = len(await list_installed_plugins())
    active_plugins = len(await list_active_plugins())
    
    return {
        "total_plugins": total_plugins,
        "installed_plugins": installed_plugins,
        "active_plugins": active_plugins,
        "hook_count": len(_plugin_hooks),
        "hooks": list(_plugin_hooks.keys())
    }


# Helper functions
async def _simulate_plugin_installation(plugin_name: str) -> None:
    """Simulate plugin installation."""
    await asyncio.sleep(0.1)  # Simulate installation time
    _plugin_instances[plugin_name] = {"installed": True}


async def _simulate_plugin_activation(plugin_name: str) -> None:
    """Simulate plugin activation."""
    await asyncio.sleep(0.05)  # Simulate activation time
    if plugin_name in _plugin_instances:
        _plugin_instances[plugin_name]["active"] = True


async def _simulate_plugin_deactivation(plugin_name: str) -> None:
    """Simulate plugin deactivation."""
    await asyncio.sleep(0.05)  # Simulate deactivation time
    if plugin_name in _plugin_instances:
        _plugin_instances[plugin_name]["active"] = False


async def _simulate_plugin_uninstallation(plugin_name: str) -> None:
    """Simulate plugin uninstallation."""
    await asyncio.sleep(0.1)  # Simulate uninstallation time
    if plugin_name in _plugin_instances:
        del _plugin_instances[plugin_name]




