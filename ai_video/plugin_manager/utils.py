from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime
from .core.models import OnyxPluginContext
from .core.exceptions import AIVideoError
from .manager import OnyxPluginManager
from typing import Any, List, Dict, Optional
import logging
"""
Onyx Plugin Manager - Utilities

Utility functions and helpers for the Onyx plugin management system.
"""




# Global plugin manager instance
onyx_plugin_manager = OnyxPluginManager()


async def initialize_onyx_plugins() -> None:
    """Initialize Onyx plugin system."""
    await onyx_plugin_manager.initialize()


async def execute_onyx_plugins(context: OnyxPluginContext, plugin_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Execute Onyx plugins."""
    return await onyx_plugin_manager.execute_plugins(context, plugin_names)


async def get_onyx_plugin_status() -> Dict[str, Any]:
    """Get Onyx plugin system status."""
    status = await onyx_plugin_manager.get_manager_status()
    return status.__dict__


async def enable_plugin(plugin_name: str) -> bool:
    """Enable a specific plugin."""
    return await onyx_plugin_manager.enable_plugin(plugin_name)


async def disable_plugin(plugin_name: str) -> bool:
    """Disable a specific plugin."""
    return await onyx_plugin_manager.disable_plugin(plugin_name)


async def reload_plugin(plugin_name: str) -> bool:
    """Reload a specific plugin."""
    return await onyx_plugin_manager.reload_plugin(plugin_name)


async def get_plugin_info(plugin_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific plugin."""
    info = await onyx_plugin_manager.get_plugin_info(plugin_name)
    return info.__dict__ if info else None


async def get_all_plugins_info() -> List[Dict[str, Any]]:
    """Get information about all plugins."""
    plugin_infos = await onyx_plugin_manager.get_all_plugins_info()
    return [info.__dict__ for info in plugin_infos]


async def cleanup_onyx_plugins() -> None:
    """Cleanup Onyx plugin system."""
    await onyx_plugin_manager.cleanup()


def create_plugin_context(request: Any, llm: Optional[Any] = None, gpu_available: bool = False) -> OnyxPluginContext:
    """Create a plugin context from request data."""
    return OnyxPluginContext(
        request=request,
        llm=llm,
        gpu_available=gpu_available,
        shared_data={},
        metadata={}
    )


async def execute_plugin_pipeline(context: OnyxPluginContext, pipeline: List[str]) -> Dict[str, Any]:
    """Execute a specific plugin pipeline."""
    try:
        results = await onyx_plugin_manager.execute_plugins(context, pipeline)
        
        # Combine results
        combined_result = {
            "pipeline": pipeline,
            "results": results,
            "execution_time": datetime.now().isoformat(),
            "success": all("error" not in result for result in results.values())
        }
        
        return combined_result
        
    except Exception as e:
        return {
            "pipeline": pipeline,
            "error": str(e),
            "execution_time": datetime.now().isoformat(),
            "success": False
        }


async def validate_plugin_pipeline(pipeline: List[str]) -> Dict[str, Any]:
    """Validate a plugin pipeline."""
    try:
        all_plugins = await onyx_plugin_manager.get_all_plugins_info()
        available_plugins = {plugin.name for plugin in all_plugins}
        
        validation_result = {
            "valid": True,
            "missing_plugins": [],
            "disabled_plugins": [],
            "gpu_required_plugins": []
        }
        
        for plugin_name in pipeline:
            if plugin_name not in available_plugins:
                validation_result["valid"] = False
                validation_result["missing_plugins"].append(plugin_name)
                continue
            
            plugin_info = await onyx_plugin_manager.get_plugin_info(plugin_name)
            if plugin_info and not plugin_info.enabled:
                validation_result["disabled_plugins"].append(plugin_name)
            
            if plugin_info and plugin_info.gpu_required:
                validation_result["gpu_required_plugins"].append(plugin_name)
        
        return validation_result
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }


async def get_plugin_statistics() -> Dict[str, Any]:
    """Get plugin execution statistics."""
    try:
        status = await onyx_plugin_manager.get_manager_status()
        
        stats = {
            "total_plugins": status.total_plugins,
            "enabled_plugins": status.enabled_plugins,
            "initialized_plugins": status.initialized_plugins,
            "gpu_available": status.gpu_available,
            "plugin_categories": {},
            "execution_stats": {}
        }
        
        # Calculate category distribution
        for plugin in status.plugins:
            category = plugin["category"]
            if category not in stats["plugin_categories"]:
                stats["plugin_categories"][category] = 0
            stats["plugin_categories"][category] += 1
        
        # Calculate execution statistics
        for plugin in status.plugins:
            plugin_name = plugin["name"]
            plugin_status = plugin["status"]
            
            stats["execution_stats"][plugin_name] = {
                "execution_count": plugin_status.get("execution_count", 0),
                "total_execution_time": plugin_status.get("total_execution_time", 0.0),
                "last_execution": plugin_status.get("last_execution"),
                "average_execution_time": (
                    plugin_status.get("total_execution_time", 0.0) / 
                    max(plugin_status.get("execution_count", 1), 1)
                )
            }
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}


async def health_check() -> Dict[str, Any]:
    """Perform a health check on the plugin system."""
    try:
        status = await onyx_plugin_manager.get_manager_status()
        
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "total_plugins": status.total_plugins,
            "enabled_plugins": status.enabled_plugins,
            "initialized_plugins": status.initialized_plugins,
            "gpu_available": status.gpu_available,
            "issues": []
        }
        
        # Check for issues
        if status.total_plugins == 0:
            health["status"] = "warning"
            health["issues"].append("No plugins loaded")
        
        if status.initialized_plugins < status.enabled_plugins:
            health["status"] = "warning"
            health["issues"].append("Some plugins failed to initialize")
        
        # Check for plugins with errors
        for plugin in status.plugins:
            if plugin["status"].get("error"):
                health["status"] = "warning"
                health["issues"].append(f"Plugin {plugin['name']} has error: {plugin['status']['error']}")
        
        return health
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        } 