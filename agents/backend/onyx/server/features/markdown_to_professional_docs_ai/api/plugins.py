"""Plugin system endpoints"""
from fastapi import APIRouter
from utils.plugin_system import get_plugin_manager

router = APIRouter(prefix="/plugins", tags=["Plugins"])


@router.get("")
async def list_plugins():
    """List all plugins"""
    plugin_manager = get_plugin_manager()
    plugins = plugin_manager.list_plugins()
    
    return {
        "plugins": plugins,
        "total": len(plugins)
    }


@router.get("/{plugin_name}")
async def get_plugin_info(plugin_name: str):
    """Get plugin information"""
    plugin_manager = get_plugin_manager()
    plugin = plugin_manager.get_plugin(plugin_name)
    
    if not plugin:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    return {
        "name": plugin.name,
        "version": plugin.version,
        "description": plugin.description,
        "enabled": plugin.enabled
    }

