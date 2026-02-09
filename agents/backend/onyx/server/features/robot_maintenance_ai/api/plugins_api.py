"""
Plugin management API endpoints.
Refactored to use BaseRouter for reduced duplication.
"""

from fastapi import Depends
from pydantic import BaseModel, Field
from typing import Dict, Any

from .base_router import BaseRouter
from .exceptions import NotFoundError, ValidationError
from ..core.plugin_system import PluginManager, Plugin

# Create base router instance
base = BaseRouter(
    prefix="/api/plugins",
    tags=["Plugins"],
    require_authentication=True,
    require_rate_limit=False
)

router = base.router

plugin_manager = PluginManager()


class PluginConfig(BaseModel):
    """Plugin configuration."""
    config: Dict[str, Any] = Field(default_factory=dict, description="Plugin configuration")


class RegisterPluginRequest(BaseModel):
    """Request to register a plugin."""
    name: str = Field(..., description="Plugin name")
    version: str = Field("1.0.0", description="Plugin version")
    config: Optional[Dict[str, Any]] = Field(None, description="Plugin configuration")


@router.get("/list")
@base.timed_endpoint("list_plugins")
async def list_plugins(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    List all registered plugins.
    """
    base.log_request("list_plugins")
    
    plugins = plugin_manager.list_plugins()
    return base.success({
        "plugins": plugins,
        "count": len(plugins)
    })


@router.get("/{plugin_name}")
@base.timed_endpoint("get_plugin_info")
async def get_plugin_info(
    plugin_name: str,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Get information about a specific plugin.
    """
    base.log_request("get_plugin_info", plugin_name=plugin_name)
    
    plugin = plugin_manager.get_plugin(plugin_name)
    if not plugin:
        raise NotFoundError(f"Plugin '{plugin_name}' not found")
    
    return base.success({
        "name": plugin.name,
        "version": plugin.version,
        "enabled": plugin.enabled
    })


@router.post("/register")
@base.timed_endpoint("register_plugin")
async def register_plugin(
    request: RegisterPluginRequest,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Register a new plugin.
    
    Note: This is a simplified version. In production, plugins would be
    loaded from files or modules.
    """
    base.log_request("register_plugin", plugin_name=request.name)
    
    # Check if plugin already exists
    if plugin_manager.get_plugin(request.name):
        raise ValidationError(f"Plugin '{request.name}' already registered")
    
    # Create a simple plugin instance
    class SimplePlugin(Plugin):
        def initialize(self, config: Dict[str, Any]) -> bool:
            return True
        
        def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
            return {"processed": True, "data": data}
    
    plugin = SimplePlugin(request.name, request.version)
    
    if request.config:
        plugin.initialize(request.config)
    
    plugin_manager.register_plugin(plugin)
    
    return base.success({
        "name": plugin.name,
        "version": plugin.version,
        "message": f"Plugin '{request.name}' registered successfully"
    })


@router.delete("/{plugin_name}")
@base.timed_endpoint("unregister_plugin")
async def unregister_plugin(
    plugin_name: str,
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Unregister a plugin.
    """
    base.log_request("unregister_plugin", plugin_name=plugin_name)
    
    plugin = plugin_manager.get_plugin(plugin_name)
    if not plugin:
        raise NotFoundError(f"Plugin '{plugin_name}' not found")
    
    plugin_manager.unregister_plugin(plugin_name)
    
    return base.success(None, message=f"Plugin '{plugin_name}' unregistered successfully")


@router.post("/{plugin_name}/execute")
@base.timed_endpoint("execute_plugin")
async def execute_plugin(
    plugin_name: str,
    data: Dict[str, Any] = Field(..., description="Data to process"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Execute a plugin with given data.
    """
    base.log_request("execute_plugin", plugin_name=plugin_name)
    
    plugin = plugin_manager.get_plugin(plugin_name)
    if not plugin:
        raise NotFoundError(f"Plugin '{plugin_name}' not found")
    
    if not plugin.enabled:
        raise ValidationError(f"Plugin '{plugin_name}' is disabled")
    
    result = plugin.execute(data)
    
    return base.success({
        "plugin": plugin_name,
        "result": result
    })


@router.get("/hooks/list")
@base.timed_endpoint("list_hooks")
async def list_hooks(
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    List all registered hooks.
    """
    base.log_request("list_hooks")
    
    hooks = list(plugin_manager.hooks.keys())
    hook_info = {}
    for hook_name in hooks:
        hook_info[hook_name] = {
            "name": hook_name,
            "callbacks": len(plugin_manager.hooks[hook_name])
        }
    
    return base.success({
        "hooks": hook_info,
        "count": len(hooks)
    })


@router.post("/hooks/{hook_name}/execute")
@base.timed_endpoint("execute_hook")
async def execute_hook(
    hook_name: str,
    data: Dict[str, Any] = Field(None, description="Data to pass to hook"),
    _: Dict = Depends(base.get_auth_dependency())
) -> Dict[str, Any]:
    """
    Execute a hook with given data.
    """
    base.log_request("execute_hook", hook_name=hook_name)
    
    results = plugin_manager.execute_hook(hook_name, data)
    
    return base.success({
        "hook": hook_name,
        "results": results,
        "count": len(results)
    })




