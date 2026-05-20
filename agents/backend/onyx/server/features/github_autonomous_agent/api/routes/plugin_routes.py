"""
Plugin Routes - Rutas para gestión de plugins.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from core.plugins.plugin_system import get_plugin_manager
from core.auth.authentication import get_current_user, require_permission, Permission, User
from config.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


class PluginEnableRequest(BaseModel):
    """Request para habilitar plugin."""
    enabled: bool = Field(default=True)


@router.get("/")
@handle_api_errors
@require_permission(Permission.SYSTEM_ADMIN)
async def list_plugins(user: User = Depends(get_current_user)):
    """
    Listar todos los plugins.
    
    Args:
        user: Usuario actual
        
    Returns:
        Lista de plugins
    """
    manager = get_plugin_manager()
    return {"plugins": manager.get_plugins()}


@router.get("/{plugin_name}")
@handle_api_errors
@require_permission(Permission.SYSTEM_ADMIN)
async def get_plugin(plugin_name: str, user: User = Depends(get_current_user)):
    """
    Obtener información de un plugin.
    
    Args:
        plugin_name: Nombre del plugin
        user: Usuario actual
        
    Returns:
        Información del plugin
    """
    manager = get_plugin_manager()
    plugin = manager.get_plugin(plugin_name)
    
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin no encontrado: {plugin_name}")
    
    return plugin


@router.post("/{plugin_name}/enable")
@handle_api_errors
@require_permission(Permission.SYSTEM_ADMIN)
async def enable_plugin(
    plugin_name: str,
    request: PluginEnableRequest,
    user: User = Depends(get_current_user)
):
    """
    Habilitar o deshabilitar plugin.
    
    Args:
        plugin_name: Nombre del plugin
        request: Request con estado
        user: Usuario actual
        
    Returns:
        Estado del plugin
    """
    manager = get_plugin_manager()
    
    if request.enabled:
        success = manager.enable_plugin(plugin_name)
    else:
        success = manager.disable_plugin(plugin_name)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"No se pudo cambiar estado del plugin: {plugin_name}")
    
    return {
        "plugin": plugin_name,
        "enabled": request.enabled,
        "message": f"Plugin {plugin_name} {'habilitado' if request.enabled else 'deshabilitado'}"
    }



