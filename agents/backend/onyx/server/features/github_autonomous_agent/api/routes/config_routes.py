"""
Config Routes - Rutas para gestión de configuración dinámica.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from core.config.dynamic_config import get_config_manager
from core.auth.authentication import get_current_user, require_permission, Permission, User
from config.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


class ConfigUpdateRequest(BaseModel):
    """Request para actualizar configuración."""
    config: Dict[str, Any] = Field(..., description="Configuración a actualizar")
    save: bool = Field(default=True, description="Guardar en archivo")


@router.get("/")
@handle_api_errors
@require_permission(Permission.SYSTEM_ADMIN)
async def list_configs(user: User = Depends(get_current_user)):
    """
    Listar todas las configuraciones.
    
    Args:
        user: Usuario actual
        
    Returns:
        Lista de configuraciones
    """
    manager = get_config_manager()
    configs = manager.get_all_configs()
    
    return {
        "configs": {
            name: {
                "loaded_at": config.get("_loaded_at"),
                "updated_at": config.get("_updated_at"),
                "file_path": config.get("_file_path")
            }
            for name, config in configs.items()
        }
    }


@router.get("/{config_name}")
@handle_api_errors
@require_permission(Permission.SYSTEM_ADMIN)
async def get_config(config_name: str, user: User = Depends(get_current_user)):
    """
    Obtener configuración.
    
    Args:
        config_name: Nombre de la configuración
        user: Usuario actual
        
    Returns:
        Configuración
    """
    manager = get_config_manager()
    config = manager.get_config(config_name)
    
    if not config:
        raise HTTPException(status_code=404, detail=f"Configuración no encontrada: {config_name}")
    
    # Remover metadatos internos
    clean_config = {k: v for k, v in config.items() if not k.startswith("_")}
    return clean_config


@router.put("/{config_name}")
@handle_api_errors
@require_permission(Permission.SYSTEM_ADMIN)
async def update_config(
    config_name: str,
    request: ConfigUpdateRequest,
    user: User = Depends(get_current_user)
):
    """
    Actualizar configuración.
    
    Args:
        config_name: Nombre de la configuración
        request: Nueva configuración
        user: Usuario actual
        
    Returns:
        Configuración actualizada
    """
    manager = get_config_manager()
    manager.set_config(config_name, request.config, save=request.save)
    
    return {
        "name": config_name,
        "config": request.config,
        "saved": request.save,
        "message": f"Configuración {config_name} actualizada"
    }


@router.post("/{config_name}/reload")
@handle_api_errors
@require_permission(Permission.SYSTEM_ADMIN)
async def reload_config(config_name: str, user: User = Depends(get_current_user)):
    """
    Recargar configuración desde archivo.
    
    Args:
        config_name: Nombre de la configuración
        user: Usuario actual
        
    Returns:
        Configuración recargada
    """
    manager = get_config_manager()
    config = manager.load_config(config_name)
    
    if not config:
        raise HTTPException(status_code=404, detail=f"Configuración no encontrada: {config_name}")
    
    return {
        "name": config_name,
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "message": f"Configuración {config_name} recargada"
    }



