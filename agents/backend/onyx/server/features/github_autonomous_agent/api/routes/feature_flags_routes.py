"""
Feature Flags Routes - Rutas para gestión de feature flags.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Header
from typing import Optional, List
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from core.services import FeatureFlagService
from config.logging_config import get_logger
from config.di_setup import get_service
from datetime import datetime

router = APIRouter()
logger = get_logger(__name__)


class UpdateFeatureFlagRequest(BaseModel):
    """Request para actualizar feature flag."""
    enabled: Optional[bool] = None
    rollout_percentage: Optional[int] = Field(None, ge=0, le=100)
    target_users: Optional[List[str]] = None


@router.get("/flags")
@handle_api_errors
async def list_feature_flags(
    user_id: Optional[str] = Query(None, description="ID de usuario para verificar flags")
):
    """
    Listar todos los feature flags.
    
    Args:
        user_id: ID de usuario (opcional)
        
    Returns:
        Lista de feature flags con estado para el usuario
    """
    try:
        feature_flags: FeatureFlagService = get_service("feature_flags_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Feature flags service no disponible")
    
    flags = feature_flags.list_flags()
    
    return {
        "total": len(flags),
        "flags": [
            {
                "name": flag.name,
                "enabled": flag.enabled,
                "enabled_for_user": flag.is_enabled_for(user_id),
                "description": flag.description,
                "rollout_percentage": flag.rollout_percentage,
                "target_users_count": len(flag.target_users),
                "created_at": flag.created_at.isoformat(),
                "updated_at": flag.updated_at.isoformat()
            }
            for flag in flags
        ]
    }


@router.get("/flags/{flag_name}")
@handle_api_errors
async def get_feature_flag(
    flag_name: str,
    user_id: Optional[str] = Query(None, description="ID de usuario")
):
    """
    Obtener estado de un feature flag.
    
    Args:
        flag_name: Nombre del flag
        user_id: ID de usuario (opcional)
        
    Returns:
        Estado del flag
    """
    try:
        feature_flags: FeatureFlagService = get_service("feature_flags_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Feature flags service no disponible")
    
    flag = feature_flags.get_flag(flag_name)
    if not flag:
        raise HTTPException(status_code=404, detail=f"Feature flag '{flag_name}' no encontrado")
    
    return {
        "name": flag.name,
        "enabled": flag.enabled,
        "enabled_for_user": flag.is_enabled_for(user_id),
        "description": flag.description,
        "rollout_percentage": flag.rollout_percentage,
        "target_users": flag.target_users,
        "metadata": flag.metadata,
        "created_at": flag.created_at.isoformat(),
        "updated_at": flag.updated_at.isoformat()
    }


@router.post("/flags/{flag_name}/enable")
@handle_api_errors
async def enable_feature_flag(flag_name: str):
    """
    Habilitar feature flag.
    
    Args:
        flag_name: Nombre del flag
        
    Returns:
        Confirmación
    """
    try:
        feature_flags: FeatureFlagService = get_service("feature_flags_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Feature flags service no disponible")
    
    success = feature_flags.enable(flag_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Feature flag '{flag_name}' no encontrado")
    
    return {"message": f"Feature flag '{flag_name}' habilitado"}


@router.post("/flags/{flag_name}/disable")
@handle_api_errors
async def disable_feature_flag(flag_name: str):
    """
    Deshabilitar feature flag.
    
    Args:
        flag_name: Nombre del flag
        
    Returns:
        Confirmación
    """
    try:
        feature_flags: FeatureFlagService = get_service("feature_flags_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Feature flags service no disponible")
    
    success = feature_flags.disable(flag_name)
    if not success:
        raise HTTPException(status_code=404, detail=f"Feature flag '{flag_name}' no encontrado")
    
    return {"message": f"Feature flag '{flag_name}' deshabilitado"}


@router.patch("/flags/{flag_name}")
@handle_api_errors
async def update_feature_flag(
    flag_name: str,
    request: UpdateFeatureFlagRequest
):
    """
    Actualizar feature flag.
    
    Args:
        flag_name: Nombre del flag
        request: Datos a actualizar
        
    Returns:
        Flag actualizado
    """
    try:
        feature_flags: FeatureFlagService = get_service("feature_flags_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Feature flags service no disponible")
    
    flag = feature_flags.get_flag(flag_name)
    if not flag:
        raise HTTPException(status_code=404, detail=f"Feature flag '{flag_name}' no encontrado")
    
    if request.enabled is not None:
        flag.enabled = request.enabled
    
    if request.rollout_percentage is not None:
        flag.rollout_percentage = request.rollout_percentage
    
    if request.target_users is not None:
        flag.target_users = request.target_users
    
    flag.updated_at = datetime.now()
    
    return {
        "name": flag.name,
        "enabled": flag.enabled,
        "rollout_percentage": flag.rollout_percentage,
        "target_users": flag.target_users
    }


@router.get("/flags/stats")
@handle_api_errors
async def get_feature_flags_stats():
    """
    Obtener estadísticas de feature flags.
    
    Returns:
        Estadísticas
    """
    try:
        feature_flags: FeatureFlagService = get_service("feature_flags_service")
    except Exception:
        raise HTTPException(status_code=503, detail="Feature flags service no disponible")
    
    return feature_flags.get_stats()

