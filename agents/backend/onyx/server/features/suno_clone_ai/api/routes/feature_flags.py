"""
API de Feature Flags

Endpoints para:
- Listar feature flags
- Crear feature flags
- Actualizar feature flags
- Verificar estado de flags
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from middleware.auth_middleware import require_role, get_current_user
from utils.feature_flags import (
    get_feature_flag_service,
    FlagType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/feature-flags",
    tags=["feature-flags"]
)


@router.get("/check/{flag_name}")
async def check_feature_flag(
    flag_name: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Verifica si un feature flag está habilitado para el usuario actual.
    """
    try:
        service = get_feature_flag_service()
        user_id = None
        if current_user:
            user_id = current_user.get("user_id") or current_user.get("sub")
        
        enabled = service.is_enabled(flag_name, user_id=user_id)
        
        return {
            "flag_name": flag_name,
            "enabled": enabled,
            "user_id": user_id
        }
    except Exception as e:
        logger.error(f"Error checking feature flag: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking feature flag: {str(e)}"
        )


@router.get("/list")
async def list_feature_flags(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Lista todos los feature flags disponibles.
    """
    try:
        service = get_feature_flag_service()
        flags = service.list_flags()
        
        # Si hay usuario, verificar estado para cada flag
        user_id = None
        if current_user:
            user_id = current_user.get("user_id") or current_user.get("sub")
        
        if user_id:
            for flag in flags:
                flag["enabled_for_user"] = service.is_enabled(
                    flag["name"],
                    user_id=user_id
                )
        
        return {
            "flags": flags,
            "total": len(flags)
        }
    except Exception as e:
        logger.error(f"Error listing feature flags: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing feature flags: {str(e)}"
        )


@router.post("/create")
async def create_feature_flag(
    name: str = Body(..., description="Nombre del flag"),
    flag_type: str = Body(..., description="Tipo de flag (boolean, percentage, user_list, attribute)"),
    enabled: bool = Body(True, description="Si está habilitado"),
    description: Optional[str] = Body(None, description="Descripción"),
    percentage: int = Body(100, ge=0, le=100, description="Porcentaje para flags de porcentaje"),
    user_list: Optional[List[str]] = Body(None, description="Lista de usuarios para flags de lista"),
    attributes: Optional[Dict[str, Any]] = Body(None, description="Atributos para flags por atributos"),
    current_user: Dict[str, Any] = Depends(require_role("admin"))
) -> Dict[str, Any]:
    """
    Crea un nuevo feature flag (requiere rol admin).
    """
    try:
        # Validar tipo
        try:
            flag_type_enum = FlagType(flag_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid flag type: {flag_type}"
            )
        
        service = get_feature_flag_service()
        flag = service.create_flag(
            name=name,
            flag_type=flag_type_enum,
            enabled=enabled,
            description=description,
            percentage=percentage,
            user_list=user_list,
            attributes=attributes
        )
        
        return {
            "message": "Feature flag created successfully",
            "flag": {
                "name": flag.name,
                "type": flag.flag_type.value,
                "enabled": flag.enabled
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating feature flag: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating feature flag: {str(e)}"
        )


@router.put("/{flag_name}")
async def update_feature_flag(
    flag_name: str,
    enabled: Optional[bool] = Body(None, description="Nuevo estado"),
    percentage: Optional[int] = Body(None, ge=0, le=100, description="Nuevo porcentaje"),
    user_list: Optional[List[str]] = Body(None, description="Nueva lista de usuarios"),
    attributes: Optional[Dict[str, Any]] = Body(None, description="Nuevos atributos"),
    current_user: Dict[str, Any] = Depends(require_role("admin"))
) -> Dict[str, Any]:
    """
    Actualiza un feature flag (requiere rol admin).
    """
    try:
        service = get_feature_flag_service()
        success = service.update_flag(
            flag_name=flag_name,
            enabled=enabled,
            percentage=percentage,
            user_list=user_list,
            attributes=attributes
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Feature flag {flag_name} not found"
            )
        
        return {
            "message": f"Feature flag {flag_name} updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating feature flag: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating feature flag: {str(e)}"
        )


@router.get("/stats")
async def get_feature_flag_stats(
    current_user: Dict[str, Any] = Depends(require_role("admin"))
) -> Dict[str, Any]:
    """
    Obtiene estadísticas de feature flags (requiere rol admin).
    """
    try:
        service = get_feature_flag_service()
        stats = service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting feature flag stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving stats: {str(e)}"
        )

