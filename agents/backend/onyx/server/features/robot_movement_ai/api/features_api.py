"""
Features API Endpoints
======================

Endpoints para feature flags y experimentos.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from ..core.feature_flags import get_feature_flag_manager, FeatureStatus
from ..core.experiment_manager import get_experiment_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/features", tags=["features"])


@router.get("/flags")
async def list_feature_flags() -> Dict[str, Any]:
    """Listar todos los feature flags."""
    try:
        manager = get_feature_flag_manager()
        flags = manager.list_flags()
        return {
            "flags": [
                {
                    "flag_id": f.flag_id,
                    "name": f.name,
                    "description": f.description,
                    "status": f.status.value,
                    "enabled": f.enabled,
                    "rollout_percentage": f.rollout_percentage
                }
                for f in flags
            ],
            "count": len(flags)
        }
    except Exception as e:
        logger.error(f"Error listing feature flags: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/flags/{flag_id}/check")
async def check_feature_flag(
    flag_id: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Verificar si feature flag está habilitado."""
    try:
        manager = get_feature_flag_manager()
        enabled = manager.is_enabled(flag_id, context=context)
        return {
            "flag_id": flag_id,
            "enabled": enabled
        }
    except Exception as e:
        logger.error(f"Error checking feature flag: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flags/{flag_id}/enable")
async def enable_feature_flag(flag_id: str) -> Dict[str, Any]:
    """Habilitar feature flag."""
    try:
        manager = get_feature_flag_manager()
        if manager.enable_flag(flag_id):
            return {"message": f"Feature flag {flag_id} enabled"}
        raise HTTPException(status_code=404, detail="Feature flag not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling feature flag: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flags/{flag_id}/disable")
async def disable_feature_flag(flag_id: str) -> Dict[str, Any]:
    """Deshabilitar feature flag."""
    try:
        manager = get_feature_flag_manager()
        if manager.disable_flag(flag_id):
            return {"message": f"Feature flag {flag_id} disabled"}
        raise HTTPException(status_code=404, detail="Feature flag not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling feature flag: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments")
async def list_experiments() -> Dict[str, Any]:
    """Listar todos los experimentos."""
    try:
        manager = get_experiment_manager()
        experiments = manager.list_experiments()
        return {
            "experiments": [
                {
                    "experiment_id": e.experiment_id,
                    "name": e.name,
                    "description": e.description,
                    "enabled": e.enabled,
                    "variants_count": len(e.variants)
                }
                for e in experiments
            ],
            "count": len(experiments)
        }
    except Exception as e:
        logger.error(f"Error listing experiments: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/assign")
async def assign_variant(
    experiment_id: str,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Asignar variante de experimento."""
    try:
        manager = get_experiment_manager()
        variant_id = manager.assign_variant(experiment_id, user_id=user_id)
        
        if variant_id:
            return {
                "experiment_id": experiment_id,
                "variant_id": variant_id,
                "user_id": user_id
            }
        raise HTTPException(status_code=404, detail="Experiment not found or not enabled")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error assigning variant: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments/{experiment_id}/statistics")
async def get_experiment_statistics(experiment_id: str) -> Dict[str, Any]:
    """Obtener estadísticas de experimento."""
    try:
        manager = get_experiment_manager()
        stats = manager.get_experiment_statistics(experiment_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting experiment statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






