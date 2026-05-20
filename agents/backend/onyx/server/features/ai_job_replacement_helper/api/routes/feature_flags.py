"""
Feature Flags endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.feature_flags import FeatureFlagsService, FeatureFlagStatus

router = APIRouter()
flags_service = FeatureFlagsService()


@router.post("/create")
async def create_feature_flag(
    name: str,
    description: str,
    status: str = "disabled"
) -> Dict[str, Any]:
    """Crear feature flag"""
    try:
        status_enum = FeatureFlagStatus(status)
        flag = flags_service.create_feature_flag(name, description, status_enum)
        return {
            "id": flag.id,
            "name": flag.name,
            "status": flag.status.value,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluate/{flag_id}")
async def evaluate_flag(
    flag_id: str,
    user_id: str,
    user_segments: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Evaluar feature flag para usuario"""
    try:
        evaluation = flags_service.evaluate_flag(flag_id, user_id, user_segments)
        return {
            "flag_id": evaluation.flag_id,
            "user_id": evaluation.user_id,
            "enabled": evaluation.enabled,
            "reason": evaluation.reason,
            "variant": evaluation.variant,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all")
async def get_all_flags() -> Dict[str, Any]:
    """Obtener todos los feature flags"""
    try:
        flags = flags_service.get_all_flags()
        return {
            "flags": flags,
            "total": len(flags),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




