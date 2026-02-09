"""
Models router for Multi-Model API
Handles model-related endpoints
"""

from fastapi import APIRouter, Depends, Path
from typing import List

from ...api.schemas import (
    ModelsListResponse,
    ModelStatus,
    ModelType
)
from ...api.dependencies import get_model_repository
from ...core.repositories import ModelRepository
from ...core.models import ModelMetadata

router = APIRouter(prefix="/multi-model/models", tags=["Models"])


@router.get("", response_model=ModelsListResponse)
async def list_models(
    repository: ModelRepository = Depends(get_model_repository)
):
    """
    List all available models with status
    
    Returns:
    - List of all registered models
    - Availability status
    - Performance metrics (success rate, latency)
    - Last used timestamp
    """
    available_models = repository.get_available_models()
    
    model_statuses = []
    # Note: This requires access to all models, not just available ones
    # For now, we'll use available models. In full implementation,
    # repository should have a method to get all models
    for model_meta in available_models:
        model_statuses.append(ModelStatus(
            model_type=model_meta.model_type,
            is_available=model_meta.is_available,
            is_enabled=model_meta.is_available,
            multiplier=1,
            last_used=model_meta.last_used.isoformat() if model_meta.last_used else None,
            success_rate=round(model_meta.success_rate, 2),
            avg_latency_ms=round(model_meta.avg_latency_ms, 2)
        ))
    
    return ModelsListResponse(
        models=model_statuses,
        total_available=len(available_models),
        total_enabled=len([m for m in model_statuses if m.is_enabled])
    )


@router.get("/{model_type}/health")
async def get_model_health(
    model_type: ModelType = Path(..., description="Model type identifier"),
    repository: ModelRepository = Depends(get_model_repository)
):
    """
    Get health metrics for a specific model
    
    Returns detailed health information including:
    - Availability status
    - Success rate and error count
    - Latency metrics (average, p95)
    - Circuit breaker state
    - Last used timestamp
    """
    health = repository.get_model_health(model_type)
    if "error" in health:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=health["error"])
    return health




