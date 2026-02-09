"""
Models Router - Endpoints for model registry.

This module provides REST API endpoints for managing
models in the registry.
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends

from ..models import ModelRegisterRequest, ModelResponse, ErrorResponse
from ..auth import verify_token
from core.model_registry import ModelRegistry, ModelMetadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["models"])

# Initialize manager (should be injected in production)
model_registry = ModelRegistry()


@router.get("", response_model=dict)
async def list_models(
    status: Optional[str] = None,
    token: str = Depends(verify_token),
):
    """
    List models in registry.
    
    Args:
        status: Filter by status (optional)
        token: Authentication token
    
    Returns:
        Dictionary with models list
    """
    try:
        models = model_registry.list_models()
        if status:
            models = [m for m in models if m.status.value == status]
        return {"models": [m.to_dict() if hasattr(m, 'to_dict') else m for m in models]}
    except Exception as e:
        logger.exception("Error listing models")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=dict)
async def register_model(
    request: ModelRegisterRequest,
    token: str = Depends(verify_token),
):
    """
    Register a new model.
    
    Args:
        request: Model registration request
        token: Authentication token
    
    Returns:
        Model version dictionary
    """
    try:
        metadata = ModelMetadata(
            name=request.name,
            version=request.version,
            description=request.description,
            architecture=request.architecture,
            parameters=request.parameters,
            tags=request.tags,
        )
        version = model_registry.register_model(metadata, request.path)
        return version.to_dict() if hasattr(version, 'to_dict') else version
    except Exception as e:
        logger.exception("Error registering model")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{model_name}", response_model=dict)
async def get_model(
    model_name: str,
    version: Optional[str] = None,
    token: str = Depends(verify_token),
):
    """
    Get model version.
    
    Args:
        model_name: Model name
        version: Optional version (defaults to latest)
        token: Authentication token
    
    Returns:
        Model dictionary
    
    Raises:
        HTTPException: If model not found
    """
    try:
        model = model_registry.get_model(model_name, version)
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        return model.to_dict() if hasattr(model, 'to_dict') else model
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error getting model")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/best/{benchmark_name}", response_model=dict)
async def get_best_models(
    benchmark_name: str,
    top_k: int = 5,
    token: str = Depends(verify_token),
):
    """
    Get best models for a benchmark.
    
    Args:
        benchmark_name: Name of the benchmark
        top_k: Number of top models to return
        token: Authentication token
    
    Returns:
        Dictionary with top models list
    """
    try:
        models = model_registry.get_best_models(benchmark_name, top_k)
        return {"models": [m.to_dict() if hasattr(m, 'to_dict') else m for m in models]}
    except Exception as e:
        logger.exception("Error getting best models")
        raise HTTPException(status_code=500, detail=str(e))












