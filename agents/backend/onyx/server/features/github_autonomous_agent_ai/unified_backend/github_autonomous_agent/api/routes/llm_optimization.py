"""
LLM Optimization Routes - Rutas para optimización de prompts y modelos.

Incluye:
- Optimización de prompts
- Análisis de prompts
- Configuración de fallback
- Estadísticas de optimización
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from config.logging_config import get_logger
from core.services.llm import (
    get_prompt_optimizer,
    get_model_fallback_system,
    OptimizationGoal,
    FallbackStrategy
)
from core.constants import ErrorMessages

logger = get_logger(__name__)

router = APIRouter(
    prefix="/llm/optimization",
    tags=["LLM Optimization"],
    responses={
        400: {"description": "Bad request"},
        500: {"description": "Internal server error"}
    }
)


class OptimizePromptRequest(BaseModel):
    """Request para optimizar un prompt."""
    prompt: str = Field(..., description="Prompt a optimizar")
    goal: str = Field("balanced", description="Objetivo de optimización (clarity, efficiency, quality, cost, balanced)")


class AnalyzePromptRequest(BaseModel):
    """Request para analizar un prompt."""
    prompt: str = Field(..., description="Prompt a analizar")
    goal: Optional[str] = Field(None, description="Objetivo de optimización")


class ConfigureFallbackRequest(BaseModel):
    """Request para configurar fallback."""
    primary_model: str = Field(..., description="Modelo principal")
    fallback_models: Optional[list[str]] = Field(None, description="Modelos de fallback (opcional)")
    strategy: str = Field("balanced", description="Estrategia de fallback")
    max_fallbacks: int = Field(3, description="Máximo de fallbacks")
    retry_on_error: bool = Field(True, description="Reintentar en errores")
    retry_on_timeout: bool = Field(True, description="Reintentar en timeouts")
    retry_on_rate_limit: bool = Field(True, description="Reintentar en rate limits")


@router.post("/prompt/optimize")
@handle_api_errors
async def optimize_prompt(request: OptimizePromptRequest):
    """
    Optimizar un prompt automáticamente.
    
    Args:
        request: Request con prompt y objetivo
        
    Returns:
        Prompt optimizado con análisis
    """
    try:
        goal = OptimizationGoal(request.goal)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Objetivo inválido: {request.goal}"
        )
    
    optimizer = get_prompt_optimizer()
    analysis = optimizer.analyze_prompt(request.prompt, goal)
    
    return {
        "success": True,
        "analysis": analysis.to_dict()
    }


@router.post("/prompt/analyze")
@handle_api_errors
async def analyze_prompt(request: AnalyzePromptRequest):
    """
    Analizar un prompt y obtener sugerencias.
    
    Args:
        request: Request con prompt
        
    Returns:
        Análisis y sugerencias de mejora
    """
    optimizer = get_prompt_optimizer()
    
    goal = OptimizationGoal.BALANCED
    if request.goal:
        try:
            goal = OptimizationGoal(request.goal)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Objetivo inválido: {request.goal}"
            )
    
    analysis = optimizer.analyze_prompt(request.prompt, goal)
    suggestions = optimizer.suggest_improvements(request.prompt)
    
    return {
        "success": True,
        "analysis": analysis.to_dict(),
        "suggestions": suggestions
    }


@router.post("/fallback/configure")
@handle_api_errors
async def configure_fallback(request: ConfigureFallbackRequest):
    """
    Configurar sistema de fallback para un modelo.
    
    Args:
        request: Request con configuración de fallback
        
    Returns:
        Confirmación de configuración
    """
    try:
        strategy = FallbackStrategy(request.strategy)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Estrategia inválida: {request.strategy}"
        )
    
    fallback_system = get_model_fallback_system()
    
    fallback_system.configure_fallback(
        primary_model=request.primary_model,
        fallback_models=request.fallback_models,
        strategy=strategy,
        max_fallbacks=request.max_fallbacks,
        retry_on_error=request.retry_on_error,
        retry_on_timeout=request.retry_on_timeout,
        retry_on_rate_limit=request.retry_on_rate_limit
    )
    
    return {
        "success": True,
        "message": f"Fallback configurado para {request.primary_model}",
        "config": fallback_system.fallback_configs[request.primary_model].to_dict()
    }


@router.get("/fallback/stats")
@handle_api_errors
async def get_fallback_stats(model: Optional[str] = None):
    """
    Obtener estadísticas de fallbacks.
    
    Args:
        model: Modelo específico (opcional)
        
    Returns:
        Estadísticas de fallbacks
    """
    fallback_system = get_model_fallback_system()
    stats = fallback_system.get_fallback_stats(model=model)
    
    return {
        "success": True,
        "stats": stats,
        "model": model or "all"
    }


@router.get("/fallback/models/{primary_model}")
@handle_api_errors
async def get_fallback_models(
    primary_model: str,
    error_type: Optional[str] = None
):
    """
    Obtener modelos de fallback para un modelo.
    
    Args:
        primary_model: Modelo principal
        error_type: Tipo de error (opcional)
        
    Returns:
        Lista de modelos de fallback
    """
    fallback_system = get_model_fallback_system()
    fallback_models = fallback_system.get_fallback_models(primary_model, error_type)
    
    return {
        "success": True,
        "primary_model": primary_model,
        "fallback_models": fallback_models,
        "count": len(fallback_models)
    }
