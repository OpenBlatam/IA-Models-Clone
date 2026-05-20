"""
Rutas para Advanced Prompt Optimization
==========================================

Endpoints para optimización avanzada de prompts.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.advanced_prompt_optimization import (
    get_advanced_prompt_optimization,
    AdvancedPromptOptimization,
    OptimizationMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/advanced-prompt-optimization",
    tags=["Advanced Prompt Optimization"]
)


class OptimizePromptRequest(BaseModel):
    """Request para optimizar prompt"""
    original_prompt: str = Field(..., description="Prompt original")
    method: str = Field("genetic_algorithm", description="Método")
    iterations: int = Field(10, description="Iteraciones")


@router.post("/optimize")
async def optimize_prompt(
    request: OptimizePromptRequest,
    system: AdvancedPromptOptimization = Depends(get_advanced_prompt_optimization)
):
    """Optimizar prompt"""
    try:
        method = OptimizationMethod(request.method)
        optimized = system.optimize_prompt(
            request.original_prompt,
            method,
            request.iterations
        )
        
        return {
            "prompt_id": optimized.prompt_id,
            "original_prompt": optimized.original_prompt,
            "optimized_prompt": optimized.optimized_prompt,
            "performance_improvement": optimized.performance_improvement,
            "method": optimized.method.value
        }
    except Exception as e:
        logger.error(f"Error optimizando prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate_prompt(
    prompt: str = Field(..., description="Prompt"),
    test_cases: List[Dict[str, Any]] = Field(..., description="Casos de prueba"),
    system: AdvancedPromptOptimization = Depends(get_advanced_prompt_optimization)
):
    """Evaluar rendimiento de prompt"""
    try:
        performance = system.evaluate_prompt_performance(prompt, test_cases)
        
        return performance
    except Exception as e:
        logger.error(f"Error evaluando prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_prompts(
    prompts: List[str] = Field(..., description="Prompts"),
    test_cases: List[Dict[str, Any]] = Field(..., description="Casos de prueba"),
    system: AdvancedPromptOptimization = Depends(get_advanced_prompt_optimization)
):
    """Comparar múltiples prompts"""
    try:
        comparison = system.compare_prompts(prompts, test_cases)
        
        return comparison
    except Exception as e:
        logger.error(f"Error comparando prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


