"""
Rutas para Prompt Engineering
================================

Endpoints para prompt engineering.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.prompt_engineering import (
    get_prompt_engineering,
    PromptEngineering,
    PromptType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/prompt-engineering",
    tags=["Prompt Engineering"]
)


class CreatePromptRequest(BaseModel):
    """Request para crear prompt"""
    base_prompt: str = Field(..., description="Prompt base")
    prompt_type: str = Field("zero_shot", description="Tipo de prompt")
    examples: Optional[List[str]] = Field(None, description="Ejemplos")


@router.post("/prompts")
async def create_prompt(
    request: CreatePromptRequest,
    system: PromptEngineering = Depends(get_prompt_engineering)
):
    """Crear prompt optimizado"""
    try:
        prompt_type = PromptType(request.prompt_type)
        prompt = system.create_prompt(
            request.base_prompt,
            prompt_type,
            request.examples
        )
        
        return {
            "prompt_id": prompt.prompt_id,
            "prompt_text": prompt.prompt_text,
            "prompt_type": prompt.prompt_type.value,
            "performance_score": prompt.performance_score
        }
    except Exception as e:
        logger.error(f"Error creando prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prompts/{prompt_id}/evaluate")
async def evaluate_prompt(
    prompt_id: str,
    test_cases: List[Dict[str, Any]] = Field(..., description="Casos de prueba"),
    system: PromptEngineering = Depends(get_prompt_engineering)
):
    """Evaluar rendimiento de prompt"""
    try:
        evaluation = system.evaluate_prompt(prompt_id, test_cases)
        
        return evaluation
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error evaluando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_prompts(
    prompt_ids: List[str] = Field(..., description="IDs de prompts"),
    test_cases: List[Dict[str, Any]] = Field(..., description="Casos de prueba"),
    system: PromptEngineering = Depends(get_prompt_engineering)
):
    """Comparar múltiples prompts"""
    try:
        comparison = system.compare_prompts(prompt_ids, test_cases)
        
        return comparison
    except Exception as e:
        logger.error(f"Error comparando: {e}")
        raise HTTPException(status_code=500, detail=str(e))



