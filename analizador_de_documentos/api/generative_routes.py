"""
Rutas para Generative AI
==========================

Endpoints para generative AI.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.generative_ai import (
    get_generative_ai,
    GenerativeAI,
    GenerationType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/generative-ai",
    tags=["Generative AI"]
)


class GenerateRequest(BaseModel):
    """Request para generar contenido"""
    prompt: str = Field(..., description="Prompt")
    generation_type: str = Field("text", description="Tipo de generación")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parámetros")


@router.post("/generate")
async def generate_content(
    request: GenerateRequest,
    system: GenerativeAI = Depends(get_generative_ai)
):
    """Generar contenido"""
    try:
        gen_type = GenerationType(request.generation_type)
        result = system.generate(
            request.prompt,
            gen_type,
            request.parameters
        )
        
        return {
            "request_id": result.request_id,
            "generated_content": result.generated_content,
            "confidence": result.confidence,
            "tokens_used": result.tokens_used
        }
    except Exception as e:
        logger.error(f"Error generando contenido: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-batch")
async def generate_batch(
    prompts: List[str] = Field(..., description="Prompts"),
    generation_type: str = Field("text", description="Tipo de generación"),
    system: GenerativeAI = Depends(get_generative_ai)
):
    """Generar contenido en batch"""
    try:
        gen_type = GenerationType(generation_type)
        results = system.generate_batch(prompts, gen_type)
        
        return {
            "num_results": len(results),
            "results": [
                {
                    "request_id": r.request_id,
                    "generated_content": r.generated_content,
                    "confidence": r.confidence
                }
                for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Error generando batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{request_id}")
async def get_result(
    request_id: str,
    system: GenerativeAI = Depends(get_generative_ai)
):
    """Obtener resultado de generación"""
    if request_id not in system.results:
        raise HTTPException(status_code=404, detail="Resultado no encontrado")
    
    result = system.results[request_id]
    
    return {
        "request_id": result.request_id,
        "generated_content": result.generated_content,
        "confidence": result.confidence,
        "tokens_used": result.tokens_used
    }



