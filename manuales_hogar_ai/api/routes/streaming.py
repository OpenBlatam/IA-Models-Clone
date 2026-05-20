"""
Rutas de Streaming
==================

Endpoints para generación con streaming.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel

from ...ml.inference.streaming_generator import StreamingGenerator
from ...ml.models.manual_generator_model import ManualGeneratorModel
from ...ml.config.ml_config import get_ml_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/streaming", tags=["streaming"])

# Instancia global
_streaming_generator: Optional[StreamingGenerator] = None


def get_streaming_generator() -> StreamingGenerator:
    """Obtener generador con streaming."""
    global _streaming_generator
    if _streaming_generator is None:
        config = get_ml_config()
        model = ManualGeneratorModel(
            model_name=config.generation_model,
            use_lora=config.use_lora,
            device=config.device
        )
        _streaming_generator = StreamingGenerator(
            model=model.model,
            tokenizer=model.tokenizer,
            device=model.device
        )
    return _streaming_generator


class StreamingRequest(BaseModel):
    """Request para streaming."""
    prompt: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class ManualStreamingRequest(BaseModel):
    """Request para streaming de manual."""
    problem_description: str
    category: str = "general"
    max_length: int = 512
    temperature: float = 0.7


@router.post("/generate")
async def stream_generation(
    request: StreamingRequest,
    generator: StreamingGenerator = Depends(get_streaming_generator)
):
    """
    Generar texto con streaming.
    
    - **prompt**: Prompt de entrada
    - **max_length**: Longitud máxima
    - **temperature**: Temperatura
    - **top_p**: Top-p sampling
    """
    try:
        async def generate():
            async for token in generator.generate_stream(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p
            ):
                yield f"data: {token}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        logger.error(f"Error en streaming: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en streaming: {str(e)}")


@router.post("/generate-manual")
async def stream_manual_generation(
    request: ManualStreamingRequest,
    generator: StreamingGenerator = Depends(get_streaming_generator)
):
    """
    Generar manual con streaming.
    
    - **problem_description**: Descripción del problema
    - **category**: Categoría del oficio
    - **max_length**: Longitud máxima
    - **temperature**: Temperatura
    """
    try:
        async def generate():
            async for token in generator.generate_stream_manual(
                problem_description=request.problem_description,
                category=request.category,
                max_length=request.max_length,
                temperature=request.temperature
            ):
                yield f"data: {token}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        logger.error(f"Error en streaming de manual: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en streaming: {str(e)}")




