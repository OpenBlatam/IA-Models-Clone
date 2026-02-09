"""
Rutas para Análisis de Audio
==============================

Endpoints para análisis de audio.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.audio_analysis import get_audio_analyzer, AudioAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/audio",
    tags=["Audio Analysis"]
)


class AnalyzeAudioRequest(BaseModel):
    """Request para analizar audio"""
    audio_path: str = Field(..., description="Ruta del audio")
    options: Optional[Dict[str, Any]] = Field(None, description="Opciones de análisis")


@router.post("/analyze")
async def analyze_audio(
    request: AnalyzeAudioRequest,
    analyzer: AudioAnalyzer = Depends(get_audio_analyzer)
):
    """Analizar audio"""
    try:
        result = analyzer.analyze_audio(request.audio_path, request.options)
        
        return result
    except Exception as e:
        logger.error(f"Error analizando audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses/{audio_id}/summary")
async def get_audio_summary(
    audio_id: str,
    analyzer: AudioAnalyzer = Depends(get_audio_analyzer)
):
    """Obtener resumen del audio"""
    try:
        summary = analyzer.get_audio_summary(audio_id)
        
        return summary
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo resumen: {e}")
        raise HTTPException(status_code=500, detail=str(e))



