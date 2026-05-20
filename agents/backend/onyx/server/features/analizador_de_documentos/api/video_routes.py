"""
Rutas para Análisis de Video
==============================

Endpoints para análisis de video.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.video_analysis import get_video_analyzer, VideoAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/video",
    tags=["Video Analysis"]
)


class AnalyzeVideoRequest(BaseModel):
    """Request para analizar video"""
    video_path: str = Field(..., description="Ruta del video")
    options: Optional[Dict[str, Any]] = Field(None, description="Opciones de análisis")


@router.post("/analyze")
async def analyze_video(
    request: AnalyzeVideoRequest,
    analyzer: VideoAnalyzer = Depends(get_video_analyzer)
):
    """Analizar video"""
    try:
        result = analyzer.analyze_video(request.video_path, request.options)
        
        return result
    except Exception as e:
        logger.error(f"Error analizando video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses/{video_id}/summary")
async def get_video_summary(
    video_id: str,
    analyzer: VideoAnalyzer = Depends(get_video_analyzer)
):
    """Obtener resumen del video"""
    try:
        summary = analyzer.get_video_summary(video_id)
        
        return summary
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error obteniendo resumen: {e}")
        raise HTTPException(status_code=500, detail=str(e))



