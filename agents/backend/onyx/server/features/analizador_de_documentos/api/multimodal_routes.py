"""
Rutas para Análisis Multimodal
=================================

Endpoints para análisis multimodal.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.multimodal_analysis import get_multimodal_analyzer, MultimodalAnalyzer, MultimodalContent, ModalityType

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/multimodal",
    tags=["Multimodal Analysis"]
)


class AnalyzeContentRequest(BaseModel):
    """Request para analizar contenido multimodal"""
    content_id: str = Field(..., description="ID del contenido")
    modalities: List[str] = Field(..., description="Modalidades")
    text_content: Optional[str] = Field(None, description="Contenido de texto")
    image_paths: Optional[List[str]] = Field(None, description="Rutas de imágenes")
    audio_path: Optional[str] = Field(None, description="Ruta de audio")
    video_path: Optional[str] = Field(None, description="Ruta de video")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata")


@router.post("/analyze")
async def analyze_multimodal_content(
    request: AnalyzeContentRequest,
    analyzer: MultimodalAnalyzer = Depends(get_multimodal_analyzer)
):
    """Analizar contenido multimodal"""
    try:
        modalities = [ModalityType(m) for m in request.modalities]
        
        content = MultimodalContent(
            content_id=request.content_id,
            modalities=modalities,
            text_content=request.text_content,
            image_paths=request.image_paths,
            audio_path=request.audio_path,
            video_path=request.video_path,
            metadata=request.metadata or {}
        )
        
        result = analyzer.analyze_content(content)
        
        return result
    except Exception as e:
        logger.error(f"Error analizando contenido multimodal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyses/{content_id}")
async def get_analysis(
    content_id: str,
    analyzer: MultimodalAnalyzer = Depends(get_multimodal_analyzer)
):
    """Obtener análisis multimodal"""
    if content_id not in analyzer.analyses:
        raise HTTPException(status_code=404, detail="Análisis no encontrado")
    
    return analyzer.analyses[content_id]



