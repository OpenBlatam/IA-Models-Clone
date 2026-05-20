"""
Rutas para Resúmenes Ejecutivos
=================================

Endpoints para generar resúmenes ejecutivos inteligentes.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.executive_summary import ExecutiveSummaryGenerator
from .routes import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/summary",
    tags=["Executive Summary"]
)


class SummaryRequest(BaseModel):
    """Request para generar resumen ejecutivo"""
    analysis_result: Dict[str, Any] = Field(..., description="Resultado de análisis completo")
    document_content: Optional[str] = Field(None, description="Contenido del documento")
    max_findings: int = Field(5, description="Número máximo de hallazgos")
    max_recommendations: int = Field(5, description="Número máximo de recomendaciones")


@router.post("/executive")
async def generate_executive_summary(
    request: SummaryRequest,
    analyzer = Depends(get_analyzer)
):
    """Generar resumen ejecutivo"""
    try:
        generator = ExecutiveSummaryGenerator(analyzer)
        summary = await generator.generate_summary(
            request.analysis_result,
            request.document_content,
            request.max_findings,
            request.max_recommendations
        )
        
        return {
            "title": summary.title,
            "overview": summary.overview,
            "key_findings": summary.key_findings,
            "recommendations": summary.recommendations,
            "metrics": summary.metrics,
            "insights": summary.insights,
            "confidence": summary.confidence,
            "timestamp": summary.timestamp
        }
    except Exception as e:
        logger.error(f"Error generando resumen ejecutivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















