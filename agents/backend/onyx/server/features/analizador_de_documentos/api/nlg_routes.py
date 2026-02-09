"""
Rutas para Natural Language Generation
========================================

Endpoints para generación de lenguaje natural.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.natural_language_generation import (
    get_nlg,
    NaturalLanguageGeneration,
    NLGType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/nlg",
    tags=["Natural Language Generation"]
)


class GenerateTextRequest(BaseModel):
    """Request para generar texto"""
    input_data: Dict[str, Any] = Field(..., description="Datos de entrada")
    nlg_type: str = Field("summary", description="Tipo")
    style: str = Field("formal", description="Estilo")
    length: str = Field("medium", description="Longitud")


@router.post("/generate")
async def generate_text(
    request: GenerateTextRequest,
    nlg: NaturalLanguageGeneration = Depends(get_nlg)
):
    """Generar texto"""
    try:
        nlg_type = NLGType(request.nlg_type)
        result = nlg.generate_text(
            request.input_data,
            nlg_type,
            request.style,
            request.length
        )
        
        return {
            "request_id": result.request_id,
            "generated_text": result.generated_text,
            "quality_score": result.quality_score,
            "readability_score": result.readability_score
        }
    except Exception as e:
        logger.error(f"Error generando texto: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/summarize")
async def generate_summary(
    content: str = Field(..., description="Contenido"),
    max_length: int = Field(200, description="Longitud máxima"),
    nlg: NaturalLanguageGeneration = Depends(get_nlg)
):
    """Generar resumen"""
    try:
        summary = nlg.generate_summary(content, max_length)
        
        return {
            "summary": summary,
            "original_length": len(content),
            "summary_length": len(summary)
        }
    except Exception as e:
        logger.error(f"Error generando resumen: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report")
async def generate_report(
    data: Dict[str, Any] = Field(..., description="Datos"),
    sections: List[str] = Field(..., description="Secciones"),
    nlg: NaturalLanguageGeneration = Depends(get_nlg)
):
    """Generar reporte"""
    try:
        report = nlg.generate_report(data, sections)
        
        return {
            "report": report,
            "sections": sections,
            "num_sections": len(sections)
        }
    except Exception as e:
        logger.error(f"Error generando reporte: {e}")
        raise HTTPException(status_code=500, detail=str(e))


