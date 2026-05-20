"""
Rutas para Generación de Reportes
===================================

Endpoints para generar reportes avanzados.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..core.report_generator import (
    get_report_generator,
    AdvancedReportGenerator,
    ReportType
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/reports",
    tags=["Reports"]
)


class GenerateReportRequest(BaseModel):
    """Request para generar reporte"""
    data: Dict[str, Any] = Field(..., description="Datos para el reporte")
    report_type: str = Field("summary", description="Tipo de reporte")
    template: Optional[str] = Field(None, description="Plantilla personalizada")


@router.post("/generate")
async def generate_report(
    request: GenerateReportRequest,
    generator: AdvancedReportGenerator = Depends(get_report_generator)
):
    """Generar reporte"""
    try:
        report_type = ReportType(request.report_type)
        report = generator.generate_report(
            request.data,
            report_type,
            request.template
        )
        
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Tipo de reporte inválido: {e}")
    except Exception as e:
        logger.error(f"Error generando reporte: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_report(
    request: GenerateReportRequest,
    format: str = "json",
    generator: AdvancedReportGenerator = Depends(get_report_generator)
):
    """Generar y exportar reporte"""
    try:
        report_type = ReportType(request.report_type)
        report = generator.generate_report(
            request.data,
            report_type,
            request.template
        )
        
        exported = generator.export_report(report, format)
        
        content_type_map = {
            "json": "application/json",
            "markdown": "text/markdown",
            "html": "text/html"
        }
        
        return Response(
            content=exported,
            media_type=content_type_map.get(format, "text/plain")
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error exportando reporte: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















