"""
Reports API Endpoints
====================

Endpoints para generación de reportes.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from ..core.report_generator import get_report_generator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.post("/create")
async def create_report(
    title: str,
    author: str = "System",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Crear nuevo reporte."""
    try:
        generator = get_report_generator()
        report = generator.create_report(title, author, metadata)
        return {
            "report_id": report.report_id,
            "title": report.title,
            "author": report.author,
            "created_at": report.created_at
        }
    except Exception as e:
        logger.error(f"Error creating report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{report_id}/sections")
async def add_section(
    report_id: str,
    title: str,
    content: str,
    level: int = Query(1, ge=1, le=6)
) -> Dict[str, Any]:
    """Agregar sección al reporte."""
    try:
        generator = get_report_generator()
        report = next((r for r in generator.reports if r.report_id == report_id), None)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        section = generator.add_section(report, title, content, level)
        return {
            "section_title": section.title,
            "level": section.level
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding section: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{report_id}/export")
async def export_report(
    report_id: str,
    format: str = Query("markdown", regex="^(markdown|html|pdf)$")
) -> Dict[str, Any]:
    """Exportar reporte."""
    try:
        generator = get_report_generator()
        report = next((r for r in generator.reports if r.report_id == report_id), None)
        
        if not report:
            raise HTTPException(status_code=404, detail="Report not found")
        
        output_file = f"reports/{report_id}.{format}"
        exported_path = generator.export_report(report, output_file, format=format)
        
        return {
            "report_id": report_id,
            "format": format,
            "file_path": exported_path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_reports() -> Dict[str, Any]:
    """Listar todos los reportes."""
    try:
        generator = get_report_generator()
        return {
            "reports": [
                {
                    "report_id": r.report_id,
                    "title": r.title,
                    "author": r.author,
                    "created_at": r.created_at,
                    "sections_count": len(r.sections)
                }
                for r in generator.reports
            ],
            "count": len(generator.reports)
        }
    except Exception as e:
        logger.error(f"Error listing reports: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))






