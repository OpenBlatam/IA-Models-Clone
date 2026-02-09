"""
Reports endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.reports import ReportsService, ReportType, ExportFormat

router = APIRouter()
reports_service = ReportsService()


@router.get("/progress/{user_id}")
async def get_progress_report(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """Generar reporte de progreso"""
    try:
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None
        
        report = reports_service.generate_progress_report(user_id, start_dt, end_dt)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/applications/{user_id}")
async def get_applications_report(user_id: str) -> Dict[str, Any]:
    """Generar reporte de aplicaciones"""
    try:
        report = reports_service.generate_applications_report(user_id)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comprehensive/{user_id}")
async def get_comprehensive_report(user_id: str) -> Dict[str, Any]:
    """Generar reporte comprensivo"""
    try:
        report = reports_service.generate_comprehensive_report(user_id)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export")
async def export_report(
    report: Dict[str, Any],
    format: str
) -> Dict[str, Any]:
    """Exportar reporte en formato específico"""
    try:
        format_enum = ExportFormat(format)
        exported = reports_service.export_report(report, format_enum)
        return {
            "format": format,
            "content": exported,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




