"""
Advanced Reports endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.advanced_reports import AdvancedReportsService, ReportType, ReportFormat

router = APIRouter()
reports_service = AdvancedReportsService()


@router.post("/generate/{user_id}")
async def generate_report(
    user_id: str,
    report_type: str,
    format: str,
    date_range: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Generar reporte"""
    try:
        type_enum = ReportType(report_type)
        format_enum = ReportFormat(format)
        
        date_range_obj = None
        if date_range:
            date_range_obj = {
                "start": datetime.fromisoformat(date_range["start"]),
                "end": datetime.fromisoformat(date_range["end"]),
            }
        
        report = reports_service.generate_report(user_id, type_enum, format_enum, date_range_obj)
        
        return {
            "id": report.id,
            "report_type": report.report_type.value,
            "format": report.format.value,
            "file_url": report.file_url,
            "file_size": report.file_size,
            "generated_at": report.generated_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{user_id}")
async def get_user_reports(user_id: str) -> Dict[str, Any]:
    """Obtener reportes del usuario"""
    try:
        reports = reports_service.get_user_reports(user_id)
        return {
            "user_id": user_id,
            "reports": reports,
            "total": len(reports),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




