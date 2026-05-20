"""
Advanced reporting routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import Optional, Dict

try:
    from services.advanced_reporting_service import AdvancedReportingService
except ImportError:
    from ...services.advanced_reporting_service import AdvancedReportingService

router = APIRouter()

advanced_reporting = AdvancedReportingService()


@router.post("/reports/comprehensive")
async def generate_comprehensive_report(
    user_id: str = Body(...),
    start_date: Optional[str] = Body(None),
    end_date: Optional[str] = Body(None)
):
    """Genera reporte comprensivo"""
    try:
        report = advanced_reporting.generate_comprehensive_report(
            user_id, start_date, end_date
        )
        return JSONResponse(content=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando reporte: {str(e)}")


@router.post("/reports/export")
async def export_report(
    user_id: str = Body(...),
    report_data: Dict = Body(...),
    format: str = Body("pdf")
):
    """Exporta reporte en formato específico"""
    try:
        export_result = advanced_reporting.export_report(user_id, report_data, format)
        return JSONResponse(content=export_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando reporte: {str(e)}")



