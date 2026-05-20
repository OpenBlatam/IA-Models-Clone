"""
Reports Router - Handles report generation and visualization endpoints
"""

from fastapi import APIRouter, HTTPException, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
import json

from ...api.services_locator import get_service
from ...utils.logger import logger

router = APIRouter(prefix="/dermatology", tags=["reports"])


@router.post("/report/json")
async def generate_json_report(
    analysis_id: str = Form(...),
    include_history: bool = Form(False)
):
    """Genera reporte en formato JSON"""
    try:
        report_generator = get_service("report_generator")
        report = report_generator.generate_json_report(analysis_id, include_history)
        return JSONResponse(content={"success": True, "report": report})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/report/pdf")
async def generate_pdf_report(
    analysis_id: str = Form(...),
    template: Optional[str] = Form(None)
):
    """Genera reporte en formato PDF"""
    try:
        report_generator = get_service("report_generator")
        pdf_path = report_generator.generate_pdf_report(analysis_id, template)
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"report_{analysis_id}.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/report/html")
async def generate_html_report(
    analysis_id: str = Form(...),
    template: Optional[str] = Form(None)
):
    """Genera reporte en formato HTML"""
    try:
        report_generator = get_service("report_generator")
        html_content = report_generator.generate_html_report(analysis_id, template)
        return JSONResponse(content={"success": True, "html": html_content})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/reports/advanced")
async def generate_advanced_report(
    user_id: str = Form(...),
    config: str = Form(...)
):
    """Genera reporte avanzado"""
    try:
        advanced_reporting = get_service("advanced_reporting")
        config_dict = json.loads(config)
        report = advanced_reporting.generate_report(user_id, config_dict)
        return JSONResponse(content={
            "success": True,
            "report": report.to_dict() if hasattr(report, 'to_dict') else report
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/reports/formats")
async def get_report_formats():
    """Obtiene formatos de reporte disponibles"""
    try:
        advanced_reporting = get_service("advanced_reporting")
        formats = advanced_reporting.get_available_formats()
        return JSONResponse(content={"success": True, "formats": formats})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/visualization/radar")
async def generate_radar_visualization(
    analysis_data: str = Form(...)
):
    """Genera visualización tipo radar"""
    try:
        visualization_generator = get_service("visualization_generator")
        data_dict = json.loads(analysis_data)
        visualization = visualization_generator.generate_radar_chart(data_dict)
        return JSONResponse(content={"success": True, "visualization": visualization})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/visualization/timeline")
async def generate_timeline_visualization(
    user_id: str = Form(...),
    days: int = Form(90)
):
    """Genera visualización de línea de tiempo"""
    try:
        visualization_generator = get_service("visualization_generator")
        visualization = visualization_generator.generate_timeline(user_id, days)
        return JSONResponse(content={"success": True, "visualization": visualization})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/visualization/comparison")
async def generate_comparison_visualization(
    analysis_ids: str = Form(...)
):
    """Genera visualización de comparación"""
    try:
        visualization_generator = get_service("visualization_generator")
        ids_list = json.loads(analysis_ids)
        visualization = visualization_generator.generate_comparison(ids_list)
        return JSONResponse(content={"success": True, "visualization": visualization})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")




