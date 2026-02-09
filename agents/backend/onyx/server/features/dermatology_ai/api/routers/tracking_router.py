"""
Tracking Router - Handles progress, habits, side effects, and other tracking endpoints
"""

from fastapi import APIRouter, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import Optional
import json

from ...api.services_locator import get_service
from ...utils.logger import logger

router = APIRouter(prefix="/dermatology", tags=["tracking"])


@router.post("/progress/add-data")
async def add_progress_data(
    user_id: str = Form(...),
    date: str = Form(...),
    metrics: str = Form(...),
    notes: Optional[str] = Form(None)
):
    """Agrega punto de datos de progreso"""
    try:
        progress_visualization = get_service("progress_visualization")
        metrics_dict = json.loads(metrics)
        progress_visualization.add_data_point(user_id, date, metrics_dict, notes)
        return JSONResponse(content={"success": True})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/progress/report/{user_id}")
async def get_progress_report(user_id: str, days: int = Query(90)):
    """Genera reporte completo de progreso"""
    try:
        progress_visualization = get_service("progress_visualization")
        report = progress_visualization.generate_progress_report(user_id, days)
        return JSONResponse(content={"success": True, "report": report.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/side-effect/add")
async def add_side_effect(
    user_id: str = Form(...),
    record_date: str = Form(...),
    product_name: str = Form(...),
    product_category: str = Form(...),
    side_effect_type: str = Form(...),
    severity: str = Form(...),
    location: Optional[str] = Form(None),
    duration_hours: Optional[int] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega registro de efecto secundario"""
    try:
        side_effect_tracker = get_service("side_effect_tracker")
        record = side_effect_tracker.add_side_effect(
            user_id, record_date, product_name, product_category,
            side_effect_type, severity, location, duration_hours, notes
        )
        return JSONResponse(content={"success": True, "record": record.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/side-effect/analyze/{user_id}")
async def analyze_side_effects(user_id: str, days: int = Query(90)):
    """Analiza efectos secundarios"""
    try:
        side_effect_tracker = get_service("side_effect_tracker")
        analysis = side_effect_tracker.analyze_side_effects(user_id, days)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/habits/record")
async def record_habit(
    user_id: str = Form(...),
    habit_type: str = Form(...),
    date: str = Form(...),
    value: Optional[float] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Registra un hábito"""
    try:
        habit_analyzer = get_service("habit_analyzer")
        record = habit_analyzer.record_habit(user_id, habit_type, date, value, notes)
        return JSONResponse(content={"success": True, "record": record.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/habits/analyze/{user_id}")
async def analyze_habits(user_id: str, days: int = Query(90)):
    """Analiza hábitos del usuario"""
    try:
        habit_analyzer = get_service("habit_analyzer")
        analysis = habit_analyzer.analyze_habits(user_id, days)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/sleep/add-record")
async def add_sleep_record(
    user_id: str = Form(...),
    date: str = Form(...),
    hours: float = Form(...),
    quality: Optional[str] = Form(None),
    notes: Optional[str] = Form(None)
):
    """Agrega registro de sueño"""
    try:
        sleep_tracker = get_service("sleep_habit_tracker")
        record = sleep_tracker.add_record(user_id, date, hours, quality, notes)
        return JSONResponse(content={"success": True, "record": record.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/sleep/analyze/{user_id}")
async def analyze_sleep(user_id: str, days: int = Query(90)):
    """Analiza patrones de sueño"""
    try:
        sleep_tracker = get_service("sleep_habit_tracker")
        analysis = sleep_tracker.analyze_sleep(user_id, days)
        return JSONResponse(content={"success": True, "analysis": analysis.to_dict()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")




