"""
Analytics and reporting routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import JSONResponse
from datetime import datetime

try:
    from services.analytics_service import AnalyticsService
    from core.progress_tracker import ProgressTracker
except ImportError:
    from ...services.analytics_service import AnalyticsService
    from ...core.progress_tracker import ProgressTracker

router = APIRouter()

analytics = AnalyticsService()
tracker = ProgressTracker()


@router.get("/analytics/{user_id}")
async def get_analytics(user_id: str):
    """Obtiene análisis completo del usuario"""
    try:
        progress = tracker.get_progress(user_id, None, [])
        stats = tracker.get_stats(user_id, [])
        
        return JSONResponse(content={
            "user_id": user_id,
            "progress": progress,
            "statistics": stats,
            "status": "success"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo análisis: {str(e)}")


@router.post("/generate-report")
async def generate_report(
    user_id: str = Body(...),
    report_type: str = Body("summary")
):
    """Genera reporte detallado"""
    try:
        progress = tracker.get_progress(user_id, None, [])
        stats = tracker.get_stats(user_id, [])
        
        report = {
            "user_id": user_id,
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "progress_summary": progress,
            "detailed_statistics": stats,
            "recommendations": []
        }
        
        return JSONResponse(content=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando reporte: {str(e)}")


@router.get("/insights/{user_id}")
async def get_insights(user_id: str):
    """Obtiene insights personalizados"""
    try:
        progress = tracker.get_progress(user_id, None, [])
        stats = tracker.get_stats(user_id, [])
        
        insights = {
            "user_id": user_id,
            "key_insights": [
                f"Has estado sobrio por {progress.get('days_sober', 0)} días",
                "Continúa con tu plan de recuperación",
                "Mantén contacto regular con tu sistema de apoyo"
            ],
            "trends": stats.get("trends", {}),
            "recommendations": []
        }
        
        return JSONResponse(content=insights)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo insights: {str(e)}")


@router.get("/analytics/advanced/{user_id}")
async def get_advanced_analytics(user_id: str):
    """Obtiene análisis avanzado completo del usuario"""
    try:
        entries = []
        analytics_data = analytics.generate_comprehensive_analytics(user_id, entries)
        return JSONResponse(content=analytics_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo análisis avanzado: {str(e)}")



