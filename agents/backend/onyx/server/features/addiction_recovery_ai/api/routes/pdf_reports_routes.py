"""
PDF reports routes for addiction recovery API
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

try:
    from services.report_service import ReportService
    from models.database import DatabaseManager
    from core.progress_tracker import ProgressTracker
    from services.analytics_service import AnalyticsService
except ImportError:
    from ...services.report_service import ReportService
    from ...models.database import DatabaseManager
    from ...core.progress_tracker import ProgressTracker
    from ...services.analytics_service import AnalyticsService

router = APIRouter()

reports = ReportService()
db_manager = DatabaseManager()
tracker = ProgressTracker()
analytics = AnalyticsService()


@router.get("/reports/pdf/{user_id}")
async def generate_pdf_report(user_id: str):
    """Genera reporte PDF del usuario"""
    try:
        user = db_manager.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        progress = tracker.get_progress(user_id, None, [])
        analytics_data = analytics.generate_comprehensive_analytics(user_id, [])
        
        user_data = {
            "name": user.name,
            "email": user.email
        }
        
        pdf_bytes = reports.generate_pdf_report(user_id, user_data, progress, analytics_data)
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=recovery_report_{user_id}.pdf"}
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generando PDF: {str(e)}")



