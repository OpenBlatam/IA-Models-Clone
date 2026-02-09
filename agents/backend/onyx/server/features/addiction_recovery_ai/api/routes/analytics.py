"""
Analytics and reporting routes - Refactored with best practices
"""

from fastapi import APIRouter, HTTPException, status

try:
    from schemas.analytics import (
        AnalyticsResponse,
        GenerateReportRequest,
        ReportResponse,
        InsightsResponse
    )
    from schemas.common import ErrorResponse
    from dependencies import (
        ProgressTrackerDep,
        AnalyticsServiceDep
    )
except ImportError:
    from ...schemas.analytics import (
        AnalyticsResponse,
        GenerateReportRequest,
        ReportResponse,
        InsightsResponse
    )
    from ...schemas.common import ErrorResponse
    from ...dependencies import (
        ProgressTrackerDep,
        AnalyticsServiceDep
    )

router = APIRouter(prefix="/analytics", tags=["Analytics & Reporting"])


@router.get(
    "/{user_id}",
    response_model=AnalyticsResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_analytics(
    user_id: str,
    tracker: ProgressTrackerDep,
    analytics_service: AnalyticsServiceDep
) -> AnalyticsResponse:
    """
    Obtiene análisis completo del usuario
    
    - **user_id**: ID del usuario
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    try:
        progress = tracker.get_progress(user_id, None, [])
        stats = tracker.get_stats(user_id, [])
        
        return AnalyticsResponse(
            user_id=user_id,
            progress_summary=progress,
            statistics=stats,
            trends=stats.get("trends", {})
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo análisis: {str(e)}"
        )


@router.post(
    "/generate-report",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def generate_report(
    request: GenerateReportRequest,
    tracker: ProgressTrackerDep,
    analytics_service: AnalyticsServiceDep
) -> ReportResponse:
    """
    Genera reporte detallado
    
    - **user_id**: ID del usuario
    - **report_type**: Tipo de reporte (summary, detailed, comprehensive)
    - **include_charts**: Incluir gráficos en el reporte
    """
    # Guard clause: Validate user_id
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    # Guard clause: Validate report_type
    valid_types = ['summary', 'detailed', 'comprehensive']
    if request.report_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"report_type debe ser uno de: {valid_types}"
        )
    
    try:
        progress = tracker.get_progress(request.user_id, None, [])
        stats = tracker.get_stats(request.user_id, [])
        
        # Generate report ID
        import uuid
        report_id = str(uuid.uuid4())
        
        return ReportResponse(
            user_id=request.user_id,
            report_id=report_id,
            report_type=request.report_type,
            progress_summary=progress,
            detailed_statistics=stats,
            recommendations=[],
            charts=[] if not request.include_charts else None
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generando reporte: {str(e)}"
        )


@router.get(
    "/insights/{user_id}",
    response_model=InsightsResponse,
    status_code=status.HTTP_200_OK,
    responses={
        404: {"model": ErrorResponse, "description": "User not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def get_insights(
    user_id: str,
    tracker: ProgressTrackerDep,
    analytics_service: AnalyticsServiceDep
) -> InsightsResponse:
    """
    Obtiene insights personalizados
    
    - **user_id**: ID del usuario
    """
    # Guard clause: Validate user_id
    if not user_id or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id es requerido"
        )
    
    try:
        progress = tracker.get_progress(user_id, None, [])
        stats = tracker.get_stats(user_id, [])
        
        days_sober = progress.get('days_sober', 0)
        key_insights = [
            f"Has estado sobrio por {days_sober} días",
            "Continúa con tu plan de recuperación",
            "Mantén contacto regular con tu sistema de apoyo"
        ]
        
        return InsightsResponse(
            user_id=user_id,
            key_insights=key_insights,
            trends=stats.get("trends", {}),
            recommendations=[]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo insights: {str(e)}"
        )

