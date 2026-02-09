"""
Analytics Routes - Endpoints de analytics
==========================================

Endpoints para analytics y métricas.
"""

import logging
from fastapi import APIRouter, Depends

from ...services.analytics_service import AnalyticsService
from ...infrastructure.dependencies import get_analytics_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/stats")
async def get_analytics_stats(
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Obtiene estadísticas de analytics"""
    return await analytics_service.get_stats()















