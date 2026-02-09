"""Analytics endpoints"""
from fastapi import APIRouter, Query
from typing import Optional
from utils.analytics import get_analytics_engine

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/report")
async def get_analytics_report(
    period: str = Query("daily", regex="^(daily|weekly|monthly)$")
):
    """Get analytics report"""
    analytics = get_analytics_engine()
    report = analytics.generate_report(period)
    
    return report

