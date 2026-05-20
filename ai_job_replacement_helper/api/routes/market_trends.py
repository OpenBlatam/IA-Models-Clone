"""
Market Trends endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.market_trends import MarketTrendsService

router = APIRouter()
trends_service = MarketTrendsService()


@router.get("/skill/{skill}")
async def analyze_skill_trend(
    skill: str,
    industry: Optional[str] = None
) -> Dict[str, Any]:
    """Analizar tendencia de una habilidad"""
    try:
        trend = trends_service.analyze_skill_trend(skill, industry)
        return {
            "skill": trend.skill,
            "industry": trend.industry,
            "demand_score": trend.demand_score,
            "growth_rate": trend.growth_rate,
            "salary_trend": trend.salary_trend,
            "future_outlook": trend.future_outlook,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/industry/{industry}")
async def analyze_industry(industry: str) -> Dict[str, Any]:
    """Analizar industria completa"""
    try:
        analysis = trends_service.analyze_industry(industry)
        return {
            "industry": analysis.industry,
            "growth_rate": analysis.growth_rate,
            "top_skills": analysis.top_skills,
            "job_openings_trend": analysis.job_openings_trend,
            "salary_range": analysis.salary_range,
            "competition_level": analysis.competition_level,
            "recommendations": analysis.recommendations,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/emerging-skills/{industry}")
async def get_emerging_skills(
    industry: str,
    timeframe_days: int = 90
) -> Dict[str, Any]:
    """Obtener habilidades emergentes"""
    try:
        emerging = trends_service.get_emerging_skills(industry, timeframe_days)
        return {
            "emerging_skills": emerging,
            "timeframe_days": timeframe_days,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predict/{skill}")
async def predict_skill_demand(
    skill: str,
    months_ahead: int = 12
) -> Dict[str, Any]:
    """Predecir demanda futura de habilidad"""
    try:
        prediction = trends_service.predict_skill_demand(skill, months_ahead)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare-skills")
async def compare_skills(
    skills: List[str],
    industry: Optional[str] = None
) -> Dict[str, Any]:
    """Comparar múltiples habilidades"""
    try:
        comparison = trends_service.compare_skills(skills, industry)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/insights")
async def get_market_insights(
    user_skills: List[str],
    target_industry: str
) -> Dict[str, Any]:
    """Obtener insights de mercado personalizados"""
    try:
        insights = trends_service.get_market_insights(user_skills, target_industry)
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




