"""
Rutas para Análisis de Tendencias
===================================

Endpoints para analizar tendencias y estadísticas temporales.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.trend_analyzer import TrendAnalyzer
from .routes import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/trends",
    tags=["Trend Analysis"]
)


class TrendAnalysisRequest(BaseModel):
    """Request para análisis de tendencias"""
    documents: List[Dict[str, Any]] = Field(
        ...,
        description="Lista de documentos con timestamp y content"
    )
    period: str = Field("day", description="Período: day, week, month")


@router.post("/sentiment")
async def analyze_sentiment_trend(
    request: TrendAnalysisRequest,
    analyzer = Depends(get_analyzer)
):
    """Analizar tendencia de sentimiento"""
    try:
        trend_analyzer = TrendAnalyzer(analyzer)
        result = await trend_analyzer.analyze_sentiment_trend(
            request.documents,
            request.period
        )
        
        return {
            "metric_name": result.metric_name,
            "trend_direction": result.trend_direction,
            "trend_strength": result.trend_strength,
            "average_value": result.average_value,
            "min_value": result.min_value,
            "max_value": result.max_value,
            "variance": result.variance,
            "period": result.period,
            "data_points": [
                {
                    "timestamp": dp.timestamp.isoformat() if hasattr(dp.timestamp, 'isoformat') else str(dp.timestamp),
                    "value": dp.value,
                    "metadata": dp.metadata
                }
                for dp in result.data_points
            ]
        }
    except Exception as e:
        logger.error(f"Error analizando tendencia de sentimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/keywords")
async def analyze_keyword_trend(
    request: TrendAnalysisRequest,
    top_k: int = 10,
    analyzer = Depends(get_analyzer)
):
    """Analizar tendencia de keywords"""
    try:
        trend_analyzer = TrendAnalyzer(analyzer)
        results = await trend_analyzer.analyze_keyword_trend(
            request.documents,
            request.period,
            top_k
        )
        
        return {
            "trends": {
                keyword: {
                    "trend_direction": trend.trend_direction,
                    "trend_strength": trend.trend_strength,
                    "average_value": trend.average_value,
                    "data_points": [
                        {
                            "timestamp": dp.timestamp.isoformat() if hasattr(dp.timestamp, 'isoformat') else str(dp.timestamp),
                            "value": dp.value
                        }
                        for dp in trend.data_points
                    ]
                }
                for keyword, trend in results.items()
            }
        }
    except Exception as e:
        logger.error(f"Error analizando tendencia de keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report")
async def generate_trend_report(
    request: TrendAnalysisRequest,
    analyzer = Depends(get_analyzer)
):
    """Generar reporte completo de tendencias"""
    try:
        trend_analyzer = TrendAnalyzer(analyzer)
        report = await trend_analyzer.generate_trend_report(
            request.documents,
            request.period
        )
        
        # Convertir datetime a string para JSON
        def convert_datetime(obj):
            if isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return obj
        
        return convert_datetime(report)
    except Exception as e:
        logger.error(f"Error generando reporte de tendencias: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















