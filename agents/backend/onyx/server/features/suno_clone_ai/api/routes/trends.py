"""
API de Análisis de Tendencias

Endpoints para:
- Analizar tendencias
- Obtener reportes
- Predecir tendencias
- Comparar períodos
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body

from services.trend_analysis import get_trend_analysis_service
from middleware.auth_middleware import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/trends",
    tags=["trends"]
)


@router.post("/analyze")
async def analyze_trends(
    songs: List[Dict[str, Any]] = Body(..., description="Lista de canciones"),
    period_days: int = Body(30, description="Período de análisis en días"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Analiza tendencias musicales.
    """
    try:
        service = get_trend_analysis_service()
        report = service.analyze_trends(songs, period_days)
        
        return {
            "period_start": report.period_start.isoformat(),
            "period_end": report.period_end.isoformat(),
            "genre_trends": [
                {
                    "genre": gt.genre,
                    "popularity": gt.popularity,
                    "growth_rate": gt.growth_rate,
                    "sample_count": gt.sample_count
                }
                for gt in report.genre_trends
            ],
            "bpm_trends": report.bpm_trends,
            "key_trends": report.key_trends,
            "top_tags": report.top_tags
        }
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing trends: {str(e)}"
        )


@router.post("/predict")
async def predict_trends(
    current_trends: Dict[str, Any] = Body(..., description="Tendencias actuales"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Predice tendencias futuras.
    """
    try:
        from services.trend_analysis import TrendReport, GenreTrend
        
        # Convertir dict a TrendReport (simplificado)
        # En producción, esto sería más robusto
        service = get_trend_analysis_service()
        
        # Crear TrendReport desde dict
        genre_trends = [
            GenreTrend(
                genre=gt["genre"],
                popularity=gt["popularity"],
                growth_rate=gt.get("growth_rate", 0),
                sample_count=gt.get("sample_count", 0)
            )
            for gt in current_trends.get("genre_trends", [])
        ]
        
        report = TrendReport(
            period_start=datetime.fromisoformat(current_trends.get("period_start", datetime.now().isoformat())),
            period_end=datetime.fromisoformat(current_trends.get("period_end", datetime.now().isoformat())),
            genre_trends=genre_trends,
            bpm_trends=current_trends.get("bpm_trends", {}),
            key_trends=current_trends.get("key_trends", {}),
            top_tags=current_trends.get("top_tags", [])
        )
        
        predictions = service.predict_trends(report)
        return predictions
    except Exception as e:
        logger.error(f"Error predicting trends: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error predicting trends: {str(e)}"
        )

