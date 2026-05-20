"""
Rutas para Análisis Predictivo
===============================

Endpoints para análisis predictivo y forecasting.
"""

import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.predictive_analyzer import PredictiveAnalyzer
from .routes import get_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/predictive",
    tags=["Predictive Analysis"]
)


class PredictiveRequest(BaseModel):
    """Request para análisis predictivo"""
    historical_documents: List[Dict[str, Any]] = Field(
        ...,
        description="Documentos históricos con timestamp y content"
    )
    timeframe: str = Field("1week", description="Período de predicción: 1day, 1week, 1month")


@router.post("/sentiment")
async def predict_sentiment(
    request: PredictiveRequest,
    analyzer = Depends(get_analyzer)
):
    """Predecir sentimiento futuro"""
    try:
        predictive_analyzer = PredictiveAnalyzer(analyzer)
        prediction = await predictive_analyzer.predict_sentiment(
            request.historical_documents,
            request.timeframe
        )
        
        return {
            "metric": prediction.metric,
            "current_value": prediction.current_value,
            "predicted_value": prediction.predicted_value,
            "confidence": prediction.confidence,
            "timeframe": prediction.timeframe,
            "trend": prediction.trend,
            "factors": prediction.factors,
            "timestamp": prediction.timestamp
        }
    except Exception as e:
        logger.error(f"Error prediciendo sentimiento: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/topics")
async def predict_topics(
    request: PredictiveRequest,
    analyzer = Depends(get_analyzer)
):
    """Predecir temas futuros"""
    try:
        predictive_analyzer = PredictiveAnalyzer(analyzer)
        predictions = await predictive_analyzer.predict_topics(
            request.historical_documents,
            request.timeframe
        )
        
        return {
            "predictions": [
                {
                    "metric": p.metric,
                    "current_value": p.current_value,
                    "predicted_value": p.predicted_value,
                    "confidence": p.confidence,
                    "trend": p.trend
                }
                for p in predictions
            ],
            "timeframe": request.timeframe
        }
    except Exception as e:
        logger.error(f"Error prediciendo temas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report")
async def generate_predictive_report(
    request: PredictiveRequest,
    analyzer = Depends(get_analyzer)
):
    """Generar reporte predictivo completo"""
    try:
        predictive_analyzer = PredictiveAnalyzer(analyzer)
        report = await predictive_analyzer.generate_predictive_report(
            request.historical_documents,
            request.timeframe
        )
        
        return {
            "predictions": [
                {
                    "metric": p.metric,
                    "current_value": p.current_value,
                    "predicted_value": p.predicted_value,
                    "confidence": p.confidence,
                    "trend": p.trend,
                    "timeframe": p.timeframe
                }
                for p in report.predictions
            ],
            "insights": report.insights,
            "recommendations": report.recommendations,
            "confidence_score": report.confidence_score,
            "timestamp": report.timestamp
        }
    except Exception as e:
        logger.error(f"Error generando reporte predictivo: {e}")
        raise HTTPException(status_code=500, detail=str(e))
















