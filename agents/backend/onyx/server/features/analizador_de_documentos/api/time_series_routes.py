"""
Rutas para Time Series Analysis
==================================

Endpoints para análisis de series temporales.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..core.time_series_analysis import (
    get_time_series_analyzer,
    TimeSeriesAnalyzer,
    TimeSeriesData,
    TimeSeriesMethod
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/analizador-documentos/time-series",
    tags=["Time Series Analysis"]
)


class AddSeriesRequest(BaseModel):
    """Request para agregar serie"""
    data: List[Dict[str, Any]] = Field(..., description="Datos de la serie")


@router.post("/series/{series_id}")
async def add_series(
    series_id: str,
    request: AddSeriesRequest,
    analyzer: TimeSeriesAnalyzer = Depends(get_time_series_analyzer)
):
    """Agregar serie temporal"""
    try:
        time_series_data = [
            TimeSeriesData(
                timestamp=item.get("timestamp", ""),
                value=item.get("value", 0.0),
                metadata=item.get("metadata", {})
            )
            for item in request.data
        ]
        
        analyzer.add_series(series_id, time_series_data)
        
        return {"status": "added", "series_id": series_id, "points": len(time_series_data)}
    except Exception as e:
        logger.error(f"Error agregando serie: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/series/{series_id}/trends")
async def analyze_trends(
    series_id: str,
    analyzer: TimeSeriesAnalyzer = Depends(get_time_series_analyzer)
):
    """Analizar tendencias"""
    try:
        trends = analyzer.analyze_trends(series_id)
        
        return trends
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analizando tendencias: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/series/{series_id}/forecast")
async def forecast(
    series_id: str,
    steps: int = Field(10, description="Número de pasos"),
    method: str = Field("arima", description="Método"),
    analyzer: TimeSeriesAnalyzer = Depends(get_time_series_analyzer)
):
    """Pronosticar valores futuros"""
    try:
        ts_method = TimeSeriesMethod(method)
        result = analyzer.forecast(series_id, steps, ts_method)
        
        return {
            "forecast_id": result.forecast_id,
            "predictions": result.predictions,
            "confidence_intervals": result.confidence_intervals,
            "method": result.method.value
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error pronosticando: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/series/{series_id}/anomalies")
async def detect_anomalies(
    series_id: str,
    threshold: float = Field(2.0, description="Umbral"),
    analyzer: TimeSeriesAnalyzer = Depends(get_time_series_analyzer)
):
    """Detectar anomalías temporales"""
    try:
        anomalies = analyzer.detect_anomalies(series_id, threshold)
        
        return {"anomalies": anomalies, "count": len(anomalies)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error detectando anomalías: {e}")
        raise HTTPException(status_code=500, detail=str(e))


