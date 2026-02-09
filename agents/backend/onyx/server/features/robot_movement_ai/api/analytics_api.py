"""
Analytics API Endpoints
=======================

Endpoints para adaptive learning y predictive analytics.
"""

from fastapi import APIRouter, HTTPException, Body, Query
from typing import Dict, Any, Optional
import logging

try:
    from ..core.adaptive_learning import get_adaptive_learning_system
except ImportError:
    def get_adaptive_learning_system():
        return None
try:
    from ..core.predictive_analytics import get_predictive_analytics_system
except ImportError:
    def get_predictive_analytics_system():
        return None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.post("/learning/metrics")
async def record_learning_metric(
    name: str,
    value: float,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Registrar métrica de aprendizaje."""
    try:
        system = get_adaptive_learning_system()
        metric_id = system.record_metric(name, value, metadata)
        
        return {
            "metric_id": metric_id,
            "name": name,
            "value": value
        }
    except Exception as e:
        logger.error(f"Error recording learning metric: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learning/detect-patterns")
async def detect_learning_patterns() -> Dict[str, Any]:
    """Detectar patrones de aprendizaje."""
    try:
        system = get_adaptive_learning_system()
        patterns = system.detect_patterns()
        
        return {
            "patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "name": p.name,
                    "pattern_type": p.pattern_type,
                    "confidence": p.confidence,
                    "parameters": p.parameters
                }
                for p in patterns
            ],
            "count": len(patterns)
        }
    except Exception as e:
        logger.error(f"Error detecting patterns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/learning/adapt-parameter")
async def adapt_parameter(
    parameter_name: str,
    current_value: float,
    performance_metric: float,
    target_metric: float = 1.0
) -> Dict[str, Any]:
    """Adaptar parámetro."""
    try:
        system = get_adaptive_learning_system()
        new_value = system.adapt_parameters(
            parameter_name=parameter_name,
            current_value=current_value,
            performance_metric=performance_metric,
            target_metric=target_metric
        )
        
        return {
            "parameter_name": parameter_name,
            "old_value": current_value,
            "new_value": new_value
        }
    except Exception as e:
        logger.error(f"Error adapting parameter: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning/recommendations")
async def get_learning_recommendations() -> Dict[str, Any]:
    """Obtener recomendaciones de aprendizaje."""
    try:
        system = get_adaptive_learning_system()
        recommendations = system.get_recommendations()
        
        return {
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/learning/statistics")
async def get_learning_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de aprendizaje."""
    try:
        system = get_adaptive_learning_system()
        stats = system.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting learning statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predictive/data-points")
async def add_data_point(
    metric_name: str,
    value: float,
    timestamp: Optional[str] = None
) -> Dict[str, Any]:
    """Agregar punto de datos para predicción."""
    try:
        system = get_predictive_analytics_system()
        system.add_data_point(metric_name, value, timestamp)
        
        return {
            "metric_name": metric_name,
            "value": value,
            "added": True
        }
    except Exception as e:
        logger.error(f"Error adding data point: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predictive/predict")
async def predict_value(
    metric_name: str,
    horizon: float = 1.0,
    method: str = "linear"
) -> Dict[str, Any]:
    """Predecir valor futuro."""
    try:
        system = get_predictive_analytics_system()
        prediction = system.predict(metric_name, horizon, method)
        
        return {
            "prediction_id": prediction.prediction_id,
            "metric_name": prediction.metric_name,
            "predicted_value": prediction.predicted_value,
            "confidence": prediction.confidence,
            "horizon": prediction.horizon
        }
    except Exception as e:
        logger.error(f"Error making prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predictive/forecast")
async def generate_forecast(
    metric_name: str,
    steps: int = 10,
    method: str = "linear"
) -> Dict[str, Any]:
    """Generar pronóstico."""
    try:
        system = get_predictive_analytics_system()
        forecast = system.forecast(metric_name, steps, method)
        
        return {
            "forecast_id": forecast.forecast_id,
            "metric_name": forecast.metric_name,
            "predictions": forecast.predictions,
            "confidence_intervals": forecast.confidence_intervals,
            "timestamps": forecast.timestamps,
            "steps": steps
        }
    except Exception as e:
        logger.error(f"Error generating forecast: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictive/statistics")
async def get_predictive_statistics() -> Dict[str, Any]:
    """Obtener estadísticas de análisis predictivo."""
    try:
        system = get_predictive_analytics_system()
        stats = system.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting predictive statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
