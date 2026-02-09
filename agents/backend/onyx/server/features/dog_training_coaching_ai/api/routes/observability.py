"""
Observability Endpoints
=======================
Endpoints para observabilidad y tracing.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
from datetime import datetime

from ...utils.observability import get_metrics_collector, TraceContext, Span
from ...utils.logger import get_logger

router = APIRouter(prefix="/api/v1/observability", tags=["observability"])
logger = get_logger(__name__)


@router.get("/metrics")
async def get_metrics(
    metric_name: Optional[str] = Query(None, description="Nombre de métrica específica")
) -> Dict[str, Any]:
    """
    Obtener métricas personalizadas.
    
    Returns:
        Métricas registradas
    """
    try:
        collector = get_metrics_collector()
        
        if metric_name:
            metric_data = collector.get_metric(metric_name)
            summary = collector.get_summary(metric_name)
            
            return {
                "success": True,
                "metric": metric_name,
                "data": metric_data,
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }
        
        # Retornar todas las métricas
        all_metrics = {}
        for name in collector.metrics.keys():
            all_metrics[name] = {
                "data": collector.get_metric(name),
                "summary": collector.get_summary(name)
            }
        
        return {
            "success": True,
            "metrics": all_metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/record")
async def record_metric(
    name: str,
    value: float,
    tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Registrar una métrica manualmente.
    
    Returns:
        Confirmación de registro
    """
    try:
        collector = get_metrics_collector()
        collector.record(name, value, tags)
        
        return {
            "success": True,
            "message": f"Metric '{name}' recorded",
            "value": value,
            "tags": tags,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error recording metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trace/start")
async def start_trace(
    trace_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Iniciar un nuevo trace.
    
    Returns:
        Información del trace iniciado
    """
    try:
        context = TraceContext(trace_id=trace_id)
        
        return {
            "success": True,
            "trace": context.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting trace: {e}")
        raise HTTPException(status_code=500, detail=str(e))

