"""
API endpoints para métricas y dashboards
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, Dict
from datetime import datetime, timedelta
from pydantic import BaseModel

from core.architecture.metrics_dashboard import get_metrics_collector, MetricType

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


class RecordMetricRequest(BaseModel):
    """Request para registrar métrica"""
    name: str
    value: float
    labels: Optional[Dict[str, str]] = None
    metric_type: MetricType = MetricType.GAUGE


@router.post("/record")
async def record_metric(
    request: RecordMetricRequest,
    collector=Depends(get_metrics_collector)
):
    """Registrar métrica"""
    try:
        collector.record(
            name=request.name,
            value=request.value,
            labels=request.labels,
            metric_type=request.metric_type
        )
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query")
async def query_metrics(
    name: Optional[str] = None,
    labels: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    collector=Depends(get_metrics_collector)
):
    """Consultar métricas"""
    try:
        labels_dict = {}
        if labels:
            # Parsear labels desde query string (formato: "key1=value1,key2=value2")
            for pair in labels.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    labels_dict[k.strip()] = v.strip()
        
        metrics = collector.query(
            name=name,
            labels=labels_dict if labels_dict else None,
            start_time=start_time,
            end_time=end_time
        )
        
        return {
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "labels": m.labels,
                    "timestamp": m.timestamp.isoformat(),
                    "metric_type": m.metric_type.value
                }
                for m in metrics
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aggregate")
async def aggregate_metrics(
    name: str,
    aggregation: str = "sum",
    labels: Optional[str] = None,
    time_range_minutes: Optional[int] = None,
    collector=Depends(get_metrics_collector)
):
    """Agregar métricas"""
    try:
        labels_dict = {}
        if labels:
            for pair in labels.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    labels_dict[k.strip()] = v.strip()
        
        time_range = timedelta(minutes=time_range_minutes) if time_range_minutes else None
        
        value = collector.aggregate(
            name=name,
            aggregation=aggregation,
            labels=labels_dict if labels_dict else None,
            time_range=time_range
        )
        
        return {
            "name": name,
            "aggregation": aggregation,
            "value": value
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_dashboard(
    collector=Depends(get_metrics_collector)
):
    """Obtener datos del dashboard"""
    try:
        dashboard_data = collector.get_dashboard_data()
        return dashboard_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
