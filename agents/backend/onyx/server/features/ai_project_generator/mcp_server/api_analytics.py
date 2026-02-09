"""
MCP API Usage Analytics - Analytics avanzado de uso de API
===========================================================
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class APIUsageMetric(BaseModel):
    """Métrica de uso de API"""
    endpoint: str = Field(..., description="Endpoint")
    method: str = Field(..., description="Método HTTP")
    count: int = Field(default=0, description="Número de requests")
    total_duration_ms: float = Field(default=0.0, description="Duración total en ms")
    error_count: int = Field(default=0, description="Número de errores")
    user_ids: List[str] = Field(default_factory=list, description="Usuarios únicos")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class APIUsageAnalytics:
    """
    Analytics avanzado de uso de API
    
    Proporciona métricas detalladas de uso de endpoints.
    """
    
    def __init__(self):
        self._metrics: Dict[str, APIUsageMetric] = {}
        self._hourly_metrics: Dict[str, List[APIUsageMetric]] = defaultdict(list)
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        duration_ms: float,
        status_code: int,
        user_id: Optional[str] = None,
    ):
        """
        Registra un request
        
        Args:
            endpoint: Endpoint
            method: Método HTTP
            duration_ms: Duración en milisegundos
            status_code: Código de estado
            user_id: ID del usuario (opcional)
        """
        key = f"{method}:{endpoint}"
        
        if key not in self._metrics:
            self._metrics[key] = APIUsageMetric(
                endpoint=endpoint,
                method=method,
            )
        
        metric = self._metrics[key]
        metric.count += 1
        metric.total_duration_ms += duration_ms
        
        if status_code >= 400:
            metric.error_count += 1
        
        if user_id and user_id not in metric.user_ids:
            metric.user_ids.append(user_id)
        
        # Registrar métrica horaria
        hour_key = f"{key}:{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H')}"
        if hour_key not in self._hourly_metrics:
            self._hourly_metrics[hour_key] = []
        
        self._hourly_metrics[hour_key].append(APIUsageMetric(
            endpoint=endpoint,
            method=method,
            count=1,
            total_duration_ms=duration_ms,
            error_count=1 if status_code >= 400 else 0,
        ))
    
    def get_endpoint_stats(
        self,
        endpoint: str,
        method: Optional[str] = None,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Obtiene estadísticas de un endpoint
        
        Args:
            endpoint: Endpoint
            method: Método HTTP (opcional)
            hours: Número de horas a analizar
            
        Returns:
            Diccionario con estadísticas
        """
        if method:
            key = f"{method}:{endpoint}"
            metric = self._metrics.get(key)
        else:
            # Agregar todos los métodos
            metrics = [
                m for k, m in self._metrics.items()
                if m.endpoint == endpoint
            ]
            if not metrics:
                return {"endpoint": endpoint, "stats": {}}
            
            metric = APIUsageMetric(
                endpoint=endpoint,
                method="ALL",
                count=sum(m.count for m in metrics),
                total_duration_ms=sum(m.total_duration_ms for m in metrics),
                error_count=sum(m.error_count for m in metrics),
            )
            metric.user_ids = set()
            for m in metrics:
                metric.user_ids.update(m.user_ids)
        
        if not metric:
            return {"endpoint": endpoint, "stats": {}}
        
        avg_duration = (
            metric.total_duration_ms / metric.count
            if metric.count > 0 else 0
        )
        error_rate = (
            metric.error_count / metric.count * 100
            if metric.count > 0 else 0
        )
        
        return {
            "endpoint": endpoint,
            "method": method or "ALL",
            "stats": {
                "total_requests": metric.count,
                "unique_users": len(metric.user_ids),
                "avg_duration_ms": avg_duration,
                "total_duration_ms": metric.total_duration_ms,
                "error_count": metric.error_count,
                "error_rate_percent": error_rate,
                "success_rate_percent": 100 - error_rate,
            },
        }
    
    def get_top_endpoints(
        self,
        limit: int = 10,
        sort_by: str = "count",
    ) -> List[Dict[str, Any]]:
        """
        Obtiene top endpoints
        
        Args:
            limit: Número de endpoints a retornar
            sort_by: Campo para ordenar (count, duration, errors)
            
        Returns:
            Lista de endpoints ordenados
        """
        endpoints = []
        
        for metric in self._metrics.values():
            endpoints.append({
                "endpoint": metric.endpoint,
                "method": metric.method,
                "count": metric.count,
                "avg_duration_ms": (
                    metric.total_duration_ms / metric.count
                    if metric.count > 0 else 0
                ),
                "error_count": metric.error_count,
                "unique_users": len(metric.user_ids),
            })
        
        # Ordenar
        reverse = True
        if sort_by == "duration":
            endpoints.sort(key=lambda x: x["avg_duration_ms"], reverse=reverse)
        elif sort_by == "errors":
            endpoints.sort(key=lambda x: x["error_count"], reverse=reverse)
        else:
            endpoints.sort(key=lambda x: x["count"], reverse=reverse)
        
        return endpoints[:limit]
    
    def get_trends(
        self,
        endpoint: str,
        hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Obtiene tendencias de un endpoint
        
        Args:
            endpoint: Endpoint
            hours: Número de horas
            
        Returns:
            Diccionario con tendencias
        """
        now = datetime.now(timezone.utc)
        trends = []
        
        for i in range(hours):
            hour_start = now - timedelta(hours=i+1)
            hour_end = now - timedelta(hours=i)
            hour_key = hour_start.strftime('%Y-%m-%d-%H')
            
            metrics = self._hourly_metrics.get(hour_key, [])
            endpoint_metrics = [m for m in metrics if m.endpoint == endpoint]
            
            if endpoint_metrics:
                total_count = sum(m.count for m in endpoint_metrics)
                total_duration = sum(m.total_duration_ms for m in endpoint_metrics)
                total_errors = sum(m.error_count for m in endpoint_metrics)
                
                trends.append({
                    "hour": hour_key,
                    "count": total_count,
                    "avg_duration_ms": total_duration / total_count if total_count > 0 else 0,
                    "error_count": total_errors,
                })
        
        return {
            "endpoint": endpoint,
            "trends": list(reversed(trends)),
            "period_hours": hours,
        }

