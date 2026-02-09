"""
Realtime Analytics Service - Análisis de datos en tiempo real
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class RealtimeAnalyticsService:
    """Servicio para análisis en tiempo real"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.dashboards: Dict[str, Dict[str, Any]] = {}
    
    def record_metric(
        self,
        store_id: str,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Registrar métrica en tiempo real"""
        
        metric = {
            "metric_id": f"metric_{store_id}_{len(self.metrics.get(store_id, [])) + 1}",
            "store_id": store_id,
            "name": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.now().isoformat()
        }
        
        if store_id not in self.metrics:
            self.metrics[store_id] = []
        
        self.metrics[store_id].append(metric)
        
        # Mantener solo últimas 10000 métricas por tienda
        if len(self.metrics[store_id]) > 10000:
            self.metrics[store_id] = self.metrics[store_id][-10000:]
        
        return metric
    
    def get_realtime_dashboard(
        self,
        store_id: str,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """Obtener dashboard en tiempo real"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        
        store_metrics = self.metrics.get(store_id, [])
        recent_metrics = [
            m for m in store_metrics
            if start_time <= datetime.fromisoformat(m["timestamp"]) <= end_time
        ]
        
        # Agrupar por nombre de métrica
        metrics_by_name = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_name[metric["name"]].append(metric)
        
        dashboard = {
            "store_id": store_id,
            "time_window_minutes": time_window_minutes,
            "last_updated": datetime.now().isoformat(),
            "metrics": {}
        }
        
        for metric_name, metrics in metrics_by_name.items():
            values = [m["value"] for m in metrics]
            
            dashboard["metrics"][metric_name] = {
                "count": len(metrics),
                "current": values[-1] if values else None,
                "average": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "trend": self._calculate_trend(values),
                "latest_timestamp": metrics[-1]["timestamp"] if metrics else None
            }
        
        return dashboard
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calcular tendencia"""
        if len(values) < 2:
            return "stable"
        
        recent = values[-5:] if len(values) >= 5 else values
        older = values[-10:-5] if len(values) >= 10 else values[:len(values)//2]
        
        if not older:
            return "stable"
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        change = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        
        if change > 5:
            return "increasing"
        elif change < -5:
            return "decreasing"
        else:
            return "stable"
    
    def get_metric_history(
        self,
        store_id: str,
        metric_name: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Obtener historial de métrica"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        store_metrics = self.metrics.get(store_id, [])
        metric_data = [
            m for m in store_metrics
            if m["name"] == metric_name
            and start_time <= datetime.fromisoformat(m["timestamp"]) <= end_time
        ]
        
        values = [m["value"] for m in metric_data]
        
        return {
            "store_id": store_id,
            "metric_name": metric_name,
            "period_hours": hours,
            "data_points": len(metric_data),
            "values": values,
            "timestamps": [m["timestamp"] for m in metric_data],
            "statistics": {
                "average": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "current": values[-1] if values else None
            }
        }
    
    def detect_anomalies(
        self,
        store_id: str,
        metric_name: str,
        threshold_std: float = 2.0
    ) -> List[Dict[str, Any]]:
        """Detectar anomalías en métricas"""
        
        store_metrics = self.metrics.get(store_id, [])
        metric_data = [m for m in store_metrics if m["name"] == metric_name]
        
        if len(metric_data) < 20:
            return []
        
        values = [m["value"] for m in metric_data[-100:]]  # Últimas 100
        mean = sum(values) / len(values)
        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        
        threshold = threshold_std * std_dev
        anomalies = []
        
        for metric in metric_data[-100:]:
            deviation = abs(metric["value"] - mean)
            if deviation > threshold:
                anomalies.append({
                    "metric": metric,
                    "deviation": deviation,
                    "severity": "high" if deviation > 3 * std_dev else "medium"
                })
        
        return anomalies
    
    def create_custom_dashboard(
        self,
        store_id: str,
        dashboard_name: str,
        metrics: List[str],
        refresh_interval_seconds: int = 60
    ) -> Dict[str, Any]:
        """Crear dashboard personalizado"""
        
        dashboard_id = f"dashboard_{store_id}_{len(self.dashboards.get(store_id, [])) + 1}"
        
        dashboard = {
            "dashboard_id": dashboard_id,
            "store_id": store_id,
            "name": dashboard_name,
            "metrics": metrics,
            "refresh_interval_seconds": refresh_interval_seconds,
            "created_at": datetime.now().isoformat(),
            "is_active": True
        }
        
        if store_id not in self.dashboards:
            self.dashboards[store_id] = []
        
        self.dashboards[store_id].append(dashboard)
        
        return dashboard
    
    def get_dashboards(self, store_id: str) -> List[Dict[str, Any]]:
        """Obtener dashboards"""
        return self.dashboards.get(store_id, [])




