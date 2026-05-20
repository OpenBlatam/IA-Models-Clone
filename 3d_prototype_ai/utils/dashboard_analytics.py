"""
Dashboard Analytics - Sistema de analytics avanzado con dashboards
===================================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class DashboardAnalytics:
    """Sistema de analytics avanzado con dashboards"""
    
    def __init__(self):
        self.metrics_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.dashboards: Dict[str, Dict[str, Any]] = {}
    
    def record_metric(self, metric_name: str, value: float, 
                     tags: Optional[Dict[str, str]] = None, timestamp: Optional[datetime] = None):
        """Registra una métrica"""
        if not timestamp:
            timestamp = datetime.now()
        
        metric_entry = {
            "name": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": timestamp.isoformat()
        }
        
        self.metrics_data[metric_name].append(metric_entry)
        
        # Mantener solo últimas 10000 entradas por métrica
        if len(self.metrics_data[metric_name]) > 10000:
            self.metrics_data[metric_name] = self.metrics_data[metric_name][-10000:]
    
    def create_dashboard(self, dashboard_id: str, name: str, 
                        widgets: List[Dict[str, Any]]):
        """Crea un dashboard"""
        self.dashboards[dashboard_id] = {
            "id": dashboard_id,
            "name": name,
            "widgets": widgets,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        logger.info(f"Dashboard creado: {dashboard_id}")
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene un dashboard"""
        return self.dashboards.get(dashboard_id)
    
    def get_overview_dashboard(self) -> Dict[str, Any]:
        """Obtiene dashboard de overview"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)
        
        # Calcular métricas
        prototypes_24h = sum(
            1 for m in self.metrics_data.get("prototypes_generated", [])
            if datetime.fromisoformat(m["timestamp"]) > last_24h
        )
        
        prototypes_7d = sum(
            1 for m in self.metrics_data.get("prototypes_generated", [])
            if datetime.fromisoformat(m["timestamp"]) > last_7d
        )
        
        avg_cost = self._calculate_avg_metric("prototype_cost", last_24h)
        avg_time = self._calculate_avg_metric("prototype_generation_time", last_24h)
        
        return {
            "dashboard_id": "overview",
            "name": "Overview Dashboard",
            "widgets": [
                {
                    "type": "stat",
                    "title": "Prototipos (24h)",
                    "value": prototypes_24h,
                    "change": prototypes_24h - prototypes_7d // 7,
                    "trend": "up" if prototypes_24h > prototypes_7d // 7 else "down"
                },
                {
                    "type": "stat",
                    "title": "Costo Promedio",
                    "value": f"${avg_cost:.2f}",
                    "change": 0,
                    "trend": "stable"
                },
                {
                    "type": "stat",
                    "title": "Tiempo Promedio",
                    "value": f"{avg_time:.2f}s",
                    "change": 0,
                    "trend": "stable"
                },
                {
                    "type": "chart",
                    "title": "Prototipos por Día",
                    "chart_type": "line",
                    "data": self._get_daily_prototypes(last_7d)
                },
                {
                    "type": "chart",
                    "title": "Tipos de Productos",
                    "chart_type": "pie",
                    "data": self._get_product_types_distribution(last_7d)
                }
            ],
            "updated_at": now.isoformat()
        }
    
    def _calculate_avg_metric(self, metric_name: str, since: datetime) -> float:
        """Calcula promedio de una métrica"""
        metrics = [
            m for m in self.metrics_data.get(metric_name, [])
            if datetime.fromisoformat(m["timestamp"]) > since
        ]
        
        if not metrics:
            return 0.0
        
        return sum(m["value"] for m in metrics) / len(metrics)
    
    def _get_daily_prototypes(self, since: datetime) -> List[Dict[str, Any]]:
        """Obtiene prototipos por día"""
        daily_count = defaultdict(int)
        
        for metric in self.metrics_data.get("prototypes_generated", []):
            timestamp = datetime.fromisoformat(metric["timestamp"])
            if timestamp > since:
                day = timestamp.strftime("%Y-%m-%d")
                daily_count[day] += 1
        
        return [
            {"date": date, "count": count}
            for date, count in sorted(daily_count.items())
        ]
    
    def _get_product_types_distribution(self, since: datetime) -> List[Dict[str, Any]]:
        """Obtiene distribución de tipos de productos"""
        type_count = defaultdict(int)
        
        for metric in self.metrics_data.get("prototype_type", []):
            timestamp = datetime.fromisoformat(metric["timestamp"])
            if timestamp > since:
                product_type = metric.get("tags", {}).get("type", "unknown")
                type_count[product_type] += 1
        
        total = sum(type_count.values())
        
        return [
            {
                "label": ptype,
                "value": count,
                "percentage": (count / total * 100) if total > 0 else 0
            }
            for ptype, count in type_count.items()
        ]
    
    def get_metrics_summary(self, metric_name: str, 
                           time_range: timedelta = timedelta(hours=24)) -> Dict[str, Any]:
        """Obtiene resumen de métricas"""
        cutoff = datetime.now() - time_range
        metrics = [
            m for m in self.metrics_data.get(metric_name, [])
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        
        if not metrics:
            return {
                "metric": metric_name,
                "count": 0,
                "avg": 0,
                "min": 0,
                "max": 0
            }
        
        values = [m["value"] for m in metrics]
        
        return {
            "metric": metric_name,
            "count": len(metrics),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values)
        }




