"""
Sistema de métricas de negocio
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class BusinessMetric:
    """Métrica de negocio"""
    name: str
    value: float
    unit: str
    timestamp: str = None
    category: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario"""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "category": self.category,
            "metadata": self.metadata or {}
        }


class BusinessMetrics:
    """Sistema de métricas de negocio"""
    
    def __init__(self):
        """Inicializa el sistema"""
        self.metrics: List[BusinessMetric] = []
        self.max_metrics = 10000
    
    def record_metric(self, name: str, value: float, unit: str = "",
                     category: Optional[str] = None,
                     metadata: Optional[Dict] = None):
        """
        Registra una métrica
        
        Args:
            name: Nombre de la métrica
            value: Valor
            unit: Unidad
            category: Categoría
            metadata: Metadatos
        """
        metric = BusinessMetric(
            name=name,
            value=value,
            unit=unit,
            category=category,
            metadata=metadata
        )
        
        self.metrics.append(metric)
        
        # Mantener límite
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def get_metrics(self, name: Optional[str] = None,
                   category: Optional[str] = None,
                   days: int = 30) -> List[BusinessMetric]:
        """
        Obtiene métricas
        
        Args:
            name: Nombre de la métrica (opcional)
            category: Categoría (opcional)
            days: Días de historial
            
        Returns:
            Lista de métricas
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        filtered = [
            m for m in self.metrics
            if datetime.fromisoformat(m.timestamp) >= cutoff
        ]
        
        if name:
            filtered = [m for m in filtered if m.name == name]
        
        if category:
            filtered = [m for m in filtered if m.category == category]
        
        return filtered
    
    def get_kpis(self) -> Dict:
        """Obtiene KPIs principales"""
        # Calcular KPIs comunes
        total_analyses = len([m for m in self.metrics if m.name == "analysis_completed"])
        total_users = len(set(
            m.metadata.get("user_id") for m in self.metrics
            if m.metadata and "user_id" in m.metadata
        ))
        
        # Revenue (si está disponible)
        revenue_metrics = [m for m in self.metrics if m.name == "revenue"]
        total_revenue = sum(m.value for m in revenue_metrics)
        
        return {
            "total_analyses": total_analyses,
            "total_users": total_users,
            "total_revenue": total_revenue,
            "average_analysis_per_user": total_analyses / total_users if total_users > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_category_metrics(self, category: str, days: int = 30) -> Dict:
        """Obtiene métricas por categoría"""
        metrics = self.get_metrics(category=category, days=days)
        
        if not metrics:
            return {
                "category": category,
                "count": 0,
                "total": 0,
                "average": 0
            }
        
        values = [m.value for m in metrics]
        
        return {
            "category": category,
            "count": len(metrics),
            "total": sum(values),
            "average": sum(values) / len(values),
            "min": min(values),
            "max": max(values)
        }






