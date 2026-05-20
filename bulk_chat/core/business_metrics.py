"""
Business Metrics - Métricas de Negocio
======================================

Sistema de métricas de negocio con KPIs, análisis de conversión y insights empresariales.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class MetricCategory(Enum):
    """Categoría de métrica."""
    REVENUE = "revenue"
    USERS = "users"
    ENGAGEMENT = "engagement"
    CONVERSION = "conversion"
    RETENTION = "retention"
    PERFORMANCE = "performance"


@dataclass
class BusinessMetric:
    """Métrica de negocio."""
    metric_id: str
    category: MetricCategory
    name: str
    value: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KPI:
    """KPI (Key Performance Indicator)."""
    kpi_id: str
    name: str
    target_value: float
    current_value: float
    unit: str = ""
    trend: str = "stable"  # "increasing", "decreasing", "stable"
    status: str = "on_track"  # "on_track", "at_risk", "off_track"
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ConversionFunnel:
    """Embudo de conversión."""
    funnel_id: str
    name: str
    stages: List[str]
    stage_values: Dict[str, int] = field(default_factory=dict)
    conversion_rates: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class BusinessMetrics:
    """Gestor de métricas de negocio."""
    
    def __init__(self):
        self.metrics: Dict[str, List[BusinessMetric]] = defaultdict(list)
        self.kpis: Dict[str, KPI] = {}
        self.conversion_funnels: Dict[str, ConversionFunnel] = {}
        self.metric_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    def record_metric(
        self,
        metric_name: str,
        category: MetricCategory,
        value: float,
        unit: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Registrar métrica."""
        metric = BusinessMetric(
            metric_id=f"metric_{metric_name}_{datetime.now().timestamp()}",
            category=category,
            name=metric_name,
            value=value,
            unit=unit,
            metadata=metadata or {},
        )
        
        self.metrics[metric_name].append(metric)
        
        # Mantener solo últimos 10000 registros por métrica
        if len(self.metrics[metric_name]) > 10000:
            self.metrics[metric_name].pop(0)
        
        logger.debug(f"Recorded metric: {metric_name} = {value} {unit}")
    
    def define_kpi(
        self,
        kpi_id: str,
        name: str,
        target_value: float,
        unit: str = "",
        current_value: float = 0.0,
    ):
        """Definir KPI."""
        kpi = KPI(
            kpi_id=kpi_id,
            name=name,
            target_value=target_value,
            current_value=current_value,
            unit=unit,
        )
        
        self.kpis[kpi_id] = kpi
        logger.info(f"Defined KPI: {kpi_id} - {name}")
    
    def update_kpi(self, kpi_id: str, current_value: float):
        """Actualizar KPI."""
        kpi = self.kpis.get(kpi_id)
        if not kpi:
            return
        
        old_value = kpi.current_value
        kpi.current_value = current_value
        kpi.last_updated = datetime.now()
        
        # Calcular tendencia
        if current_value > old_value:
            kpi.trend = "increasing"
        elif current_value < old_value:
            kpi.trend = "decreasing"
        else:
            kpi.trend = "stable"
        
        # Calcular estado
        progress = (current_value / kpi.target_value * 100) if kpi.target_value > 0 else 0
        if progress >= 90:
            kpi.status = "on_track"
        elif progress >= 70:
            kpi.status = "at_risk"
        else:
            kpi.status = "off_track"
    
    def create_conversion_funnel(
        self,
        funnel_id: str,
        name: str,
        stages: List[str],
    ) -> str:
        """Crear embudo de conversión."""
        funnel = ConversionFunnel(
            funnel_id=funnel_id,
            name=name,
            stages=stages,
        )
        
        self.conversion_funnels[funnel_id] = funnel
        logger.info(f"Created conversion funnel: {funnel_id} - {name}")
        return funnel_id
    
    def record_funnel_stage(self, funnel_id: str, stage_name: str, value: int):
        """Registrar valor de stage en embudo."""
        funnel = self.conversion_funnels.get(funnel_id)
        if not funnel:
            return
        
        funnel.stage_values[stage_name] = value
        
        # Calcular tasas de conversión
        if len(funnel.stages) > 1:
            for i in range(1, len(funnel.stages)):
                current_stage = funnel.stages[i]
                previous_stage = funnel.stages[i - 1]
                
                current_value = funnel.stage_values.get(current_stage, 0)
                previous_value = funnel.stage_values.get(previous_stage, 0)
                
                if previous_value > 0:
                    conversion_rate = (current_value / previous_value) * 100
                    funnel.conversion_rates[f"{previous_stage}->{current_stage}"] = conversion_rate
    
    def get_metric_trend(
        self,
        metric_name: str,
        period_days: int = 30,
    ) -> Dict[str, Any]:
        """Obtener tendencia de métrica."""
        metrics = self.metrics.get(metric_name, [])
        
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_metrics = [
            m for m in metrics
            if m.timestamp >= cutoff_date
        ]
        
        if not recent_metrics:
            return {
                "metric_name": metric_name,
                "period_days": period_days,
                "data_points": 0,
            }
        
        values = [m.value for m in recent_metrics]
        
        return {
            "metric_name": metric_name,
            "period_days": period_days,
            "data_points": len(recent_metrics),
            "current_value": values[-1] if values else 0.0,
            "average": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "trend": "increasing" if len(values) > 1 and values[-1] > values[0] else "decreasing",
        }
    
    def get_kpi_status(self, kpi_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de KPI."""
        kpi = self.kpis.get(kpi_id)
        if not kpi:
            return None
        
        progress = (kpi.current_value / kpi.target_value * 100) if kpi.target_value > 0 else 0
        
        return {
            "kpi_id": kpi.kpi_id,
            "name": kpi.name,
            "target_value": kpi.target_value,
            "current_value": kpi.current_value,
            "unit": kpi.unit,
            "progress": progress,
            "trend": kpi.trend,
            "status": kpi.status,
            "last_updated": kpi.last_updated.isoformat(),
        }
    
    def get_funnel_analysis(self, funnel_id: str) -> Optional[Dict[str, Any]]:
        """Obtener análisis de embudo."""
        funnel = self.conversion_funnels.get(funnel_id)
        if not funnel:
            return None
        
        return {
            "funnel_id": funnel.funnel_id,
            "name": funnel.name,
            "stages": funnel.stages,
            "stage_values": funnel.stage_values,
            "conversion_rates": funnel.conversion_rates,
            "overall_conversion": (
                (list(funnel.stage_values.values())[-1] / list(funnel.stage_values.values())[0] * 100)
                if len(funnel.stage_values) >= 2 and list(funnel.stage_values.values())[0] > 0
                else 0.0
            ),
        }
    
    def get_business_summary(self) -> Dict[str, Any]:
        """Obtener resumen de negocio."""
        kpis_by_status: Dict[str, int] = defaultdict(int)
        metrics_by_category: Dict[str, int] = defaultdict(int)
        
        for kpi in self.kpis.values():
            kpis_by_status[kpi.status] += 1
        
        for metric_name, metric_list in self.metrics.items():
            if metric_list:
                category = metric_list[0].category.value
                metrics_by_category[category] += len(metric_list)
        
        return {
            "total_metrics": sum(len(m) for m in self.metrics.values()),
            "metrics_by_category": dict(metrics_by_category),
            "total_kpis": len(self.kpis),
            "kpis_by_status": dict(kpis_by_status),
            "total_funnels": len(self.conversion_funnels),
        }
















