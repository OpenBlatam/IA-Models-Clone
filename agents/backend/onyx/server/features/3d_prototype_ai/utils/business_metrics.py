"""
Business Metrics - Sistema de métricas de negocio
===================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class BusinessMetrics:
    """Sistema de métricas de negocio"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.kpis: Dict[str, float] = {}
    
    def record_metric(self, metric_name: str, value: float,
                     category: str = "general", metadata: Optional[Dict[str, Any]] = None):
        """Registra una métrica de negocio"""
        metric_entry = {
            "name": metric_name,
            "value": value,
            "category": category,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat()
        }
        
        self.metrics[metric_name].append(metric_entry)
        
        # Actualizar KPI si aplica
        if metric_name.startswith("kpi_"):
            self.kpis[metric_name] = value
        
        logger.debug(f"Métrica de negocio registrada: {metric_name} = {value}")
    
    def calculate_mrr(self, subscriptions: List[Dict[str, Any]]) -> float:
        """Calcula Monthly Recurring Revenue"""
        mrr = sum(sub.get("plan", {}).get("price", 0) for sub in subscriptions)
        self.record_metric("kpi_mrr", mrr, "revenue")
        return mrr
    
    def calculate_cac(self, marketing_spend: float, new_customers: int) -> float:
        """Calcula Customer Acquisition Cost"""
        cac = marketing_spend / new_customers if new_customers > 0 else 0
        self.record_metric("kpi_cac", cac, "marketing")
        return cac
    
    def calculate_ltv(self, avg_revenue_per_user: float, avg_customer_lifespan_months: float) -> float:
        """Calcula Lifetime Value"""
        ltv = avg_revenue_per_user * avg_customer_lifespan_months
        self.record_metric("kpi_ltv", ltv, "revenue")
        return ltv
    
    def calculate_churn_rate(self, lost_customers: int, total_customers: int) -> float:
        """Calcula tasa de churn"""
        churn = (lost_customers / total_customers * 100) if total_customers > 0 else 0
        self.record_metric("kpi_churn_rate", churn, "retention")
        return churn
    
    def get_business_dashboard(self) -> Dict[str, Any]:
        """Obtiene dashboard de métricas de negocio"""
        return {
            "kpis": self.kpis,
            "revenue_metrics": self._get_revenue_metrics(),
            "user_metrics": self._get_user_metrics(),
            "product_metrics": self._get_product_metrics(),
            "generated_at": datetime.now().isoformat()
        }
    
    def _get_revenue_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de ingresos"""
        revenue_metrics = [
            m for m in self.metrics.get("kpi_mrr", [])
            if m["category"] == "revenue"
        ]
        
        return {
            "mrr": self.kpis.get("kpi_mrr", 0),
            "ltv": self.kpis.get("kpi_ltv", 0),
            "total_revenue": sum(m["value"] for m in revenue_metrics)
        }
    
    def _get_user_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de usuarios"""
        return {
            "total_users": 0,  # Vendría de base de datos
            "active_users": 0,
            "new_users_today": 0,
            "churn_rate": self.kpis.get("kpi_churn_rate", 0)
        }
    
    def _get_product_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de producto"""
        return {
            "prototypes_generated": 0,
            "average_prototype_cost": 0,
            "most_popular_product_type": "licuadora"
        }
    
    def get_kpi_trends(self, kpi_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Obtiene tendencias de un KPI"""
        cutoff = datetime.now() - timedelta(days=days)
        
        kpi_metrics = [
            m for m in self.metrics.get(kpi_name, [])
            if datetime.fromisoformat(m["timestamp"]) > cutoff
        ]
        
        return sorted(kpi_metrics, key=lambda x: x["timestamp"])




