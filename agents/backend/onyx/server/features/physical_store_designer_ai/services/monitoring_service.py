"""
Monitoring Service - Sistema de alertas y monitoreo
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MonitoringService:
    """Servicio para monitoreo y alertas"""
    
    def __init__(self):
        self.alerts: Dict[str, List[Dict[str, Any]]] = {}
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
    
    def create_alert(
        self,
        store_id: str,
        level: AlertLevel,
        title: str,
        message: str,
        metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """Crear alerta"""
        
        alert = {
            "id": f"alert_{store_id}_{len(self.alerts.get(store_id, [])) + 1}",
            "store_id": store_id,
            "level": level.value,
            "title": title,
            "message": message,
            "metric": metric,
            "created_at": datetime.now().isoformat(),
            "acknowledged": False,
            "resolved": False
        }
        
        if store_id not in self.alerts:
            self.alerts[store_id] = []
        
        self.alerts[store_id].append(alert)
        
        logger.warning(f"Alerta creada: {title} para diseño {store_id}")
        return alert
    
    def check_design_health(
        self,
        design: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verificar salud del diseño"""
        alerts = []
        warnings = []
        
        store_id = design.get("store_id")
        
        # Verificar análisis financiero
        if not design.get("financial_analysis"):
            warnings.append("Análisis financiero faltante")
        
        # Verificar rentabilidad
        financial = design.get("financial_analysis", {})
        monthly_profit = financial.get("profitability", {}).get("monthly_profit", 0)
        
        if monthly_profit < 0:
            alerts.append({
                "level": AlertLevel.CRITICAL,
                "title": "Rentabilidad Negativa",
                "message": "El diseño proyecta pérdidas mensuales",
                "metric": "monthly_profit"
            })
        elif monthly_profit < 1000:
            alerts.append({
                "level": AlertLevel.WARNING,
                "title": "Rentabilidad Baja",
                "message": "Margen de ganancia muy bajo",
                "metric": "monthly_profit"
            })
        
        # Verificar punto de equilibrio
        break_even = financial.get("break_even", {}).get("months")
        if break_even and break_even > 24:
            alerts.append({
                "level": AlertLevel.WARNING,
                "title": "Punto de Equilibrio Largo",
                "message": f"Punto de equilibrio en {break_even} meses",
                "metric": "break_even"
            })
        
        # Verificar análisis de competencia
        if not design.get("competitor_analysis"):
            warnings.append("Análisis de competencia faltante")
        
        # Crear alertas
        for alert_data in alerts:
            self.create_alert(
                store_id=store_id,
                level=alert_data["level"],
                title=alert_data["title"],
                message=alert_data["message"],
                metric=alert_data.get("metric")
            )
        
        return {
            "store_id": store_id,
            "health_score": self._calculate_health_score(design, alerts),
            "alerts": alerts,
            "warnings": warnings,
            "status": "healthy" if not alerts else "needs_attention"
        }
    
    def _calculate_health_score(
        self,
        design: Dict[str, Any],
        alerts: List[Dict[str, Any]]
    ) -> float:
        """Calcular score de salud"""
        score = 100.0
        
        # Penalizar por alertas críticas
        for alert in alerts:
            if alert["level"] == AlertLevel.CRITICAL:
                score -= 20
            elif alert["level"] == AlertLevel.WARNING:
                score -= 10
        
        # Bonificar por tener análisis completos
        if design.get("financial_analysis"):
            score += 5
        if design.get("competitor_analysis"):
            score += 5
        if design.get("inventory_recommendations"):
            score += 5
        
        return max(0, min(100, score))
    
    def get_alerts(
        self,
        store_id: str,
        level: Optional[AlertLevel] = None,
        unresolved_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Obtener alertas"""
        alerts = self.alerts.get(store_id, [])
        
        if level:
            alerts = [a for a in alerts if a["level"] == level.value]
        
        if unresolved_only:
            alerts = [a for a in alerts if not a.get("resolved", False)]
        
        return alerts
    
    def acknowledge_alert(
        self,
        store_id: str,
        alert_id: str
    ) -> bool:
        """Reconocer alerta"""
        alerts = self.alerts.get(store_id, [])
        
        for alert in alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.now().isoformat()
                return True
        
        return False
    
    def resolve_alert(
        self,
        store_id: str,
        alert_id: str
    ) -> bool:
        """Resolver alerta"""
        alerts = self.alerts.get(store_id, [])
        
        for alert in alerts:
            if alert["id"] == alert_id:
                alert["resolved"] = True
                alert["resolved_at"] = datetime.now().isoformat()
                return True
        
        return False
    
    def track_metric(
        self,
        store_id: str,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Rastrear métrica"""
        metric = {
            "store_id": store_id,
            "metric_name": metric_name,
            "value": value,
            "timestamp": (timestamp or datetime.now()).isoformat()
        }
        
        if store_id not in self.metrics:
            self.metrics[store_id] = []
        
        self.metrics[store_id].append(metric)
        
        return metric
    
    def get_metrics(
        self,
        store_id: str,
        metric_name: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Obtener métricas"""
        metrics = self.metrics.get(store_id, [])
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        filtered = [
            m for m in metrics
            if datetime.fromisoformat(m["timestamp"]) >= cutoff
        ]
        
        if metric_name:
            filtered = [m for m in filtered if m["metric_name"] == metric_name]
        
        return filtered




