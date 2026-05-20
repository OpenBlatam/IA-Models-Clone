"""
Intelligent Alerts System
=========================

Sistema de alertas inteligentes con priorización y agrupación.
"""

import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severidad de alerta."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Estado de alerta."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alerta del sistema."""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    category: str
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class IntelligentAlertSystem:
    """Sistema de alertas inteligentes."""
    
    def __init__(
        self,
        deduplication_window: int = 300,  # segundos
        max_alerts: int = 1000
    ):
        self.deduplication_window = deduplication_window
        self.max_alerts = max_alerts
        self.alerts: Dict[str, Alert] = {}
        self.alert_groups: Dict[str, List[str]] = defaultdict(list)
        self.alert_callbacks: List[Any] = []
        self.suppressed_alerts: Set[str] = set()
    
    def register_callback(self, callback: Any) -> None:
        """Registra callback para alertas."""
        self.alert_callbacks.append(callback)
    
    def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        category: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Crea una nueva alerta."""
        # Generar ID único
        alert_id = f"{source}_{category}_{int(time.time())}"
        
        # Verificar si hay alerta similar reciente (deduplicación)
        similar_alert = self._find_similar_alert(title, message, category)
        if similar_alert:
            # Actualizar alerta existente
            similar_alert.timestamp = datetime.now()
            similar_alert.metadata.update(metadata or {})
            return similar_alert
        
        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            severity=severity,
            category=category,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        
        # Agrupar por categoría
        self.alert_groups[category].append(alert_id)
        
        # Limpiar alertas antiguas
        self._cleanup_old_alerts()
        
        # Notificar callbacks
        self._notify_callbacks(alert)
        
        return alert
    
    def _find_similar_alert(
        self,
        title: str,
        message: str,
        category: str
    ) -> Optional[Alert]:
        """Encuentra alerta similar reciente."""
        cutoff_time = datetime.now() - timedelta(seconds=self.deduplication_window)
        
        for alert in self.alerts.values():
            if (alert.category == category and
                alert.title == title and
                alert.timestamp > cutoff_time and
                alert.status == AlertStatus.ACTIVE):
                return alert
        
        return None
    
    def _cleanup_old_alerts(self) -> None:
        """Limpia alertas antiguas."""
        if len(self.alerts) <= self.max_alerts:
            return
        
        # Ordenar por timestamp y eliminar las más antiguas
        sorted_alerts = sorted(
            self.alerts.items(),
            key=lambda x: x[1].timestamp
        )
        
        to_remove = len(self.alerts) - self.max_alerts
        for alert_id, _ in sorted_alerts[:to_remove]:
            del self.alerts[alert_id]
            # Remover de grupos
            for group in self.alert_groups.values():
                if alert_id in group:
                    group.remove(alert_id)
    
    def _notify_callbacks(self, alert: Alert) -> None:
        """Notifica callbacks sobre nueva alerta."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Marca alerta como reconocida."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = user
        alert.acknowledged_at = datetime.now()
        
        return True
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resuelve una alerta."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        return True
    
    def suppress_alert(self, alert_id: str) -> bool:
        """Suprime una alerta."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.SUPPRESSED
        self.suppressed_alerts.add(alert_id)
        
        return True
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        category: Optional[str] = None
    ) -> List[Alert]:
        """Obtiene alertas activas."""
        alerts = [
            alert for alert in self.alerts.values()
            if alert.status == AlertStatus.ACTIVE
        ]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        return sorted(alerts, key=lambda x: (
            x.severity.value,
            x.timestamp
        ), reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Obtiene resumen de alertas."""
        active = self.get_active_alerts()
        
        by_severity = defaultdict(int)
        by_category = defaultdict(int)
        
        for alert in active:
            by_severity[alert.severity.value] += 1
            by_category[alert.category] += 1
        
        return {
            "total_active": len(active),
            "by_severity": dict(by_severity),
            "by_category": dict(by_category),
            "critical_count": len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
            "high_count": len([a for a in active if a.severity == AlertSeverity.HIGH])
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas."""
        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(self.get_active_alerts()),
            "summary": self.get_alert_summary()
        }


# Factory function
_alert_system = None

def get_alert_system() -> IntelligentAlertSystem:
    """Obtiene instancia global del sistema de alertas."""
    global _alert_system
    if _alert_system is None:
        _alert_system = IntelligentAlertSystem()
    return _alert_system


