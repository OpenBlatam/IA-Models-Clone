"""
Alerts - Sistema de alertas y notificaciones proactivas
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severidad de alertas"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Tipos de alertas"""
    PERFORMANCE = "performance"
    ERROR_RATE = "error_rate"
    USAGE_LIMIT = "usage_limit"
    QUALITY = "quality"
    SECURITY = "security"
    SYSTEM = "system"


@dataclass
class Alert:
    """Alerta del sistema"""
    id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False


class AlertManager:
    """Gestor de alertas"""

    def __init__(self):
        """Inicializar gestor de alertas"""
        self.alerts: List[Alert] = []
        self.rules: List[Dict[str, Any]] = []
        self.handlers: Dict[AlertType, List[Callable]] = {}
        self.max_alerts = 1000

    def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Crear una alerta.

        Args:
            alert_type: Tipo de alerta
            severity: Severidad
            title: Título
            message: Mensaje
            metadata: Metadatos

        Returns:
            Alerta creada
        """
        alert_id = str(uuid.uuid4())
        alert = Alert(
            id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        
        # Limitar tamaño
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # Ejecutar handlers
        self._trigger_handlers(alert)
        
        logger.warning(f"Alerta creada: {title} ({severity.value})")
        return alert

    def register_handler(self, alert_type: AlertType, handler: Callable):
        """
        Registrar handler para un tipo de alerta.

        Args:
            alert_type: Tipo de alerta
            handler: Función handler
        """
        if alert_type not in self.handlers:
            self.handlers[alert_type] = []
        
        self.handlers[alert_type].append(handler)
        logger.info(f"Handler registrado para: {alert_type.value}")

    def _trigger_handlers(self, alert: Alert):
        """Ejecutar handlers para una alerta"""
        handlers = self.handlers.get(alert.alert_type, [])
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error en handler de alerta: {e}")

    def check_performance_threshold(
        self,
        metric: str,
        value: float,
        threshold: float,
        operator: str = "gt"
    ):
        """
        Verificar umbral de rendimiento.

        Args:
            metric: Nombre de la métrica
            value: Valor actual
            threshold: Umbral
            operator: Operador (gt, lt, eq)
        """
        should_alert = False
        
        if operator == "gt" and value > threshold:
            should_alert = True
        elif operator == "lt" and value < threshold:
            should_alert = True
        elif operator == "eq" and value == threshold:
            should_alert = True
        
        if should_alert:
            self.create_alert(
                AlertType.PERFORMANCE,
                AlertSeverity.WARNING,
                f"Umbral de rendimiento excedido: {metric}",
                f"El valor de {metric} ({value}) excede el umbral ({threshold})",
                {"metric": metric, "value": value, "threshold": threshold}
            )

    def check_error_rate(
        self,
        error_rate: float,
        threshold: float = 0.1
    ):
        """
        Verificar tasa de errores.

        Args:
            error_rate: Tasa de errores (0-1)
            threshold: Umbral
        """
        if error_rate > threshold:
            severity = AlertSeverity.CRITICAL if error_rate > 0.5 else AlertSeverity.ERROR
            self.create_alert(
                AlertType.ERROR_RATE,
                severity,
                "Tasa de errores alta",
                f"La tasa de errores ({error_rate:.2%}) excede el umbral ({threshold:.2%})",
                {"error_rate": error_rate, "threshold": threshold}
            )

    def get_alerts(
        self,
        alert_type: Optional[AlertType] = None,
        severity: Optional[AlertSeverity] = None,
        unresolved_only: bool = False,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Obtener alertas.

        Args:
            alert_type: Filtrar por tipo
            severity: Filtrar por severidad
            unresolved_only: Solo no resueltas
            limit: Límite de resultados

        Returns:
            Lista de alertas
        """
        alerts = self.alerts
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if unresolved_only:
            alerts = [a for a in alerts if not a.resolved]
        
        alerts = alerts[-limit:][::-1]
        
        return [
            {
                "id": a.id,
                "type": a.alert_type.value,
                "severity": a.severity.value,
                "title": a.title,
                "message": a.message,
                "metadata": a.metadata,
                "created_at": a.created_at.isoformat(),
                "acknowledged": a.acknowledged,
                "resolved": a.resolved
            }
            for a in alerts
        ]

    def acknowledge_alert(self, alert_id: str):
        """
        Reconocer una alerta.

        Args:
            alert_id: ID de la alerta
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                break

    def resolve_alert(self, alert_id: str):
        """
        Resolver una alerta.

        Args:
            alert_id: ID de la alerta
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                break






