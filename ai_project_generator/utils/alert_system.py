"""
Alert System - Sistema de Alertas
==================================

Sistema de alertas y notificaciones avanzado.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertSystem:
    """Sistema de alertas"""

    def __init__(self):
        """Inicializa el sistema de alertas"""
        self.alerts: List[Dict[str, Any]] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history = 1000

    def create_alert_rule(
        self,
        rule_name: str,
        condition: str,
        level: AlertLevel,
        message: str,
        enabled: bool = True,
    ) -> str:
        """
        Crea una regla de alerta.

        Args:
            rule_name: Nombre de la regla
            condition: Condición a evaluar
            level: Nivel de alerta
            message: Mensaje de la alerta
            enabled: Si está habilitada

        Returns:
            ID de la regla
        """
        rule_id = f"rule_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        self.alert_rules[rule_id] = {
            "id": rule_id,
            "name": rule_name,
            "condition": condition,
            "level": level.value,
            "message": message,
            "enabled": enabled,
            "created_at": datetime.now().isoformat(),
        }

        logger.info(f"Regla de alerta creada: {rule_name}")
        return rule_id

    def trigger_alert(
        self,
        level: AlertLevel,
        message: str,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dispara una alerta.

        Args:
            level: Nivel de alerta
            message: Mensaje
            source: Fuente de la alerta
            metadata: Metadata adicional

        Returns:
            Información de la alerta
        """
        alert = {
            "id": f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "level": level.value,
            "message": message,
            "source": source or "system",
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "acknowledged": False,
        }

        self.alerts.append(alert)
        self.alert_history.append(alert)

        # Limitar historial
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

        logger.warning(f"Alerta {level.value}: {message}")
        return alert

    def get_active_alerts(
        self,
        level: Optional[AlertLevel] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Obtiene alertas activas.

        Args:
            level: Filtrar por nivel
            limit: Límite de resultados

        Returns:
            Lista de alertas activas
        """
        alerts = [a for a in self.alerts if not a.get("acknowledged", False)]

        if level:
            alerts = [a for a in alerts if a["level"] == level.value]

        return alerts[:limit]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Marca una alerta como reconocida.

        Args:
            alert_id: ID de la alerta

        Returns:
            True si se reconoció exitosamente
        """
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.now().isoformat()
                logger.info(f"Alerta {alert_id} reconocida")
                return True
        return False

    def get_alert_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de alertas"""
        level_counts = defaultdict(int)
        for alert in self.alert_history:
            level_counts[alert["level"]] += 1

        return {
            "total_alerts": len(self.alert_history),
            "active_alerts": len([a for a in self.alerts if not a.get("acknowledged", False)]),
            "by_level": dict(level_counts),
            "rules_count": len(self.alert_rules),
        }


