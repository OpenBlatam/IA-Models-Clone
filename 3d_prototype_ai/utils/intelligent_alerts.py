"""
Intelligent Alerts - Sistema de alertas inteligentes
=====================================================
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Severidad de alertas"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class IntelligentAlerts:
    """Sistema de alertas inteligentes"""
    
    def __init__(self):
        self.alerts: List[Dict[str, Any]] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.suppressions: Dict[str, datetime] = {}
    
    def create_alert_rule(self, rule_id: str, name: str, condition: Callable,
                         severity: AlertSeverity = AlertSeverity.WARNING,
                         action: Optional[Callable] = None):
        """Crea una regla de alerta"""
        self.alert_rules[rule_id] = {
            "id": rule_id,
            "name": name,
            "condition": condition,
            "severity": severity.value,
            "action": action,
            "enabled": True,
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Regla de alerta creada: {rule_id}")
    
    def evaluate_alert(self, rule_id: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evalúa una regla de alerta"""
        rule = self.alert_rules.get(rule_id)
        if not rule or not rule["enabled"]:
            return None
        
        # Verificar supresión
        if rule_id in self.suppressions:
            suppress_until = self.suppressions[rule_id]
            if datetime.now() < suppress_until:
                return None
        
        try:
            condition = rule["condition"]
            if condition(context):
                alert = {
                    "id": f"alert_{len(self.alerts)}",
                    "rule_id": rule_id,
                    "rule_name": rule["name"],
                    "severity": rule["severity"],
                    "message": f"Alerta: {rule['name']}",
                    "context": context,
                    "timestamp": datetime.now().isoformat(),
                    "acknowledged": False
                }
                
                self.alerts.append(alert)
                self.alert_history.append(alert)
                
                # Mantener solo últimas 1000 alertas
                if len(self.alerts) > 1000:
                    self.alerts = self.alerts[-1000:]
                
                # Ejecutar acción si existe
                if rule["action"]:
                    try:
                        rule["action"](alert)
                    except Exception as e:
                        logger.error(f"Error ejecutando acción de alerta: {e}")
                
                logger.warning(f"Alerta disparada: {rule_id} - {rule['name']}")
                return alert
        
        except Exception as e:
            logger.error(f"Error evaluando regla de alerta {rule_id}: {e}")
        
        return None
    
    def suppress_alert(self, rule_id: str, duration_minutes: int = 60):
        """Suprime una alerta temporalmente"""
        self.suppressions[rule_id] = datetime.now() + timedelta(minutes=duration_minutes)
        logger.info(f"Alerta suprimida: {rule_id} por {duration_minutes} minutos")
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None,
                         unacknowledged_only: bool = False) -> List[Dict[str, Any]]:
        """Obtiene alertas activas"""
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity.value]
        
        if unacknowledged_only:
            alerts = [a for a in alerts if not a["acknowledged"]]
        
        return sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
    
    def acknowledge_alert(self, alert_id: str):
        """Reconoce una alerta"""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_at"] = datetime.now().isoformat()
                logger.info(f"Alerta reconocida: {alert_id}")
                return True
        return False
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de alertas"""
        severity_counts = defaultdict(int)
        for alert in self.alerts:
            severity_counts[alert["severity"]] += 1
        
        return {
            "total_alerts": len(self.alerts),
            "active_alerts": sum(1 for a in self.alerts if not a["acknowledged"]),
            "by_severity": dict(severity_counts),
            "total_rules": len(self.alert_rules),
            "active_rules": sum(1 for r in self.alert_rules.values() if r["enabled"])
        }




