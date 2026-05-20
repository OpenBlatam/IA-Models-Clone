"""
Alerting - Sistema de alertas avanzado
=======================================

Sistema de alertas con reglas y acciones automáticas.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severidad de alerta"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertCondition(Enum):
    """Condiciones de alerta"""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    CONTAINS = "contains"
    MATCHES = "matches"


@dataclass
class AlertRule:
    """Regla de alerta"""
    id: str
    name: str
    condition: AlertCondition
    threshold: Any
    metric_name: str
    severity: AlertSeverity
    enabled: bool = True
    action: Optional[Callable] = None
    cooldown_seconds: int = 300  # 5 minutos
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:
    """Alerta"""
    id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    value: Any
    threshold: Any
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False


class AlertManager:
    """Gestor de alertas"""
    
    def __init__(self, agent):
        self.agent = agent
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_history = 1000
        self.running = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    def add_rule(
        self,
        name: str,
        metric_name: str,
        condition: AlertCondition,
        threshold: Any,
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        action: Optional[Callable] = None,
        cooldown_seconds: int = 300
    ) -> str:
        """Agregar regla de alerta"""
        rule_id = f"alert_rule_{datetime.now().timestamp()}_{len(self.rules)}"
        
        rule = AlertRule(
            id=rule_id,
            name=name,
            condition=condition,
            threshold=threshold,
            metric_name=metric_name,
            severity=severity,
            action=action,
            cooldown_seconds=cooldown_seconds
        )
        
        self.rules[rule_id] = rule
        logger.info(f"🚨 Alert rule added: {name}")
        return rule_id
    
    async def start(self):
        """Iniciar monitor de alertas"""
        if self.running:
            return
        
        self.running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("🚨 Alert manager started")
    
    async def stop(self):
        """Detener monitor de alertas"""
        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Loop de monitoreo"""
        while self.running:
            try:
                await self._check_rules()
                await asyncio.sleep(10)  # Verificar cada 10 segundos
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert monitor loop: {e}")
                await asyncio.sleep(30)
    
    async def _check_rules(self):
        """Verificar todas las reglas"""
        if not self.agent.metrics:
            return
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # Verificar cooldown
            if rule.last_triggered:
                time_since = (datetime.now() - rule.last_triggered).total_seconds()
                if time_since < rule.cooldown_seconds:
                    continue
            
            # Obtener valor de métrica
            metric_value = self._get_metric_value(rule.metric_name)
            if metric_value is None:
                continue
            
            # Verificar condición
            if self._check_condition(metric_value, rule.condition, rule.threshold):
                await self._trigger_alert(rule, metric_value)
    
    def _get_metric_value(self, metric_name: str) -> Optional[Any]:
        """Obtener valor de métrica"""
        if not self.agent.metrics:
            return None
        
        # Intentar obtener de diferentes fuentes
        if metric_name.startswith("counter_"):
            counter_name = metric_name.replace("counter_", "")
            return self.agent.metrics.get_counter(counter_name)
        elif metric_name.startswith("gauge_"):
            gauge_name = metric_name.replace("gauge_", "")
            return self.agent.metrics.get_gauge(gauge_name)
        else:
            # Buscar en estado del agente
            return None
    
    def _check_condition(self, value: Any, condition: AlertCondition, threshold: Any) -> bool:
        """Verificar condición"""
        try:
            if condition == AlertCondition.GREATER_THAN:
                return float(value) > float(threshold)
            elif condition == AlertCondition.LESS_THAN:
                return float(value) < float(threshold)
            elif condition == AlertCondition.EQUAL:
                return value == threshold
            elif condition == AlertCondition.NOT_EQUAL:
                return value != threshold
            elif condition == AlertCondition.CONTAINS:
                return str(threshold) in str(value)
            elif condition == AlertCondition.MATCHES:
                import re
                return bool(re.search(str(threshold), str(value)))
            return False
        except Exception as e:
            logger.error(f"Error checking condition: {e}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule, value: Any):
        """Disparar alerta"""
        alert_id = f"alert_{datetime.now().timestamp()}"
        
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            rule_name=rule.name,
            severity=rule.severity,
            message=f"{rule.name}: {value} {rule.condition.value} {rule.threshold}",
            value=value,
            threshold=rule.threshold
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Limitar historial
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]
        
        rule.last_triggered = datetime.now()
        
        logger.warning(f"🚨 ALERT [{rule.severity.value.upper()}]: {alert.message}")
        
        # Ejecutar acción si existe
        if rule.action:
            try:
                if asyncio.iscoroutinefunction(rule.action):
                    await rule.action(alert)
                else:
                    rule.action(alert)
            except Exception as e:
                logger.error(f"Error executing alert action: {e}")
        
        # Notificar si hay sistema de notificaciones
        if self.agent.notifications:
            await self.agent.notifications.notify(
                f"Alert: {rule.name}",
                alert.message,
                level=self.agent.notifications.NotificationLevel.ERROR if rule.severity == AlertSeverity.CRITICAL
                else self.agent.notifications.NotificationLevel.WARNING,
                metadata={"alert_id": alert_id, "severity": rule.severity.value}
            )
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Reconocer alerta"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolver alerta"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            del self.active_alerts[alert_id]
            return True
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Obtener alertas activas"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Obtener historial de alertas"""
        return sorted(self.alert_history, key=lambda x: x.timestamp, reverse=True)[:limit]


