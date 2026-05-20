"""
Alerting System
===============

System for alerts and notifications based on conditions.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert definition."""
    id: str
    name: str
    message: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "message": self.message,
            "severity": self.severity.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
            "tags": self.tags
        }


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: Callable[[], Awaitable[bool]]
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: float = 300.0  # 5 minutes default
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize rule."""
        self.last_triggered: Optional[datetime] = None


class AlertManager:
    """Alert manager."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.alerts: Dict[str, Alert] = {}
        self.rules: Dict[str, AlertRule] = {}
        self.handlers: List[Callable[[Alert], Awaitable[None]]] = []
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
    
    def register_rule(self, rule: AlertRule):
        """
        Register an alert rule.
        
        Args:
            rule: Alert rule
        """
        self.rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")
    
    def register_handler(self, handler: Callable[[Alert], Awaitable[None]]):
        """
        Register an alert handler.
        
        Args:
            handler: Async function to handle alerts
        """
        self.handlers.append(handler)
        logger.info(f"Registered alert handler: {handler.__name__}")
    
    async def check_rules(self):
        """Check all alert rules."""
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule.last_triggered:
                elapsed = (datetime.now() - rule.last_triggered).total_seconds()
                if elapsed < rule.cooldown_seconds:
                    continue
            
            try:
                # Check condition
                condition_met = await rule.condition()
                
                if condition_met:
                    # Create alert
                    alert_id = f"{rule_name}_{datetime.now().timestamp()}"
                    alert = Alert(
                        id=alert_id,
                        name=rule_name,
                        message=rule.message_template,
                        severity=rule.severity,
                        tags=rule.tags
                    )
                    
                    # Store alert
                    self.alerts[alert_id] = alert
                    rule.last_triggered = datetime.now()
                    
                    # Trigger handlers
                    for handler in self.handlers:
                        try:
                            await handler(alert)
                        except Exception as e:
                            logger.error(f"Error in alert handler: {e}")
                    
                    logger.warning(f"Alert triggered: {rule_name}")
            
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")
    
    async def start_monitoring(self, interval_seconds: float = 60.0):
        """
        Start monitoring alert rules.
        
        Args:
            interval_seconds: Check interval in seconds
        """
        if self._running:
            return
        
        self._running = True
        
        async def monitor_loop():
            while self._running:
                await self.check_rules()
                await asyncio.sleep(interval_seconds)
        
        self._check_task = asyncio.create_task(monitor_loop())
        logger.info("Alert monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Alert monitoring stopped")
    
    def acknowledge_alert(self, alert_id: str):
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
        """
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            logger.info(f"Alert acknowledged: {alert_id}")
    
    def resolve_alert(self, alert_id: str):
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID
        """
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.now()
            logger.info(f"Alert resolved: {alert_id}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return [
            alert for alert in self.alerts.values()
            if alert.status == AlertStatus.ACTIVE
        ]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """
        Get alerts by severity.
        
        Args:
            severity: Alert severity
            
        Returns:
            List of alerts
        """
        return [
            alert for alert in self.alerts.values()
            if alert.severity == severity
        ]

