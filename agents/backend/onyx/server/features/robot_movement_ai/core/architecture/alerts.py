"""
Sistema de alertas y notificaciones para Robot Movement AI v2.0
Alertas configurables con múltiples canales
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class AlertLevel(str, Enum):
    """Niveles de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Canales de alerta disponibles"""
    LOG = "log"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"


@dataclass
class Alert:
    """Representa una alerta"""
    title: str
    message: str
    level: AlertLevel
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])


class AlertManager:
    """Gestor de alertas"""
    
    def __init__(self):
        """Inicializar gestor"""
        self.handlers: Dict[AlertChannel, List[Callable]] = {}
        self.alert_history: List[Alert] = []
        self.max_history: int = 1000
        self.enabled_channels: List[AlertChannel] = [AlertChannel.LOG]
    
    def register_handler(self, channel: AlertChannel, handler: Callable):
        """Registrar handler para un canal"""
        if channel not in self.handlers:
            self.handlers[channel] = []
        self.handlers[channel].append(handler)
    
    async def send_alert(self, alert: Alert):
        """Enviar alerta a todos los canales configurados"""
        # Agregar a historial
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        # Enviar a cada canal
        channels_to_use = alert.channels or self.enabled_channels
        
        for channel in channels_to_use:
            if channel in self.handlers:
                for handler in self.handlers[channel]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(alert)
                        else:
                            handler(alert)
                    except Exception as e:
                        print(f"Error sending alert via {channel}: {e}")
    
    def get_history(self, level: Optional[AlertLevel] = None, limit: int = 100) -> List[Alert]:
        """Obtener historial de alertas"""
        alerts = self.alert_history
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts[-limit:]
    
    def clear_history(self):
        """Limpiar historial"""
        self.alert_history.clear()


# Handlers por defecto
async def log_handler(alert: Alert):
    """Handler por defecto para logging"""
    from core.architecture.logging_config import get_logger
    logger = get_logger()
    
    level_map = {
        AlertLevel.INFO: logger.info,
        AlertLevel.WARNING: logger.warning,
        AlertLevel.ERROR: logger.error,
        AlertLevel.CRITICAL: logger.critical
    }
    
    log_func = level_map.get(alert.level, logger.info)
    log_func(f"[ALERT] {alert.title}: {alert.message}")


async def webhook_handler(alert: Alert, webhook_url: str):
    """Handler para webhook"""
    if not HTTPX_AVAILABLE:
        return
    
    async with httpx.AsyncClient() as client:
        payload = {
            "title": alert.title,
            "message": alert.message,
            "level": alert.level.value,
            "source": alert.source,
            "timestamp": alert.timestamp.isoformat(),
            "metadata": alert.metadata
        }
        
        try:
            await client.post(webhook_url, json=payload, timeout=5.0)
        except Exception as e:
            print(f"Error sending webhook alert: {e}")


# Instancia global
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Obtener instancia global del gestor de alertas"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
        # Registrar handler por defecto
        _alert_manager.register_handler(AlertChannel.LOG, log_handler)
    return _alert_manager


async def send_alert(
    title: str,
    message: str,
    level: AlertLevel = AlertLevel.INFO,
    channels: Optional[List[AlertChannel]] = None
):
    """Helper para enviar alerta"""
    manager = get_alert_manager()
    alert = Alert(
        title=title,
        message=message,
        level=level,
        channels=channels
    )
    await manager.send_alert(alert)




