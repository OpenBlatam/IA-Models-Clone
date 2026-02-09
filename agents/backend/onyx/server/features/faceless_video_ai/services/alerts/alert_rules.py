"""
Alert Rules
Alert rule definitions
"""

from typing import Callable, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertRule:
    """Represents an alert rule"""
    
    def __init__(
        self,
        name: str,
        condition: Callable[[Dict[str, Any]], bool],
        severity: AlertSeverity,
        message: str,
        cooldown_seconds: int = 300
    ):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.message = message
        self.cooldown_seconds = cooldown_seconds

