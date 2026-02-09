"""
Tracing Service - Servicio de tracing
"""

from typing import Any, Dict, Optional
from .logger import StructuredLogger
from .metrics import Metrics


class TracingService:
    """Servicio centralizado de tracing y observabilidad"""

    def __init__(self):
        """Inicializa el servicio de tracing"""
        self.logger = StructuredLogger()
        self.metrics = Metrics()

    def trace(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Traza una operación"""
        self.logger.info(f"Tracing: {operation}", extra=metadata or {})

    def log(self, level: str, message: str, **kwargs) -> None:
        """Registra un log"""
        getattr(self.logger, level.lower())(message, **kwargs)

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Registra una métrica"""
        self.metrics.record(name, value, tags)

