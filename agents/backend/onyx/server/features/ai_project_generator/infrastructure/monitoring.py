"""
Monitoring Service - Servicio de monitoreo
==========================================

Servicio de monitoreo que abstrae Prometheus y otras herramientas.
"""

import logging
from typing import Dict, Any

from ..core.prometheus_metrics import (
    record_project_generation,
    record_cache_operation,
    update_queue_size,
    update_resource_metrics
)

logger = logging.getLogger(__name__)


class MonitoringService:
    """Servicio de monitoreo"""
    
    def record_metric(self, metric_name: str, value: Any, **labels):
        """Registra una métrica"""
        # Implementación básica, puede extenderse
        pass
    
    def record_project_generation(
        self,
        status: str,
        ai_type: str = "unknown",
        framework: str = "unknown",
        duration: float = None
    ):
        """Registra generación de proyecto"""
        record_project_generation(status, ai_type, framework, duration)
    
    def record_cache_operation(self, cache_type: str, hit: bool):
        """Registra operación de cache"""
        record_cache_operation(cache_type, hit)
    
    def update_queue_size(self, size: int):
        """Actualiza tamaño de cola"""
        update_queue_size(size)
    
    def update_resources(self, cpu: float, memory: int, disk: int):
        """Actualiza métricas de recursos"""
        update_resource_metrics(cpu, memory, disk)















