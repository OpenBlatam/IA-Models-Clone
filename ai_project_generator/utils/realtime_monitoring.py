"""
Realtime Monitoring - Monitoreo en Tiempo Real
==============================================

Sistema de monitoreo en tiempo real del sistema.
"""

import logging
import asyncio
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


class RealtimeMonitor:
    """Monitor en tiempo real"""

    def __init__(self):
        """Inicializa el monitor"""
        self.metrics_buffer: deque = deque(maxlen=1000)
        self.alerts: List[Dict[str, Any]] = []
        self.monitoring_active = False

    async def start_monitoring(
        self,
        interval_seconds: int = 5,
    ):
        """
        Inicia el monitoreo en tiempo real.

        Args:
            interval_seconds: Intervalo de monitoreo en segundos
        """
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_buffer.append(metrics)
                
                # Verificar alertas
                self._check_alerts(metrics)
                
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error en monitoreo: {e}")

    def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.monitoring_active = False

    def _collect_metrics(self) -> Dict[str, Any]:
        """Recolecta métricas del sistema"""
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0,
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
        }

    def _check_alerts(self, metrics: Dict[str, Any]):
        """Verifica alertas basadas en métricas"""
        # Alerta de CPU alto
        if metrics["cpu_percent"] > 90:
            self.alerts.append({
                "type": "high_cpu",
                "value": metrics["cpu_percent"],
                "timestamp": metrics["timestamp"],
                "severity": "warning",
            })

        # Alerta de memoria alta
        if metrics["memory_percent"] > 90:
            self.alerts.append({
                "type": "high_memory",
                "value": metrics["memory_percent"],
                "timestamp": metrics["timestamp"],
                "severity": "critical",
            })

    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Obtiene métricas actuales"""
        if self.metrics_buffer:
            return self.metrics_buffer[-1]
        return self._collect_metrics()

    def get_metrics_history(
        self,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtiene historial de métricas"""
        return list(self.metrics_buffer)[-limit:]

    def get_recent_alerts(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Obtiene alertas recientes"""
        return self.alerts[-limit:]


