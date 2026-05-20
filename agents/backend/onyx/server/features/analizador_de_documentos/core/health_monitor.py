"""
Monitor de Salud Avanzado
==========================

Sistema avanzado para monitoreo de salud del sistema.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estado de salud"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Check de salud"""
    component: str
    status: HealthStatus
    message: str
    timestamp: str
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class AdvancedHealthMonitor:
    """
    Monitor de salud avanzado
    
    Proporciona:
    - Health checks de componentes
    - Monitoreo continuo
    - Alertas automáticas
    - Métricas de salud
    - Diagnóstico automático
    """
    
    def __init__(self):
        """Inicializar monitor"""
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_history: List[Dict[str, Any]] = []
        self.check_interval = 60  # segundos
        logger.info("AdvancedHealthMonitor inicializado")
    
    def register_component(
        self,
        component: str,
        health_check_func: callable
    ):
        """Registrar componente para health check"""
        # En producción, se almacenarían las funciones de check
        logger.info(f"Componente registrado para health check: {component}")
    
    def perform_health_check(
        self,
        component: str,
        status: HealthStatus,
        message: str,
        metrics: Optional[Dict[str, Any]] = None
    ):
        """Realizar health check"""
        check = HealthCheck(
            component=component,
            status=status,
            message=message,
            timestamp=datetime.now().isoformat(),
            metrics=metrics or {}
        )
        
        self.health_checks[component] = check
        
        # Guardar en historial
        self.health_history.append({
            "component": component,
            "status": status.value,
            "message": message,
            "timestamp": check.timestamp,
            "metrics": metrics or {}
        })
        
        # Mantener solo últimos 10000 registros
        if len(self.health_history) > 10000:
            self.health_history = self.health_history[-10000:]
    
    def get_component_health(self, component: str) -> Optional[HealthCheck]:
        """Obtener salud de componente"""
        return self.health_checks.get(component)
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Obtener salud general del sistema"""
        if not self.health_checks:
            return {
                "status": "unknown",
                "message": "No hay componentes registrados"
            }
        
        # Determinar estado general
        statuses = [check.status for check in self.health_checks.values()]
        
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "status": overall_status.value,
            "components": {
                component: {
                    "status": check.status.value,
                    "message": check.message,
                    "timestamp": check.timestamp
                }
                for component, check in self.health_checks.items()
            },
            "total_components": len(self.health_checks),
            "healthy_components": sum(1 for s in statuses if s == HealthStatus.HEALTHY),
            "degraded_components": sum(1 for s in statuses if s == HealthStatus.DEGRADED),
            "unhealthy_components": sum(1 for s in statuses if s in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL])
        }
    
    def get_health_history(
        self,
        component: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Obtener historial de salud"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        filtered = [
            h for h in self.health_history
            if datetime.fromisoformat(h["timestamp"]) >= cutoff
        ]
        
        if component:
            filtered = [h for h in filtered if h["component"] == component]
        
        return filtered


# Instancia global
_health_monitor: Optional[AdvancedHealthMonitor] = None


def get_health_monitor() -> AdvancedHealthMonitor:
    """Obtener instancia global del monitor"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = AdvancedHealthMonitor()
    return _health_monitor
















