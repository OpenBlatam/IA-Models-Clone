"""
Intelligent Health Checker - Health Checks Inteligentes
=======================================================

Sistema de health checks inteligente con auto-diagnóstico y auto-recovery.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estado de salud."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Check de salud."""
    check_id: str
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthMetric:
    """Métrica de salud."""
    metric_name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class IntelligentHealthChecker:
    """Health checker inteligente."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.metrics: Dict[str, HealthMetric] = {}
        self.check_history: List[HealthCheck] = []
        self.auto_recovery_actions: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
    
    def register_check(
        self,
        check_id: str,
        check_func: Callable,
        auto_recovery: Optional[Callable] = None,
    ):
        """
        Registrar check de salud.
        
        Args:
            check_id: ID único del check
            check_func: Función async que retorna (status, message, details)
            auto_recovery: Función async para auto-recovery (opcional)
        """
        self.checks[check_id] = check_func
        if auto_recovery:
            self.auto_recovery_actions[check_id] = auto_recovery
        
        logger.info(f"Registered health check: {check_id}")
    
    def register_metric(
        self,
        metric_name: str,
        threshold_warning: float,
        threshold_critical: float,
        unit: str = "",
    ):
        """Registrar métrica de salud."""
        self.metrics[metric_name] = HealthMetric(
            metric_name=metric_name,
            value=0.0,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical,
            unit=unit,
        )
    
    async def update_metric(
        self,
        metric_name: str,
        value: float,
    ):
        """Actualizar valor de métrica."""
        if metric_name in self.metrics:
            self.metrics[metric_name].value = value
            self.metrics[metric_name].timestamp = datetime.now()
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Ejecutar todos los checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for check_id, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    status, message, details = await check_func()
                else:
                    status, message, details = check_func()
                
                health_status = HealthStatus(status) if isinstance(status, str) else status
                
                # Verificar si necesita auto-recovery
                if health_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    if check_id in self.auto_recovery_actions:
                        recovery_func = self.auto_recovery_actions[check_id]
                        try:
                            if asyncio.iscoroutinefunction(recovery_func):
                                recovery_result = await recovery_func()
                            else:
                                recovery_result = recovery_func()
                            
                            if recovery_result:
                                logger.info(f"Auto-recovery successful for {check_id}")
                                health_status = HealthStatus.HEALTHY
                                message = f"Auto-recovered: {message}"
                        except Exception as e:
                            logger.error(f"Auto-recovery failed for {check_id}: {e}")
                
                # Actualizar estado general
                if health_status == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif health_status == HealthStatus.UNHEALTHY and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif health_status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                
                check_result = HealthCheck(
                    check_id=check_id,
                    name=check_id,
                    status=health_status,
                    message=message,
                    timestamp=datetime.now(),
                    details=details or {},
                )
                
                self.check_history.append(check_result)
                results[check_id] = {
                    "status": health_status.value,
                    "message": message,
                    "details": details,
                }
                
            except Exception as e:
                logger.error(f"Error running check {check_id}: {e}")
                results[check_id] = {
                    "status": HealthStatus.CRITICAL.value,
                    "message": f"Check failed: {str(e)}",
                    "details": {},
                }
        
        # Evaluar métricas
        for metric_name, metric in self.metrics.items():
            if metric.value >= metric.threshold_critical:
                overall_status = HealthStatus.CRITICAL
                results[f"metric_{metric_name}"] = {
                    "status": HealthStatus.CRITICAL.value,
                    "message": f"{metric_name} exceeds critical threshold",
                    "details": {
                        "value": metric.value,
                        "threshold": metric.threshold_critical,
                        "unit": metric.unit,
                    },
                }
            elif metric.value >= metric.threshold_warning:
                if overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                results[f"metric_{metric_name}"] = {
                    "status": HealthStatus.DEGRADED.value,
                    "message": f"{metric_name} exceeds warning threshold",
                    "details": {
                        "value": metric.value,
                        "threshold": metric.threshold_warning,
                        "unit": metric.unit,
                    },
                }
        
        # Mantener solo últimos 1000 checks
        if len(self.check_history) > 1000:
            self.check_history = self.check_history[-1000:]
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": results,
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Obtener resumen de salud."""
        if not self.check_history:
            return {
                "status": "unknown",
                "message": "No checks performed yet",
            }
        
        latest_checks = self.check_history[-len(self.checks):] if len(self.check_history) >= len(self.checks) else self.check_history
        
        statuses = [check.status for check in latest_checks]
        
        if HealthStatus.CRITICAL in statuses:
            overall = HealthStatus.CRITICAL
        elif HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY
        
        return {
            "overall_status": overall.value,
            "total_checks": len(self.checks),
            "metrics": {
                name: {
                    "value": metric.value,
                    "threshold_warning": metric.threshold_warning,
                    "threshold_critical": metric.threshold_critical,
                    "unit": metric.unit,
                }
                for name, metric in self.metrics.items()
            },
            "recent_checks": [
                {
                    "check_id": check.check_id,
                    "status": check.status.value,
                    "message": check.message,
                    "timestamp": check.timestamp.isoformat(),
                }
                for check in latest_checks
            ],
        }
















