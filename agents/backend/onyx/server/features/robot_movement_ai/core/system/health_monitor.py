"""
Health Monitor System
=====================

Sistema avanzado de monitoreo de salud.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estado de salud."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check."""
    check_id: str
    name: str
    check_func: Callable
    interval: float = 30.0  # Segundos
    timeout: float = 5.0
    enabled: bool = True
    last_check: Optional[str] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Reporte de salud."""
    overall_status: HealthStatus
    timestamp: str
    checks: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """
    Monitor de salud.
    
    Gestiona health checks y reportes de salud.
    """
    
    def __init__(self):
        """Inicializar monitor de salud."""
        self.checks: Dict[str, HealthCheck] = {}
        self.check_tasks: Dict[str, asyncio.Task] = {}
        self.metrics: Dict[str, Any] = {}
    
    def register_check(
        self,
        check_id: str,
        name: str,
        check_func: Callable,
        interval: float = 30.0,
        timeout: float = 5.0,
        enabled: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HealthCheck:
        """
        Registrar health check.
        
        Args:
            check_id: ID único del check
            name: Nombre
            check_func: Función de verificación
            interval: Intervalo en segundos
            timeout: Timeout en segundos
            enabled: Si está habilitado
            metadata: Metadata adicional
            
        Returns:
            Health check registrado
        """
        check = HealthCheck(
            check_id=check_id,
            name=name,
            check_func=check_func,
            interval=interval,
            timeout=timeout,
            enabled=enabled,
            metadata=metadata or {}
        )
        
        self.checks[check_id] = check
        
        if enabled:
            self._start_check_task(check)
        
        logger.info(f"Registered health check: {name} ({check_id})")
        
        return check
    
    def _start_check_task(self, check: HealthCheck) -> None:
        """Iniciar tarea de health check."""
        if check.check_id in self.check_tasks:
            return
        
        async def check_loop():
            while check.enabled:
                try:
                    await self._run_check(check)
                    await asyncio.sleep(check.interval)
                except Exception as e:
                    logger.error(f"Error in health check loop for {check.name}: {e}")
                    await asyncio.sleep(check.interval)
        
        task = asyncio.create_task(check_loop())
        self.check_tasks[check.check_id] = task
    
    async def _run_check(self, check: HealthCheck) -> None:
        """Ejecutar health check."""
        try:
            if asyncio.iscoroutinefunction(check.check_func):
                result = await asyncio.wait_for(
                    check.check_func(),
                    timeout=check.timeout
                )
            else:
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, check.check_func),
                    timeout=check.timeout
                )
            
            # Interpretar resultado
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            elif isinstance(result, dict):
                status = HealthStatus(result.get("status", "unknown"))
            else:
                status = HealthStatus.HEALTHY
            
            check.last_status = status
            check.last_check = datetime.now().isoformat()
            check.last_error = None
        except asyncio.TimeoutError:
            check.last_status = HealthStatus.UNHEALTHY
            check.last_check = datetime.now().isoformat()
            check.last_error = "Timeout"
        except Exception as e:
            check.last_status = HealthStatus.UNHEALTHY
            check.last_check = datetime.now().isoformat()
            check.last_error = str(e)
            logger.error(f"Health check {check.name} failed: {e}")
    
    async def get_health_report(self) -> HealthReport:
        """
        Obtener reporte de salud.
        
        Returns:
            Reporte de salud
        """
        # Ejecutar todos los checks
        for check in self.checks.values():
            if check.enabled:
                await self._run_check(check)
        
        # Determinar estado general
        statuses = [check.last_status for check in self.checks.values() if check.enabled]
        
        if not statuses:
            overall_status = HealthStatus.UNKNOWN
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED
        
        checks_data = [
            {
                "check_id": check.check_id,
                "name": check.name,
                "status": check.last_status.value,
                "last_check": check.last_check,
                "error": check.last_error
            }
            for check in self.checks.values()
        ]
        
        return HealthReport(
            overall_status=overall_status,
            timestamp=datetime.now().isoformat(),
            checks=checks_data,
            metrics=self.metrics
        )
    
    def get_check(self, check_id: str) -> Optional[HealthCheck]:
        """Obtener health check por ID."""
        return self.checks.get(check_id)
    
    def list_checks(self) -> List[HealthCheck]:
        """Listar todos los health checks."""
        return list(self.checks.values())


# Instancia global
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Obtener instancia global del monitor de salud."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor






