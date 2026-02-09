"""
Health Check System
===================

Sistema de health checks para el sistema.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estados de salud."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Resultado de health check."""
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        """Inicializar después de creación."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.details is None:
            self.details = {}


class HealthCheck:
    """
    Health check individual.
    
    Verifica un aspecto específico del sistema.
    """
    
    def __init__(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult],
        timeout: float = 5.0,
        critical: bool = False
    ):
        """
        Inicializar health check.
        
        Args:
            name: Nombre del check
            check_func: Función que realiza el check
            timeout: Timeout en segundos
            critical: Si es crítico (afecta estado general)
        """
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
        self.critical = critical
        self.enabled = True
        self.last_result: Optional[HealthCheckResult] = None
    
    def run(self) -> HealthCheckResult:
        """
        Ejecutar health check.
        
        Returns:
            Resultado del check
        """
        if not self.enabled:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message=f"Health check {self.name} is disabled"
            )
        
        start_time = time.time()
        try:
            result = self.check_func()
            duration = time.time() - start_time
            
            result.details = result.details or {}
            result.details["duration"] = duration
            result.details["check_name"] = self.name
            
            self.last_result = result
            return result
        
        except Exception as e:
            logger.error(f"Error in health check {self.name}: {e}")
            result = HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}",
                details={"error": str(e), "check_name": self.name}
            )
            self.last_result = result
            return result


class HealthCheckSystem:
    """
    Sistema de health checks.
    
    Gestiona múltiples health checks y proporciona estado general.
    """
    
    def __init__(self):
        """Inicializar sistema de health checks."""
        self.checks: Dict[str, HealthCheck] = {}
    
    def register_check(self, check: HealthCheck) -> None:
        """Registrar health check."""
        self.checks[check.name] = check
        logger.info(f"Registered health check: {check.name}")
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Ejecutar todos los health checks.
        
        Returns:
            Diccionario de {check_name: result}
        """
        results = {}
        for name, check in self.checks.items():
            results[name] = check.run()
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """
        Obtener estado general del sistema.
        
        Returns:
            Estado general
        """
        results = self.run_all_checks()
        
        if not results:
            return HealthStatus.UNKNOWN
        
        # Verificar checks críticos
        critical_checks = [
            name for name, check in self.checks.items()
            if check.critical
        ]
        
        for name in critical_checks:
            result = results.get(name)
            if result and result.status == HealthStatus.UNHEALTHY:
                return HealthStatus.UNHEALTHY
        
        # Verificar si hay algún check unhealthy
        unhealthy_count = sum(
            1 for r in results.values()
            if r.status == HealthStatus.UNHEALTHY
        )
        
        if unhealthy_count > 0:
            return HealthStatus.DEGRADED
        
        # Verificar si hay checks degraded
        degraded_count = sum(
            1 for r in results.values()
            if r.status == HealthStatus.DEGRADED
        )
        
        if degraded_count > len(results) / 2:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Obtener reporte completo de salud.
        
        Returns:
            Diccionario con reporte completo
        """
        results = self.run_all_checks()
        overall = self.get_overall_status()
        
        return {
            "status": overall.value,
            "timestamp": time.time(),
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details
                }
                for name, result in results.items()
            },
            "summary": {
                "total": len(results),
                "healthy": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY),
            }
        }


# Instancia global
_health_check_system: Optional[HealthCheckSystem] = None


def get_health_check_system() -> HealthCheckSystem:
    """Obtener instancia global del sistema de health checks."""
    global _health_check_system
    if _health_check_system is None:
        _health_check_system = HealthCheckSystem()
    return _health_check_system


def create_basic_health_checks() -> List[HealthCheck]:
    """
    Crear health checks básicos del sistema.
    
    Returns:
        Lista de health checks básicos
    """
    checks = []
    
    # Check de métricas
    def metrics_check() -> HealthCheckResult:
        from .metrics import get_metrics_collector
        try:
            collector = get_metrics_collector()
            metrics = collector.get_all_metrics()
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message=f"Metrics system operational ({len(metrics)} metrics)",
                details={"metric_count": len(metrics)}
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Metrics system error: {e}"
            )
    
    checks.append(HealthCheck("metrics", metrics_check, critical=False))
    
    # Check de caché
    def cache_check() -> HealthCheckResult:
        from .cache import get_cache_manager
        try:
            manager = get_cache_manager()
            stats = manager.get_all_statistics()
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message=f"Cache system operational ({len(stats)} caches)",
                details={"cache_count": len(stats)}
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.DEGRADED,
                message=f"Cache system warning: {e}"
            )
    
    checks.append(HealthCheck("cache", cache_check, critical=False))
    
    return checks






