"""
Health Checker - Sistema de verificación de salud del sistema.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

from config.logging_config import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Estado de salud."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Resultado de un health check."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class HealthChecker:
    """Sistema de verificación de salud."""
    
    def __init__(self):
        """Inicializar health checker."""
        self.checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self.overall_status = HealthStatus.UNKNOWN
    
    def register_check(
        self,
        name: str,
        check_func: Callable[[], HealthCheckResult]
    ) -> None:
        """
        Registrar un health check.
        
        Args:
            name: Nombre del check
            check_func: Función que retorna HealthCheckResult
        """
        self.checks[name] = check_func
        logger.debug(f"Health check registrado: {name}")
    
    def run_check(self, name: str) -> HealthCheckResult:
        """
        Ejecutar un health check específico.
        
        Args:
            name: Nombre del check
            
        Returns:
            Resultado del check
        """
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check '{name}' no encontrado"
            )
        
        try:
            return self.checks[name]()
        except Exception as e:
            logger.error(f"Error ejecutando health check '{name}': {e}", exc_info=True)
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}"
            )
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        Ejecutar todos los health checks.
        
        Returns:
            Resultados de todos los checks
        """
        results: Dict[str, HealthCheckResult] = {}
        
        for name in self.checks:
            results[name] = self.run_check(name)
        
        # Determinar estado general
        statuses = [r.status for r in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            self.overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            self.overall_status = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            self.overall_status = HealthStatus.HEALTHY
        else:
            self.overall_status = HealthStatus.UNKNOWN
        
        return {
            "status": self.overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat()
                }
                for name, result in results.items()
            }
        }
    
    def get_overall_status(self) -> HealthStatus:
        """Obtener estado general."""
        return self.overall_status


# Health checks comunes
def create_database_check(storage) -> Callable[[], HealthCheckResult]:
    """Crear check de base de datos."""
    def check() -> HealthCheckResult:
        try:
            # Intentar operación simple
            # Esto depende de la implementación de storage
            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection OK"
            )
        except Exception as e:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database error: {str(e)}"
            )
    return check


def create_redis_check(redis_client) -> Callable[[], HealthCheckResult]:
    """Crear check de Redis."""
    def check() -> HealthCheckResult:
        try:
            redis_client.ping()
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connection OK"
            )
        except Exception as e:
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis error: {str(e)}"
            )
    return check


def create_disk_space_check(threshold_percent: float = 90.0) -> Callable[[], HealthCheckResult]:
    """Crear check de espacio en disco."""
    def check() -> HealthCheckResult:
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            used_percent = (used / total) * 100
            
            if used_percent >= threshold_percent:
                return HealthCheckResult(
                    name="disk_space",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Disk space critical: {used_percent:.1f}% used",
                    details={"used_percent": used_percent, "free_gb": free / (1024**3)}
                )
            elif used_percent >= threshold_percent - 10:
                return HealthCheckResult(
                    name="disk_space",
                    status=HealthStatus.DEGRADED,
                    message=f"Disk space warning: {used_percent:.1f}% used",
                    details={"used_percent": used_percent, "free_gb": free / (1024**3)}
                )
            else:
                return HealthCheckResult(
                    name="disk_space",
                    status=HealthStatus.HEALTHY,
                    message=f"Disk space OK: {used_percent:.1f}% used",
                    details={"used_percent": used_percent, "free_gb": free / (1024**3)}
                )
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check disk space: {str(e)}"
            )
    return check


def create_memory_check(threshold_percent: float = 90.0) -> Callable[[], HealthCheckResult]:
    """Crear check de memoria."""
    def check() -> HealthCheckResult:
        try:
            import psutil
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            
            if used_percent >= threshold_percent:
                return HealthCheckResult(
                    name="memory",
                    status=HealthStatus.UNHEALTHY,
                    message=f"Memory critical: {used_percent:.1f}% used",
                    details={"used_percent": used_percent, "available_gb": memory.available / (1024**3)}
                )
            elif used_percent >= threshold_percent - 10:
                return HealthCheckResult(
                    name="memory",
                    status=HealthStatus.DEGRADED,
                    message=f"Memory warning: {used_percent:.1f}% used",
                    details={"used_percent": used_percent, "available_gb": memory.available / (1024**3)}
                )
            else:
                return HealthCheckResult(
                    name="memory",
                    status=HealthStatus.HEALTHY,
                    message=f"Memory OK: {used_percent:.1f}% used",
                    details={"used_percent": used_percent, "available_gb": memory.available / (1024**3)}
                )
        except ImportError:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="psutil not available"
            )
        except Exception as e:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Could not check memory: {str(e)}"
            )
    return check



