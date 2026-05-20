"""
Health Checks
=============
Utilidades para health checks avanzados.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum
import asyncio


class HealthStatus(str, Enum):
    """Estados de salud."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheck:
    """Representación de un health check."""
    
    def __init__(
        self,
        name: str,
        check_func: Callable,
        critical: bool = False,
        timeout: float = 5.0
    ):
        self.name = name
        self.check_func = check_func
        self.critical = critical
        self.timeout = timeout
        self.last_check: Optional[datetime] = None
        self.last_result: Optional[Dict[str, Any]] = None
    
    async def run(self) -> Dict[str, Any]:
        """
        Ejecutar health check.
        
        Returns:
            Resultado del check
        """
        try:
            result = await asyncio.wait_for(
                self.check_func(),
                timeout=self.timeout
            )
            
            self.last_check = datetime.now()
            self.last_result = result
            
            return {
                "status": HealthStatus.HEALTHY,
                "name": self.name,
                "critical": self.critical,
                "result": result,
                "timestamp": self.last_check.isoformat()
            }
        except asyncio.TimeoutError:
            self.last_check = datetime.now()
            status = HealthStatus.UNHEALTHY if self.critical else HealthStatus.DEGRADED
            return {
                "status": status,
                "name": self.name,
                "critical": self.critical,
                "error": "Health check timed out",
                "timestamp": self.last_check.isoformat()
            }
        except Exception as e:
            self.last_check = datetime.now()
            status = HealthStatus.UNHEALTHY if self.critical else HealthStatus.DEGRADED
            return {
                "status": status,
                "name": self.name,
                "critical": self.critical,
                "error": str(e),
                "timestamp": self.last_check.isoformat()
            }


class HealthChecker:
    """Checker de salud del sistema."""
    
    def __init__(self):
        self.checks: List[HealthCheck] = []
    
    def register(self, check: HealthCheck):
        """Registrar un health check."""
        self.checks.append(check)
    
    def register_simple(
        self,
        name: str,
        check_func: Callable,
        critical: bool = False,
        timeout: float = 5.0
    ):
        """Registrar un health check simple."""
        check = HealthCheck(name, check_func, critical, timeout)
        self.register(check)
    
    async def run_all(self) -> Dict[str, Any]:
        """
        Ejecutar todos los health checks.
        
        Returns:
            Resultado de todos los checks
        """
        results = await asyncio.gather(*[check.run() for check in self.checks])
        
        # Determinar estado general
        has_unhealthy = any(
            r["status"] == HealthStatus.UNHEALTHY
            for r in results
        )
        has_degraded = any(
            r["status"] == HealthStatus.DEGRADED
            for r in results
        )
        
        if has_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif has_degraded:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": results,
            "summary": {
                "total": len(results),
                "healthy": sum(1 for r in results if r["status"] == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in results if r["status"] == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for r in results if r["status"] == HealthStatus.UNHEALTHY)
            }
        }


# Instancia global
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Obtener instancia global del health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker

