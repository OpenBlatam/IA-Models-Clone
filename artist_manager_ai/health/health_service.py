"""
Health Service
==============

Servicio de health checks avanzados.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Estado de salud."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthService:
    """Servicio de health checks."""
    
    def __init__(self):
        """Inicializar servicio de health."""
        self.checks: Dict[str, callable] = {}
        self._logger = logger
    
    def register_check(self, name: str, check_func: callable):
        """
        Registrar health check.
        
        Args:
            name: Nombre del check
            check_func: Función que retorna (status, details)
        """
        self.checks[name] = check_func
        self._logger.info(f"Registered health check: {name}")
    
    async def check_all(self) -> Dict[str, Any]:
        """
        Ejecutar todos los health checks.
        
        Returns:
            Estado completo de salud
        """
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for name, check_func in self.checks.items():
            try:
                if hasattr(check_func, '__call__'):
                    if hasattr(check_func, '__code__') and check_func.__code__.co_flags & 0x80:
                        status, details = await check_func()
                    else:
                        status, details = check_func()
                else:
                    status, details = HealthStatus.HEALTHY, {}
                
                results[name] = {
                    "status": status.value if isinstance(status, HealthStatus) else status,
                    "details": details,
                    "checked_at": datetime.now().isoformat()
                }
                
                # Determinar estado general
                if status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            
            except Exception as e:
                self._logger.error(f"Health check {name} failed: {str(e)}")
                results[name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "error": str(e),
                    "checked_at": datetime.now().isoformat()
                }
                overall_status = HealthStatus.UNHEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": results
        }
    
    async def check_database(self, db_service) -> tuple[HealthStatus, Dict[str, Any]]:
        """Check de base de datos."""
        try:
            if not db_service:
                return HealthStatus.DEGRADED, {"message": "Database service not configured"}
            
            # Intentar operación simple
            with db_service.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            
            return HealthStatus.HEALTHY, {"message": "Database connection OK"}
        except Exception as e:
            return HealthStatus.UNHEALTHY, {"error": str(e)}
    
    async def check_openrouter(self, openrouter_client) -> tuple[HealthStatus, Dict[str, Any]]:
        """Check de OpenRouter."""
        try:
            if not openrouter_client:
                return HealthStatus.DEGRADED, {"message": "OpenRouter not configured"}
            
            health = await openrouter_client.health_check()
            if health.get("healthy"):
                return HealthStatus.HEALTHY, health
            return HealthStatus.DEGRADED, health
        except Exception as e:
            return HealthStatus.UNHEALTHY, {"error": str(e)}




