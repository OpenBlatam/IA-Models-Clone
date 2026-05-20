"""
Health Checker - Health checks robustos
=======================================

Sistema de health checks robusto para todos los componentes.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Estados de salud"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth:
    """Health check de un componente"""
    
    def __init__(self, name: str):
        self.name = name
        self.status = HealthStatus.UNKNOWN
        self.message = ""
        self.details: Dict[str, Any] = {}
        self.timestamp = datetime.now()
        self.response_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "response_time": self.response_time
        }


class RobustHealthChecker:
    """
    Health checker robusto que verifica:
    - Servicios
    - Dependencias
    - Recursos del sistema
    - Conectividad
    """
    
    def __init__(self):
        self.components: List[ComponentHealth] = []
        self.checks: Dict[str, Callable] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """
        Registra un health check.
        
        Args:
            name: Nombre del check
            check_func: Función que ejecuta el check
        """
        self.checks[name] = check_func
    
    async def check_component(
        self,
        name: str,
        check_func: Callable,
        timeout: float = 5.0
    ) -> ComponentHealth:
        """
        Ejecuta health check de un componente.
        
        Args:
            name: Nombre del componente
            check_func: Función de check
            timeout: Timeout en segundos
        
        Returns:
            ComponentHealth
        """
        health = ComponentHealth(name)
        start_time = datetime.now()
        
        try:
            # Ejecutar check con timeout
            if asyncio.iscoroutinefunction(check_func):
                result = await asyncio.wait_for(check_func(), timeout=timeout)
            else:
                result = check_func()
            
            duration = (datetime.now() - start_time).total_seconds()
            health.response_time = duration
            
            # Interpretar resultado
            if isinstance(result, dict):
                health.status = HealthStatus(result.get("status", "unknown"))
                health.message = result.get("message", "")
                health.details = result.get("details", {})
            elif isinstance(result, bool):
                health.status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                health.message = "Check passed" if result else "Check failed"
            else:
                health.status = HealthStatus.HEALTHY
                health.message = "Check completed"
            
        except asyncio.TimeoutError:
            health.status = HealthStatus.UNHEALTHY
            health.message = f"Health check timed out after {timeout}s"
            health.response_time = timeout
        
        except Exception as e:
            health.status = HealthStatus.UNHEALTHY
            health.message = f"Health check failed: {str(e)}"
            logger.error(f"Health check failed for {name}: {e}", exc_info=True)
        
        health.timestamp = datetime.now()
        return health
    
    async def check_all(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Ejecuta todos los health checks.
        
        Args:
            timeout: Timeout por check
        
        Returns:
            Resultado de todos los checks
        """
        results = []
        
        for name, check_func in self.checks.items():
            health = await self.check_component(name, check_func, timeout)
            results.append(health.to_dict())
        
        # Determinar estado general
        statuses = [r["status"] for r in results]
        if all(s == "healthy" for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == "unhealthy" for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "components": results
        }
    
    async def check_cache(self) -> Dict[str, Any]:
        """Health check de cache"""
        try:
            from ..infrastructure.cache import get_cache_service
            cache = get_cache_service()
            
            # Test de escritura/lectura
            test_key = "health_check_test"
            test_value = {"test": True, "timestamp": datetime.now().isoformat()}
            
            await cache.set(test_key, test_value, ttl=10)
            retrieved = await cache.get(test_key)
            
            if retrieved and retrieved.get("test"):
                await cache.delete(test_key)
                return {
                    "status": "healthy",
                    "message": "Cache is operational",
                    "details": {}
                }
            else:
                return {
                    "status": "degraded",
                    "message": "Cache read/write test failed",
                    "details": {}
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Cache check failed: {str(e)}",
                "details": {}
            }
    
    async def check_database(self) -> Dict[str, Any]:
        """Health check de base de datos (si está disponible)"""
        # Implementación específica según el backend de datos
        return {
            "status": "unknown",
            "message": "Database check not implemented",
            "details": {}
        }
    
    async def check_external_services(self) -> Dict[str, Any]:
        """Health check de servicios externos"""
        services_status = {}
        
        # Check Redis
        try:
            from ..core.redis_client import get_redis_client
            redis = get_redis_client()
            if redis.sync_client:
                redis.sync_client.ping()
                services_status["redis"] = "healthy"
            else:
                services_status["redis"] = "unavailable"
        except Exception as e:
            services_status["redis"] = f"unhealthy: {str(e)}"
        
        return {
            "status": "healthy" if all(s == "healthy" for s in services_status.values()) else "degraded",
            "message": "External services check",
            "details": services_status
        }


def get_health_checker() -> RobustHealthChecker:
    """Obtiene instancia de health checker"""
    checker = RobustHealthChecker()
    
    # Registrar checks por defecto
    checker.register_check("cache", checker.check_cache)
    checker.register_check("external_services", checker.check_external_services)
    
    return checker

