"""
Dependency Validator - Validador de dependencias
================================================

Valida que todas las dependencias estén disponibles y funcionando.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DependencyValidator:
    """
    Validador de dependencias que verifica:
    - Disponibilidad de servicios
    - Conectividad
    - Configuración correcta
    """
    
    def __init__(self):
        self.dependencies: Dict[str, Dict[str, Any]] = {}
    
    def register_dependency(
        self,
        name: str,
        check_func: callable,
        required: bool = True,
        timeout: float = 5.0
    ):
        """
        Registra una dependencia para validar.
        
        Args:
            name: Nombre de la dependencia
            check_func: Función que verifica la dependencia
            required: Si es requerida
            timeout: Timeout para el check
        """
        self.dependencies[name] = {
            "check": check_func,
            "required": required,
            "timeout": timeout,
            "status": "unknown"
        }
    
    async def validate_all(self) -> Dict[str, Any]:
        """
        Valida todas las dependencias.
        
        Returns:
            Estado de todas las dependencias
        """
        results = {}
        all_healthy = True
        required_failed = False
        
        for name, dep_info in self.dependencies.items():
            try:
                check_func = dep_info["check"]
                timeout = dep_info["timeout"]
                
                if asyncio.iscoroutinefunction(check_func):
                    result = await asyncio.wait_for(check_func(), timeout=timeout)
                else:
                    result = check_func()
                
                is_healthy = result if isinstance(result, bool) else result.get("status") == "healthy"
                
                results[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "required": dep_info["required"],
                    "timestamp": datetime.now().isoformat()
                }
                
                if not is_healthy:
                    all_healthy = False
                    if dep_info["required"]:
                        required_failed = True
                
            except asyncio.TimeoutError:
                results[name] = {
                    "status": "timeout",
                    "required": dep_info["required"],
                    "timestamp": datetime.now().isoformat()
                }
                all_healthy = False
                if dep_info["required"]:
                    required_failed = True
            
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "required": dep_info["required"],
                    "timestamp": datetime.now().isoformat()
                }
                all_healthy = False
                if dep_info["required"]:
                    required_failed = True
        
        return {
            "all_healthy": all_healthy,
            "required_failed": required_failed,
            "dependencies": results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def check_cache(self) -> bool:
        """Check de cache"""
        try:
            from ..infrastructure.cache import get_cache_service
            cache = get_cache_service()
            # Test simple
            await cache.set("dependency_check", True, ttl=1)
            result = await cache.get("dependency_check")
            await cache.delete("dependency_check")
            return result is True
        except Exception:
            return False
    
    async def check_redis(self) -> Dict[str, Any]:
        """Check de Redis"""
        try:
            from ..core.redis_client import get_redis_client
            redis = get_redis_client()
            if redis.sync_client:
                redis.sync_client.ping()
                return {"status": "healthy"}
            return {"status": "unhealthy", "message": "Redis client not available"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}
    
    async def check_workers(self) -> Dict[str, Any]:
        """Check de workers"""
        try:
            from ..infrastructure.workers import get_worker_service
            worker = get_worker_service()
            if worker.worker_manager:
                return {"status": "healthy"}
            return {"status": "unavailable", "message": "Workers not configured"}
        except Exception as e:
            return {"status": "unhealthy", "message": str(e)}


def get_dependency_validator() -> DependencyValidator:
    """Obtiene validador de dependencias con checks por defecto"""
    validator = DependencyValidator()
    
    # Registrar checks por defecto
    validator.register_dependency("cache", validator.check_cache, required=False)
    validator.register_dependency("redis", validator.check_redis, required=False)
    validator.register_dependency("workers", validator.check_workers, required=False)
    
    return validator















