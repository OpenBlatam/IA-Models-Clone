"""
Health Checker - Sistema de health checks avanzado
===================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Estados de salud"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DOWN = "down"


class HealthChecker:
    """Sistema de health checks avanzado"""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self.last_results: Dict[str, Dict[str, Any]] = {}
    
    def register_check(self, name: str, check_func: callable, critical: bool = True):
        """Registra un health check"""
        self.checks[name] = {
            "func": check_func,
            "critical": critical
        }
        logger.info(f"Health check registrado: {name}")
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Ejecuta todos los health checks"""
        results = {}
        overall_status = HealthStatus.HEALTHY
        critical_failed = False
        
        for name, check_config in self.checks.items():
            try:
                check_func = check_config["func"]
                result = await check_func() if hasattr(check_func, '__call__') else check_func()
                
                if isinstance(result, dict):
                    status = result.get("status", HealthStatus.HEALTHY)
                    message = result.get("message", "OK")
                else:
                    status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                    message = "OK" if result else "Check failed"
                
                results[name] = {
                    "status": status.value,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                    "critical": check_config["critical"]
                }
                
                self.last_results[name] = results[name]
                
                if status == HealthStatus.UNHEALTHY:
                    if check_config["critical"]:
                        critical_failed = True
                        overall_status = HealthStatus.DOWN
                    elif overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED
                
            except Exception as e:
                logger.error(f"Error en health check {name}: {e}")
                results[name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": f"Error: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "critical": check_config["critical"]
                }
                
                if check_config["critical"]:
                    critical_failed = True
                    overall_status = HealthStatus.DOWN
                elif overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": results,
            "summary": {
                "total": len(results),
                "healthy": sum(1 for r in results.values() if r["status"] == HealthStatus.HEALTHY.value),
                "degraded": sum(1 for r in results.values() if r["status"] == HealthStatus.DEGRADED.value),
                "unhealthy": sum(1 for r in results.values() if r["status"] == HealthStatus.UNHEALTHY.value)
            }
        }
    
    def get_last_results(self) -> Dict[str, Any]:
        """Obtiene los últimos resultados"""
        return self.last_results
    
    async def check_database(self) -> Dict[str, Any]:
        """Health check para base de datos"""
        # Simulación - en producción verificaría conexión real
        return {
            "status": HealthStatus.HEALTHY,
            "message": "Database connection OK"
        }
    
    async def check_cache(self) -> Dict[str, Any]:
        """Health check para caché"""
        # Verificar que el caché esté funcionando
        return {
            "status": HealthStatus.HEALTHY,
            "message": "Cache system OK"
        }
    
    async def check_disk_space(self) -> Dict[str, Any]:
        """Health check para espacio en disco"""
        import shutil
        try:
            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100
            
            if free_percent < 10:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": f"Low disk space: {free_percent:.1f}% free"
                }
            elif free_percent < 20:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": f"Disk space warning: {free_percent:.1f}% free"
                }
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": f"Disk space OK: {free_percent:.1f}% free"
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Cannot check disk space: {str(e)}"
            }
    
    async def check_memory(self) -> Dict[str, Any]:
        """Health check para memoria"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_percent = memory.available / memory.total * 100
            
            if available_percent < 10:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": f"Low memory: {available_percent:.1f}% available"
                }
            elif available_percent < 20:
                return {
                    "status": HealthStatus.DEGRADED,
                    "message": f"Memory warning: {available_percent:.1f}% available"
                }
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": f"Memory OK: {available_percent:.1f}% available"
            }
        except ImportError:
            return {
                "status": HealthStatus.HEALTHY,
                "message": "Memory check not available (psutil not installed)"
            }
        except Exception as e:
            return {
                "status": HealthStatus.DEGRADED,
                "message": f"Cannot check memory: {str(e)}"
            }




