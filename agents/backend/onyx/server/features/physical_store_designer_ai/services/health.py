"""
Health Check Service
"""

import logging
import os
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from ..config.settings import settings

logger = logging.getLogger(__name__)


class HealthChecker:
    """Health checker para el servicio"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.checks: List[Dict[str, Any]] = []
    
    def check_health(self) -> Dict[str, Any]:
        """Verificar salud básica del servicio"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }
    
    def check_storage(self) -> Dict[str, Any]:
        """Verificar acceso a almacenamiento"""
        try:
            storage_path = Path(settings.storage_path)
            designs_path = Path(settings.designs_path)
            
            # Verificar que los directorios existen o pueden crearse
            storage_path.mkdir(parents=True, exist_ok=True)
            designs_path.mkdir(parents=True, exist_ok=True)
            
            # Verificar permisos de escritura
            test_file = storage_path / ".health_check"
            test_file.write_text("test")
            test_file.unlink()
            
            return {
                "status": "healthy",
                "message": "Storage accessible",
                "storage_path": str(storage_path),
                "designs_path": str(designs_path)
            }
        except Exception as e:
            logger.error(f"Storage check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Storage check failed: {str(e)}"
            }
    
    def check_openai(self) -> Dict[str, Any]:
        """Verificar configuración de OpenAI"""
        if not settings.openai_api_key:
            return {
                "status": "degraded",
                "message": "OpenAI API key not configured (optional)"
            }
        
        # Verificar formato básico de la key
        if not settings.openai_api_key.startswith("sk-"):
            return {
                "status": "degraded",
                "message": "OpenAI API key format may be invalid"
            }
        
        return {
            "status": "healthy",
            "message": "OpenAI API key configured"
        }
    
    def check_memory(self) -> Dict[str, Any]:
        """Verificar uso de memoria"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            status = "healthy"
            if usage_percent > 90:
                status = "unhealthy"
            elif usage_percent > 75:
                status = "degraded"
            
            return {
                "status": status,
                "usage_percent": usage_percent,
                "available_mb": memory.available / (1024 * 1024),
                "total_mb": memory.total / (1024 * 1024)
            }
        except ImportError:
            return {
                "status": "degraded",
                "message": "psutil not available for memory checks"
            }
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return {
                "status": "degraded",
                "message": f"Memory check failed: {str(e)}"
            }
    
    def get_status(self) -> str:
        """Obtener estado general del servicio"""
        checks = self.run_all_checks()
        statuses = [check.get("status", "unknown") for check in checks.get("checks", [])]
        
        if "unhealthy" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        return "healthy"
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Ejecutar todas las verificaciones"""
        checks = [
            {
                "name": "basic",
                **self.check_health()
            },
            {
                "name": "storage",
                **self.check_storage()
            },
            {
                "name": "openai",
                **self.check_openai()
            },
            {
                "name": "memory",
                **self.check_memory()
            }
        ]
        
        overall_status = self.get_status()
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "checks": checks
        }


_health_checker: HealthChecker = None


def get_health_checker() -> HealthChecker:
    """Obtener instancia del health checker"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker




