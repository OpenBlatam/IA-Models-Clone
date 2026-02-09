"""
Health check del sistema
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from ..db.base import get_db_session, engine
from ..config import get_settings
from sqlalchemy import text

logger = logging.getLogger(__name__)


class HealthCheck:
    """Health check del sistema"""
    
    def __init__(self):
        self.settings = get_settings()
    
    def check_database(self) -> Dict[str, Any]:
        """Verifica salud de la base de datos"""
        try:
            with get_db_session() as db:
                db.execute(text("SELECT 1"))
                db.commit()
            
            return {
                "status": "healthy",
                "message": "Database connection OK"
            }
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Database error: {str(e)}"
            }
    
    def check_storage(self) -> Dict[str, Any]:
        """Verifica salud del storage"""
        try:
            from pathlib import Path
            storage_path = Path(self.settings.storage_path)
            
            if not storage_path.exists():
                return {
                    "status": "unhealthy",
                    "message": "Storage path does not exist"
                }
            
            # Verificar espacio disponible
            import shutil
            total, used, free = shutil.disk_usage(storage_path)
            free_percent = (free / total) * 100
            
            if free_percent < 10:
                return {
                    "status": "warning",
                    "message": f"Low disk space: {free_percent:.1f}% free"
                }
            
            return {
                "status": "healthy",
                "message": f"Storage OK ({free_percent:.1f}% free)"
            }
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Storage error: {str(e)}"
            }
    
    def check_openai(self) -> Dict[str, Any]:
        """Verifica conexión con OpenAI"""
        try:
            if not self.settings.openai_api_key:
                return {
                    "status": "warning",
                    "message": "OpenAI API key not configured"
                }
            
            # Verificación básica (no hacer llamada real)
            return {
                "status": "healthy",
                "message": "OpenAI API key configured"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"OpenAI check error: {str(e)}"
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Obtiene estado completo de salud"""
        checks = {
            "database": self.check_database(),
            "storage": self.check_storage(),
            "openai": self.check_openai()
        }
        
        # Determinar estado general
        statuses = [check["status"] for check in checks.values()]
        if "unhealthy" in statuses:
            overall_status = "unhealthy"
        elif "warning" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }


# Singleton global
_health_check: Optional[HealthCheck] = None


def get_health_check() -> HealthCheck:
    """Obtiene instancia singleton del health check"""
    global _health_check
    if _health_check is None:
        _health_check = HealthCheck()
    return _health_check




