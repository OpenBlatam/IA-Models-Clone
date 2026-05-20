"""
Health Checker
==============

Checker principal que compone checkers especializados.
"""

from typing import Dict, Any
from datetime import datetime
from ...core.base.service_base import BaseService
from ...config.settings import get_settings
from .database_checker import DatabaseChecker
from .redis_checker import RedisChecker
from .openrouter_checker import OpenRouterChecker


class HealthChecker(BaseService):
    """Checker principal de salud."""
    
    def __init__(self):
        """Inicializar checker."""
        super().__init__(logger_name=__name__)
        self.settings = get_settings()
        self.database_checker = DatabaseChecker()
        self.redis_checker = RedisChecker()
        self.openrouter_checker = OpenRouterChecker()
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Obtener estado de salud comprehensivo.
        
        Returns:
            Diccionario con status completo
        """
        checks = {
            "database": await self.database_checker.check(),
            "redis": await self.redis_checker.check(),
            "openrouter": await self.openrouter_checker.check(),
        }
        
        statuses = [check["status"] for check in checks.values()]
        if "unhealthy" in statuses:
            overall_status = "unhealthy"
        elif "degraded" in statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "environment": self.settings.environment,
            "checks": checks,
        }

