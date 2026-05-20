"""
Health Check System
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthChecker:
    """Sistema de health checks"""
    
    def __init__(self):
        """Inicializar health checker"""
        self.checks: Dict[str, callable] = {}
        logger.info("HealthChecker initialized")
    
    def register_check(self, name: str, check_function: callable):
        """Registrar un health check"""
        self.checks[name] = check_function
    
    def run_checks(self) -> Dict[str, Any]:
        """Ejecutar todos los health checks"""
        results = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {},
        }
        
        all_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                check_result = check_func()
                results["checks"][name] = {
                    "status": "healthy" if check_result else "unhealthy",
                    "result": check_result,
                }
                if not check_result:
                    all_healthy = False
            except Exception as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e),
                }
                all_healthy = False
        
        if not all_healthy:
            results["status"] = "unhealthy"
        
        return results


# Instancia global
health_checker = HealthChecker()




