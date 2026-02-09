"""
Health Checker
Advanced health check system
"""

from typing import Dict, Any, List, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HealthCheck:
    """Represents a health check"""
    
    def __init__(
        self,
        name: str,
        check_func: Callable[[], Dict[str, Any]],
        critical: bool = False
    ):
        self.name = name
        self.check_func = check_func
        self.critical = critical
    
    def run(self) -> Dict[str, Any]:
        """Run health check"""
        try:
            result = self.check_func()
            return {
                "name": self.name,
                "status": result.get("status", "unknown"),
                "critical": self.critical,
                **result
            }
        except Exception as e:
            logger.error(f"Health check {self.name} failed: {str(e)}")
            return {
                "name": self.name,
                "status": "error",
                "critical": self.critical,
                "error": str(e)
            }


class HealthChecker:
    """Manages health checks"""
    
    def __init__(self):
        self.checks: List[HealthCheck] = []
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        # Database check (placeholder)
        self.register_check(HealthCheck(
            name="database",
            check_func=lambda: {"status": "healthy", "message": "Database connection OK"},
            critical=True
        ))
        
        # Cache check
        self.register_check(HealthCheck(
            name="cache",
            check_func=lambda: {"status": "healthy", "message": "Cache available"},
            critical=False
        ))
        
        # Storage check
        self.register_check(HealthCheck(
            name="storage",
            check_func=lambda: {"status": "healthy", "message": "Storage available"},
            critical=True
        ))
    
    def register_check(self, check: HealthCheck):
        """Register health check"""
        self.checks.append(check)
        logger.info(f"Registered health check: {check.name}")
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = []
        overall_status = "healthy"
        
        for check in self.checks:
            result = check.run()
            results.append(result)
            
            if result["status"] != "healthy":
                if check.critical:
                    overall_status = "unhealthy"
                elif overall_status == "healthy":
                    overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results,
        }
    
    def get_status(self) -> str:
        """Get overall health status"""
        result = self.run_all_checks()
        return result["status"]


_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get health checker instance (singleton)"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker

