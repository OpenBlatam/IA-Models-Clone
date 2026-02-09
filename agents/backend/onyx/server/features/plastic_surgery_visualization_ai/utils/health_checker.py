"""Health check utilities."""

from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from utils.logger import get_logger

logger = get_logger(__name__)


class HealthChecker:
    """Utility for performing health checks."""
    
    def __init__(self):
        self.checks: Dict[str, callable] = {}
    
    def register_check(self, name: str, check_func: callable) -> None:
        """
        Register a health check.
        
        Args:
            name: Name of the check
            check_func: Async function that returns (bool, Optional[str])
        """
        self.checks[name] = check_func
    
    async def run_all_checks(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Run all registered health checks.
        
        Args:
            timeout: Timeout for each check in seconds
            
        Returns:
            Dictionary with check results
        """
        results = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        for name, check_func in self.checks.items():
            try:
                check_result = await asyncio.wait_for(
                    check_func(),
                    timeout=timeout
                )
                
                if isinstance(check_result, tuple):
                    is_healthy, message = check_result
                else:
                    is_healthy = bool(check_result)
                    message = None
                
                results["checks"][name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "message": message
                }
                
                if not is_healthy:
                    results["status"] = "unhealthy"
                    
            except asyncio.TimeoutError:
                results["checks"][name] = {
                    "status": "timeout",
                    "message": f"Check timed out after {timeout}s"
                }
                results["status"] = "unhealthy"
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results["checks"][name] = {
                    "status": "error",
                    "message": str(e)
                }
                results["status"] = "unhealthy"
        
        return results


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker

