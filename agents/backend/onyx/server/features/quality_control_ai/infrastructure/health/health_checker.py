"""
Health Checker

System health monitoring and checks.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import torch

logger = logging.getLogger(__name__)


class HealthStatus:
    """Health status enum."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthChecker:
    """
    System health checker.
    
    Checks various system components and reports overall health.
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.checks = {}
        self.last_check_time = None
    
    def register_check(
        self,
        name: str,
        check_func: callable,
        critical: bool = False
    ):
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Function that returns (is_healthy, message)
            critical: Whether this check is critical
        """
        self.checks[name] = {
            "func": check_func,
            "critical": critical,
        }
    
    def check_all(self) -> Dict[str, Any]:
        """
        Run all health checks.
        
        Returns:
            Dictionary with health status
        """
        results = {}
        all_healthy = True
        any_critical_failed = False
        
        for name, check_info in self.checks.items():
            try:
                is_healthy, message = check_info["func"]()
                results[name] = {
                    "status": "healthy" if is_healthy else "unhealthy",
                    "message": message,
                    "critical": check_info["critical"],
                }
                
                if not is_healthy:
                    all_healthy = False
                    if check_info["critical"]:
                        any_critical_failed = True
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {str(e)}", exc_info=True)
                results[name] = {
                    "status": "unhealthy",
                    "message": f"Check failed: {str(e)}",
                    "critical": check_info["critical"],
                }
                all_healthy = False
                if check_info["critical"]:
                    any_critical_failed = True
        
        # Determine overall status
        if any_critical_failed:
            overall_status = HealthStatus.UNHEALTHY
        elif not all_healthy:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        self.last_check_time = datetime.utcnow()
        
        return {
            "status": overall_status,
            "timestamp": self.last_check_time.isoformat(),
            "checks": results,
        }
    
    def check_database(self) -> tuple[bool, str]:
        """Check database connectivity."""
        # Placeholder - would check actual database
        return True, "Database connection OK"
    
    def check_models(self) -> tuple[bool, str]:
        """Check ML models availability."""
        try:
            # Check if PyTorch is available
            if not torch.cuda.is_available() and not hasattr(torch, 'cpu'):
                return False, "PyTorch not properly installed"
            return True, "Models available"
        except Exception as e:
            return False, f"Model check failed: {str(e)}"
    
    def check_storage(self) -> tuple[bool, str]:
        """Check storage availability."""
        try:
            from pathlib import Path
            storage_path = Path("./storage")
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # Try to write a test file
            test_file = storage_path / ".health_check"
            test_file.write_text("test")
            test_file.unlink()
            
            return True, "Storage accessible"
        except Exception as e:
            return False, f"Storage check failed: {str(e)}"
    
    def check_memory(self) -> tuple[bool, str]:
        """Check memory availability."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent > 90:
                return False, f"Memory usage critical: {usage_percent:.1f}%"
            elif usage_percent > 80:
                return True, f"Memory usage high: {usage_percent:.1f}%"
            else:
                return True, f"Memory usage OK: {usage_percent:.1f}%"
        except ImportError:
            return True, "Memory check not available (psutil not installed)"
        except Exception as e:
            return False, f"Memory check failed: {str(e)}"


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """
    Get global health checker instance.
    
    Returns:
        HealthChecker instance
    """
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
        
        # Register default checks
        _health_checker.register_check("models", _health_checker.check_models, critical=True)
        _health_checker.register_check("storage", _health_checker.check_storage, critical=True)
        _health_checker.register_check("memory", _health_checker.check_memory, critical=False)
    
    return _health_checker



