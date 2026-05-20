"""Advanced health check system"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    checked_at: datetime = None
    
    def __post_init__(self):
        if self.checked_at is None:
            self.checked_at = datetime.now()


class HealthChecker:
    """Advanced health check system"""
    
    def __init__(self):
        self.checks: List[callable] = []
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.checks.append(self._check_disk_space)
        self.checks.append(self._check_memory)
        self.checks.append(self._check_cache)
        self.checks.append(self._check_dependencies)
    
    def register_check(self, check_func: callable):
        """Register custom health check"""
        self.checks.append(check_func)
    
    async def run_checks(self) -> Dict[str, Any]:
        """
        Run all health checks
        
        Returns:
            Health check results
        """
        results = []
        overall_status = HealthStatus.HEALTHY
        
        for check_func in self.checks:
            try:
                result = await check_func() if callable(check_func) else check_func()
                if isinstance(result, HealthCheck):
                    results.append(result)
                    
                    # Determine overall status
                    if result.status == HealthStatus.UNHEALTHY:
                        overall_status = HealthStatus.UNHEALTHY
                    elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.DEGRADED
                else:
                    # Convert dict to HealthCheck
                    health_check = HealthCheck(
                        name=result.get("name", "unknown"),
                        status=HealthStatus(result.get("status", "unknown")),
                        message=result.get("message"),
                        details=result.get("details")
                    )
                    results.append(health_check)
            except Exception as e:
                logger.error(f"Error running health check: {e}")
                results.append(HealthCheck(
                    name=check_func.__name__ if hasattr(check_func, '__name__') else "unknown",
                    status=HealthStatus.UNKNOWN,
                    message=str(e)
                ))
                overall_status = HealthStatus.DEGRADED
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                    "checked_at": r.checked_at.isoformat()
                }
                for r in results
            ]
        }
    
    async def _check_disk_space(self) -> HealthCheck:
        """Check disk space"""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_percent = (free / total) * 100
            
            if free_percent < 5:
                status = HealthStatus.UNHEALTHY
                message = "Critical: Less than 5% disk space available"
            elif free_percent < 10:
                status = HealthStatus.DEGRADED
                message = "Warning: Less than 10% disk space available"
            else:
                status = HealthStatus.HEALTHY
                message = "Disk space OK"
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                details={
                    "free_percent": round(free_percent, 2),
                    "free_bytes": free,
                    "total_bytes": total
                }
            )
        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking disk space: {e}"
            )
    
    async def _check_memory(self) -> HealthCheck:
        """Check memory usage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            
            if used_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "Critical: Memory usage above 90%"
            elif used_percent > 80:
                status = HealthStatus.DEGRADED
                message = "Warning: Memory usage above 80%"
            else:
                status = HealthStatus.HEALTHY
                message = "Memory usage OK"
            
            return HealthCheck(
                name="memory",
                status=status,
                message=message,
                details={
                    "used_percent": round(used_percent, 2),
                    "total_bytes": memory.total,
                    "available_bytes": memory.available
                }
            )
        except ImportError:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="psutil not available"
            )
        except Exception as e:
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking memory: {e}"
            )
    
    async def _check_cache(self) -> HealthCheck:
        """Check cache health"""
        try:
            from utils.cache import get_cache
            cache = get_cache()
            stats = cache.get_stats()
            
            hit_rate = (stats.get("hits", 0) / max(stats.get("total", 1), 1)) * 100
            
            if hit_rate < 20:
                status = HealthStatus.DEGRADED
                message = "Cache hit rate below 20%"
            else:
                status = HealthStatus.HEALTHY
                message = "Cache OK"
            
            return HealthCheck(
                name="cache",
                status=status,
                message=message,
                details=stats
            )
        except Exception as e:
            return HealthCheck(
                name="cache",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking cache: {e}"
            )
    
    async def _check_dependencies(self) -> HealthCheck:
        """Check external dependencies"""
        dependencies = {
            "pandas": False,
            "openpyxl": False,
            "reportlab": False,
            "python-docx": False
        }
        
        for dep in dependencies:
            try:
                __import__(dep.replace("-", "_"))
                dependencies[dep] = True
            except ImportError:
                pass
        
        missing = [dep for dep, available in dependencies.items() if not available]
        
        if missing:
            status = HealthStatus.DEGRADED
            message = f"Missing dependencies: {', '.join(missing)}"
        else:
            status = HealthStatus.HEALTHY
            message = "All dependencies available"
        
        return HealthCheck(
            name="dependencies",
            status=status,
            message=message,
            details=dependencies
        )


# Global health checker
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get global health checker"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker

