"""
Health Check Utilities
"""

from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


class HealthStatus(str, Enum):
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
    timestamp: datetime = None
    response_time_ms: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class SystemHealth:
    """System health status"""
    overall_status: HealthStatus
    checks: List[HealthCheck]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "message": check.message,
                    "details": check.details,
                    "response_time_ms": check.response_time_ms
                }
                for check in self.checks
            ]
        }


class HealthChecker:
    """Health checker utility"""
    
    def __init__(self):
        self._checks: Dict[str, Callable] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """Register health check function"""
        self._checks[name] = check_func
    
    async def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check"""
        import time
        from .service import UtilityService
        
        if name not in self._checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check '{name}' not registered"
            )
        
        start_time = time.time()
        try:
            check_func = self._checks[name]
            result = await check_func() if callable(check_func) else check_func
            
            if isinstance(result, HealthCheck):
                result.response_time_ms = (time.time() - start_time) * 1000
                return result
            elif isinstance(result, bool):
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    response_time_ms=(time.time() - start_time) * 1000
                )
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    response_time_ms=(time.time() - start_time) * 1000
                )
        except Exception as e:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )
    
    async def run_all_checks(self) -> SystemHealth:
        """Run all registered health checks"""
        checks = []
        for name in self._checks.keys():
            check = await self.run_check(name)
            checks.append(check)
        
        # Determine overall status
        statuses = [check.status for check in checks]
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return SystemHealth(
            overall_status=overall_status,
            checks=checks
        )

