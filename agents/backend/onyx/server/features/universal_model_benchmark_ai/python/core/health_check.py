"""
Health Check Module - Advanced health checking.

Provides:
- Health check endpoints
- Dependency health checks
- Health status aggregation
- Health check scheduling
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str = ""
    response_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "response_time_ms": self.response_time_ms,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class HealthChecker:
    """Health checker."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheckResult] = {}
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """
        Register health check.
        
        Args:
            name: Check name
            check_func: Check function (returns HealthCheckResult or bool)
        """
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check.
        
        Args:
            name: Check name
            
        Returns:
            Health check result
        """
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check {name} not found",
            )
        
        check_func = self.checks[name]
        start_time = time.time()
        
        try:
            result = check_func()
            
            # Handle different return types
            if isinstance(result, HealthCheckResult):
                result.response_time_ms = (time.time() - start_time) * 1000
                self.results[name] = result
                return result
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                response_time = (time.time() - start_time) * 1000
                result_obj = HealthCheckResult(
                    name=name,
                    status=status,
                    response_time_ms=response_time,
                )
                self.results[name] = result_obj
                return result_obj
            else:
                return HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message="Invalid check result",
                )
        
        except Exception as e:
            logger.error(f"Health check {name} failed: {e}")
            response_time = (time.time() - start_time) * 1000
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                response_time_ms=response_time,
            )
            self.results[name] = result
            return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks.
        
        Returns:
            Dictionary of check results
        """
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """
        Get overall health status.
        
        Returns:
            Overall health status
        """
        if not self.results:
            return HealthStatus.UNKNOWN
        
        statuses = [r.status for r in self.results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Get comprehensive health report.
        
        Returns:
            Health report
        """
        results = self.run_all_checks()
        overall_status = self.get_overall_status()
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {name: result.to_dict() for name, result in results.items()},
            "summary": {
                "total": len(results),
                "healthy": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY),
            },
        }


# Common health checks

def database_health_check(db_path: str) -> Callable:
    """Create database health check."""
    def check() -> HealthCheckResult:
        try:
            import sqlite3
            conn = sqlite3.connect(db_path, timeout=1.0)
            conn.execute("SELECT 1")
            conn.close()
            return HealthCheckResult(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection OK",
            )
        except Exception as e:
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
    
    return check


def memory_health_check(threshold_mb: float = 8192) -> Callable:
    """Create memory health check."""
    def check() -> HealthCheckResult:
        try:
            import psutil
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            
            status = HealthStatus.HEALTHY
            if mem_mb > threshold_mb:
                status = HealthStatus.DEGRADED
            
            return HealthCheckResult(
                name="memory",
                status=status,
                message=f"Memory usage: {mem_mb:.2f} MB",
                details={"memory_mb": mem_mb, "threshold_mb": threshold_mb},
            )
        except Exception as e:
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message=str(e),
            )
    
    return check












