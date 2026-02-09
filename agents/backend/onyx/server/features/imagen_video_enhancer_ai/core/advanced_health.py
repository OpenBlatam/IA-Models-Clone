"""
Advanced Health Check System
============================

Advanced health check system with dependencies and status aggregation.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
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
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    response_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms
        }


class HealthCheck:
    """Health check definition."""
    
    def __init__(
        self,
        name: str,
        check_func: Callable[[], Awaitable[HealthCheckResult]],
        critical: bool = True,
        timeout: float = 5.0
    ):
        """
        Initialize health check.
        
        Args:
            name: Check name
            check_func: Async check function
            critical: Whether check is critical
            timeout: Check timeout in seconds
        """
        self.name = name
        self.check_func = check_func
        self.critical = critical
        self.timeout = timeout


class AdvancedHealthChecker:
    """Advanced health checker with dependencies."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, HealthCheck] = {}
        self.dependencies: Dict[str, List[str]] = {}
    
    def register(
        self,
        name: str,
        check_func: Callable[[], Awaitable[HealthCheckResult]],
        critical: bool = True,
        timeout: float = 5.0,
        depends_on: Optional[List[str]] = None
    ):
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Async check function
            critical: Whether check is critical
            timeout: Check timeout
            depends_on: Optional list of dependency check names
        """
        check = HealthCheck(name, check_func, critical, timeout)
        self.checks[name] = check
        
        if depends_on:
            self.dependencies[name] = depends_on
        
        logger.debug(f"Registered health check: {name}")
    
    async def check(self, name: str) -> HealthCheckResult:
        """
        Run a single health check.
        
        Args:
            name: Check name
            
        Returns:
            Health check result
        """
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check {name} not found"
            )
        
        check = self.checks[name]
        start = datetime.now()
        
        try:
            # Check dependencies first
            if name in self.dependencies:
                for dep_name in self.dependencies[name]:
                    dep_result = await self.check(dep_name)
                    if dep_result.status == HealthStatus.UNHEALTHY:
                        return HealthCheckResult(
                            name=name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"Dependency {dep_name} is unhealthy",
                            details={"dependency": dep_name, "dependency_status": dep_result.status.value}
                        )
            
            # Run check with timeout
            result = await asyncio.wait_for(
                check.check_func(),
                timeout=check.timeout
            )
            
            elapsed = (datetime.now() - start).total_seconds() * 1000
            result.response_time_ms = elapsed
            
            return result
            
        except asyncio.TimeoutError:
            elapsed = (datetime.now() - start).total_seconds() * 1000
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check timed out after {check.timeout}s",
                response_time_ms=elapsed
            )
        except Exception as e:
            elapsed = (datetime.now() - start).total_seconds() * 1000
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                response_time_ms=elapsed
            )
    
    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks.
        
        Returns:
            Dictionary of check results
        """
        results = {}
        
        for name in self.checks.keys():
            results[name] = await self.check(name)
        
        return results
    
    async def get_overall_status(self) -> Dict[str, Any]:
        """
        Get overall health status.
        
        Returns:
            Overall status dictionary
        """
        results = await self.check_all()
        
        # Determine overall status
        critical_checks = [r for name, r in results.items() if self.checks[name].critical]
        unhealthy_critical = [r for r in critical_checks if r.status == HealthStatus.UNHEALTHY]
        degraded_critical = [r for r in critical_checks if r.status == HealthStatus.DEGRADED]
        
        if unhealthy_critical:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_critical:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {name: r.to_dict() for name, r in results.items()},
            "summary": {
                "total": len(results),
                "healthy": len([r for r in results.values() if r.status == HealthStatus.HEALTHY]),
                "degraded": len([r for r in results.values() if r.status == HealthStatus.DEGRADED]),
                "unhealthy": len([r for r in results.values() if r.status == HealthStatus.UNHEALTHY])
            }
        }
    
    def unregister(self, name: str):
        """
        Unregister a health check.
        
        Args:
            name: Check name
        """
        self.checks.pop(name, None)
        self.dependencies.pop(name, None)
        
        # Remove from other dependencies
        for deps in self.dependencies.values():
            if name in deps:
                deps.remove(name)




