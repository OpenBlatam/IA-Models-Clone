"""
Advanced Health Monitoring System
==================================

Advanced health monitoring system with dependency checks and status aggregation.
"""

import asyncio
import logging
import time
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
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable
    timeout: float = 5.0
    critical: bool = True
    dependencies: List[str] = field(default_factory=list)


@dataclass
class HealthCheckResult:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str
    duration: float
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SystemHealth:
    """System health status."""
    overall_status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedHealthMonitor:
    """Advanced health monitor with dependency resolution."""
    
    def __init__(self, start_time: Optional[datetime] = None):
        """
        Initialize advanced health monitor.
        
        Args:
            start_time: Optional start time for uptime calculation
        """
        self.start_time = start_time or datetime.now()
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.lock = asyncio.Lock()
    
    def register_check(self, check: HealthCheck):
        """
        Register health check.
        
        Args:
            check: Health check definition
        """
        self.checks[check.name] = check
        logger.info(f"Registered health check: {check.name}")
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check.
        
        Args:
            name: Health check name
            
        Returns:
            Health check result
        """
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check not found: {name}",
                duration=0.0
            )
        
        check = self.checks[name]
        start_time = time.time()
        
        try:
            # Check dependencies first
            for dep_name in check.dependencies:
                if dep_name in self.results:
                    dep_result = self.results[dep_name]
                    if dep_result.status != HealthStatus.HEALTHY:
                        return HealthCheckResult(
                            name=name,
                            status=HealthStatus.UNHEALTHY,
                            message=f"Dependency '{dep_name}' is not healthy",
                            duration=time.time() - start_time
                        )
            
            # Run check with timeout
            if asyncio.iscoroutinefunction(check.check_function):
                result = await asyncio.wait_for(
                    check.check_function(),
                    timeout=check.timeout
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(check.check_function),
                    timeout=check.timeout
                )
            
            duration = time.time() - start_time
            
            # Parse result
            if isinstance(result, dict):
                status = HealthStatus(result.get('status', 'healthy'))
                message = result.get('message', 'Check passed')
                details = result.get('details', {})
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "Check passed" if result else "Check failed"
                details = {}
            else:
                status = HealthStatus.HEALTHY
                message = "Check passed"
                details = {}
            
            check_result = HealthCheckResult(
                name=name,
                status=status,
                message=message,
                duration=duration,
                details=details
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            check_result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {check.timeout}s",
                duration=duration,
                error="timeout"
            )
        except Exception as e:
            duration = time.time() - start_time
            check_result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration=duration,
                error=str(e)
            )
        
        # Store result
        async with self.lock:
            self.results[name] = check_result
        
        return check_result
    
    async def run_all_checks(self) -> SystemHealth:
        """
        Run all health checks.
        
        Returns:
            System health status
        """
        # Sort checks by dependencies
        sorted_checks = self._sort_by_dependencies(list(self.checks.values()))
        
        # Run checks
        results = []
        for check in sorted_checks:
            result = await self.run_check(check.name)
            results.append(result)
        
        # Determine overall status
        overall_status = self._determine_overall_status(results)
        
        # Calculate uptime
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        return SystemHealth(
            overall_status=overall_status,
            checks=results,
            uptime_seconds=uptime_seconds
        )
    
    def _sort_by_dependencies(self, checks: List[HealthCheck]) -> List[HealthCheck]:
        """Sort checks by dependencies."""
        sorted_checks = []
        remaining = checks.copy()
        added = set()
        
        while remaining:
            progress = False
            for check in remaining[:]:
                # Check if all dependencies are satisfied
                deps_satisfied = all(dep in added for dep in check.dependencies)
                
                if deps_satisfied:
                    sorted_checks.append(check)
                    remaining.remove(check)
                    added.add(check.name)
                    progress = True
            
            if not progress:
                # Circular dependency or missing dependency
                logger.warning("Could not resolve all dependencies")
                sorted_checks.extend(remaining)
                break
        
        return sorted_checks
    
    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall health status."""
        if not results:
            return HealthStatus.UNKNOWN
        
        # Check for any unhealthy critical checks
        critical_unhealthy = any(
            r.status == HealthStatus.UNHEALTHY
            for r in results
            if self.checks.get(r.name, HealthCheck("", lambda: True, critical=True)).critical
        )
        
        if critical_unhealthy:
            return HealthStatus.UNHEALTHY
        
        # Check for any unhealthy checks
        any_unhealthy = any(r.status == HealthStatus.UNHEALTHY for r in results)
        if any_unhealthy:
            return HealthStatus.DEGRADED
        
        # Check for any degraded checks
        any_degraded = any(r.status == HealthStatus.DEGRADED for r in results)
        if any_degraded:
            return HealthStatus.DEGRADED
        
        # All healthy
        return HealthStatus.HEALTHY
    
    def get_check_result(self, name: str) -> Optional[HealthCheckResult]:
        """
        Get health check result.
        
        Args:
            name: Health check name
            
        Returns:
            Health check result or None
        """
        return self.results.get(name)
    
    def get_all_results(self) -> Dict[str, HealthCheckResult]:
        """
        Get all health check results.
        
        Returns:
            Dictionary of name -> result
        """
        return self.results.copy()



