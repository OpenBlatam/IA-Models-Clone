"""
Advanced Health Check Service
==============================
Service for comprehensive health checks
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check definition"""
    name: str
    check_fn: Callable[[], Awaitable[bool]]
    timeout: float = 5.0
    critical: bool = True
    enabled: bool = True
    last_check: Optional[datetime] = None
    last_result: Optional[bool] = None
    last_error: Optional[str] = None


@dataclass
class HealthCheckResult:
    """Health check result"""
    name: str
    status: bool
    duration_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceHealth:
    """Service health status"""
    service_name: str
    overall_status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: datetime
    uptime_seconds: float
    version: Optional[str] = None
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy"""
        return self.overall_status == HealthStatus.HEALTHY


class HealthCheckService:
    """
    Service for comprehensive health checks.
    
    Features:
    - Multiple health checks
    - Async check execution
    - Timeout handling
    - Critical vs non-critical checks
    - Health aggregation
    - Statistics
    """
    
    def __init__(self, service_name: str):
        """
        Initialize health check service.
        
        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        self._checks: Dict[str, HealthCheck] = {}
        self._start_time = time.time()
        self._version: Optional[str] = None
    
    def set_version(self, version: str):
        """Set service version"""
        self._version = version
    
    def register_check(
        self,
        name: str,
        check_fn: Callable[[], Awaitable[bool]],
        timeout: float = 5.0,
        critical: bool = True,
        enabled: bool = True
    ):
        """
        Register a health check.
        
        Args:
            name: Check name
            check_fn: Async function that returns bool
            timeout: Check timeout in seconds
            critical: Whether check is critical
            enabled: Whether check is enabled
        """
        check = HealthCheck(
            name=name,
            check_fn=check_fn,
            timeout=timeout,
            critical=critical,
            enabled=enabled
        )
        
        self._checks[name] = check
        logger.info(f"Registered health check: {name} (critical: {critical})")
    
    def unregister_check(self, name: str) -> bool:
        """Unregister a health check"""
        if name in self._checks:
            del self._checks[name]
            logger.info(f"Unregistered health check: {name}")
            return True
        return False
    
    async def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """
        Run a single health check.
        
        Args:
            name: Check name
        
        Returns:
            HealthCheckResult or None if check not found
        """
        check = self._checks.get(name)
        if not check or not check.enabled:
            return None
        
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                check.check_fn(),
                timeout=check.timeout
            )
            
            duration_ms = (time.time() - start_time) * 1000
            check.last_check = datetime.now()
            check.last_result = result
            check.last_error = None
            
            return HealthCheckResult(
                name=name,
                status=result,
                duration_ms=duration_ms
            )
        
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            check.last_check = datetime.now()
            check.last_result = False
            check.last_error = "Check timeout"
            
            return HealthCheckResult(
                name=name,
                status=False,
                duration_ms=duration_ms,
                error="Check timeout"
            )
        
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            check.last_check = datetime.now()
            check.last_result = False
            check.last_error = str(e)
            
            return HealthCheckResult(
                name=name,
                status=False,
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def run_all_checks(self) -> ServiceHealth:
        """
        Run all health checks.
        
        Returns:
            ServiceHealth with aggregated results
        """
        checks = [c for c in self._checks.values() if c.enabled]
        
        # Run all checks in parallel
        tasks = [self.run_check(name) for name in self._checks.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        check_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check error: {result}")
                continue
            if result:
                check_results.append(result)
        
        # Determine overall status
        critical_checks = [c for c in check_results if self._checks[c.name].critical]
        critical_failed = [r for r in critical_checks if not r.status]
        
        if critical_failed:
            overall_status = HealthStatus.UNHEALTHY
        elif any(not r.status for r in check_results):
            overall_status = HealthStatus.DEGRADED
        elif all(r.status for r in check_results):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        uptime = time.time() - self._start_time
        
        return ServiceHealth(
            service_name=self.service_name,
            overall_status=overall_status,
            checks=check_results,
            timestamp=datetime.now(),
            uptime_seconds=uptime,
            version=self._version
        )
    
    def get_check_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific check"""
        check = self._checks.get(name)
        if not check:
            return None
        
        return {
            'name': name,
            'enabled': check.enabled,
            'critical': check.critical,
            'last_check': check.last_check.isoformat() if check.last_check else None,
            'last_result': check.last_result,
            'last_error': check.last_error
        }
    
    def list_checks(self) -> List[str]:
        """List all registered check names"""
        return list(self._checks.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get health check statistics"""
        enabled_checks = [c for c in self._checks.values() if c.enabled]
        critical_checks = [c for c in enabled_checks if c.critical]
        
        return {
            'service_name': self.service_name,
            'total_checks': len(self._checks),
            'enabled_checks': len(enabled_checks),
            'critical_checks': len(critical_checks),
            'uptime_seconds': time.time() - self._start_time,
            'version': self._version
        }


# Global health check service instance
_health_check_service: Optional[HealthCheckService] = None


def get_health_check_service(service_name: str = "default") -> HealthCheckService:
    """Get or create health check service instance"""
    global _health_check_service
    if _health_check_service is None:
        _health_check_service = HealthCheckService(service_name)
    return _health_check_service

