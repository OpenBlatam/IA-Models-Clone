"""
Health Check System
==================

System for health checks and service status monitoring.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "duration_ms": self.duration_ms
        }


class HealthChecker:
    """Health check manager."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable[[], Awaitable[HealthCheckResult]]] = {}
        self.cache_ttl: float = 5.0  # Cache results for 5 seconds
        self._cache: Dict[str, tuple[HealthCheckResult, float]] = {}
    
    def register(
        self,
        name: str,
        check_func: Callable[[], Awaitable[HealthCheckResult]]
    ):
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Async function that returns HealthCheckResult
        """
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def check(self, name: str, use_cache: bool = True) -> HealthCheckResult:
        """
        Run a specific health check.
        
        Args:
            name: Check name
            use_cache: Whether to use cached result
            
        Returns:
            HealthCheckResult
        """
        # Check cache
        if use_cache and name in self._cache:
            result, cached_time = self._cache[name]
            age = (datetime.now().timestamp() - cached_time)
            if age < self.cache_ttl:
                return result
        
        # Run check
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found"
            )
        
        try:
            start_time = datetime.now()
            result = await self.checks[name]()
            duration = (datetime.now() - start_time).total_seconds() * 1000
            result.duration_ms = duration
            
            # Cache result
            self._cache[name] = (result, datetime.now().timestamp())
            
            return result
        except Exception as e:
            logger.error(f"Error running health check '{name}': {e}")
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}"
            )
    
    async def check_all(self, use_cache: bool = True) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks.
        
        Args:
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary of check results
        """
        results = {}
        
        # Run checks in parallel
        tasks = [
            self.check(name, use_cache=use_cache)
            for name in self.checks.keys()
        ]
        
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for name, result in zip(self.checks.keys(), check_results):
            if isinstance(result, Exception):
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Exception: {str(result)}"
                )
            else:
                results[name] = result
        
        return results
    
    async def get_overall_status(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Overall health status dictionary
        """
        results = await self.check_all()
        
        statuses = [result.status for result in results.values()]
        
        # Determine overall status
        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {name: result.to_dict() for name, result in results.items()}
        }
    
    def clear_cache(self):
        """Clear health check cache."""
        self._cache.clear()

