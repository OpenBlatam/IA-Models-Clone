"""
Advanced Health Checker for Piel Mejorador AI SAM3
==================================================

Comprehensive health checking with dependencies.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: HealthStatus
    message: str = ""
    response_time_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class HealthChecker:
    """
    Advanced health checker.
    
    Features:
    - Dependency checking
    - Response time tracking
    - Status aggregation
    - Custom health checks
    """
    
    def __init__(self):
        """Initialize health checker."""
        self._checks: Dict[str, Callable] = {}
        self._check_results: Dict[str, HealthCheck] = {}
        self._timeout: float = 5.0
        
        self._stats = {
            "total_checks": 0,
            "healthy_checks": 0,
            "unhealthy_checks": 0,
        }
    
    def register_check(
        self,
        name: str,
        check_func: Callable,
        timeout: Optional[float] = None
    ):
        """
        Register a health check.
        
        Args:
            name: Check name
            check_func: Async function returning HealthCheck
            timeout: Optional timeout override
        """
        self._checks[name] = {
            "func": check_func,
            "timeout": timeout or self._timeout,
        }
        logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> Optional[HealthCheck]:
        """
        Run a specific health check.
        
        Args:
            name: Check name
            
        Returns:
            HealthCheck result or None
        """
        if name not in self._checks:
            return None
        
        check_config = self._checks[name]
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                check_config["func"](),
                timeout=check_config["timeout"]
            )
            
            response_time = (time.time() - start_time) * 1000
            result.response_time_ms = response_time
            
            self._check_results[name] = result
            self._stats["total_checks"] += 1
            
            if result.status == HealthStatus.HEALTHY:
                self._stats["healthy_checks"] += 1
            else:
                self._stats["unhealthy_checks"] += 1
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Health check {name} timed out")
            result = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                response_time_ms=check_config["timeout"] * 1000
            )
            self._check_results[name] = result
            return result
            
        except Exception as e:
            logger.error(f"Health check {name} failed: {e}")
            result = HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000
            )
            self._check_results[name] = result
            return result
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of check results
        """
        tasks = [
            self.run_check(name)
            for name in self._checks.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None and exceptions
        check_results = {}
        for name, result in zip(self._checks.keys(), results):
            if result and isinstance(result, HealthCheck):
                check_results[name] = result
        
        return check_results
    
    def get_overall_status(self) -> HealthStatus:
        """
        Get overall health status.
        
        Returns:
            Aggregated health status
        """
        if not self._check_results:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in self._check_results.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        overall = self.get_overall_status()
        
        return {
            "status": overall.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "response_time_ms": check.response_time_ms,
                    "details": check.details,
                }
                for name, check in self._check_results.items()
            },
            "statistics": self._stats,
        }




