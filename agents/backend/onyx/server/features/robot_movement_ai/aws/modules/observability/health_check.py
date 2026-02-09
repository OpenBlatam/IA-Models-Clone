"""
Health Check
============

Advanced health checking.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check: Callable
    timeout: float = 5.0
    critical: bool = True


class HealthChecker:
    """Health checker."""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self._checks: List[HealthCheck] = []
    
    def register_check(
        self,
        name: str,
        check: Callable,
        timeout: float = 5.0,
        critical: bool = True
    ):
        """Register health check."""
        self._checks.append(HealthCheck(
            name=name,
            check=check,
            timeout=timeout,
            critical=critical
        ))
        logger.info(f"Registered health check: {name}")
    
    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for health_check in self._checks:
            try:
                if asyncio.iscoroutinefunction(health_check.check):
                    result = await asyncio.wait_for(
                        health_check.check(),
                        timeout=health_check.timeout
                    )
                else:
                    result = health_check.check()
                
                results[health_check.name] = {
                    "status": "healthy" if result else "unhealthy",
                    "critical": health_check.critical
                }
                
                if not result and health_check.critical:
                    overall_status = HealthStatus.UNHEALTHY
                elif not result and overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
                
            except asyncio.TimeoutError:
                results[health_check.name] = {
                    "status": "timeout",
                    "critical": health_check.critical
                }
                if health_check.critical:
                    overall_status = HealthStatus.UNHEALTHY
                elif overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
            
            except Exception as e:
                results[health_check.name] = {
                    "status": "error",
                    "error": str(e),
                    "critical": health_check.critical
                }
                if health_check.critical:
                    overall_status = HealthStatus.UNHEALTHY
                elif overall_status == HealthStatus.HEALTHY:
                    overall_status = HealthStatus.DEGRADED
        
        return {
            "service": self.service_name,
            "status": overall_status.value,
            "checks": results
        }
    
    async def liveness(self) -> Dict[str, str]:
        """Liveness probe."""
        return {
            "status": "alive",
            "service": self.service_name
        }
    
    async def readiness(self) -> Dict[str, Any]:
        """Readiness probe."""
        return await self.check_all()















