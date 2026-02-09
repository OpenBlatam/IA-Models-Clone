"""
Health Checker for Document Analyzer
====================================

Comprehensive health checking system for all components.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: HealthStatus
    message: str = ""
    response_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class HealthChecker:
    """Comprehensive health checker"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.check_history: List[HealthCheck] = []
        self.max_history = 100
        logger.info("HealthChecker initialized")
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def check(self, name: str) -> HealthCheck:
        """Run a specific health check"""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found"
            )
        
        start_time = time.time()
        try:
            check_func = self.checks[name]
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
            
            response_time = time.time() - start_time
            
            if isinstance(result, HealthCheck):
                result.response_time = response_time
                return result
            elif isinstance(result, dict):
                status = HealthStatus(result.get("status", "unknown"))
                return HealthCheck(
                    name=name,
                    status=status,
                    message=result.get("message", ""),
                    response_time=response_time,
                    details=result.get("details", {})
                )
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    response_time=response_time
                )
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                response_time=response_time
            )
    
    async def check_all(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks"""
        results = {}
        
        for name in self.checks.keys():
            results[name] = await self.check(name)
            self.check_history.append(results[name])
            
            # Limit history size
            if len(self.check_history) > self.max_history:
                self.check_history = self.check_history[-self.max_history:]
        
        return results
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health status"""
        checks = await self.check_all()
        
        statuses = [check.status for check in checks.values()]
        
        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNKNOWN
        
        return {
            "status": overall_status.value,
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "response_time": check.response_time,
                    "details": check.details
                }
                for name, check in checks.items()
            },
            "timestamp": datetime.now().isoformat()
        }

# Global instance
health_checker = HealthChecker()
















