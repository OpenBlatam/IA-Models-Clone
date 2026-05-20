"""
Health Monitor for Color Grading AI
====================================

Comprehensive health monitoring for all services.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_func: Callable
    timeout: float = 5.0
    critical: bool = True
    interval: float = 30.0
    last_check: Optional[datetime] = None
    last_status: HealthStatus = HealthStatus.UNKNOWN
    last_error: Optional[str] = None


class HealthMonitor:
    """
    Comprehensive health monitor.
    
    Features:
    - Multiple health checks
    - Automatic monitoring
    - Status aggregation
    - Alerting
    - Statistics
    """
    
    def __init__(self):
        """Initialize health monitor."""
        self._checks: Dict[str, HealthCheck] = {}
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._status_history: List[Dict[str, Any]] = []
        self._max_history = 100
    
    def register_check(
        self,
        name: str,
        check_func: Callable,
        timeout: float = 5.0,
        critical: bool = True,
        interval: float = 30.0
    ):
        """
        Register health check.
        
        Args:
            name: Check name
            check_func: Check function (async or sync)
            timeout: Check timeout
            critical: Whether check is critical
            interval: Check interval in seconds
        """
        check = HealthCheck(
            name=name,
            check_func=check_func,
            timeout=timeout,
            critical=critical,
            interval=interval
        )
        self._checks[name] = check
        logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> HealthStatus:
        """
        Run single health check.
        
        Args:
            name: Check name
            
        Returns:
            Health status
        """
        check = self._checks.get(name)
        if not check:
            return HealthStatus.UNKNOWN
        
        try:
            if asyncio.iscoroutinefunction(check.check_func):
                result = await asyncio.wait_for(
                    check.check_func(),
                    timeout=check.timeout
                )
            else:
                result = check.check_func()
            
            # Result should be bool or HealthStatus
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            elif isinstance(result, HealthStatus):
                status = result
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
            
            check.last_status = status
            check.last_error = None
            check.last_check = datetime.now()
            
            return status
        
        except asyncio.TimeoutError:
            check.last_status = HealthStatus.UNHEALTHY
            check.last_error = f"Timeout after {check.timeout}s"
            check.last_check = datetime.now()
            return HealthStatus.UNHEALTHY
        
        except Exception as e:
            check.last_status = HealthStatus.UNHEALTHY
            check.last_error = str(e)
            check.last_check = datetime.now()
            logger.error(f"Health check {name} failed: {e}")
            return HealthStatus.UNHEALTHY
    
    async def run_all_checks(self) -> Dict[str, HealthStatus]:
        """
        Run all health checks.
        
        Returns:
            Dictionary of check results
        """
        results = {}
        tasks = []
        
        for name in self._checks.keys():
            tasks.append(self.run_check(name))
        
        statuses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for name, status in zip(self._checks.keys(), statuses):
            if isinstance(status, Exception):
                results[name] = HealthStatus.UNHEALTHY
            else:
                results[name] = status
        
        # Record in history
        self._record_status(results)
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """
        Get overall health status.
        
        Returns:
            Overall status
        """
        if not self._checks:
            return HealthStatus.UNKNOWN
        
        critical_failed = False
        any_failed = False
        
        for check in self._checks.values():
            if check.last_status == HealthStatus.UNHEALTHY:
                any_failed = True
                if check.critical:
                    critical_failed = True
        
        if critical_failed:
            return HealthStatus.UNHEALTHY
        elif any_failed:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Get comprehensive status report.
        
        Returns:
            Status report dictionary
        """
        overall = self.get_overall_status()
        checks_status = {}
        
        for name, check in self._checks.items():
            checks_status[name] = {
                "status": check.last_status.value,
                "critical": check.critical,
                "last_check": check.last_check.isoformat() if check.last_check else None,
                "last_error": check.last_error,
                "timeout": check.timeout,
                "interval": check.interval,
            }
        
        return {
            "overall_status": overall.value,
            "checks": checks_status,
            "total_checks": len(self._checks),
            "healthy_checks": sum(1 for c in self._checks.values() if c.last_status == HealthStatus.HEALTHY),
            "unhealthy_checks": sum(1 for c in self._checks.values() if c.last_status == HealthStatus.UNHEALTHY),
        }
    
    async def start_monitoring(self):
        """Start automatic health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop automatic health monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Monitor health in loop."""
        while self._running:
            try:
                await self.run_all_checks()
                
                # Wait for next check (use minimum interval)
                min_interval = min(
                    (c.interval for c in self._checks.values()),
                    default=30.0
                )
                await asyncio.sleep(min_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5.0)
    
    def _record_status(self, results: Dict[str, HealthStatus]):
        """Record status in history."""
        self._status_history.append({
            "timestamp": datetime.now().isoformat(),
            "results": {k: v.value for k, v in results.items()},
            "overall": self.get_overall_status().value,
        })
        
        # Keep only last N entries
        if len(self._status_history) > self._max_history:
            self._status_history = self._status_history[-self._max_history:]
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get status history."""
        return self._status_history[-limit:]




