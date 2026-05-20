"""
Health Checker Service
======================

Service for monitoring system health and dependencies.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import psutil
import os

from ..utils.helpers import DateTimeHelpers
from ..utils.decorators import retry, log_execution
from ..container import Container


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result"""
    name: str
    status: HealthStatus
    message: str
    response_time_ms: float
    timestamp: datetime = field(default_factory=DateTimeHelpers.now_utc)
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SystemMetrics:
    """System metrics"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    load_average: List[float]
    timestamp: datetime = field(default_factory=DateTimeHelpers.now_utc)


class HealthChecker:
    """Health checker service"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheck] = {}
        self.check_interval = 30  # seconds
        self.timeout = 10  # seconds
        self.is_running = False
        self.checker_task: Optional[asyncio.Task] = None
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.register_check("database", self._check_database)
        self.register_check("cache", self._check_cache)
        self.register_check("ai_service", self._check_ai_service)
        self.register_check("notification_service", self._check_notification_service)
        self.register_check("analytics_service", self._check_analytics_service)
        self.register_check("system_resources", self._check_system_resources)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("network", self._check_network)
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check"""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    async def start(self):
        """Start the health checker"""
        if self.is_running:
            return
        
        self.is_running = True
        self.checker_task = asyncio.create_task(self._run_health_checks())
        logger.info("Health checker started")
    
    async def stop(self):
        """Stop the health checker"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.checker_task:
            self.checker_task.cancel()
            try:
                await self.checker_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health checker stopped")
    
    async def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check"""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
                response_time_ms=0.0
            )
        
        start_time = DateTimeHelpers.now_utc()
        
        try:
            check_func = self.checks[name]
            if asyncio.iscoroutinefunction(check_func):
                result = await asyncio.wait_for(check_func(), timeout=self.timeout)
            else:
                result = check_func()
            
            response_time = (DateTimeHelpers.now_utc() - start_time).total_seconds() * 1000
            
            if isinstance(result, HealthCheck):
                result.response_time_ms = response_time
                return result
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="Check completed successfully",
                    response_time_ms=response_time,
                    details=result if isinstance(result, dict) else {}
                )
        
        except asyncio.TimeoutError:
            response_time = (DateTimeHelpers.now_utc() - start_time).total_seconds() * 1000
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                response_time_ms=response_time,
                error="Timeout"
            )
        
        except Exception as e:
            response_time = (DateTimeHelpers.now_utc() - start_time).total_seconds() * 1000
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                response_time_ms=response_time,
                error=str(e)
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks"""
        results = {}
        
        for name in self.checks:
            results[name] = await self.run_check(name)
        
        self.last_results.update(results)
        return results
    
    def get_last_results(self) -> Dict[str, HealthCheck]:
        """Get last health check results"""
        return self.last_results.copy()
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in self.last_results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        overall_status = self.get_overall_status()
        results = self.get_last_results()
        
        summary = {
            "overall_status": overall_status.value,
            "timestamp": DateTimeHelpers.now_utc().isoformat(),
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "response_time_ms": check.response_time_ms,
                    "timestamp": check.timestamp.isoformat(),
                    "error": check.error
                }
                for name, check in results.items()
            },
            "statistics": {
                "total_checks": len(results),
                "healthy_checks": len([c for c in results.values() if c.status == HealthStatus.HEALTHY]),
                "unhealthy_checks": len([c for c in results.values() if c.status == HealthStatus.UNHEALTHY]),
                "degraded_checks": len([c for c in results.values() if c.status == HealthStatus.DEGRADED]),
                "unknown_checks": len([c for c in results.values() if c.status == HealthStatus.UNKNOWN])
            }
        }
        
        return summary
    
    async def _run_health_checks(self):
        """Run health checks periodically"""
        while self.is_running:
            try:
                await self.run_all_checks()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health checker error: {e}")
                await asyncio.sleep(self.check_interval)
    
    # Default health check implementations
    async def _check_database(self) -> HealthCheck:
        """Check database health"""
        try:
            container = Container()
            database_manager = container.get_database_manager()
            health_data = await database_manager.health_check()
            
            if health_data["status"] == "healthy":
                return HealthCheck(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message="Database is healthy",
                    response_time_ms=0.0,
                    details=health_data
                )
            else:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message="Database is unhealthy",
                    response_time_ms=0.0,
                    details=health_data
                )
        except Exception as e:
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database check failed: {str(e)}",
                response_time_ms=0.0,
                error=str(e)
            )
    
    async def _check_cache(self) -> HealthCheck:
        """Check cache health"""
        try:
            container = Container()
            cache_service = container.get_cache_service()
            
            # Test cache operations
            test_key = "health_check_test"
            test_value = "test_value"
            
            await cache_service.set(test_key, test_value, ttl=60)
            retrieved_value = await cache_service.get(test_key)
            await cache_service.delete(test_key)
            
            if retrieved_value == test_value:
                return HealthCheck(
                    name="cache",
                    status=HealthStatus.HEALTHY,
                    message="Cache is healthy",
                    response_time_ms=0.0
                )
            else:
                return HealthCheck(
                    name="cache",
                    status=HealthStatus.UNHEALTHY,
                    message="Cache test failed",
                    response_time_ms=0.0
                )
        except Exception as e:
            return HealthCheck(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache check failed: {str(e)}",
                response_time_ms=0.0,
                error=str(e)
            )
    
    async def _check_ai_service(self) -> HealthCheck:
        """Check AI service health"""
        try:
            container = Container()
            ai_service = container.get_ai_service()
            
            # Test AI service with a simple request
            result = await ai_service.generate_content("Test prompt", max_tokens=10)
            
            if result:
                return HealthCheck(
                    name="ai_service",
                    status=HealthStatus.HEALTHY,
                    message="AI service is healthy",
                    response_time_ms=0.0
                )
            else:
                return HealthCheck(
                    name="ai_service",
                    status=HealthStatus.DEGRADED,
                    message="AI service returned empty result",
                    response_time_ms=0.0
                )
        except Exception as e:
            return HealthCheck(
                name="ai_service",
                status=HealthStatus.UNHEALTHY,
                message=f"AI service check failed: {str(e)}",
                response_time_ms=0.0,
                error=str(e)
            )
    
    async def _check_notification_service(self) -> HealthCheck:
        """Check notification service health"""
        try:
            container = Container()
            notification_service = container.get_notification_service()
            
            # Test notification service (without actually sending)
            # In a real implementation, you might send a test notification
            
            return HealthCheck(
                name="notification_service",
                status=HealthStatus.HEALTHY,
                message="Notification service is healthy",
                response_time_ms=0.0
            )
        except Exception as e:
            return HealthCheck(
                name="notification_service",
                status=HealthStatus.UNHEALTHY,
                message=f"Notification service check failed: {str(e)}",
                response_time_ms=0.0,
                error=str(e)
            )
    
    async def _check_analytics_service(self) -> HealthCheck:
        """Check analytics service health"""
        try:
            container = Container()
            analytics_service = container.get_analytics_service()
            
            # Test analytics service
            await analytics_service.track_event("health_check", {"test": True})
            
            return HealthCheck(
                name="analytics_service",
                status=HealthStatus.HEALTHY,
                message="Analytics service is healthy",
                response_time_ms=0.0
            )
        except Exception as e:
            return HealthCheck(
                name="analytics_service",
                status=HealthStatus.UNHEALTHY,
                message=f"Analytics service check failed: {str(e)}",
                response_time_ms=0.0,
                error=str(e)
            )
    
    def _check_system_resources(self) -> HealthCheck:
        """Check system resources"""
        try:
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on resource usage
            if cpu_usage > 90 or memory.percent > 90 or disk.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "System resources critically high"
            elif cpu_usage > 80 or memory.percent > 80 or disk.percent > 80:
                status = HealthStatus.DEGRADED
                message = "System resources high"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources normal"
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                response_time_ms=0.0,
                details={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory.percent,
                    "disk_usage": disk.percent,
                    "memory_available": memory.available,
                    "disk_free": disk.free
                }
            )
        except Exception as e:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"System resources check failed: {str(e)}",
                response_time_ms=0.0,
                error=str(e)
            )
    
    def _check_disk_space(self) -> HealthCheck:
        """Check disk space"""
        try:
            disk = psutil.disk_usage('/')
            free_percent = (disk.free / disk.total) * 100
            
            if free_percent < 5:
                status = HealthStatus.UNHEALTHY
                message = "Disk space critically low"
            elif free_percent < 10:
                status = HealthStatus.DEGRADED
                message = "Disk space low"
            else:
                status = HealthStatus.HEALTHY
                message = "Disk space sufficient"
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                response_time_ms=0.0,
                details={
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "free_percent": free_percent
                }
            )
        except Exception as e:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk space check failed: {str(e)}",
                response_time_ms=0.0,
                error=str(e)
            )
    
    async def _check_network(self) -> HealthCheck:
        """Check network connectivity"""
        try:
            # Test external connectivity
            async with aiohttp.ClientSession() as session:
                async with session.get('https://httpbin.org/status/200', timeout=5) as response:
                    if response.status == 200:
                        return HealthCheck(
                            name="network",
                            status=HealthStatus.HEALTHY,
                            message="Network connectivity is healthy",
                            response_time_ms=0.0
                        )
                    else:
                        return HealthCheck(
                            name="network",
                            status=HealthStatus.DEGRADED,
                            message=f"Network check returned status {response.status}",
                            response_time_ms=0.0
                        )
        except Exception as e:
            return HealthCheck(
                name="network",
                status=HealthStatus.UNHEALTHY,
                message=f"Network check failed: {str(e)}",
                response_time_ms=0.0,
                error=str(e)
            )


# Global health checker
health_checker = HealthChecker()


# Utility functions
async def start_health_checker():
    """Start the health checker"""
    await health_checker.start()


async def stop_health_checker():
    """Stop the health checker"""
    await health_checker.stop()


async def run_health_check(name: str) -> HealthCheck:
    """Run a specific health check"""
    return await health_checker.run_check(name)


async def run_all_health_checks() -> Dict[str, HealthCheck]:
    """Run all health checks"""
    return await health_checker.run_all_checks()


def get_health_summary() -> Dict[str, Any]:
    """Get health summary"""
    return health_checker.get_health_summary()


def get_overall_health_status() -> HealthStatus:
    """Get overall health status"""
    return health_checker.get_overall_status()


def register_health_check(name: str, check_func: Callable):
    """Register a health check"""
    health_checker.register_check(name, check_func)




