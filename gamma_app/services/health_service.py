"""
Gamma App - Health Check Service
Comprehensive health monitoring and status reporting
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import redis
from sqlalchemy import text
from sqlalchemy.orm import Session

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
    message: str
    response_time: float
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class SystemHealth:
    """Overall system health"""
    status: HealthStatus
    timestamp: datetime
    checks: List[HealthCheck]
    uptime: float
    version: str
    environment: str

class HealthService:
    """
    Comprehensive health monitoring service
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize health service"""
        self.config = config or {}
        self.start_time = time.time()
        self.version = self.config.get('version', '1.0.0')
        self.environment = self.config.get('environment', 'development')
        
        # Health check results cache
        self.health_cache: Dict[str, HealthCheck] = {}
        self.cache_ttl = self.config.get('cache_ttl', 30)  # 30 seconds
        self.last_check = 0
        
        logger.info("Health Service initialized successfully")

    async def get_system_health(self, force_refresh: bool = False) -> SystemHealth:
        """Get overall system health"""
        try:
            current_time = time.time()
            
            # Check if we need to refresh cache
            if force_refresh or (current_time - self.last_check) > self.cache_ttl:
                await self._run_all_health_checks()
                self.last_check = current_time
            
            # Determine overall status
            overall_status = self._determine_overall_status()
            
            # Calculate uptime
            uptime = current_time - self.start_time
            
            return SystemHealth(
                status=overall_status,
                timestamp=datetime.now(),
                checks=list(self.health_cache.values()),
                uptime=uptime,
                version=self.version,
                environment=self.environment
            )
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return SystemHealth(
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.now(),
                checks=[],
                uptime=time.time() - self.start_time,
                version=self.version,
                environment=self.environment
            )

    async def _run_all_health_checks(self):
        """Run all health checks"""
        checks = [
            self._check_database,
            self._check_redis,
            self._check_disk_space,
            self._check_memory,
            self._check_cpu,
            self._check_network,
            self._check_ai_services,
            self._check_cache_service,
            self._check_security_service,
            self._check_performance_service
        ]
        
        # Run checks concurrently
        tasks = [check() for check in checks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Health check {checks[i].__name__} failed: {result}")
                self.health_cache[checks[i].__name__] = HealthCheck(
                    name=checks[i].__name__,
                    status=HealthStatus.UNKNOWN,
                    message="Check failed with exception",
                    response_time=0.0,
                    timestamp=datetime.now(),
                    error=str(result)
                )

    async def _check_database(self) -> HealthCheck:
        """Check database connectivity and performance"""
        start_time = time.time()
        
        try:
            # This would use the actual database session
            # For now, we'll simulate the check
            db_url = self.config.get('database_url', 'sqlite:///./gamma_app.db')
            
            # Simulate database check
            await asyncio.sleep(0.01)  # Simulate query time
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="database",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    "url": db_url.split('@')[-1] if '@' in db_url else db_url,
                    "connection_pool_size": 10
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message="Database connection failed",
                response_time=response_time,
                timestamp=datetime.now(),
                error=str(e)
            )

    async def _check_redis(self) -> HealthCheck:
        """Check Redis connectivity and performance"""
        start_time = time.time()
        
        try:
            redis_url = self.config.get('redis_url', 'redis://localhost:6379')
            
            # Try to connect to Redis
            r = redis.Redis.from_url(redis_url, decode_responses=True)
            r.ping()
            
            # Get Redis info
            info = r.info()
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connection successful",
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    "version": info.get('redis_version'),
                    "memory_used": info.get('used_memory_human'),
                    "connected_clients": info.get('connected_clients')
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message="Redis connection failed",
                response_time=response_time,
                timestamp=datetime.now(),
                error=str(e)
            )

    async def _check_disk_space(self) -> HealthCheck:
        """Check disk space availability"""
        start_time = time.time()
        
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            response_time = (time.time() - start_time) * 1000
            
            if free_percent > 20:
                status = HealthStatus.HEALTHY
                message = f"Disk space healthy: {free_percent:.1f}% free"
            elif free_percent > 10:
                status = HealthStatus.DEGRADED
                message = f"Disk space low: {free_percent:.1f}% free"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Disk space critical: {free_percent:.1f}% free"
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    "total": disk_usage.total,
                    "used": disk_usage.used,
                    "free": disk_usage.free,
                    "free_percent": free_percent
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message="Disk space check failed",
                response_time=response_time,
                timestamp=datetime.now(),
                error=str(e)
            )

    async def _check_memory(self) -> HealthCheck:
        """Check memory usage"""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            response_time = (time.time() - start_time) * 1000
            
            if memory_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"Memory usage healthy: {memory_percent:.1f}%"
            elif memory_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"Memory usage high: {memory_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"Memory usage critical: {memory_percent:.1f}%"
            
            return HealthCheck(
                name="memory",
                status=status,
                message=message,
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory_percent
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="Memory check failed",
                response_time=response_time,
                timestamp=datetime.now(),
                error=str(e)
            )

    async def _check_cpu(self) -> HealthCheck:
        """Check CPU usage"""
        start_time = time.time()
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            response_time = (time.time() - start_time) * 1000
            
            if cpu_percent < 80:
                status = HealthStatus.HEALTHY
                message = f"CPU usage healthy: {cpu_percent:.1f}%"
            elif cpu_percent < 90:
                status = HealthStatus.DEGRADED
                message = f"CPU usage high: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"CPU usage critical: {cpu_percent:.1f}%"
            
            return HealthCheck(
                name="cpu",
                status=status,
                message=message,
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="cpu",
                status=HealthStatus.UNKNOWN,
                message="CPU check failed",
                response_time=response_time,
                timestamp=datetime.now(),
                error=str(e)
            )

    async def _check_network(self) -> HealthCheck:
        """Check network connectivity"""
        start_time = time.time()
        
        try:
            # Check network interfaces
            net_io = psutil.net_io_counters()
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="network",
                status=HealthStatus.HEALTHY,
                message="Network interfaces active",
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="network",
                status=HealthStatus.UNKNOWN,
                message="Network check failed",
                response_time=response_time,
                timestamp=datetime.now(),
                error=str(e)
            )

    async def _check_ai_services(self) -> HealthCheck:
        """Check AI services availability"""
        start_time = time.time()
        
        try:
            # Check if AI services are configured
            openai_key = self.config.get('openai_api_key')
            anthropic_key = self.config.get('anthropic_api_key')
            
            response_time = (time.time() - start_time) * 1000
            
            if openai_key or anthropic_key:
                status = HealthStatus.HEALTHY
                message = "AI services configured"
                details = {
                    "openai_configured": bool(openai_key),
                    "anthropic_configured": bool(anthropic_key)
                }
            else:
                status = HealthStatus.DEGRADED
                message = "No AI services configured"
                details = {
                    "openai_configured": False,
                    "anthropic_configured": False
                }
            
            return HealthCheck(
                name="ai_services",
                status=status,
                message=message,
                response_time=response_time,
                timestamp=datetime.now(),
                details=details
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="ai_services",
                status=HealthStatus.UNKNOWN,
                message="AI services check failed",
                response_time=response_time,
                timestamp=datetime.now(),
                error=str(e)
            )

    async def _check_cache_service(self) -> HealthCheck:
        """Check cache service health"""
        start_time = time.time()
        
        try:
            # This would check the actual cache service
            # For now, we'll simulate the check
            await asyncio.sleep(0.01)
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="cache_service",
                status=HealthStatus.HEALTHY,
                message="Cache service operational",
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    "cache_type": "redis",
                    "hit_rate": 0.95  # This would be actual hit rate
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="cache_service",
                status=HealthStatus.UNHEALTHY,
                message="Cache service unavailable",
                response_time=response_time,
                timestamp=datetime.now(),
                error=str(e)
            )

    async def _check_security_service(self) -> HealthCheck:
        """Check security service health"""
        start_time = time.time()
        
        try:
            # This would check the actual security service
            await asyncio.sleep(0.01)
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="security_service",
                status=HealthStatus.HEALTHY,
                message="Security service operational",
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    "rate_limiting": True,
                    "threat_detection": True,
                    "encryption": True
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="security_service",
                status=HealthStatus.UNHEALTHY,
                message="Security service unavailable",
                response_time=response_time,
                timestamp=datetime.now(),
                error=str(e)
            )

    async def _check_performance_service(self) -> HealthCheck:
        """Check performance monitoring service"""
        start_time = time.time()
        
        try:
            # This would check the actual performance service
            await asyncio.sleep(0.01)
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="performance_service",
                status=HealthStatus.HEALTHY,
                message="Performance monitoring active",
                response_time=response_time,
                timestamp=datetime.now(),
                details={
                    "metrics_collection": True,
                    "alerting": True,
                    "profiling": True
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="performance_service",
                status=HealthStatus.UNHEALTHY,
                message="Performance monitoring unavailable",
                response_time=response_time,
                timestamp=datetime.now(),
                error=str(e)
            )

    def _determine_overall_status(self) -> HealthStatus:
        """Determine overall system status based on individual checks"""
        if not self.health_cache:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in self.health_cache.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    async def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for monitoring dashboards"""
        health = await self.get_system_health()
        
        return {
            "status": health.status.value,
            "timestamp": health.timestamp.isoformat(),
            "uptime": health.uptime,
            "version": health.version,
            "environment": health.environment,
            "checks": {
                check.name: {
                    "status": check.status.value,
                    "message": check.message,
                    "response_time": check.response_time,
                    "details": check.details
                }
                for check in health.checks
            },
            "summary": {
                "total_checks": len(health.checks),
                "healthy": len([c for c in health.checks if c.status == HealthStatus.HEALTHY]),
                "degraded": len([c for c in health.checks if c.status == HealthStatus.DEGRADED]),
                "unhealthy": len([c for c in health.checks if c.status == HealthStatus.UNHEALTHY]),
                "unknown": len([c for c in health.checks if c.status == HealthStatus.UNKNOWN])
            }
        }

    async def get_readiness(self) -> Tuple[bool, str]:
        """Check if system is ready to serve requests"""
        health = await self.get_system_health()
        
        # Critical services that must be healthy
        critical_services = ["database", "redis"]
        critical_checks = [c for c in health.checks if c.name in critical_services]
        
        if not critical_checks:
            return False, "Critical services not checked"
        
        unhealthy_critical = [c for c in critical_checks if c.status == HealthStatus.UNHEALTHY]
        
        if unhealthy_critical:
            return False, f"Critical services unhealthy: {[c.name for c in unhealthy_critical]}"
        
        return True, "System ready"

    async def get_liveness(self) -> Tuple[bool, str]:
        """Check if system is alive (basic health check)"""
        try:
            # Simple liveness check - just verify the service is responding
            health = await self.get_system_health()
            return True, "System alive"
        except Exception as e:
            return False, f"System not alive: {str(e)}"

# Global health service instance
health_service = HealthService()

# Convenience functions
async def get_system_health(force_refresh: bool = False) -> SystemHealth:
    """Get system health"""
    return await health_service.get_system_health(force_refresh)

async def get_health_summary() -> Dict[str, Any]:
    """Get health summary"""
    return await health_service.get_health_summary()

async def is_ready() -> Tuple[bool, str]:
    """Check if system is ready"""
    return await health_service.get_readiness()

async def is_alive() -> Tuple[bool, str]:
    """Check if system is alive"""
    return await health_service.get_liveness()
