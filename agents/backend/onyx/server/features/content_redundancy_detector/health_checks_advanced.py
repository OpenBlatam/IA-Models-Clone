"""
Advanced Health Checks with Dependency Monitoring
Comprehensive health monitoring for microservices architecture
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import json

try:
    import redis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import psycopg2
    from psycopg2 import OperationalError
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    response_time: float
    details: Optional[Dict[str, Any]] = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class HealthChecker:
    """Advanced health checker with dependency monitoring"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.dependencies: Dict[str, Dict[str, Any]] = {}
        self.overall_status = HealthStatus.UNKNOWN
        self.last_check = None
        self.check_interval = 30  # seconds
        
        # Register default checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        self.register_check("system", self._check_system_health)
        self.register_check("memory", self._check_memory_health)
        self.register_check("disk", self._check_disk_health)
        self.register_check("cpu", self._check_cpu_health)
        
        if REDIS_AVAILABLE:
            self.register_check("redis", self._check_redis_health)
        
        if POSTGRES_AVAILABLE:
            self.register_check("database", self._check_database_health)
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check function"""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")
    
    def add_dependency(self, name: str, check_func: Callable, 
                      timeout: float = 5.0, critical: bool = True):
        """Add an external dependency to monitor"""
        self.dependencies[name] = {
            "check_func": check_func,
            "timeout": timeout,
            "critical": critical,
            "last_status": HealthStatus.UNKNOWN,
            "last_check": None
        }
        logger.info(f"Added dependency: {name}")
    
    async def _check_system_health(self) -> HealthCheckResult:
        """Check basic system health"""
        start_time = time.time()
        
        try:
            # Check if system is responsive
            uptime = time.time() - psutil.boot_time()
            
            # Basic system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            status = HealthStatus.HEALTHY
            message = "System is healthy"
            details = {
                "uptime_seconds": uptime,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "available_memory_gb": memory.available / (1024**3)
            }
            
            # Check for warning conditions
            if cpu_percent > 80:
                status = HealthStatus.DEGRADED
                message = "High CPU usage detected"
            elif memory.percent > 85:
                status = HealthStatus.DEGRADED
                message = "High memory usage detected"
            
            response_time = time.time() - start_time
            
            return HealthCheckResult(
                name="system",
                status=status,
                message=message,
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return HealthCheckResult(
                name="system",
                status=HealthStatus.UNHEALTHY,
                message=f"System health check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def _check_memory_health(self) -> HealthCheckResult:
        """Check memory health"""
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            status = HealthStatus.HEALTHY
            message = "Memory is healthy"
            details = {
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
                "used_memory_gb": memory.used / (1024**3),
                "memory_percent": memory.percent,
                "swap_total_gb": swap.total / (1024**3),
                "swap_used_gb": swap.used / (1024**3),
                "swap_percent": swap.percent
            }
            
            # Check memory thresholds
            if memory.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "Critical memory usage"
            elif memory.percent > 80:
                status = HealthStatus.DEGRADED
                message = "High memory usage"
            
            # Check swap usage
            if swap.percent > 50:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                    message = "High swap usage"
            
            response_time = time.time() - start_time
            
            return HealthCheckResult(
                name="memory",
                status=status,
                message=message,
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Memory health check failed: {e}")
            return HealthCheckResult(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                message=f"Memory health check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def _check_disk_health(self) -> HealthCheckResult:
        """Check disk health"""
        start_time = time.time()
        
        try:
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            status = HealthStatus.HEALTHY
            message = "Disk is healthy"
            details = {
                "total_disk_gb": disk_usage.total / (1024**3),
                "used_disk_gb": disk_usage.used / (1024**3),
                "free_disk_gb": disk_usage.free / (1024**3),
                "disk_percent": (disk_usage.used / disk_usage.total) * 100,
                "read_bytes": disk_io.read_bytes if disk_io else 0,
                "write_bytes": disk_io.write_bytes if disk_io else 0
            }
            
            # Check disk space thresholds
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            if disk_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "Critical disk space"
            elif disk_percent > 85:
                status = HealthStatus.DEGRADED
                message = "Low disk space"
            
            response_time = time.time() - start_time
            
            return HealthCheckResult(
                name="disk",
                status=status,
                message=message,
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Disk health check failed: {e}")
            return HealthCheckResult(
                name="disk",
                status=HealthStatus.UNHEALTHY,
                message=f"Disk health check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def _check_cpu_health(self) -> HealthCheckResult:
        """Check CPU health"""
        start_time = time.time()
        
        try:
            # Get CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            
            status = HealthStatus.HEALTHY
            message = "CPU is healthy"
            details = {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "cpu_freq_mhz": cpu_freq.current if cpu_freq else None,
                "load_avg_1min": load_avg[0] if load_avg else None,
                "load_avg_5min": load_avg[1] if load_avg else None,
                "load_avg_15min": load_avg[2] if load_avg else None
            }
            
            # Check CPU thresholds
            if cpu_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "Critical CPU usage"
            elif cpu_percent > 80:
                status = HealthStatus.DEGRADED
                message = "High CPU usage"
            
            # Check load average if available
            if load_avg and load_avg[0] > cpu_count * 2:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                    message = "High system load"
            
            response_time = time.time() - start_time
            
            return HealthCheckResult(
                name="cpu",
                status=status,
                message=message,
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            logger.error(f"CPU health check failed: {e}")
            return HealthCheckResult(
                name="cpu",
                status=HealthStatus.UNHEALTHY,
                message=f"CPU health check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def _check_redis_health(self) -> HealthCheckResult:
        """Check Redis health"""
        start_time = time.time()
        
        try:
            import redis
            client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # Test basic operations
            client.ping()
            client.set('health_check', 'test', ex=10)
            value = client.get('health_check')
            client.delete('health_check')
            
            if value != 'test':
                raise Exception("Redis read/write test failed")
            
            # Get Redis info
            info = client.info()
            
            status = HealthStatus.HEALTHY
            message = "Redis is healthy"
            details = {
                "version": info.get('redis_version'),
                "uptime_seconds": info.get('uptime_in_seconds'),
                "connected_clients": info.get('connected_clients'),
                "used_memory_human": info.get('used_memory_human'),
                "keyspace_hits": info.get('keyspace_hits'),
                "keyspace_misses": info.get('keyspace_misses')
            }
            
            response_time = time.time() - start_time
            
            return HealthCheckResult(
                name="redis",
                status=status,
                message=message,
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis health check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def _check_database_health(self) -> HealthCheckResult:
        """Check database health"""
        start_time = time.time()
        
        try:
            # This is a placeholder - implement based on your database
            # For PostgreSQL example:
            conn = psycopg2.connect(
                host="localhost",
                database="your_db",
                user="your_user",
                password="your_password"
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result[0] != 1:
                raise Exception("Database query test failed")
            
            status = HealthStatus.HEALTHY
            message = "Database is healthy"
            details = {
                "connection_test": "passed",
                "query_test": "passed"
            }
            
            response_time = time.time() - start_time
            
            return HealthCheckResult(
                name="database",
                status=status,
                message=message,
                response_time=response_time,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return HealthCheckResult(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database health check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def check_dependency(self, name: str) -> HealthCheckResult:
        """Check a specific dependency"""
        if name not in self.dependencies:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Dependency '{name}' not found",
                response_time=0.0
            )
        
        dep = self.dependencies[name]
        start_time = time.time()
        
        try:
            # Run dependency check with timeout
            result = await asyncio.wait_for(
                dep["check_func"](),
                timeout=dep["timeout"]
            )
            
            dep["last_status"] = result.status
            dep["last_check"] = time.time()
            
            return result
            
        except asyncio.TimeoutError:
            dep["last_status"] = HealthStatus.UNHEALTHY
            dep["last_check"] = time.time()
            
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Dependency '{name}' check timed out",
                response_time=time.time() - start_time
            )
        except Exception as e:
            dep["last_status"] = HealthStatus.UNHEALTHY
            dep["last_check"] = time.time()
            
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Dependency '{name}' check failed: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status"""
        start_time = time.time()
        results = {}
        
        # Run basic health checks
        for name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[name] = {
                    "status": result.status.value,
                    "message": result.message,
                    "response_time": result.response_time,
                    "details": result.details,
                    "timestamp": result.timestamp
                }
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results[name] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": f"Health check failed: {str(e)}",
                    "response_time": 0.0,
                    "details": None,
                    "timestamp": time.time()
                }
        
        # Run dependency checks
        for name in self.dependencies:
            try:
                result = await self.check_dependency(name)
                results[f"dependency_{name}"] = {
                    "status": result.status.value,
                    "message": result.message,
                    "response_time": result.response_time,
                    "details": result.details,
                    "timestamp": result.timestamp,
                    "critical": self.dependencies[name]["critical"]
                }
            except Exception as e:
                logger.error(f"Dependency check '{name}' failed: {e}")
                results[f"dependency_{name}"] = {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": f"Dependency check failed: {str(e)}",
                    "response_time": 0.0,
                    "details": None,
                    "timestamp": time.time(),
                    "critical": self.dependencies[name]["critical"]
                }
        
        # Determine overall status
        overall_status = self._determine_overall_status(results)
        
        self.overall_status = overall_status
        self.last_check = time.time()
        
        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "response_time": time.time() - start_time,
            "checks": results,
            "summary": {
                "total_checks": len(results),
                "healthy": len([r for r in results.values() if r["status"] == HealthStatus.HEALTHY.value]),
                "degraded": len([r for r in results.values() if r["status"] == HealthStatus.DEGRADED.value]),
                "unhealthy": len([r for r in results.values() if r["status"] == HealthStatus.UNHEALTHY.value]),
                "unknown": len([r for r in results.values() if r["status"] == HealthStatus.UNKNOWN.value])
            }
        }
    
    def _determine_overall_status(self, results: Dict[str, Any]) -> HealthStatus:
        """Determine overall health status based on individual checks"""
        if not results:
            return HealthStatus.UNKNOWN
        
        # Check for any unhealthy critical dependencies
        for name, result in results.items():
            if name.startswith("dependency_") and result.get("critical", False):
                if result["status"] == HealthStatus.UNHEALTHY.value:
                    return HealthStatus.UNHEALTHY
        
        # Check for any unhealthy basic checks
        unhealthy_count = len([r for r in results.values() if r["status"] == HealthStatus.UNHEALTHY.value])
        degraded_count = len([r for r in results.values() if r["status"] == HealthStatus.DEGRADED.value])
        
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


# Global health checker instance
health_checker = HealthChecker()


def create_health_response(status_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized health response"""
    return {
        "success": True,
        "data": status_data,
        "error": None,
        "timestamp": time.time(),
        "message": f"Health status: {status_data['status']}"
    }





