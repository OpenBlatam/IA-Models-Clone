from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
BUFFER_SIZE = 1024

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import traceback
    import httpx
    import psutil
    import structlog
                    import redis
                    import asyncpg
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Production Health Check Script for Next-Level HeyGen AI FastAPI
Comprehensive health monitoring for Docker containers and Kubernetes deployments.
"""


try:
except ImportError as e:
    print(f"Missing required dependency: {e}")
    sys.exit(2)

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Health check configuration
HEALTH_CHECK_CONFIG = {
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": int(os.getenv("PORT", "8000")),
    "timeout": int(os.getenv("HEALTH_CHECK_TIMEOUT", "10")),
    "retry_attempts": int(os.getenv("HEALTH_CHECK_RETRIES", "3")),
    "retry_delay": float(os.getenv("HEALTH_CHECK_RETRY_DELAY", "1.0")),
    "critical_memory_threshold": float(os.getenv("CRITICAL_MEMORY_THRESHOLD", "90.0")),
    "critical_cpu_threshold": float(os.getenv("CRITICAL_CPU_THRESHOLD", "95.0")),
    "critical_disk_threshold": float(os.getenv("CRITICAL_DISK_THRESHOLD", "95.0")),
}

class HealthCheckResult:
    """Health check result container."""
    
    def __init__(self, name: str, status: str, message: str, details: Optional[Dict[str, Any]] = None):
        
    """__init__ function."""
self.name = name
        self.status = status  # "healthy", "unhealthy", "warning"
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp
        }

class HealthChecker:
    """Comprehensive health checker for HeyGen AI FastAPI application."""
    
    def __init__(self) -> Any:
        self.config = HEALTH_CHECK_CONFIG
        self.results: List[HealthCheckResult] = []
        self.overall_status = "healthy"
        
    async def check_application_endpoint(self) -> HealthCheckResult:
        """Check if the main application endpoint is responsive."""
        url = f"http://{self.config['host']}:{self.config['port']}/health"
        
        for attempt in range(self.config["retry_attempts"]):
            try:
                async with httpx.AsyncClient(timeout=self.config["timeout"]) as client:
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        return HealthCheckResult(
                            "application_endpoint",
                            "healthy",
                            "Application endpoint is responsive",
                            {"status_code": response.status_code, "response_data": data}
                        )
                    else:
                        return HealthCheckResult(
                            "application_endpoint",
                            "unhealthy",
                            f"Application endpoint returned status {response.status_code}",
                            {"status_code": response.status_code}
                        )
                        
            except httpx.ConnectError:
                if attempt == self.config["retry_attempts"] - 1:
                    return HealthCheckResult(
                        "application_endpoint",
                        "unhealthy",
                        "Cannot connect to application endpoint",
                        {"url": url, "attempts": attempt + 1}
                    )
                await asyncio.sleep(self.config["retry_delay"])
                
            except httpx.TimeoutException:
                if attempt == self.config["retry_attempts"] - 1:
                    return HealthCheckResult(
                        "application_endpoint",
                        "unhealthy",
                        f"Application endpoint timeout after {self.config['timeout']}s",
                        {"url": url, "timeout": self.config["timeout"]}
                    )
                await asyncio.sleep(self.config["retry_delay"])
                
            except Exception as e:
                if attempt == self.config["retry_attempts"] - 1:
                    return HealthCheckResult(
                        "application_endpoint",
                        "unhealthy",
                        f"Application endpoint check failed: {str(e)}",
                        {"error": str(e), "url": url}
                    )
                await asyncio.sleep(self.config["retry_delay"])
        
        return HealthCheckResult(
            "application_endpoint",
            "unhealthy",
            "Application endpoint check failed after all retries"
        )
    
    async def check_metrics_endpoint(self) -> HealthCheckResult:
        """Check if the metrics endpoint is responsive."""
        url = f"http://{self.config['host']}:{self.config['port']}/metrics/performance"
        
        try:
            async with httpx.AsyncClient(timeout=self.config["timeout"]) as client:
                response = await client.get(url)
                
                if response.status_code == 200:
                    return HealthCheckResult(
                        "metrics_endpoint",
                        "healthy",
                        "Metrics endpoint is responsive",
                        {"status_code": response.status_code}
                    )
                else:
                    return HealthCheckResult(
                        "metrics_endpoint",
                        "warning",
                        f"Metrics endpoint returned status {response.status_code}",
                        {"status_code": response.status_code}
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                "metrics_endpoint",
                "warning",
                f"Metrics endpoint check failed: {str(e)}",
                {"error": str(e)}
            )
    
    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource utilization."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            # Load average (Unix only)
            load_avg = None
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
            
            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk_percent,
                "disk_free_gb": disk.free / (1024**3),
                "load_average": load_avg
            }
            
            # Determine status based on thresholds
            status = "healthy"
            messages = []
            
            if cpu_percent > self.config["critical_cpu_threshold"]:
                status = "unhealthy"
                messages.append(f"Critical CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 80:
                status = "warning" if status == "healthy" else status
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > self.config["critical_memory_threshold"]:
                status = "unhealthy"
                messages.append(f"Critical memory usage: {memory_percent:.1f}%")
            elif memory_percent > 80:
                status = "warning" if status == "healthy" else status
                messages.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > self.config["critical_disk_threshold"]:
                status = "unhealthy"
                messages.append(f"Critical disk usage: {disk_percent:.1f}%")
            elif disk_percent > 85:
                status = "warning" if status == "healthy" else status
                messages.append(f"High disk usage: {disk_percent:.1f}%")
            
            message = "; ".join(messages) if messages else "System resources are within normal limits"
            
            return HealthCheckResult(
                "system_resources",
                status,
                message,
                details
            )
            
        except Exception as e:
            return HealthCheckResult(
                "system_resources",
                "unhealthy",
                f"System resource check failed: {str(e)}",
                {"error": str(e)}
            )
    
    def check_process_health(self) -> HealthCheckResult:
        """Check the health of the application process."""
        try:
            current_process = psutil.Process()
            
            # Process information
            process_info = {
                "pid": current_process.pid,
                "ppid": current_process.ppid(),
                "name": current_process.name(),
                "status": current_process.status(),
                "create_time": current_process.create_time(),
                "cpu_percent": current_process.cpu_percent(),
                "memory_info": current_process.memory_info()._asdict(),
                "num_threads": current_process.num_threads(),
                "num_fds": getattr(current_process, 'num_fds', lambda: 0)(),  # Unix only
            }
            
            # Check for zombie or defunct processes
            if process_info["status"] in ["zombie", "defunct"]:
                return HealthCheckResult(
                    "process_health",
                    "unhealthy",
                    f"Process is in {process_info['status']} state",
                    process_info
                )
            
            # Check memory usage (process-specific)
            memory_mb = process_info["memory_info"]["rss"] / (1024 * 1024)
            if memory_mb > 8192:  # 8GB limit
                return HealthCheckResult(
                    "process_health",
                    "warning",
                    f"High process memory usage: {memory_mb:.1f}MB",
                    process_info
                )
            
            return HealthCheckResult(
                "process_health",
                "healthy",
                "Process is running normally",
                process_info
            )
            
        except Exception as e:
            return HealthCheckResult(
                "process_health",
                "unhealthy",
                f"Process health check failed: {str(e)}",
                {"error": str(e)}
            )
    
    def check_file_system(self) -> HealthCheckResult:
        """Check file system health and permissions."""
        try:
            checks = []
            
            # Check required directories
            required_dirs = ["/app/logs", "/app/outputs", "/app/cache", "/app/temp"]
            for dir_path in required_dirs:
                if os.path.exists(dir_path):
                    if os.access(dir_path, os.W_OK):
                        checks.append(f"{dir_path}: ✓ writable")
                    else:
                        checks.append(f"{dir_path}: ✗ not writable")
                else:
                    checks.append(f"{dir_path}: ✗ missing")
            
            # Check disk space for critical paths
            disk_checks = []
            for path in ["/app", "/tmp"]:
                if os.path.exists(path):
                    disk_usage = psutil.disk_usage(path)
                    free_gb = disk_usage.free / (1024**3)
                    disk_checks.append(f"{path}: {free_gb:.1f}GB free")
            
            details = {
                "directory_checks": checks,
                "disk_space_checks": disk_checks
            }
            
            # Determine status
            if any("✗" in check for check in checks):
                return HealthCheckResult(
                    "file_system",
                    "unhealthy",
                    "File system issues detected",
                    details
                )
            else:
                return HealthCheckResult(
                    "file_system",
                    "healthy",
                    "File system is healthy",
                    details
                )
                
        except Exception as e:
            return HealthCheckResult(
                "file_system",
                "unhealthy",
                f"File system check failed: {str(e)}",
                {"error": str(e)}
            )
    
    async def check_external_dependencies(self) -> HealthCheckResult:
        """Check external service dependencies."""
        try:
            dependency_checks = []
            
            # Check Redis if enabled
            if os.getenv("ENABLE_REDIS", "true").lower() == "true":
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                try:
                    r = redis.from_url(redis_url, socket_timeout=5, socket_connect_timeout=5)
                    r.ping()
                    dependency_checks.append("Redis: ✓ connected")
                except Exception as e:
                    dependency_checks.append(f"Redis: ✗ failed - {str(e)}")
            
            # Check database if configured
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                try:
                    conn = await asyncpg.connect(database_url, timeout=5)
                    await conn.close()
                    dependency_checks.append("Database: ✓ connected")
                except Exception as e:
                    dependency_checks.append(f"Database: ✗ failed - {str(e)}")
            
            details = {"dependency_checks": dependency_checks}
            
            # Determine status
            if any("✗" in check for check in dependency_checks):
                return HealthCheckResult(
                    "external_dependencies",
                    "warning",
                    "Some external dependencies are unavailable",
                    details
                )
            else:
                return HealthCheckResult(
                    "external_dependencies",
                    "healthy",
                    "All external dependencies are available",
                    details
                )
                
        except Exception as e:
            return HealthCheckResult(
                "external_dependencies",
                "warning",
                f"External dependency check failed: {str(e)}",
                {"error": str(e)}
            )
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive results."""
        start_time = time.time()
        
        logger.info("Starting comprehensive health check")
        
        # Run all health checks
        checks = [
            self.check_application_endpoint(),
            self.check_metrics_endpoint(),
            self.check_external_dependencies(),
        ]
        
        # Run async checks
        async_results = await asyncio.gather(*checks, return_exceptions=True)
        
        # Run sync checks
        sync_results = [
            self.check_system_resources(),
            self.check_process_health(),
            self.check_file_system(),
        ]
        
        # Combine results
        all_results = []
        for result in async_results + sync_results:
            if isinstance(result, Exception):
                all_results.append(HealthCheckResult(
                    "unknown_check",
                    "unhealthy",
                    f"Health check exception: {str(result)}",
                    {"exception": str(result), "traceback": traceback.format_exc()}
                ))
            else:
                all_results.append(result)
        
        # Determine overall status
        overall_status = "healthy"
        unhealthy_count = 0
        warning_count = 0
        
        for result in all_results:
            if result.status == "unhealthy":
                unhealthy_count += 1
                overall_status = "unhealthy"
            elif result.status == "warning":
                warning_count += 1
                if overall_status == "healthy":
                    overall_status = "warning"
        
        execution_time = time.time() - start_time
        
        summary = {
            "overall_status": overall_status,
            "total_checks": len(all_results),
            "healthy_checks": len([r for r in all_results if r.status == "healthy"]),
            "warning_checks": warning_count,
            "unhealthy_checks": unhealthy_count,
            "execution_time_seconds": round(execution_time, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": [result.to_dict() for result in all_results]
        }
        
        logger.info(
            "Health check completed",
            extra={
                "overall_status": overall_status,
                "execution_time": execution_time,
                "total_checks": len(all_results),
                "unhealthy_checks": unhealthy_count,
                "warning_checks": warning_count
            }
        )
        
        return summary

async def main():
    """Main health check execution."""
    health_checker = HealthChecker()
    
    try:
        results = await health_checker.run_all_checks()
        
        # Output results as JSON for container orchestration
        print(json.dumps(results, indent=2))
        
        # Determine exit code based on overall status
        if results["overall_status"] == "healthy":
            sys.exit(0)  # Healthy
        elif results["overall_status"] == "warning":
            sys.exit(0)  # Warning but still operational
        else:
            sys.exit(1)  # Unhealthy
            
    except Exception as e:
        error_result = {
            "overall_status": "unhealthy",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        print(json.dumps(error_result, indent=2))
        logger.error("Health check failed with exception", exc_info=True)
        sys.exit(2)  # Health check error

match __name__:
    case "__main__":
    asyncio.run(main()) 