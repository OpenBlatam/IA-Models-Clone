"""
Advanced Health Monitoring System - Service health checks and status monitoring
Production-ready health monitoring and alerting
"""

import asyncio
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
import psutil
import os

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Individual health check definition"""
    name: str
    check_func: Callable[[], Union[bool, Dict[str, Any]]]
    timeout: float = 5.0
    interval: float = 30.0
    critical: bool = False
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class HealthResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    response_time: float = 0.0
    error: Optional[str] = None

@dataclass
class ServiceHealth:
    """Overall health status of a service"""
    service_name: str
    overall_status: HealthStatus
    checks: List[HealthResult] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    uptime: float = 0.0
    version: str = "unknown"

class HealthMonitor:
    """Advanced health monitoring system"""
    
    def __init__(
        self,
        service_name: str = "content-redundancy-detector",
        version: str = "1.0.0",
        check_interval: float = 30.0,
        timeout: float = 10.0
    ):
        self.service_name = service_name
        self.version = version
        self.check_interval = check_interval
        self.timeout = timeout
        
        # Health checks
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_results: Dict[str, HealthResult] = {}
        
        # Monitoring
        self.start_time = time.time()
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Callbacks
        self.status_callbacks: List[Callable[[ServiceHealth], None]] = []
        self.alert_callbacks: List[Callable[[HealthResult], None]] = []
        
        # Register default health checks
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default system health checks"""
        # Memory check
        self.add_health_check(
            "memory_usage",
            self._check_memory_usage,
            interval=60.0,
            critical=True
        )
        
        # CPU check
        self.add_health_check(
            "cpu_usage",
            self._check_cpu_usage,
            interval=60.0,
            critical=True
        )
        
        # Disk space check
        self.add_health_check(
            "disk_space",
            self._check_disk_space,
            interval=300.0,
            critical=True
        )
        
        # Process check
        self.add_health_check(
            "process_status",
            self._check_process_status,
            interval=30.0,
            critical=True
        )

    def add_health_check(
        self,
        name: str,
        check_func: Callable[[], Union[bool, Dict[str, Any]]],
        timeout: float = None,
        interval: float = None,
        critical: bool = False,
        tags: Dict[str, str] = None
    ):
        """Add a health check"""
        health_check = HealthCheck(
            name=name,
            check_func=check_func,
            timeout=timeout or self.timeout,
            interval=interval or self.check_interval,
            critical=critical,
            tags=tags or {}
        )
        
        with self.lock:
            self.health_checks[name] = health_check

    def remove_health_check(self, name: str):
        """Remove a health check"""
        with self.lock:
            if name in self.health_checks:
                del self.health_checks[name]
            if name in self.health_results:
                del self.health_results[name]

    def add_status_callback(self, callback: Callable[[ServiceHealth], None]):
        """Add callback for status changes"""
        self.status_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable[[HealthResult], None]):
        """Add callback for health alerts"""
        self.alert_callbacks.append(callback)

    async def start(self):
        """Start health monitoring"""
        if self.running:
            return
        
        self.running = True
        self.monitor_task = asyncio.create_task(self._monitor_worker())

    async def stop(self):
        """Stop health monitoring"""
        self.running = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_worker(self):
        """Background worker for health monitoring"""
        while self.running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Health monitor error: {e}")

    async def _run_health_checks(self):
        """Run all health checks"""
        with self.lock:
            current_time = time.time()
            checks_to_run = []
            
            # Determine which checks to run based on interval
            for name, check in self.health_checks.items():
                last_result = self.health_results.get(name)
                if (not last_result or 
                    current_time - last_result.timestamp >= check.interval):
                    checks_to_run.append(name)
            
            # Run checks concurrently
            if checks_to_run:
                tasks = [self._run_single_check(name) for name in checks_to_run]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update overall status
                await self._update_overall_status()

    async def _run_single_check(self, name: str):
        """Run a single health check"""
        check = self.health_checks[name]
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, check.check_func
                ),
                timeout=check.timeout
            )
            
            response_time = time.time() - start_time
            
            # Process result
            if isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = "Check passed" if result else "Check failed"
                details = {}
            else:
                # Result is a dictionary
                status = HealthStatus(result.get("status", "unknown"))
                message = result.get("message", "Check completed")
                details = result.get("details", {})
            
            # Create health result
            health_result = HealthResult(
                name=name,
                status=status,
                message=message,
                details=details,
                timestamp=time.time(),
                response_time=response_time
            )
            
            # Store result
            with self.lock:
                self.health_results[name] = health_result
            
            # Check for alerts
            if status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                await self._trigger_alert(health_result)
            
        except asyncio.TimeoutError:
            health_result = HealthResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message="Health check timed out",
                timestamp=time.time(),
                response_time=time.time() - start_time,
                error="Timeout"
            )
            
            with self.lock:
                self.health_results[name] = health_result
            
            await self._trigger_alert(health_result)
            
        except Exception as e:
            health_result = HealthResult(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check error: {str(e)}",
                timestamp=time.time(),
                response_time=time.time() - start_time,
                error=str(e)
            )
            
            with self.lock:
                self.health_results[name] = health_result
            
            await self._trigger_alert(health_result)

    async def _update_overall_status(self):
        """Update overall service health status"""
        with self.lock:
            if not self.health_results:
                overall_status = HealthStatus.UNKNOWN
            else:
                # Determine overall status based on individual checks
                statuses = [result.status for result in self.health_results.values()]
                
                if HealthStatus.CRITICAL in statuses:
                    overall_status = HealthStatus.CRITICAL
                elif HealthStatus.UNHEALTHY in statuses:
                    overall_status = HealthStatus.UNHEALTHY
                elif HealthStatus.DEGRADED in statuses:
                    overall_status = HealthStatus.DEGRADED
                elif all(status == HealthStatus.HEALTHY for status in statuses):
                    overall_status = HealthStatus.HEALTHY
                else:
                    overall_status = HealthStatus.UNKNOWN
            
            # Create service health
            service_health = ServiceHealth(
                service_name=self.service_name,
                overall_status=overall_status,
                checks=list(self.health_results.values()),
                last_updated=time.time(),
                uptime=time.time() - self.start_time,
                version=self.version
            )
            
            # Notify callbacks
            for callback in self.status_callbacks:
                try:
                    callback(service_health)
                except Exception as e:
                    print(f"Status callback error: {e}")

    async def _trigger_alert(self, health_result: HealthResult):
        """Trigger alert for health check failure"""
        for callback in self.alert_callbacks:
            try:
                callback(health_result)
            except Exception as e:
                print(f"Alert callback error: {e}")

    def get_health_status(self) -> ServiceHealth:
        """Get current health status"""
        with self.lock:
            if not self.health_results:
                overall_status = HealthStatus.UNKNOWN
            else:
                statuses = [result.status for result in self.health_results.values()]
                
                if HealthStatus.CRITICAL in statuses:
                    overall_status = HealthStatus.CRITICAL
                elif HealthStatus.UNHEALTHY in statuses:
                    overall_status = HealthStatus.UNHEALTHY
                elif HealthStatus.DEGRADED in statuses:
                    overall_status = HealthStatus.DEGRADED
                elif all(status == HealthStatus.HEALTHY for status in statuses):
                    overall_status = HealthStatus.HEALTHY
                else:
                    overall_status = HealthStatus.UNKNOWN
            
            return ServiceHealth(
                service_name=self.service_name,
                overall_status=overall_status,
                checks=list(self.health_results.values()),
                last_updated=time.time(),
                uptime=time.time() - self.start_time,
                version=self.version
            )

    def get_check_status(self, name: str) -> Optional[HealthResult]:
        """Get status of a specific health check"""
        with self.lock:
            return self.health_results.get(name)

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for monitoring"""
        health_status = self.get_health_status()
        
        # Count checks by status
        status_counts = defaultdict(int)
        for check in health_status.checks:
            status_counts[check.status.value] += 1
        
        # Calculate average response time
        avg_response_time = 0.0
        if health_status.checks:
            total_time = sum(check.response_time for check in health_status.checks)
            avg_response_time = total_time / len(health_status.checks)
        
        return {
            "service": {
                "name": health_status.service_name,
                "version": health_status.version,
                "status": health_status.overall_status.value,
                "uptime": health_status.uptime,
                "last_updated": health_status.last_updated
            },
            "checks": {
                "total": len(health_status.checks),
                "by_status": dict(status_counts),
                "avg_response_time": avg_response_time
            },
            "monitoring": {
                "running": self.running,
                "check_interval": self.check_interval,
                "timeout": self.timeout
            }
        }

    # Default health check implementations

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check system memory usage"""
        memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        if memory.percent > 90:
            status = "critical"
            message = f"Memory usage critical: {memory.percent:.1f}%"
        elif memory.percent > 80:
            status = "unhealthy"
            message = f"Memory usage high: {memory.percent:.1f}%"
        elif memory.percent > 70:
            status = "degraded"
            message = f"Memory usage elevated: {memory.percent:.1f}%"
        else:
            status = "healthy"
            message = f"Memory usage normal: {memory.percent:.1f}%"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "system_memory_percent": memory.percent,
                "system_memory_available": memory.available / (1024 * 1024 * 1024),  # GB
                "process_memory_mb": process_memory
            }
        }

    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 90:
            status = "critical"
            message = f"CPU usage critical: {cpu_percent:.1f}%"
        elif cpu_percent > 80:
            status = "unhealthy"
            message = f"CPU usage high: {cpu_percent:.1f}%"
        elif cpu_percent > 70:
            status = "degraded"
            message = f"CPU usage elevated: {cpu_percent:.1f}%"
        else:
            status = "healthy"
            message = f"CPU usage normal: {cpu_percent:.1f}%"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count()
            }
        }

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space"""
        disk = psutil.disk_usage('/')
        free_percent = (disk.free / disk.total) * 100
        
        if free_percent < 5:
            status = "critical"
            message = f"Disk space critical: {free_percent:.1f}% free"
        elif free_percent < 10:
            status = "unhealthy"
            message = f"Disk space low: {free_percent:.1f}% free"
        elif free_percent < 20:
            status = "degraded"
            message = f"Disk space warning: {free_percent:.1f}% free"
        else:
            status = "healthy"
            message = f"Disk space normal: {free_percent:.1f}% free"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "free_percent": free_percent,
                "free_gb": disk.free / (1024 * 1024 * 1024),
                "total_gb": disk.total / (1024 * 1024 * 1024)
            }
        }

    def _check_process_status(self) -> Dict[str, Any]:
        """Check process status"""
        try:
            process = psutil.Process(os.getpid())
            
            if not process.is_running():
                status = "critical"
                message = "Process not running"
            else:
                status = "healthy"
                message = "Process running normally"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "pid": process.pid,
                    "is_running": process.is_running(),
                    "create_time": process.create_time(),
                    "num_threads": process.num_threads()
                }
            }
        except Exception as e:
            return {
                "status": "critical",
                "message": f"Process check failed: {str(e)}",
                "details": {"error": str(e)}
            }





