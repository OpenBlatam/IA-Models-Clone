"""
Monitoring Middleware
=====================

Advanced monitoring middleware with metrics, health checks, and performance tracking.
"""

from __future__ import annotations
import asyncio
import logging
import time
import psutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ...shared.config import get_settings


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Metric data"""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str = ""


@dataclass
class HealthCheck:
    """Health check data"""
    name: str
    status: str  # healthy, unhealthy, degraded
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: float = 0.0


class MonitoringMiddleware:
    """
    Advanced monitoring middleware
    
    Provides comprehensive monitoring with metrics collection,
    health checks, and performance tracking.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._metrics: Dict[str, List[Metric]] = {}
        self._health_checks: Dict[str, HealthCheck] = {}
        self._statistics = {
            "requests_total": 0,
            "requests_duration_seconds": 0.0,
            "requests_by_status": {},
            "requests_by_method": {},
            "requests_by_path": {},
            "active_connections": 0,
            "memory_usage_bytes": 0,
            "cpu_usage_percent": 0.0,
            "disk_usage_percent": 0.0
        }
        self._start_time = datetime.utcnow()
        self._initialize_health_checks()
        self._start_background_tasks()
    
    def _initialize_health_checks(self):
        """Initialize health checks"""
        # Database health check
        self._health_checks["database"] = HealthCheck(
            name="database",
            status="healthy",
            message="Database connection is healthy"
        )
        
        # Redis health check
        self._health_checks["redis"] = HealthCheck(
            name="redis",
            status="healthy",
            message="Redis connection is healthy"
        )
        
        # AI service health check
        self._health_checks["ai_service"] = HealthCheck(
            name="ai_service",
            status="healthy",
            message="AI service is healthy"
        )
        
        # External services health check
        self._health_checks["external_services"] = HealthCheck(
            name="external_services",
            status="healthy",
            message="External services are healthy"
        )
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        if self.settings.monitoring_enabled:
            asyncio.create_task(self._collect_system_metrics())
            asyncio.create_task(self._run_health_checks())
    
    async def process_request(self, request: Request) -> Request:
        """Process incoming request"""
        try:
            # Record request start
            request.state.monitoring_start_time = time.time()
            request.state.monitoring_request_id = f"req_{int(time.time() * 1000)}"
            
            # Increment request counter
            self._increment_metric("http_requests_total", {
                "method": request.method,
                "path": request.url.path
            })
            
            return request
            
        except Exception as e:
            logger.error(f"Failed to process request for monitoring: {e}")
            return request
    
    async def process_response(
        self,
        request: Request,
        response: Response
    ) -> Response:
        """Process outgoing response"""
        try:
            # Calculate response time
            start_time = getattr(request.state, "monitoring_start_time", time.time())
            response_time = time.time() - start_time
            
            # Record metrics
            self._record_metric("http_request_duration_seconds", response_time, {
                "method": request.method,
                "path": request.url.path,
                "status_code": str(response.status_code)
            })
            
            self._increment_metric("http_requests_total", {
                "method": request.method,
                "path": request.url.path,
                "status_code": str(response.status_code)
            })
            
            # Update statistics
            self._update_statistics(request, response, response_time)
            
            # Add monitoring headers
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            response.headers["X-Request-ID"] = getattr(request.state, "monitoring_request_id", "unknown")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process response for monitoring: {e}")
            return response
    
    def _increment_metric(self, name: str, labels: Dict[str, str] = None):
        """Increment counter metric"""
        if name not in self._metrics:
            self._metrics[name] = []
        
        # Find existing metric with same labels
        for metric in self._metrics[name]:
            if metric.labels == (labels or {}):
                metric.value += 1
                metric.timestamp = datetime.utcnow()
                return
        
        # Create new metric
        self._metrics[name].append(Metric(
            name=name,
            type=MetricType.COUNTER,
            value=1,
            labels=labels or {}
        ))
    
    def _record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record gauge metric"""
        if name not in self._metrics:
            self._metrics[name] = []
        
        self._metrics[name].append(Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            labels=labels or {}
        ))
    
    def _update_statistics(self, request: Request, response: Response, response_time: float):
        """Update monitoring statistics"""
        self._statistics["requests_total"] += 1
        self._statistics["requests_duration_seconds"] += response_time
        
        # Update by status code
        status_code = response.status_code
        if status_code not in self._statistics["requests_by_status"]:
            self._statistics["requests_by_status"][status_code] = 0
        self._statistics["requests_by_status"][status_code] += 1
        
        # Update by method
        method = request.method
        if method not in self._statistics["requests_by_method"]:
            self._statistics["requests_by_method"][method] = 0
        self._statistics["requests_by_method"][method] += 1
        
        # Update by path
        path = request.url.path
        if path not in self._statistics["requests_by_path"]:
            self._statistics["requests_by_path"][path] = 0
        self._statistics["requests_by_path"][path] += 1
    
    async def _collect_system_metrics(self):
        """Collect system metrics in background"""
        while True:
            try:
                # Memory usage
                memory = psutil.virtual_memory()
                self._statistics["memory_usage_bytes"] = memory.used
                self._record_metric("system_memory_usage_bytes", memory.used)
                self._record_metric("system_memory_usage_percent", memory.percent)
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self._statistics["cpu_usage_percent"] = cpu_percent
                self._record_metric("system_cpu_usage_percent", cpu_percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self._statistics["disk_usage_percent"] = (disk.used / disk.total) * 100
                self._record_metric("system_disk_usage_percent", self._statistics["disk_usage_percent"])
                self._record_metric("system_disk_usage_bytes", disk.used)
                
                # Network connections
                connections = len(psutil.net_connections())
                self._statistics["active_connections"] = connections
                self._record_metric("system_network_connections", connections)
                
                # Process metrics
                process = psutil.Process()
                self._record_metric("process_memory_usage_bytes", process.memory_info().rss)
                self._record_metric("process_cpu_usage_percent", process.cpu_percent())
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Failed to collect system metrics: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _run_health_checks(self):
        """Run health checks in background"""
        while True:
            try:
                # Check database
                await self._check_database_health()
                
                # Check Redis
                await self._check_redis_health()
                
                # Check AI service
                await self._check_ai_service_health()
                
                # Check external services
                await self._check_external_services_health()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Failed to run health checks: {e}")
                await asyncio.sleep(120)  # Wait longer on error
    
    async def _check_database_health(self):
        """Check database health"""
        try:
            start_time = time.time()
            
            # In a real implementation, you would check database connection
            # For now, simulate a healthy database
            response_time = (time.time() - start_time) * 1000
            
            self._health_checks["database"] = HealthCheck(
                name="database",
                status="healthy",
                message="Database connection is healthy",
                response_time_ms=response_time,
                details={"connection_pool_size": 10, "active_connections": 5}
            )
            
        except Exception as e:
            self._health_checks["database"] = HealthCheck(
                name="database",
                status="unhealthy",
                message=f"Database health check failed: {e}",
                details={"error": str(e)}
            )
    
    async def _check_redis_health(self):
        """Check Redis health"""
        try:
            start_time = time.time()
            
            # In a real implementation, you would check Redis connection
            # For now, simulate a healthy Redis
            response_time = (time.time() - start_time) * 1000
            
            self._health_checks["redis"] = HealthCheck(
                name="redis",
                status="healthy",
                message="Redis connection is healthy",
                response_time_ms=response_time,
                details={"memory_usage": "50MB", "connected_clients": 3}
            )
            
        except Exception as e:
            self._health_checks["redis"] = HealthCheck(
                name="redis",
                status="unhealthy",
                message=f"Redis health check failed: {e}",
                details={"error": str(e)}
            )
    
    async def _check_ai_service_health(self):
        """Check AI service health"""
        try:
            start_time = time.time()
            
            # In a real implementation, you would check AI service
            # For now, simulate a healthy AI service
            response_time = (time.time() - start_time) * 1000
            
            self._health_checks["ai_service"] = HealthCheck(
                name="ai_service",
                status="healthy",
                message="AI service is healthy",
                response_time_ms=response_time,
                details={"provider": "openai", "model": "gpt-4", "rate_limit": "60/min"}
            )
            
        except Exception as e:
            self._health_checks["ai_service"] = HealthCheck(
                name="ai_service",
                status="unhealthy",
                message=f"AI service health check failed: {e}",
                details={"error": str(e)}
            )
    
    async def _check_external_services_health(self):
        """Check external services health"""
        try:
            start_time = time.time()
            
            # In a real implementation, you would check external services
            # For now, simulate healthy external services
            response_time = (time.time() - start_time) * 1000
            
            self._health_checks["external_services"] = HealthCheck(
                name="external_services",
                status="healthy",
                message="External services are healthy",
                response_time_ms=response_time,
                details={"services": ["notification", "analytics", "audit"]}
            )
            
        except Exception as e:
            self._health_checks["external_services"] = HealthCheck(
                name="external_services",
                status="unhealthy",
                message=f"External services health check failed: {e}",
                details={"error": str(e)}
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        return {
            "metrics": {
                name: [
                    {
                        "name": metric.name,
                        "type": metric.type.value,
                        "value": metric.value,
                        "labels": metric.labels,
                        "timestamp": metric.timestamp.isoformat()
                    }
                    for metric in metrics
                ]
                for name, metrics in self._metrics.items()
            },
            "statistics": self._statistics
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        overall_status = "healthy"
        unhealthy_checks = []
        
        for check in self._health_checks.values():
            if check.status == "unhealthy":
                overall_status = "unhealthy"
                unhealthy_checks.append(check.name)
            elif check.status == "degraded" and overall_status == "healthy":
                overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            "checks": {
                name: {
                    "status": check.status,
                    "message": check.message,
                    "last_check": check.last_check.isoformat(),
                    "response_time_ms": check.response_time_ms,
                    "details": check.details
                }
                for name, check in self._health_checks.items()
            },
            "unhealthy_checks": unhealthy_checks
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_requests = self._statistics["requests_total"]
        avg_response_time = (
            self._statistics["requests_duration_seconds"] / total_requests
            if total_requests > 0 else 0
        )
        
        return {
            "total_requests": total_requests,
            "average_response_time_seconds": round(avg_response_time, 3),
            "requests_per_second": round(total_requests / (datetime.utcnow() - self._start_time).total_seconds(), 2),
            "requests_by_status": self._statistics["requests_by_status"],
            "requests_by_method": self._statistics["requests_by_method"],
            "top_paths": dict(sorted(
                self._statistics["requests_by_path"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            "system_metrics": {
                "memory_usage_bytes": self._statistics["memory_usage_bytes"],
                "cpu_usage_percent": self._statistics["cpu_usage_percent"],
                "disk_usage_percent": self._statistics["disk_usage_percent"],
                "active_connections": self._statistics["active_connections"]
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring middleware statistics"""
        return {
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            "metrics_count": sum(len(metrics) for metrics in self._metrics.values()),
            "health_checks_count": len(self._health_checks),
            "statistics": self._statistics,
            "config": {
                "monitoring_enabled": self.settings.monitoring_enabled,
                "metrics_enabled": self.settings.metrics_enabled,
                "health_check_enabled": self.settings.health_check_enabled
            }
        }


# Global monitoring middleware instance
_monitoring_middleware: Optional[MonitoringMiddleware] = None


def get_monitoring_middleware() -> MonitoringMiddleware:
    """Get global monitoring middleware instance"""
    global _monitoring_middleware
    if _monitoring_middleware is None:
        _monitoring_middleware = MonitoringMiddleware()
    return _monitoring_middleware


# FastAPI dependency
async def get_monitoring_middleware_dependency() -> MonitoringMiddleware:
    """FastAPI dependency for monitoring middleware"""
    return get_monitoring_middleware()




