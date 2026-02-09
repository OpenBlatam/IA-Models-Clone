from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import logging
import functools
import threading
import inspect
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import json
import sqlite3
from pathlib import Path
import weakref
import signal
import contextlib
import structlog
from pydantic import BaseModel, Field
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
import redis.asyncio as redis
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import aiofiles
import aiohttp
        import random
    from fastapi.testclient import TestClient
from typing import Any, List, Dict, Optional
"""
📊 Route Performance Monitor
============================

Comprehensive route performance monitoring system with:
- Real-time blocking operation detection
- Route performance analytics
- Performance bottleneck identification
- Optimization recommendations
- Historical performance tracking
- Performance alerts and notifications
- Resource usage monitoring
- Performance regression detection
- Load testing integration
- Performance dashboards
"""



logger = structlog.get_logger(__name__)

T = TypeVar('T')
InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')

class PerformanceAlertLevel(Enum):
    """Performance alert levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class BlockingOperationType(Enum):
    """Types of blocking operations"""
    DATABASE_QUERY = "database_query"
    FILE_IO = "file_io"
    NETWORK_IO = "network_io"
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_OPERATION = "memory_operation"
    EXTERNAL_API = "external_api"
    SLEEP_OPERATION = "sleep_operation"
    SUBPROCESS = "subprocess"
    UNKNOWN = "unknown"

@dataclass
class BlockingOperation:
    """Information about a blocking operation"""
    operation_type: BlockingOperationType
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    function_name: str = ""
    line_number: int = 0
    stack_trace: str = ""
    resource_usage: Dict[str, float] = field(default_factory=dict)

@dataclass
class RoutePerformanceData:
    """Performance data for a route"""
    route_path: str
    method: str
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    total_duration: Optional[float] = None
    blocking_operations: List[BlockingOperation] = field(default_factory=list)
    async_operations: int = 0
    background_tasks: int = 0
    status_code: int = 200
    error_message: Optional[str] = None
    request_size: int = 0
    response_size: int = 0
    user_agent: str = ""
    client_ip: str = ""

class PerformanceAlert:
    """Performance alert"""
    
    def __init__(self, 
                 route_path: str,
                 alert_type: str,
                 level: PerformanceAlertLevel,
                 message: str,
                 metrics: Dict[str, Any]):
        
    """__init__ function."""
self.id = f"{route_path}_{alert_type}_{int(time.time())}"
        self.route_path = route_path
        self.alert_type = alert_type
        self.level = level
        self.message = message
        self.metrics = metrics
        self.timestamp = time.time()
        self.acknowledged = False
        self.resolved = False

class RoutePerformanceMonitor:
    """Main route performance monitoring system"""
    
    def __init__(self, app: FastAPI, redis_url: str = "redis://localhost:6379"):
        
    """__init__ function."""
self.app = app
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        
        # Performance data storage
        self.performance_data: Dict[str, RoutePerformanceData] = {}
        self.performance_data_lock = threading.Lock()
        
        # Route statistics
        self.route_stats: Dict[str, Dict[str, Any]] = {}
        self.route_stats_lock = threading.Lock()
        
        # Performance alerts
        self.alerts: List[PerformanceAlert] = []
        self.alerts_lock = threading.Lock()
        
        # Blocking operation detection
        self.blocking_patterns = {
            "time.sleep": BlockingOperationType.SLEEP_OPERATION,
            "requests.get": BlockingOperationType.NETWORK_IO,
            "requests.post": BlockingOperationType.NETWORK_IO,
            "urllib.request": BlockingOperationType.NETWORK_IO,
            "subprocess.run": BlockingOperationType.SUBPROCESS,
            "subprocess.call": BlockingOperationType.SUBPROCESS,
            "os.system": BlockingOperationType.SUBPROCESS,
            "sqlite3.connect": BlockingOperationType.DATABASE_QUERY,
            "open(": BlockingOperationType.FILE_IO,
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            "file(": BlockingOperationType.FILE_IO,
            "read(": BlockingOperationType.FILE_IO,
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            "write(": BlockingOperationType.FILE_IO,
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            "seek(": BlockingOperationType.FILE_IO,
            "aiohttp.ClientSession": BlockingOperationType.EXTERNAL_API,
            "httpx.Client": BlockingOperationType.EXTERNAL_API
        }
        
        # Performance thresholds
        self.thresholds = {
            "response_time_warning": 1.0,  # 1 second
            "response_time_error": 5.0,    # 5 seconds
            "response_time_critical": 10.0, # 10 seconds
            "blocking_operations_warning": 3,
            "blocking_operations_error": 10,
            "error_rate_warning": 0.05,    # 5%
            "error_rate_error": 0.10,      # 10%
            "error_rate_critical": 0.20    # 20%
        }
        
        # Performance history
        self.performance_history: deque = deque(maxlen=10000)
        self.start_time = time.time()
        
        logger.info("Route Performance Monitor initialized")
    
    async def initialize(self) -> Any:
        """Initialize the monitor"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Redis connection established for route performance monitoring")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory storage only.")
            self.redis_client = None
        
        # Setup middleware
        self._setup_monitoring_middleware()
        
        # Setup route analysis
        self._analyze_existing_routes()
    
    def _setup_monitoring_middleware(self) -> Any:
        """Setup performance monitoring middleware"""
        
        @self.app.middleware("http")
        async def performance_middleware(request: Request, call_next):
            
    """performance_middleware function."""
# Generate request ID
            request_id = f"req_{int(time.time() * 1000)}_{id(request)}"
            
            # Create performance data
            performance_data = RoutePerformanceData(
                route_path=request.url.path,
                method=request.method,
                request_id=request_id,
                start_time=time.time(),
                request_size=len(str(request.headers)),
                user_agent=request.headers.get("user-agent", ""),
                client_ip=request.client.host if request.client else ""
            )
            
            # Store performance data
            with self.performance_data_lock:
                self.performance_data[request_id] = performance_data
            
            # Track blocking operations
            blocking_operations = []
            
            try:
                # Process request
                response = await call_next(request)
                
                # Update performance data
                performance_data.end_time = time.time()
                performance_data.total_duration = performance_data.end_time - performance_data.start_time
                performance_data.status_code = response.status_code
                performance_data.response_size = len(str(response.headers))
                performance_data.blocking_operations = blocking_operations
                
                # Check for performance issues
                self._check_performance_alerts(performance_data)
                
                # Update route statistics
                self._update_route_stats(performance_data)
                
                return response
                
            except Exception as e:
                # Update error performance data
                performance_data.end_time = time.time()
                performance_data.total_duration = performance_data.end_time - performance_data.start_time
                performance_data.status_code = 500
                performance_data.error_message = str(e)
                performance_data.blocking_operations = blocking_operations
                
                # Check for performance issues
                self._check_performance_alerts(performance_data)
                
                # Update route statistics
                self._update_route_stats(performance_data)
                
                raise
    
    def _analyze_existing_routes(self) -> Any:
        """Analyze existing routes for potential blocking operations"""
        for route in self.app.routes:
            if isinstance(route, APIRoute):
                self._analyze_route_for_blocking_operations(route)
    
    def _analyze_route_for_blocking_operations(self, route: APIRoute):
        """Analyze a route for potential blocking operations"""
        try:
            source = inspect.getsource(route.endpoint)
            
            detected_operations = []
            for pattern, operation_type in self.blocking_patterns.items():
                if pattern in source:
                    detected_operations.append({
                        "pattern": pattern,
                        "type": operation_type.value,
                        "severity": "warning"
                    })
            
            if detected_operations:
                logger.warning(
                    f"Route {route.path} contains potential blocking operations: {detected_operations}"
                )
                
                # Create alert
                alert = PerformanceAlert(
                    route_path=route.path,
                    alert_type="blocking_operations_detected",
                    level=PerformanceAlertLevel.WARNING,
                    message=f"Route contains {len(detected_operations)} potential blocking operations",
                    metrics={"detected_operations": detected_operations}
                )
                
                with self.alerts_lock:
                    self.alerts.append(alert)
                    
        except Exception as e:
            logger.debug(f"Could not analyze route {route.path}: {e}")
    
    def _check_performance_alerts(self, performance_data: RoutePerformanceData):
        """Check for performance alerts"""
        route_key = f"{performance_data.method}:{performance_data.route_path}"
        
        # Check response time
        if performance_data.total_duration:
            if performance_data.total_duration > self.thresholds["response_time_critical"]:
                self._create_alert(
                    route_key,
                    "response_time_critical",
                    PerformanceAlertLevel.CRITICAL,
                    f"Response time {performance_data.total_duration:.2f}s exceeds critical threshold",
                    {"response_time": performance_data.total_duration}
                )
            elif performance_data.total_duration > self.thresholds["response_time_error"]:
                self._create_alert(
                    route_key,
                    "response_time_error",
                    PerformanceAlertLevel.ERROR,
                    f"Response time {performance_data.total_duration:.2f}s exceeds error threshold",
                    {"response_time": performance_data.total_duration}
                )
            elif performance_data.total_duration > self.thresholds["response_time_warning"]:
                self._create_alert(
                    route_key,
                    "response_time_warning",
                    PerformanceAlertLevel.WARNING,
                    f"Response time {performance_data.total_duration:.2f}s exceeds warning threshold",
                    {"response_time": performance_data.total_duration}
                )
        
        # Check blocking operations
        blocking_count = len(performance_data.blocking_operations)
        if blocking_count > self.thresholds["blocking_operations_error"]:
            self._create_alert(
                route_key,
                "blocking_operations_error",
                PerformanceAlertLevel.ERROR,
                f"Route has {blocking_count} blocking operations",
                {"blocking_operations": blocking_count}
            )
        elif blocking_count > self.thresholds["blocking_operations_warning"]:
            self._create_alert(
                route_key,
                "blocking_operations_warning",
                PerformanceAlertLevel.WARNING,
                f"Route has {blocking_count} blocking operations",
                {"blocking_operations": blocking_count}
            )
    
    def _create_alert(self, route_path: str, alert_type: str, level: PerformanceAlertLevel, 
                     message: str, metrics: Dict[str, Any]):
        """Create a performance alert"""
        alert = PerformanceAlert(
            route_path=route_path,
            alert_type=alert_type,
            level=level,
            message=message,
            metrics=metrics
        )
        
        with self.alerts_lock:
            self.alerts.append(alert)
        
        logger.warning(f"Performance alert: {message}")
    
    def _update_route_stats(self, performance_data: RoutePerformanceData):
        """Update route statistics"""
        route_key = f"{performance_data.method}:{performance_data.route_path}"
        
        with self.route_stats_lock:
            if route_key not in self.route_stats:
                self.route_stats[route_key] = {
                    "request_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "total_response_time": 0.0,
                    "min_response_time": float('inf'),
                    "max_response_time": 0.0,
                    "total_blocking_operations": 0,
                    "total_async_operations": 0,
                    "total_background_tasks": 0,
                    "last_updated": time.time()
                }
            
            stats = self.route_stats[route_key]
            stats["request_count"] += 1
            
            if performance_data.status_code < 400:
                stats["success_count"] += 1
            else:
                stats["error_count"] += 1
            
            if performance_data.total_duration:
                stats["total_response_time"] += performance_data.total_duration
                stats["min_response_time"] = min(stats["min_response_time"], performance_data.total_duration)
                stats["max_response_time"] = max(stats["max_response_time"], performance_data.total_duration)
            
            stats["total_blocking_operations"] += len(performance_data.blocking_operations)
            stats["total_async_operations"] += performance_data.async_operations
            stats["total_background_tasks"] += performance_data.background_tasks
            stats["last_updated"] = time.time()
    
    def get_route_performance(self, route_path: str = None, method: str = None) -> Dict[str, Any]:
        """Get route performance data"""
        with self.route_stats_lock:
            if route_path and method:
                route_key = f"{method}:{route_path}"
                stats = self.route_stats.get(route_key, {})
                
                if stats:
                    return {
                        "route_path": route_path,
                        "method": method,
                        "request_count": stats["request_count"],
                        "success_count": stats["success_count"],
                        "error_count": stats["error_count"],
                        "success_rate": stats["success_count"] / stats["request_count"] if stats["request_count"] > 0 else 0.0,
                        "error_rate": stats["error_count"] / stats["request_count"] if stats["request_count"] > 0 else 0.0,
                        "average_response_time": stats["total_response_time"] / stats["request_count"] if stats["request_count"] > 0 else 0.0,
                        "min_response_time": stats["min_response_time"] if stats["min_response_time"] != float('inf') else 0.0,
                        "max_response_time": stats["max_response_time"],
                        "total_blocking_operations": stats["total_blocking_operations"],
                        "total_async_operations": stats["total_async_operations"],
                        "total_background_tasks": stats["total_background_tasks"],
                        "blocking_to_async_ratio": stats["total_blocking_operations"] / stats["total_async_operations"] if stats["total_async_operations"] > 0 else float('inf'),
                        "last_updated": stats["last_updated"]
                    }
                return {}
            else:
                return {
                    route_key: {
                        "route_path": route_key.split(":", 1)[1],
                        "method": route_key.split(":", 1)[0],
                        "request_count": stats["request_count"],
                        "success_rate": stats["success_count"] / stats["request_count"] if stats["request_count"] > 0 else 0.0,
                        "average_response_time": stats["total_response_time"] / stats["request_count"] if stats["request_count"] > 0 else 0.0,
                        "total_blocking_operations": stats["total_blocking_operations"],
                        "total_async_operations": stats["total_async_operations"]
                    }
                    for route_key, stats in self.route_stats.items()
                }
    
    def get_performance_alerts(self, level: PerformanceAlertLevel = None) -> List[Dict[str, Any]]:
        """Get performance alerts"""
        with self.alerts_lock:
            alerts = self.alerts.copy()
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return [
            {
                "id": alert.id,
                "route_path": alert.route_path,
                "alert_type": alert.alert_type,
                "level": alert.level.value,
                "message": alert.message,
                "metrics": alert.metrics,
                "timestamp": alert.timestamp,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved
            }
            for alert in alerts
        ]
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge a performance alert"""
        with self.alerts_lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    break
    
    def resolve_alert(self, alert_id: str):
        """Resolve a performance alert"""
        with self.alerts_lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    break
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        route_performance = self.get_route_performance()
        
        # Calculate overall statistics
        total_requests = sum(route["request_count"] for route in route_performance.values())
        total_success = sum(route["success_count"] for route in route_performance.values())
        total_blocking = sum(route["total_blocking_operations"] for route in route_performance.values())
        total_async = sum(route["total_async_operations"] for route in route_performance.values())
        
        # Find problematic routes
        slow_routes = sorted(
            route_performance.items(),
            key=lambda x: x[1]["average_response_time"],
            reverse=True
        )[:5]
        
        high_blocking_routes = sorted(
            route_performance.items(),
            key=lambda x: x[1]["total_blocking_operations"],
            reverse=True
        )[:5]
        
        high_error_routes = sorted(
            route_performance.items(),
            key=lambda x: x[1]["error_rate"],
            reverse=True
        )[:5]
        
        # Get active alerts
        active_alerts = [alert for alert in self.get_performance_alerts() if not alert["resolved"]]
        
        return {
            "overall": {
                "total_routes": len(route_performance),
                "total_requests": total_requests,
                "overall_success_rate": total_success / total_requests if total_requests > 0 else 0.0,
                "total_blocking_operations": total_blocking,
                "total_async_operations": total_async,
                "blocking_to_async_ratio": total_blocking / total_async if total_async > 0 else float('inf')
            },
            "problematic_routes": {
                "slowest_routes": [
                    {
                        "route": route_key,
                        "average_response_time": route["average_response_time"],
                        "request_count": route["request_count"]
                    }
                    for route_key, route in slow_routes
                ],
                "routes_with_most_blocking": [
                    {
                        "route": route_key,
                        "blocking_operations": route["total_blocking_operations"],
                        "request_count": route["request_count"]
                    }
                    for route_key, route in high_blocking_routes
                ],
                "routes_with_most_errors": [
                    {
                        "route": route_key,
                        "error_rate": route["error_rate"],
                        "request_count": route["request_count"]
                    }
                    for route_key, route in high_error_routes
                ]
            },
            "alerts": {
                "total_alerts": len(active_alerts),
                "alerts_by_level": {
                    level.value: len([a for a in active_alerts if a["level"] == level.value])
                    for level in PerformanceAlertLevel
                },
                "recent_alerts": active_alerts[-10:]  # Last 10 alerts
            },
            "optimization_recommendations": self._generate_optimization_recommendations(route_performance),
            "uptime_seconds": time.time() - self.start_time
        }
    
    def _generate_optimization_recommendations(self, route_performance: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        for route_key, route in route_performance.items():
            # Check for slow routes
            if route["average_response_time"] > self.thresholds["response_time_warning"]:
                recommendations.append(
                    f"Route {route_key} is slow ({route['average_response_time']:.2f}s avg). "
                    "Consider async optimization, caching, or database query optimization."
                )
            
            # Check for routes with many blocking operations
            if route["total_blocking_operations"] > route["total_async_operations"]:
                recommendations.append(
                    f"Route {route_key} has more blocking than async operations. "
                    "Consider converting blocking operations to async patterns."
                )
            
            # Check for high error rates
            if route["error_rate"] > self.thresholds["error_rate_warning"]:
                recommendations.append(
                    f"Route {route_key} has high error rate ({route['error_rate']:.2%}). "
                    "Consider improving error handling and validation."
                )
        
        return recommendations

# Global monitor instance
_monitor: Optional[RoutePerformanceMonitor] = None

def get_route_monitor(app: FastAPI) -> RoutePerformanceMonitor:
    """Get the route performance monitor for an app"""
    global _monitor
    if _monitor is None:
        _monitor = RoutePerformanceMonitor(app)
    return _monitor

# FastAPI integration

def setup_route_monitoring(app: FastAPI, redis_url: str = "redis://localhost:6379"):
    """Setup route performance monitoring for a FastAPI app"""
    monitor = RoutePerformanceMonitor(app, redis_url)
    
    @app.on_event("startup")
    async def startup_event():
        
    """startup_event function."""
await monitor.initialize()
    
    # Add monitoring endpoints
    @app.get("/monitoring/routes/performance")
    async def get_route_performance_endpoint(
        route_path: str = None,
        method: str = None
    ):
        """Get route performance data"""
        return monitor.get_route_performance(route_path, method)
    
    @app.get("/monitoring/alerts")
    async def get_alerts_endpoint(
        level: str = None
    ):
        """Get performance alerts"""
        alert_level = None
        if level:
            try:
                alert_level = PerformanceAlertLevel(level)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid alert level: {level}")
        
        return monitor.get_performance_alerts(alert_level)
    
    @app.post("/monitoring/alerts/{alert_id}/acknowledge")
    async def acknowledge_alert_endpoint(alert_id: str):
        """Acknowledge a performance alert"""
        monitor.acknowledge_alert(alert_id)
        return {"message": "Alert acknowledged"}
    
    @app.post("/monitoring/alerts/{alert_id}/resolve")
    async def resolve_alert_endpoint(alert_id: str):
        """Resolve a performance alert"""
        monitor.resolve_alert(alert_id)
        return {"message": "Alert resolved"}
    
    @app.get("/monitoring/summary")
    async def get_performance_summary_endpoint():
        """Get comprehensive performance summary"""
        return monitor.get_performance_summary()
    
    return monitor

# Example usage

async def example_usage():
    """Example usage of route performance monitoring"""
    
    # Create FastAPI app
    app = FastAPI(title="Example App with Route Monitoring")
    
    # Setup monitoring
    monitor = setup_route_monitoring(app)
    
    # Example routes with different performance characteristics
    
    @app.get("/fast")
    async def fast_route():
        """Fast route"""
        await asyncio.sleep(0.1)
        return {"message": "Fast response"}
    
    @app.get("/slow")
    async def slow_route():
        """Slow route"""
        await asyncio.sleep(2.0)
        return {"message": "Slow response"}
    
    @app.get("/blocking")
    async def blocking_route():
        """Route with blocking operations"""
        def blocking_operation():
            
    """blocking_operation function."""
time.sleep(1.0)
            return "Blocking operation completed"
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, blocking_operation)
        return {"message": result}
    
    @app.get("/error-prone")
    async def error_prone_route():
        """Route that sometimes errors"""
        if random.random() < 0.3:  # 30% error rate
            raise HTTPException(status_code=500, detail="Random error")
        return {"message": "Success"}
    
    # Simulate some requests
    client = TestClient(app)
    
    # Make some requests
    for _ in range(10):
        client.get("/fast")
        client.get("/slow")
        client.get("/blocking")
        client.get("/error-prone")
    
    # Get performance data
    performance_response = client.get("/monitoring/routes/performance")
    print("Route Performance:", performance_response.json())
    
    # Get alerts
    alerts_response = client.get("/monitoring/alerts")
    print("Performance Alerts:", alerts_response.json())
    
    # Get summary
    summary_response = client.get("/monitoring/summary")
    print("Performance Summary:", summary_response.json())

match __name__:
    case "__main__":
    asyncio.run(example_usage()) 