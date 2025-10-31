from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import statistics
import threading
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
from fastapi import FastAPI, Request, Response, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict
import httpx
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.registry import CollectorRegistry
        import uuid
    from fastapi.responses import Response
    import uvicorn
from typing import Any, List, Dict, Optional
"""
API Performance Metrics Implementation
====================================

This module demonstrates:
- Response time monitoring and tracking
- Latency measurement and analysis
- Throughput calculation and optimization
- Performance metrics collection and storage
- Real-time monitoring and alerting
- Performance profiling and optimization
- Load testing and benchmarking
- Performance dashboards and reporting
"""




# ============================================================================
# PERFORMANCE METRICS DATA STRUCTURES
# ============================================================================

@dataclass
class PerformanceMetrics:
    """Container for performance metrics data."""
    
    endpoint: str
    method: str
    response_time: float
    status_code: int
    timestamp: datetime
    request_size: int = 0
    response_size: int = 0
    user_agent: str = ""
    client_ip: str = ""
    user_id: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "response_time": self.response_time,
            "status_code": self.status_code,
            "timestamp": self.timestamp.isoformat(),
            "request_size": self.request_size,
            "response_size": self.response_size,
            "user_agent": self.user_agent,
            "client_ip": self.client_ip,
            "user_id": self.user_id,
            "error_message": self.error_message
        }


@dataclass
class EndpointMetrics:
    """Metrics for a specific endpoint."""
    
    endpoint: str
    method: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    error_counts: Dict[int, int] = field(default_factory=dict)
    throughput_per_minute: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def add_metric(self, metric: PerformanceMetrics):
        """Add a new performance metric."""
        self.total_requests += 1
        self.total_response_time += metric.response_time
        
        if metric.status_code < 400:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            self.error_counts[metric.status_code] = self.error_counts.get(metric.status_code, 0) + 1
        
        self.min_response_time = min(self.min_response_time, metric.response_time)
        self.max_response_time = max(self.max_response_time, metric.response_time)
        self.response_times.append(metric.response_time)
        self.last_updated = datetime.utcnow()
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if self.total_requests == 0:
            return 0.0
        return self.total_response_time / self.total_requests
    
    @property
    def median_response_time(self) -> float:
        """Calculate median response time."""
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        """Calculate 95th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index]
    
    @property
    def p99_response_time(self) -> float:
        """Calculate 99th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.99 * len(sorted_times))
        return sorted_times[index]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "average_response_time": self.average_response_time,
            "median_response_time": self.median_response_time,
            "min_response_time": self.min_response_time if self.min_response_time != float('inf') else 0.0,
            "max_response_time": self.max_response_time,
            "p95_response_time": self.p95_response_time,
            "p99_response_time": self.p99_response_time,
            "success_rate": self.success_rate,
            "error_counts": self.error_counts,
            "throughput_per_minute": self.throughput_per_minute,
            "last_updated": self.last_updated.isoformat()
        }


# ============================================================================
# PERFORMANCE MONITORING MIDDLEWARE
# ============================================================================

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring API performance metrics."""
    
    def __init__(self, app: FastAPI, metrics_collector: 'MetricsCollector'):
        
    """__init__ function."""
super().__init__(app)
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
    
    async def dispatch(self, request: Request, call_next):
        """Process request and collect performance metrics."""
        start_time = time.time()
        
        # Extract request information
        endpoint = request.url.path
        method = request.method
        user_agent = request.headers.get("user-agent", "")
        client_ip = self._get_client_ip(request)
        
        # Get request size
        request_size = 0
        if request.body:
            body = await request.body()
            request_size = len(body)
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error_message = None
        except Exception as e:
            status_code = 500
            error_message = str(e)
            response = JSONResponse(
                status_code=status_code,
                content={"detail": "Internal server error"}
            )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Get response size
        response_size = 0
        if hasattr(response, 'body'):
            response_size = len(response.body)
        
        # Create performance metric
        metric = PerformanceMetrics(
            endpoint=endpoint,
            method=method,
            response_time=response_time,
            status_code=status_code,
            timestamp=datetime.utcnow(),
            request_size=request_size,
            response_size=response_size,
            user_agent=user_agent,
            client_ip=client_ip,
            error_message=error_message
        )
        
        # Collect metrics
        await self.metrics_collector.collect_metric(metric)
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{response_time:.4f}s"
        response.headers["X-Request-ID"] = self._generate_request_id()
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0]
        return request.client.host if request.client else "Unknown"
    
    async def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return str(uuid.uuid4())


# ============================================================================
# METRICS COLLECTOR
# ============================================================================

class MetricsCollector:
    """Collector for performance metrics."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        
    """__init__ function."""
self.redis_client = redis_client
        self.endpoint_metrics: Dict[str, EndpointMetrics] = defaultdict(
            lambda: EndpointMetrics("", "")
        )
        self.global_metrics = EndpointMetrics("global", "ALL")
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.request_counter = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        self.request_size = Histogram(
            'http_request_size_bytes',
            'HTTP request size in bytes',
            ['method', 'endpoint'],
            registry=self.registry
        )
        self.response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size in bytes',
            ['method', 'endpoint'],
            registry=self.registry
        )
        self.active_requests = Gauge(
            'http_active_requests',
            'Number of active HTTP requests',
            ['method', 'endpoint'],
            registry=self.registry
        )
    
    async def collect_metric(self, metric: PerformanceMetrics):
        """Collect a performance metric."""
        with self.lock:
            # Update endpoint-specific metrics
            endpoint_key = f"{metric.method}:{metric.endpoint}"
            if endpoint_key not in self.endpoint_metrics:
                self.endpoint_metrics[endpoint_key] = EndpointMetrics(
                    metric.endpoint, metric.method
                )
            
            self.endpoint_metrics[endpoint_key].add_metric(metric)
            self.global_metrics.add_metric(metric)
            
            # Update Prometheus metrics
            self.request_counter.labels(
                method=metric.method,
                endpoint=metric.endpoint,
                status_code=metric.status_code
            ).inc()
            
            self.request_duration.labels(
                method=metric.method,
                endpoint=metric.endpoint
            ).observe(metric.response_time)
            
            self.request_size.labels(
                method=metric.method,
                endpoint=metric.endpoint
            ).observe(metric.request_size)
            
            self.response_size.labels(
                method=metric.method,
                endpoint=metric.endpoint
            ).observe(metric.response_size)
        
        # Store in Redis if available
        if self.redis_client:
            await self._store_metric_redis(metric)
        
        # Log performance issues
        await self._check_performance_thresholds(metric)
    
    async def _store_metric_redis(self, metric: PerformanceMetrics):
        """Store metric in Redis for persistence."""
        try:
            # Store individual metric
            metric_key = f"metric:{metric.timestamp.timestamp()}"
            await self.redis_client.setex(
                metric_key,
                3600,  # 1 hour TTL
                json.dumps(metric.to_dict())
            )
            
            # Update endpoint metrics
            endpoint_key = f"endpoint_metrics:{metric.method}:{metric.endpoint}"
            endpoint_data = self.endpoint_metrics[f"{metric.method}:{metric.endpoint}"].to_dict()
            await self.redis_client.setex(
                endpoint_key,
                3600,  # 1 hour TTL
                json.dumps(endpoint_data)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store metric in Redis: {e}")
    
    async def _check_performance_thresholds(self, metric: PerformanceMetrics):
        """Check performance thresholds and log warnings."""
        # Check for slow responses
        if metric.response_time > 1.0:  # 1 second threshold
            self.logger.warning(
                f"Slow response detected: {metric.endpoint} took {metric.response_time:.3f}s"
            )
        
        # Check for errors
        if metric.status_code >= 500:
            self.logger.error(
                f"Server error detected: {metric.endpoint} returned {metric.status_code}"
            )
    
    def get_endpoint_metrics(self, endpoint: str, method: str) -> Optional[EndpointMetrics]:
        """Get metrics for a specific endpoint."""
        endpoint_key = f"{method}:{endpoint}"
        return self.endpoint_metrics.get(endpoint_key)
    
    def get_all_metrics(self) -> Dict[str, EndpointMetrics]:
        """Get all endpoint metrics."""
        return dict(self.endpoint_metrics)
    
    def get_global_metrics(self) -> EndpointMetrics:
        """Get global metrics."""
        return self.global_metrics
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry)


# ============================================================================
# PERFORMANCE ANALYZER
# ============================================================================

class PerformanceAnalyzer:
    """Analyzer for performance metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        
    """__init__ function."""
self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
    
    def analyze_endpoint_performance(self, endpoint: str, method: str) -> Dict[str, Any]:
        """Analyze performance for a specific endpoint."""
        metrics = self.metrics_collector.get_endpoint_metrics(endpoint, method)
        if not metrics:
            return {"error": "No metrics found for endpoint"}
        
        analysis = {
            "endpoint": endpoint,
            "method": method,
            "performance_summary": {
                "total_requests": metrics.total_requests,
                "success_rate": f"{metrics.success_rate:.2f}%",
                "average_response_time": f"{metrics.average_response_time:.3f}s",
                "median_response_time": f"{metrics.median_response_time:.3f}s",
                "p95_response_time": f"{metrics.p95_response_time:.3f}s",
                "p99_response_time": f"{metrics.p99_response_time:.3f}s",
                "min_response_time": f"{metrics.min_response_time:.3f}s",
                "max_response_time": f"{metrics.max_response_time:.3f}s"
            },
            "performance_grade": self._calculate_performance_grade(metrics),
            "recommendations": self._generate_recommendations(metrics),
            "error_analysis": self._analyze_errors(metrics),
            "trends": self._analyze_trends(metrics)
        }
        
        return analysis
    
    def _calculate_performance_grade(self, metrics: EndpointMetrics) -> str:
        """Calculate performance grade (A-F)."""
        avg_time = metrics.average_response_time
        success_rate = metrics.success_rate
        
        if avg_time < 0.1 and success_rate >= 99:
            return "A"
        elif avg_time < 0.3 and success_rate >= 95:
            return "B"
        elif avg_time < 0.5 and success_rate >= 90:
            return "C"
        elif avg_time < 1.0 and success_rate >= 85:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, metrics: EndpointMetrics) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if metrics.average_response_time > 0.5:
            recommendations.append("Consider optimizing database queries")
            recommendations.append("Implement caching for frequently accessed data")
        
        if metrics.p95_response_time > 1.0:
            recommendations.append("Investigate slow response outliers")
            recommendations.append("Consider implementing request timeouts")
        
        if metrics.success_rate < 95:
            recommendations.append("Investigate error patterns")
            recommendations.append("Improve error handling and validation")
        
        if metrics.total_requests > 1000:
            recommendations.append("Consider implementing rate limiting")
            recommendations.append("Monitor for potential DoS attacks")
        
        return recommendations
    
    def _analyze_errors(self, metrics: EndpointMetrics) -> Dict[str, Any]:
        """Analyze error patterns."""
        return {
            "total_errors": metrics.failed_requests,
            "error_rate": f"{((metrics.failed_requests / metrics.total_requests) * 100):.2f}%" if metrics.total_requests > 0 else "0%",
            "error_breakdown": metrics.error_counts,
            "most_common_error": max(metrics.error_counts.items(), key=lambda x: x[1]) if metrics.error_counts else None
        }
    
    def _analyze_trends(self, metrics: EndpointMetrics) -> Dict[str, Any]:
        """Analyze performance trends."""
        if len(metrics.response_times) < 10:
            return {"message": "Insufficient data for trend analysis"}
        
        recent_times = list(metrics.response_times)[-10:]
        older_times = list(metrics.response_times)[-20:-10] if len(metrics.response_times) >= 20 else []
        
        if not older_times:
            return {"message": "Insufficient historical data"}
        
        recent_avg = statistics.mean(recent_times)
        older_avg = statistics.mean(older_times)
        
        trend = "improving" if recent_avg < older_avg else "degrading" if recent_avg > older_avg else "stable"
        change_percent = ((recent_avg - older_avg) / older_avg) * 100
        
        return {
            "trend": trend,
            "change_percent": f"{change_percent:.2f}%",
            "recent_average": f"{recent_avg:.3f}s",
            "historical_average": f"{older_avg:.3f}s"
        }


# ============================================================================
# LOAD TESTER
# ============================================================================

class LoadTester:
    """Load testing utility for API endpoints."""
    
    def __init__(self, base_url: str):
        
    """__init__ function."""
self.base_url = base_url
        self.logger = logging.getLogger(__name__)
    
    async def run_load_test(
        self,
        endpoint: str,
        method: str = "GET",
        num_requests: int = 100,
        concurrent_users: int = 10,
        duration: int = 60
    ) -> Dict[str, Any]:
        """Run load test on an endpoint."""
        self.logger.info(f"Starting load test: {method} {endpoint}")
        
        start_time = time.time()
        results = []
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def make_request():
            
    """make_request function."""
async with semaphore:
                request_start = time.time()
                try:
                    async with httpx.AsyncClient() as client:
                        if method.upper() == "GET":
                            response = await client.get(f"{self.base_url}{endpoint}")
                        elif method.upper() == "POST":
                            response = await client.post(f"{self.base_url}{endpoint}")
                        else:
                            raise ValueError(f"Unsupported method: {method}")
                        
                        request_time = time.time() - request_start
                        results.append({
                            "status_code": response.status_code,
                            "response_time": request_time,
                            "success": response.status_code < 400
                        })
                        
                except Exception as e:
                    request_time = time.time() - request_start
                    results.append({
                        "status_code": 0,
                        "response_time": request_time,
                        "success": False,
                        "error": str(e)
                    })
        
        # Create tasks
        tasks = []
        for _ in range(num_requests):
            tasks.append(make_request())
        
        # Run requests
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        response_times = [r["response_time"] for r in results]
        
        analysis = {
            "test_configuration": {
                "endpoint": endpoint,
                "method": method,
                "num_requests": num_requests,
                "concurrent_users": concurrent_users,
                "duration": duration
            },
            "results": {
                "total_requests": len(results),
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": f"{(len(successful_requests) / len(results)) * 100:.2f}%",
                "total_time": f"{total_time:.2f}s",
                "requests_per_second": f"{len(results) / total_time:.2f}",
                "average_response_time": f"{statistics.mean(response_times):.3f}s",
                "median_response_time": f"{statistics.median(response_times):.3f}s",
                "min_response_time": f"{min(response_times):.3f}s",
                "max_response_time": f"{max(response_times):.3f}s",
                "p95_response_time": f"{sorted(response_times)[int(0.95 * len(response_times))]:.3f}s",
                "p99_response_time": f"{sorted(response_times)[int(0.99 * len(response_times))]:.3f}s"
            },
            "error_analysis": self._analyze_errors(failed_requests)
        }
        
        self.logger.info(f"Load test completed: {analysis['results']['requests_per_second']} req/s")
        return analysis
    
    def _analyze_errors(self, failed_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze failed requests."""
        if not failed_requests:
            return {"total_errors": 0}
        
        error_types = defaultdict(int)
        for req in failed_requests:
            error = req.get("error", "Unknown")
            error_types[error] += 1
        
        return {
            "total_errors": len(failed_requests),
            "error_breakdown": dict(error_types),
            "most_common_error": max(error_types.items(), key=lambda x: x[1])
        }


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PerformanceReport(BaseModel):
    """Performance report model."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    endpoint: str
    method: str
    total_requests: int
    average_response_time: float
    success_rate: float
    p95_response_time: float
    p99_response_time: float
    throughput_per_second: float
    error_count: int
    
    model_config = ConfigDict(from_attributes=True)


class LoadTestRequest(BaseModel):
    """Load test request model."""
    
    endpoint: str = Field(..., description="Endpoint to test")
    method: str = Field("GET", description="HTTP method")
    num_requests: int = Field(100, ge=1, le=10000, description="Number of requests")
    concurrent_users: int = Field(10, ge=1, le=100, description="Concurrent users")
    duration: int = Field(60, ge=10, le=3600, description="Test duration in seconds")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application with performance monitoring."""
    
    app = FastAPI(
        title="API Performance Metrics Demo",
        version="1.0.0",
        description="Demonstration of API performance monitoring and metrics"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()

# Initialize metrics collector
metrics_collector = MetricsCollector()
performance_analyzer = PerformanceAnalyzer(metrics_collector)
load_tester = LoadTester("http://localhost:8000")

# Add performance monitoring middleware
app.add_middleware(PerformanceMonitoringMiddleware, metrics_collector=metrics_collector)


# ============================================================================
# API ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    await asyncio.sleep(0.1)  # Simulate processing
    return {"message": "API Performance Metrics Demo"}


@app.get("/fast")
async def fast_endpoint():
    """Fast endpoint for testing."""
    return {"message": "Fast response", "timestamp": datetime.utcnow()}


@app.get("/slow")
async def slow_endpoint():
    """Slow endpoint for testing."""
    await asyncio.sleep(2.0)  # Simulate slow processing
    return {"message": "Slow response", "timestamp": datetime.utcnow()}


@app.get("/error")
async def error_endpoint():
    """Error endpoint for testing."""
    raise HTTPException(status_code=500, detail="Simulated error")


@app.get("/metrics")
async def get_metrics():
    """Get all performance metrics."""
    return {
        "global_metrics": metrics_collector.get_global_metrics().to_dict(),
        "endpoint_metrics": {
            key: metrics.to_dict() 
            for key, metrics in metrics_collector.get_all_metrics().items()
        }
    }


@app.get("/metrics/{endpoint}")
async def get_endpoint_metrics(endpoint: str, method: str = "GET"):
    """Get metrics for a specific endpoint."""
    metrics = metrics_collector.get_endpoint_metrics(endpoint, method)
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics found for endpoint")
    
    return metrics.to_dict()


@app.get("/analysis/{endpoint}")
async def analyze_endpoint(endpoint: str, method: str = "GET"):
    """Analyze performance for a specific endpoint."""
    analysis = performance_analyzer.analyze_endpoint_performance(endpoint, method)
    return analysis


@app.post("/load-test")
async def run_load_test(request: LoadTestRequest):
    """Run load test on an endpoint."""
    try:
        results = await load_tester.run_load_test(
            endpoint=request.endpoint,
            method=request.method,
            num_requests=request.num_requests,
            concurrent_users=request.concurrent_users,
            duration=request.duration
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load test failed: {str(e)}")


@app.get("/prometheus")
async def prometheus_metrics():
    """Get Prometheus metrics."""
    return Response(
        content=metrics_collector.get_prometheus_metrics(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global_metrics = metrics_collector.get_global_metrics()
    
    # Determine health based on metrics
    health_status = "healthy"
    if global_metrics.success_rate < 95:
        health_status = "degraded"
    if global_metrics.average_response_time > 1.0:
        health_status = "slow"
    
    return {
        "status": health_status,
        "timestamp": datetime.utcnow(),
        "metrics": {
            "success_rate": f"{global_metrics.success_rate:.2f}%",
            "average_response_time": f"{global_metrics.average_response_time:.3f}s",
            "total_requests": global_metrics.total_requests
        }
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def demonstrate_performance_metrics():
    """Demonstrate performance metrics collection and analysis."""
    
    print("\n=== API Performance Metrics Demonstrations ===")
    
    # 1. Simulate some API calls
    print("\n1. Simulating API calls...")
    
    # Fast endpoint
    start_time = time.time()
    response = await app.get("/fast")
    response_time = time.time() - start_time
    print(f"   - Fast endpoint: {response_time:.3f}s")
    
    # Slow endpoint
    start_time = time.time()
    response = await app.get("/slow")
    response_time = time.time() - start_time
    print(f"   - Slow endpoint: {response_time:.3f}s")
    
    # Error endpoint
    start_time = time.time()
    try:
        response = await app.get("/error")
    except:
        pass
    response_time = time.time() - start_time
    print(f"   - Error endpoint: {response_time:.3f}s")
    
    # 2. Get metrics
    print("\n2. Performance Metrics:")
    metrics = await app.get("/metrics")
    print(f"   - Global metrics: {metrics['global_metrics']['total_requests']} total requests")
    print(f"   - Average response time: {metrics['global_metrics']['average_response_time']:.3f}s")
    print(f"   - Success rate: {metrics['global_metrics']['success_rate']:.2f}%")
    
    # 3. Analyze performance
    print("\n3. Performance Analysis:")
    analysis = await app.get("/analysis/fast")
    print(f"   - Fast endpoint grade: {analysis['performance_grade']}")
    print(f"   - Recommendations: {len(analysis['recommendations'])} suggestions")
    
    # 4. Load testing
    print("\n4. Load Testing:")
    load_test_request = LoadTestRequest(
        endpoint="/fast",
        num_requests=50,
        concurrent_users=5
    )
    load_results = await app.post("/load-test", json=load_test_request.model_dump())
    print(f"   - Load test completed: {load_results['results']['requests_per_second']} req/s")
    print(f"   - Success rate: {load_results['results']['success_rate']}")
    
    # 5. Health check
    print("\n5. Health Check:")
    health = await app.get("/health")
    print(f"   - Status: {health['status']}")
    print(f"   - Success rate: {health['metrics']['success_rate']}")


if __name__ == "__main__":
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application
    uvicorn.run(
        "api_performance_metrics_implementation:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 