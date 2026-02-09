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
import uuid
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from performance_metrics import (
            import random
from typing import Any, List, Dict, Optional
import logging
"""
API Performance Metrics Demo

This demo showcases:
- Real-time performance monitoring
- Response time, latency, and throughput tracking
- System resource monitoring
- Performance alerts and thresholds
- Performance analytics and recommendations
- Integration with FastAPI
- Performance optimization insights
"""



    APIPerformanceMetrics, PerformanceMonitor, PerformanceThreshold,
    get_performance_metrics, set_performance_metrics, performance_tracking,
    track_performance, track_cache_performance, track_database_performance,
    track_external_api_performance, MetricType
)


# Pydantic models for API
class PerformanceStatsRequest(BaseModel):
    """Request model for performance statistics."""
    endpoint: Optional[str] = Field(None, description="Specific endpoint to analyze")
    time_window: Optional[int] = Field(300, description="Time window in seconds")


class ThresholdRequest(BaseModel):
    """Request model for performance thresholds."""
    metric_type: MetricType
    endpoint: str
    threshold_value: float
    comparison: str = "gt"
    alert_message: str = ""
    severity: str = "warning"


class PerformanceAlert(BaseModel):
    """Performance alert model."""
    threshold: PerformanceThreshold
    current_value: float
    timestamp: float
    message: str
    severity: str


# Create FastAPI application
app = FastAPI(
    title="API Performance Metrics Demo",
    description="Comprehensive performance monitoring and analytics",
    version="1.0.0"
)

# Initialize performance metrics
performance_metrics = APIPerformanceMetrics(
    max_metrics=10000,
    window_size=300,
    enable_alerts=True,
    enable_persistence=False
)
set_performance_metrics(performance_metrics)

# Performance monitor
monitor = PerformanceMonitor()


# Middleware for performance tracking
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Track performance for all requests."""
    async with performance_tracking(request, Response()):
        response = await call_next(request)
        return response


# Simulated services with performance tracking
class ProductService:
    """Product service with performance tracking."""
    
    def __init__(self) -> Any:
        self.cache = {}
    
    @track_performance("get_product")
    async def get_product(self, product_id: str) -> Dict[str, Any]:
        """Get product with performance tracking."""
        # Simulate database query
        await asyncio.sleep(0.1)
        
        # Simulate cache hit/miss
        if product_id in self.cache:
            performance_metrics.record_cache_hit()
        else:
            performance_metrics.record_cache_miss()
            self.cache[product_id] = {"id": product_id, "name": f"Product {product_id}"}
        
        return self.cache[product_id]
    
    @track_database_performance
    async def get_products_batch(self, product_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple products with database performance tracking."""
        # Simulate database batch query
        await asyncio.sleep(0.2)
        
        products = []
        for product_id in product_ids:
            products.append({"id": product_id, "name": f"Product {product_id}"})
        
        return products
    
    @track_external_api_performance
    async def get_product_reviews(self, product_id: str) -> List[Dict[str, Any]]:
        """Get product reviews with external API performance tracking."""
        # Simulate external API call
        await asyncio.sleep(0.15)
        
        return [
            {"id": 1, "rating": 5, "comment": "Great product!"},
            {"id": 2, "rating": 4, "comment": "Good quality"}
        ]


class UserService:
    """User service with performance tracking."""
    
    @track_performance("get_user")
    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get user with performance tracking."""
        # Simulate database query
        await asyncio.sleep(0.08)
        
        return {"id": user_id, "name": f"User {user_id}", "email": f"user{user_id}@example.com"}
    
    @track_database_performance
    async def get_users_paginated(self, page: int, page_size: int) -> Dict[str, Any]:
        """Get users with pagination and database performance tracking."""
        # Simulate database query with pagination
        await asyncio.sleep(0.12)
        
        users = []
        for i in range(page_size):
            user_id = page * page_size + i
            users.append({"id": f"user_{user_id}", "name": f"User {user_id}"})
        
        return {
            "users": users,
            "page": page,
            "page_size": page_size,
            "total_count": 1000
        }


# Global service instances
product_service = ProductService()
user_service = UserService()


# API Routes with performance monitoring

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "API Performance Metrics Demo",
        "version": "1.0.0",
        "endpoints": [
            "/products/{product_id}",
            "/products/batch",
            "/products/{product_id}/reviews",
            "/users/{user_id}",
            "/users",
            "/metrics/performance",
            "/metrics/system",
            "/metrics/alerts",
            "/metrics/recommendations"
        ]
    }


@app.get("/products/{product_id}")
async def get_product(product_id: str):
    """Get product with performance tracking."""
    try:
        product = await product_service.get_product(product_id)
        return {"product": product, "cached": product_id in product_service.cache}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/products/batch")
async def get_products_batch(product_ids: List[str]):
    """Get multiple products with performance tracking."""
    try:
        products = await product_service.get_products_batch(product_ids)
        return {"products": products, "count": len(products)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/{product_id}/reviews")
async def get_product_reviews(product_id: str):
    """Get product reviews with performance tracking."""
    try:
        reviews = await product_service.get_product_reviews(product_id)
        return {"product_id": product_id, "reviews": reviews}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}")
async def get_user(user_id: str):
    """Get user with performance tracking."""
    try:
        user = await user_service.get_user(user_id)
        return {"user": user}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users")
async def get_users(page: int = 0, page_size: int = 50):
    """Get users with pagination and performance tracking."""
    try:
        result = await user_service.get_users_paginated(page, page_size)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Performance metrics endpoints

@app.get("/metrics/performance")
async def get_performance_metrics(request: PerformanceStatsRequest = Depends()):
    """Get comprehensive performance metrics."""
    try:
        stats = performance_metrics.get_comprehensive_stats()
        
        # Add endpoint-specific stats if requested
        if request.endpoint:
            stats["endpoint_specific"] = {
                "response_time": performance_metrics.get_response_time_stats(request.endpoint).__dict__,
                "error_rate": performance_metrics.get_error_rate(request.endpoint)
            }
        
        return {
            "timestamp": time.time(),
            "time_window": request.time_window,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/system")
async def get_system_metrics():
    """Get system performance metrics."""
    try:
        return {
            "memory_usage": performance_metrics.get_memory_usage(),
            "cpu_usage": performance_metrics.get_cpu_usage(),
            "concurrent_requests": performance_metrics.get_concurrent_requests(),
            "uptime": performance_metrics.get_comprehensive_stats()["uptime"],
            "health_score": monitor.get_system_health_score()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/response-times")
async def get_response_time_metrics(endpoint: Optional[str] = None):
    """Get response time metrics."""
    try:
        stats = performance_metrics.get_response_time_stats(endpoint or "*")
        percentiles = monitor.get_response_time_percentiles(endpoint or "*")
        
        return {
            "endpoint": endpoint or "overall",
            "stats": stats.__dict__,
            "percentiles": percentiles
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/throughput")
async def get_throughput_metrics():
    """Get throughput metrics."""
    try:
        return {
            "current_throughput": performance_metrics.get_throughput(),
            "throughput_history": list(performance_metrics.throughput_history),
            "total_requests": performance_metrics.request_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/errors")
async def get_error_metrics(endpoint: Optional[str] = None):
    """Get error metrics."""
    try:
        return {
            "endpoint": endpoint or "overall",
            "error_rate": performance_metrics.get_error_rate(endpoint or "*"),
            "error_counts": dict(performance_metrics.error_counts),
            "success_counts": dict(performance_metrics.success_counts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/cache")
async def get_cache_metrics():
    """Get cache performance metrics."""
    try:
        return {
            "hit_rate": performance_metrics.get_cache_hit_rate(),
            "hits": performance_metrics.cache_hits,
            "misses": performance_metrics.cache_misses,
            "total_requests": performance_metrics.cache_hits + performance_metrics.cache_misses
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/database")
async def get_database_metrics():
    """Get database performance metrics."""
    try:
        return performance_metrics.get_database_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/external-api")
async def get_external_api_metrics():
    """Get external API performance metrics."""
    try:
        return performance_metrics.get_external_api_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/alerts")
async def get_alerts(severity: Optional[str] = None):
    """Get performance alerts."""
    try:
        alerts = performance_metrics.get_alerts(severity)
        return {
            "alerts": [alert.dict() for alert in alerts],
            "count": len(alerts)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/metrics/thresholds")
async def add_threshold(request: ThresholdRequest):
    """Add a performance threshold."""
    try:
        threshold = PerformanceThreshold(
            metric_type=request.metric_type,
            endpoint=request.endpoint,
            threshold_value=request.threshold_value,
            comparison=request.comparison,
            alert_message=request.alert_message,
            severity=request.severity
        )
        performance_metrics.add_threshold(threshold)
        return {"message": "Threshold added successfully", "threshold": threshold.dict()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/recommendations")
async def get_performance_recommendations():
    """Get performance improvement recommendations."""
    try:
        return {
            "health_score": monitor.get_system_health_score(),
            "recommendations": monitor.get_performance_recommendations(),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/metrics/alerts")
async def clear_alerts():
    """Clear all performance alerts."""
    try:
        performance_metrics.clear_alerts()
        return {"message": "All alerts cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks for performance testing
@app.post("/test/performance")
async def run_performance_test(
    background_tasks: BackgroundTasks,
    num_requests: int = 100,
    concurrent: int = 10
):
    """Run performance test."""
    try:
        background_tasks.add_task(
            simulate_load,
            num_requests=num_requests,
            concurrent=concurrent
        )
        
        return {
            "message": f"Performance test started: {num_requests} requests, {concurrent} concurrent",
            "test_id": str(uuid.uuid4())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def simulate_load(num_requests: int, concurrent: int):
    """Simulate load for performance testing."""
    semaphore = asyncio.Semaphore(concurrent)
    
    async def make_request():
        
    """make_request function."""
async with semaphore:
            # Simulate different types of requests
            request_type = random.choice(["product", "user", "batch"])
            
            if request_type == "product":
                product_id = f"prod_{random.randint(1, 1000)}"
                await product_service.get_product(product_id)
            elif request_type == "user":
                user_id = f"user_{random.randint(1, 1000)}"
                await user_service.get_user(user_id)
            else:
                product_ids = [f"prod_{random.randint(1, 1000)}" for _ in range(5)]
                await product_service.get_products_batch(product_ids)
    
    tasks = [make_request() for _ in range(num_requests)]
    await asyncio.gather(*tasks)


# Demo functions

async def demonstrate_performance_tracking():
    """Demonstrate performance tracking features."""
    print("\n=== Performance Tracking Demo ===")
    
    # Initialize metrics
    metrics = APIPerformanceMetrics()
    
    try:
        print("1. Testing response time tracking...")
        
        # Simulate requests
        for i in range(10):
            start_time = time.time()
            await asyncio.sleep(0.1 + (i * 0.01))  # Varying response times
            end_time = time.time()
            
            metrics.record_request(
                endpoint=f"/api/test/{i}",
                method="GET",
                status_code=200,
                response_time=(end_time - start_time) * 1000,
                request_id=str(uuid.uuid4())
            )
        
        # Get stats
        stats = metrics.get_response_time_stats()
        print(f"   ✅ Average response time: {stats.mean:.2f}ms")
        print(f"   ✅ P95 response time: {stats.p95:.2f}ms")
        print(f"   ✅ Total requests: {stats.count}")
        
        print("\n2. Testing throughput calculation...")
        throughput = metrics.get_throughput()
        print(f"   ✅ Current throughput: {throughput:.2f} requests/second")
        
        print("\n3. Testing error rate tracking...")
        # Simulate some errors
        for i in range(3):
            metrics.record_request(
                endpoint="/api/test/error",
                method="GET",
                status_code=500,
                response_time=100.0,
                request_id=str(uuid.uuid4())
            )
        
        error_rate = metrics.get_error_rate()
        print(f"   ✅ Error rate: {error_rate:.2%}")
        
        print("\n4. Testing cache performance...")
        for _ in range(10):
            metrics.record_cache_hit()
        for _ in range(5):
            metrics.record_cache_miss()
        
        cache_hit_rate = metrics.get_cache_hit_rate()
        print(f"   ✅ Cache hit rate: {cache_hit_rate:.2%}")
        
        print("✅ Performance tracking demo completed!")
        
    finally:
        await metrics.close()


async def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring features."""
    print("\n=== Performance Monitoring Demo ===")
    
    monitor = PerformanceMonitor()
    
    try:
        print("1. Testing response time percentiles...")
        percentiles = monitor.get_response_time_percentiles()
        print(f"   ✅ P50: {percentiles['p50']:.2f}ms")
        print(f"   ✅ P95: {percentiles['p95']:.2f}ms")
        print(f"   ✅ P99: {percentiles['p99']:.2f}ms")
        
        print("\n2. Testing system health score...")
        health_score = monitor.get_system_health_score()
        print(f"   ✅ System health score: {health_score:.1f}/100")
        
        print("\n3. Testing performance recommendations...")
        recommendations = monitor.get_performance_recommendations()
        print(f"   ✅ Recommendations count: {len(recommendations)}")
        for rec in recommendations:
            print(f"      - {rec}")
        
        print("✅ Performance monitoring demo completed!")
        
    except Exception as e:
        print(f"   ❌ Monitoring demo failed: {e}")


async def demonstrate_performance_alerts():
    """Demonstrate performance alerting features."""
    print("\n=== Performance Alerts Demo ===")
    
    metrics = APIPerformanceMetrics(enable_alerts=True)
    
    try:
        print("1. Testing threshold configuration...")
        
        # Add custom threshold
        threshold = PerformanceThreshold(
            metric_type=MetricType.RESPONSE_TIME,
            endpoint="/api/slow",
            threshold_value=500.0,  # 500ms
            comparison="gt",
            alert_message="Slow endpoint detected",
            severity="warning"
        )
        metrics.add_threshold(threshold)
        print("   ✅ Custom threshold added")
        
        print("\n2. Testing alert generation...")
        
        # Simulate slow requests
        for _ in range(5):
            metrics.record_request(
                endpoint="/api/slow",
                method="GET",
                status_code=200,
                response_time=600.0,  # Above threshold
                request_id=str(uuid.uuid4())
            )
        
        # Wait for alert processing
        await asyncio.sleep(2)
        
        alerts = metrics.get_alerts()
        print(f"   ✅ Generated {len(alerts)} alerts")
        
        for alert in alerts:
            print(f"      - {alert.message} (Severity: {alert.severity})")
        
        print("✅ Performance alerts demo completed!")
        
    finally:
        await metrics.close()


async def demonstrate_fastapi_integration():
    """Demonstrate FastAPI integration."""
    print("\n=== FastAPI Integration Demo ===")
    
    # Create test app
    test_app = FastAPI()
    
    # Add performance middleware
    @test_app.middleware("http")
    async def test_performance_middleware(request: Request, call_next):
        
    """test_performance_middleware function."""
async with performance_tracking(request, Response()):
            response = await call_next(request)
            return response
    
    # Add test endpoint
    @test_app.get("/test")
    async def test_endpoint():
        
    """test_endpoint function."""
await asyncio.sleep(0.1)  # Simulate processing
        return {"message": "Test completed"}
    
    print("1. Testing FastAPI middleware integration...")
    print("   ✅ Performance middleware configured")
    
    print("\n2. Testing endpoint performance tracking...")
    print("   ✅ Endpoint performance tracking enabled")
    
    print("\n3. Testing comprehensive metrics...")
    metrics = get_performance_metrics()
    stats = metrics.get_comprehensive_stats()
    print(f"   ✅ Metrics collected: {len(stats)} categories")
    
    print("✅ FastAPI integration demo completed!")


async def demonstrate_performance_optimization():
    """Demonstrate performance optimization insights."""
    print("\n=== Performance Optimization Demo ===")
    
    metrics = APIPerformanceMetrics()
    monitor = PerformanceMonitor()
    
    try:
        print("1. Testing performance baseline...")
        
        # Simulate baseline performance
        for i in range(20):
            response_time = 100 + (i * 10)  # Increasing response times
            metrics.record_request(
                endpoint="/api/baseline",
                method="GET",
                status_code=200,
                response_time=response_time,
                request_id=str(uuid.uuid4())
            )
        
        baseline_stats = metrics.get_response_time_stats()
        print(f"   ✅ Baseline average: {baseline_stats.mean:.2f}ms")
        
        print("\n2. Testing optimization recommendations...")
        recommendations = monitor.get_performance_recommendations()
        print(f"   ✅ Optimization recommendations:")
        for rec in recommendations:
            print(f"      - {rec}")
        
        print("\n3. Testing health score tracking...")
        health_score = monitor.get_system_health_score()
        print(f"   ✅ System health: {health_score:.1f}/100")
        
        if health_score < 80:
            print("   ⚠️  Performance optimization needed")
        else:
            print("   ✅ Performance is healthy")
        
        print("✅ Performance optimization demo completed!")
        
    finally:
        await metrics.close()


async def run_comprehensive_demo():
    """Run comprehensive performance metrics demo."""
    print("🚀 Starting API Performance Metrics Demo")
    print("=" * 60)
    
    try:
        await demonstrate_performance_tracking()
        await demonstrate_performance_monitoring()
        await demonstrate_performance_alerts()
        await demonstrate_fastapi_integration()
        await demonstrate_performance_optimization()
        
        print("\n" + "=" * 60)
        print("✅ All performance metrics demos completed successfully!")
        
        print("\n📋 Next Steps:")
        print("1. Run the FastAPI app: uvicorn performance_metrics_demo:app --reload")
        print("2. Test endpoints: http://localhost:8000/docs")
        print("3. Monitor performance: http://localhost:8000/metrics/performance")
        print("4. Check system health: http://localhost:8000/metrics/system")
        print("5. View recommendations: http://localhost:8000/metrics/recommendations")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_demo()) 