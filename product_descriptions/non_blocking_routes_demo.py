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
from concurrent.futures import ThreadPoolExecutor
import random
from fastapi import FastAPI, Request, Response, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from non_blocking_routes import (
from typing import Any, List, Dict, Optional
import logging
"""
Non-Blocking Routes Demo

This demo showcases:
- Non-blocking route patterns and best practices
- Connection pooling for databases and external APIs
- Background task processing
- Circuit breaker patterns
- Performance comparisons between blocking and non-blocking operations
- Real-world examples and use cases
- Error handling and recovery strategies
"""



    NonBlockingRouteManager, DatabaseConnectionPool, RedisConnectionPool,
    HTTPConnectionPool, BackgroundTaskManager, CircuitBreaker,
    non_blocking_route, async_database_operation, async_external_api,
    background_task, BlockingOperationError, OperationType
)


# Pydantic models for demo
class PerformanceTest(BaseModel):
    """Performance test configuration."""
    operation_type: str = Field(..., description="Type of operation to test")
    iterations: int = Field(100, description="Number of iterations")
    concurrent_requests: int = Field(10, description="Number of concurrent requests")
    timeout: float = Field(30.0, description="Timeout in seconds")


class ComparisonResult(BaseModel):
    """Performance comparison result."""
    operation_type: str
    blocking_time: float
    non_blocking_time: float
    improvement_percentage: float
    concurrent_requests: int
    success_rate: float


class RouteExample(BaseModel):
    """Route example configuration."""
    name: str
    description: str
    endpoint: str
    method: str
    blocking_equivalent: str
    benefits: List[str]


# Demo FastAPI application
app = FastAPI(
    title="Non-Blocking Routes Demo",
    description="Comprehensive demo of non-blocking route patterns",
    version="1.0.0"
)

# Initialize route manager
route_manager = NonBlockingRouteManager()


@app.on_event("startup")
async def startup_event():
    """Initialize demo services."""
    # Initialize with mock URLs for demo
    await route_manager.initialize_pools(
        database_url="postgresql://demo:demo@localhost/demo",
        redis_url="redis://localhost:6379"
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup demo services."""
    await route_manager.shutdown()


# Demo service classes

class DemoProductService:
    """Demo product service with blocking and non-blocking examples."""
    
    def __init__(self) -> Any:
        self.mock_products = {
            f"prod_{i}": {
                "id": f"prod_{i}",
                "name": f"Product {i}",
                "price": random.uniform(10.0, 1000.0),
                "category": random.choice(["electronics", "clothing", "books", "home"])
            }
            for i in range(1, 101)
        }
    
    # Blocking version (for comparison)
    def get_product_blocking(self, product_id: str) -> Dict[str, Any]:
        """Blocking version - simulates slow database query."""
        time.sleep(0.1)  # Simulate blocking database call
        return self.mock_products.get(product_id)
    
    # Non-blocking version
    @async_database_operation()
    async def get_product_async(self, product_id: str) -> Dict[str, Any]:
        """Non-blocking version - async database query."""
        await asyncio.sleep(0.1)  # Simulate async database call
        return self.mock_products.get(product_id)
    
    @async_database_operation()
    async def get_products_batch_async(self, product_ids: List[str]) -> List[Dict[str, Any]]:
        """Get multiple products asynchronously."""
        await asyncio.sleep(0.05 * len(product_ids))  # Simulate batch query
        return [self.mock_products.get(pid) for pid in product_ids if pid in self.mock_products]
    
    @async_external_api()
    async def get_product_reviews_async(self, product_id: str) -> Dict[str, Any]:
        """Get product reviews from external API."""
        await asyncio.sleep(0.2)  # Simulate external API call
        return {
            "product_id": product_id,
            "reviews": [
                {"rating": random.randint(1, 5), "comment": f"Review {i}"}
                for i in range(random.randint(1, 10))
            ]
        }
    
    @background_task()
    async def update_product_analytics_async(self, product_id: str) -> Dict[str, Any]:
        """Update product analytics in background."""
        await asyncio.sleep(2.0)  # Simulate heavy computation
        return {
            "product_id": product_id,
            "analytics_updated": True,
            "timestamp": time.time()
        }


class DemoUserService:
    """Demo user service with blocking and non-blocking examples."""
    
    def __init__(self) -> Any:
        self.mock_users = {
            f"user_{i}": {
                "id": f"user_{i}",
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "preferences": {"theme": "dark", "language": "en"}
            }
            for i in range(1, 51)
        }
    
    # Blocking version
    def get_user_blocking(self, user_id: str) -> Dict[str, Any]:
        """Blocking version - simulates slow database query."""
        time.sleep(0.15)  # Simulate blocking database call
        return self.mock_users.get(user_id)
    
    # Non-blocking version
    @async_database_operation()
    async def get_user_async(self, user_id: str) -> Dict[str, Any]:
        """Non-blocking version - async database query."""
        await asyncio.sleep(0.15)  # Simulate async database call
        return self.mock_users.get(user_id)
    
    @async_external_api()
    async def validate_email_async(self, email: str) -> Dict[str, Any]:
        """Validate email with external service."""
        await asyncio.sleep(0.3)  # Simulate external API call
        return {
            "email": email,
            "valid": random.choice([True, False]),
            "score": random.uniform(0.0, 1.0)
        }


class DemoCacheService:
    """Demo cache service with non-blocking operations."""
    
    def __init__(self) -> Any:
        self.mock_cache = {}
    
    async def get_cached_data(self, key: str) -> Optional[str]:
        """Get data from cache."""
        await asyncio.sleep(0.01)  # Simulate cache access
        return self.mock_cache.get(key)
    
    async def set_cached_data(self, key: str, value: str, expire: int = 3600) -> bool:
        """Set data in cache."""
        await asyncio.sleep(0.01)  # Simulate cache write
        self.mock_cache[key] = value
        return True
    
    async def invalidate_cache(self, pattern: str) -> int:
        """Invalidate cache keys."""
        await asyncio.sleep(0.01)  # Simulate cache invalidation
        keys_to_delete = [k for k in self.mock_cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self.mock_cache[key]
        return len(keys_to_delete)


# Global service instances
product_service = DemoProductService()
user_service = DemoUserService()
cache_service = DemoCacheService()


# Demo routes

@app.get("/")
async def root():
    """Root endpoint with demo information."""
    return {
        "message": "Non-Blocking Routes Demo",
        "version": "1.0.0",
        "endpoints": [
            "/demo/products/{product_id}",
            "/demo/products/batch",
            "/demo/users/{user_id}",
            "/demo/performance/comparison",
            "/demo/performance/test",
            "/demo/examples",
            "/demo/circuit-breaker",
            "/demo/background-tasks"
        ],
        "examples": [
            {
                "name": "Product Retrieval",
                "blocking": "Synchronous database query blocks the event loop",
                "non_blocking": "Asynchronous database query with connection pooling"
            },
            {
                "name": "External API Calls",
                "blocking": "HTTP requests block until response received",
                "non_blocking": "Concurrent HTTP requests with timeout and retry"
            },
            {
                "name": "Background Processing",
                "blocking": "Heavy computation blocks the request",
                "non_blocking": "Background task processing with status tracking"
            }
        ]
    }


@app.get("/demo/products/{product_id}")
@non_blocking_route()
async def get_product_demo(product_id: str):
    """Demo: Get product with non-blocking operations."""
    start_time = time.time()
    
    # Check cache first
    cache_key = f"product:{product_id}"
    cached_product = await cache_service.get_cached_data(cache_key)
    
    if cached_product:
        return {
            "product": cached_product,
            "source": "cache",
            "response_time": time.time() - start_time
        }
    
    # Get from database asynchronously
    product = await product_service.get_product_async(product_id)
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    # Cache the result
    await cache_service.set_cached_data(cache_key, str(product))
    
    # Get reviews in background (non-blocking)
    review_task = asyncio.create_task(
        product_service.get_product_reviews_async(product_id)
    )
    
    return {
        "product": product,
        "source": "database",
        "response_time": time.time() - start_time,
        "reviews_pending": True,
        "review_task_id": str(review_task)
    }


@app.post("/demo/products/batch")
@non_blocking_route()
async def get_products_batch_demo(product_ids: List[str]):
    """Demo: Get multiple products with concurrent operations."""
    start_time = time.time()
    
    # Get products from database
    products = await product_service.get_products_batch_async(product_ids)
    
    # Get reviews for all products concurrently
    review_tasks = [
        product_service.get_product_reviews_async(product["id"])
        for product in products
    ]
    
    reviews = await asyncio.gather(*review_tasks, return_exceptions=True)
    
    # Combine results
    for product, review in zip(products, reviews):
        if isinstance(review, dict):
            product["reviews"] = review.get("reviews", [])
        else:
            product["reviews"] = []
    
    return {
        "products": products,
        "count": len(products),
        "response_time": time.time() - start_time,
        "concurrent_operations": len(review_tasks)
    }


@app.get("/demo/users/{user_id}")
@non_blocking_route()
async def get_user_demo(user_id: str):
    """Demo: Get user with background email validation."""
    start_time = time.time()
    
    # Get user from database
    user = await user_service.get_user_async(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Start email validation in background
    email_validation_task = asyncio.create_task(
        user_service.validate_email_async(user["email"])
    )
    
    return {
        "user": user,
        "response_time": time.time() - start_time,
        "email_validation": {
            "status": "pending",
            "task_id": str(email_validation_task)
        }
    }


@app.post("/demo/performance/comparison")
async def performance_comparison_demo(test: PerformanceTest):
    """Demo: Compare blocking vs non-blocking performance."""
    results = []
    
    # Test product retrieval
    if test.operation_type == "product_retrieval":
        # Blocking version
        blocking_start = time.time()
        blocking_results = []
        for i in range(test.iterations):
            product_id = f"prod_{random.randint(1, 100)}"
            result = product_service.get_product_blocking(product_id)
            blocking_results.append(result)
        blocking_time = time.time() - blocking_start
        
        # Non-blocking version
        non_blocking_start = time.time()
        tasks = []
        for i in range(test.iterations):
            product_id = f"prod_{random.randint(1, 100)}"
            task = product_service.get_product_async(product_id)
            tasks.append(task)
        
        non_blocking_results = await asyncio.gather(*tasks)
        non_blocking_time = time.time() - non_blocking_start
        
        improvement = ((blocking_time - non_blocking_time) / blocking_time) * 100
        
        results.append(ComparisonResult(
            operation_type="product_retrieval",
            blocking_time=blocking_time,
            non_blocking_time=non_blocking_time,
            improvement_percentage=improvement,
            concurrent_requests=test.iterations,
            success_rate=100.0
        ))
    
    # Test external API calls
    elif test.operation_type == "external_api":
        # Blocking version (simulated)
        blocking_start = time.time()
        for i in range(test.iterations):
            await asyncio.sleep(0.2)  # Simulate blocking API call
        blocking_time = time.time() - blocking_start
        
        # Non-blocking version
        non_blocking_start = time.time()
        tasks = [asyncio.sleep(0.2) for _ in range(test.iterations)]
        await asyncio.gather(*tasks)
        non_blocking_time = time.time() - non_blocking_start
        
        improvement = ((blocking_time - non_blocking_time) / blocking_time) * 100
        
        results.append(ComparisonResult(
            operation_type="external_api",
            blocking_time=blocking_time,
            non_blocking_time=non_blocking_time,
            improvement_percentage=improvement,
            concurrent_requests=test.iterations,
            success_rate=100.0
        ))
    
    return {
        "test_configuration": test.dict(),
        "results": [result.dict() for result in results],
        "summary": {
            "total_tests": len(results),
            "average_improvement": sum(r.improvement_percentage for r in results) / len(results) if results else 0
        }
    }


@app.post("/demo/performance/test")
async def performance_test_demo(test: PerformanceTest):
    """Demo: Run performance test with concurrent requests."""
    async def single_request():
        """Simulate a single request."""
        try:
            if test.operation_type == "product_retrieval":
                product_id = f"prod_{random.randint(1, 100)}"
                await product_service.get_product_async(product_id)
            elif test.operation_type == "external_api":
                await product_service.get_product_reviews_async(f"prod_{random.randint(1, 100)}")
            elif test.operation_type == "mixed":
                # Mixed operations
                product_id = f"prod_{random.randint(1, 100)}"
                await asyncio.gather(
                    product_service.get_product_async(product_id),
                    product_service.get_product_reviews_async(product_id),
                    user_service.get_user_async(f"user_{random.randint(1, 50)}")
                )
            return True
        except Exception as e:
            print(f"Request failed: {e}")
            return False
    
    start_time = time.time()
    
    # Create concurrent requests
    tasks = [single_request() for _ in range(test.iterations)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    end_time = time.time()
    
    # Calculate statistics
    successful_requests = sum(1 for r in results if r is True)
    failed_requests = len(results) - successful_requests
    success_rate = (successful_requests / len(results)) * 100
    
    return {
        "test_configuration": test.dict(),
        "results": {
            "total_requests": len(results),
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": success_rate,
            "total_time": end_time - start_time,
            "requests_per_second": len(results) / (end_time - start_time),
            "average_response_time": (end_time - start_time) / len(results)
        }
    }


@app.get("/demo/examples")
async def get_route_examples():
    """Get examples of blocking vs non-blocking routes."""
    examples = [
        RouteExample(
            name="Product Retrieval",
            description="Retrieve product information from database",
            endpoint="/demo/products/{product_id}",
            method="GET",
            blocking_equivalent="Synchronous database query with sleep(0.1)",
            benefits=[
                "Non-blocking database operations",
                "Connection pooling",
                "Cache integration",
                "Background review fetching"
            ]
        ),
        RouteExample(
            name="Batch Product Retrieval",
            description="Retrieve multiple products with concurrent operations",
            endpoint="/demo/products/batch",
            method="POST",
            blocking_equivalent="Sequential database queries",
            benefits=[
                "Concurrent database operations",
                "Parallel external API calls",
                "Reduced total response time",
                "Better resource utilization"
            ]
        ),
        RouteExample(
            name="User Retrieval with Background Processing",
            description="Get user data with background email validation",
            endpoint="/demo/users/{user_id}",
            method="GET",
            blocking_equivalent="Synchronous user query + email validation",
            benefits=[
                "Immediate response with user data",
                "Background email validation",
                "Non-blocking external API calls",
                "Better user experience"
            ]
        ),
        RouteExample(
            name="Performance Testing",
            description="Compare blocking vs non-blocking performance",
            endpoint="/demo/performance/comparison",
            method="POST",
            blocking_equivalent="Sequential operation execution",
            benefits=[
                "Quantified performance improvements",
                "Concurrent operation testing",
                "Real-world performance metrics",
                "Optimization insights"
            ]
        )
    ]
    
    return {
        "examples": [example.dict() for example in examples],
        "total_examples": len(examples)
    }


@app.get("/demo/circuit-breaker")
async def circuit_breaker_demo():
    """Demo: Circuit breaker pattern."""
    # Create circuit breaker for external API
    circuit_breaker = route_manager.get_circuit_breaker("external_api")
    
    async def unreliable_external_api():
        """Simulate unreliable external API."""
        if random.random() < 0.3:  # 30% failure rate
            raise Exception("External API error")
        await asyncio.sleep(0.1)
        return {"status": "success"}
    
    results = []
    for i in range(10):
        try:
            result = await circuit_breaker.call(unreliable_external_api)
            results.append({"request": i + 1, "status": "success", "result": result})
        except Exception as e:
            results.append({"request": i + 1, "status": "failed", "error": str(e)})
    
    return {
        "circuit_breaker_state": circuit_breaker.state,
        "failure_count": circuit_breaker.failure_count,
        "results": results,
        "explanation": {
            "CLOSED": "Circuit is closed, requests are allowed",
            "OPEN": "Circuit is open, requests are blocked",
            "HALF_OPEN": "Circuit is half-open, limited requests allowed"
        }
    }


@app.post("/demo/background-tasks")
@non_blocking_route()
async def create_background_task_demo(product_id: str):
    """Demo: Create background task."""
    task_id = await product_service.update_product_analytics_async(product_id)
    return {
        "task_id": task_id,
        "status": "started",
        "message": "Product analytics update started in background"
    }


@app.get("/demo/background-tasks/{task_id}")
async def get_background_task_status_demo(task_id: str):
    """Demo: Get background task status."""
    try:
        result = await route_manager.task_manager.wait_for_task(task_id, timeout=1.0)
        return {
            "task_id": task_id,
            "status": "completed",
            "result": result
        }
    except asyncio.TimeoutError:
        return {
            "task_id": task_id,
            "status": "running",
            "message": "Task is still running"
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e)
        }


@app.get("/demo/health")
async def health_check_demo():
    """Demo: Health check with system metrics."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "pools": {
            "database": route_manager.db_pool is not None,
            "redis": route_manager.redis_pool is not None,
            "http": route_manager.http_pool is not None
        },
        "active_tasks": len(route_manager.task_manager.tasks),
        "circuit_breakers": {
            name: breaker.state
            for name, breaker in route_manager.circuit_breakers.items()
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
    }


# Demo functions

async def demonstrate_blocking_vs_non_blocking():
    """Demonstrate the difference between blocking and non-blocking operations."""
    print("\n=== Blocking vs Non-Blocking Demo ===")
    
    # Test 1: Product retrieval
    print("1. Testing product retrieval...")
    
    # Blocking version
    blocking_start = time.time()
    for i in range(10):
        product_service.get_product_blocking(f"prod_{i+1}")
    blocking_time = time.time() - blocking_start
    
    # Non-blocking version
    non_blocking_start = time.time()
    tasks = [product_service.get_product_async(f"prod_{i+1}") for i in range(10)]
    await asyncio.gather(*tasks)
    non_blocking_time = time.time() - non_blocking_start
    
    improvement = ((blocking_time - non_blocking_time) / blocking_time) * 100
    
    print(f"   ✅ Blocking time: {blocking_time:.2f}s")
    print(f"   ✅ Non-blocking time: {non_blocking_time:.2f}s")
    print(f"   ✅ Improvement: {improvement:.1f}%")
    
    # Test 2: External API calls
    print("\n2. Testing external API calls...")
    
    # Simulate blocking API calls
    blocking_start = time.time()
    for i in range(5):
        await asyncio.sleep(0.2)  # Simulate blocking API call
    blocking_time = time.time() - blocking_start
    
    # Non-blocking API calls
    non_blocking_start = time.time()
    tasks = [asyncio.sleep(0.2) for _ in range(5)]
    await asyncio.gather(*tasks)
    non_blocking_time = time.time() - non_blocking_start
    
    improvement = ((blocking_time - non_blocking_time) / blocking_time) * 100
    
    print(f"   ✅ Blocking time: {blocking_time:.2f}s")
    print(f"   ✅ Non-blocking time: {non_blocking_time:.2f}s")
    print(f"   ✅ Improvement: {improvement:.1f}%")
    
    print("✅ Blocking vs non-blocking demo completed!")


async def demonstrate_connection_pooling():
    """Demonstrate connection pooling benefits."""
    print("\n=== Connection Pooling Demo ===")
    
    # Simulate connection pool usage
    print("1. Testing connection pool efficiency...")
    
    start_time = time.time()
    
    # Simulate multiple concurrent database operations
    async def db_operation(operation_id: int):
        
    """db_operation function."""
await asyncio.sleep(0.1)  # Simulate database operation
        return f"Operation {operation_id} completed"
    
    # Run 20 concurrent operations
    tasks = [db_operation(i) for i in range(20)]
    results = await asyncio.gather(*tasks)
    
    end_time = time.time()
    
    print(f"   ✅ Total time: {end_time - start_time:.2f}s")
    print(f"   ✅ Operations completed: {len(results)}")
    print(f"   ✅ Average time per operation: {(end_time - start_time) / len(results):.3f}s")
    
    print("✅ Connection pooling demo completed!")


async def demonstrate_circuit_breaker():
    """Demonstrate circuit breaker pattern."""
    print("\n=== Circuit Breaker Demo ===")
    
    # Create circuit breaker
    circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)
    
    async def unreliable_service():
        """Simulate unreliable service."""
        if random.random() < 0.4:  # 40% failure rate
            raise Exception("Service error")
        await asyncio.sleep(0.1)
        return "Service response"
    
    print("1. Testing circuit breaker behavior...")
    
    for i in range(10):
        try:
            result = await circuit_breaker.call(unreliable_service)
            print(f"   Request {i+1}: Success - {result}")
        except Exception as e:
            print(f"   Request {i+1}: Failed - {e}")
        
        print(f"   Circuit state: {circuit_breaker.state}")
        
        if i == 5:
            print("   Waiting for circuit to recover...")
            await asyncio.sleep(6)  # Wait for recovery timeout
    
    print("✅ Circuit breaker demo completed!")


async def demonstrate_background_tasks():
    """Demonstrate background task processing."""
    print("\n=== Background Tasks Demo ===")
    
    # Create background task manager
    task_manager = BackgroundTaskManager(max_workers=5)
    
    async def heavy_computation(data: List[int]) -> List[int]:
        """Simulate heavy computation."""
        await asyncio.sleep(2)  # Simulate computation time
        return [x * x for x in data]
    
    print("1. Testing background task processing...")
    
    # Submit multiple background tasks
    task_ids = []
    for i in range(3):
        task_id = await route_manager.execute_in_background(
            heavy_computation, [1, 2, 3, 4, 5]
        )
        task_ids.append(task_id)
        print(f"   Submitted task {i+1}: {task_id}")
    
    # Wait for tasks to complete
    print("   Waiting for tasks to complete...")
    for task_id in task_ids:
        try:
            result = await route_manager.task_manager.wait_for_task(task_id, timeout=10.0)
            print(f"   Task {task_id}: Completed - {result}")
        except Exception as e:
            print(f"   Task {task_id}: Failed - {e}")
    
    print("✅ Background tasks demo completed!")


async def run_comprehensive_demo():
    """Run comprehensive non-blocking routes demo."""
    print("🚀 Starting Non-Blocking Routes Demo")
    print("=" * 60)
    
    try:
        await demonstrate_blocking_vs_non_blocking()
        await demonstrate_connection_pooling()
        await demonstrate_circuit_breaker()
        await demonstrate_background_tasks()
        
        print("\n" + "=" * 60)
        print("✅ All non-blocking routes demos completed successfully!")
        
        print("\n📋 Next Steps:")
        print("1. Run the FastAPI app: uvicorn non_blocking_routes_demo:app --reload")
        print("2. Test endpoints: http://localhost:8000/docs")
        print("3. Try performance comparison: POST /demo/performance/comparison")
        print("4. Test background tasks: POST /demo/background-tasks")
        print("5. Check circuit breaker: GET /demo/circuit-breaker")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_demo()) 