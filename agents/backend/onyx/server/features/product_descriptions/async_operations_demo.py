from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import uuid
import json
from typing import Dict, List, Any, Optional
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from async_database_api_operations import (
from typing import Any, List, Dict, Optional
import logging
"""
Dedicated Async Functions for Database and API Operations Demo

This demo showcases:
- Async database operations (PostgreSQL, SQLite, Redis)
- Async external API calls with connection pooling
- Concurrent operation execution
- Performance monitoring and optimization
- Error handling and retry mechanisms
- Real-world usage patterns
"""



    AsyncOperationOrchestrator, AsyncPostgreSQLManager, AsyncSQLiteManager,
    AsyncRedisManager, AsyncAPIManager, OperationContext, OperationResult,
    OperationType, DatabaseType, execute_with_retry, get_database_connection,
    get_api_session
)


# Pydantic models for demo
class UserData(BaseModel):
    """User data model for demo operations."""
    id: str
    name: str
    email: str
    age: int
    preferences: Dict[str, Any] = Field(default_factory=dict)


class ProductData(BaseModel):
    """Product data model for demo operations."""
    id: str
    name: str
    price: float
    category: str
    description: str
    stock: int


class DemoOperationRequest(BaseModel):
    """Request model for demo operations."""
    operation_type: str  # "database" or "api"
    operation_count: int = Field(10, description="Number of operations to perform")
    concurrent: bool = Field(True, description="Execute operations concurrently")
    include_errors: bool = Field(False, description="Include operations that might fail")


class PerformanceComparison(BaseModel):
    """Performance comparison result."""
    operation_type: str
    total_time: float
    operations_performed: int
    throughput: float
    concurrent: bool
    average_time: float


# Demo FastAPI application
app = FastAPI(
    title="Async Database and API Operations Demo",
    description="Comprehensive demo of dedicated async functions for database and API operations",
    version="1.0.0"
)

# Global orchestrator
orchestrator = AsyncOperationOrchestrator()


@app.on_event("startup")
async def startup_event():
    """Initialize demo managers on startup."""
    # Initialize database managers with demo configurations
    await orchestrator.add_database_manager(
        "demo_postgres",
        DatabaseType.POSTGRESQL,
        "postgresql://demo:demo@localhost:5432/demo_db",
        max_connections=10
    )
    
    await orchestrator.add_database_manager(
        "demo_sqlite",
        DatabaseType.SQLITE,
        "demo_cache.db",
        max_connections=5
    )
    
    await orchestrator.add_database_manager(
        "demo_redis",
        DatabaseType.REDIS,
        "redis://localhost:6379/1",
        max_connections=8
    )
    
    # Initialize API managers
    await orchestrator.add_api_manager(
        "demo_external_api",
        "https://jsonplaceholder.typicode.com",
        max_connections=15,
        timeout=30
    )
    
    await orchestrator.add_api_manager(
        "demo_internal_api",
        "http://localhost:8001",
        max_connections=10,
        timeout=20
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup demo managers on shutdown."""
    await orchestrator.close_all()


# Demo routes

@app.get("/")
async def root():
    """Root endpoint with demo information."""
    return {
        "message": "Async Database and API Operations Demo",
        "version": "1.0.0",
        "endpoints": [
            "/demo/database",
            "/demo/api",
            "/demo/batch",
            "/demo/performance",
            "/demo/retry",
            "/demo/context-managers",
            "/demo/stats",
            "/demo/health"
        ],
        "features": [
            "Async PostgreSQL operations with connection pooling",
            "Async SQLite operations for caching",
            "Async Redis operations for session storage",
            "Async external API calls with retry logic",
            "Concurrent operation execution",
            "Performance monitoring and statistics",
            "Error handling and recovery",
            "Context managers for resource management"
        ]
    }


@app.post("/demo/database")
async def demo_database_operations(request: DemoOperationRequest):
    """Demo: Async database operations."""
    start_time = time.time()
    results = []
    
    # Generate demo data
    users = [
        UserData(
            id=f"user_{i}",
            name=f"User {i}",
            email=f"user{i}@example.com",
            age=random.randint(18, 65),
            preferences={"theme": "dark", "language": "en"}
        )
        for i in range(request.operation_count)
    ]
    
    # Database operations
    operations = []
    
    for user in users:
        # PostgreSQL operation
        operations.append({
            "type": "database",
            "db_name": "demo_postgres",
            "query": """
                INSERT INTO users (id, name, email, age, preferences)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                email = EXCLUDED.email,
                age = EXCLUDED.age,
                preferences = EXCLUDED.preferences
            """,
            "params": {
                "1": user.id,
                "2": user.name,
                "3": user.email,
                "4": user.age,
                "5": json.dumps(user.preferences)
            }
        })
        
        # SQLite operation
        operations.append({
            "type": "database",
            "db_name": "demo_sqlite",
            "query": """
                INSERT OR REPLACE INTO user_cache (id, data, timestamp)
                VALUES (?, ?, ?)
            """,
            "params": {
                "1": user.id,
                "2": json.dumps(user.dict()),
                "3": int(time.time())
            }
        })
        
        # Redis operation
        operations.append({
            "type": "database",
            "db_name": "demo_redis",
            "query": f"user:{user.id}",
            "params": {
                "operation": "set",
                "key": f"user:{user.id}",
                "value": json.dumps(user.dict())
            }
        })
    
    if request.concurrent:
        # Execute operations concurrently
        batch_results = await orchestrator.execute_batch_operations(operations)
        results.extend(batch_results)
    else:
        # Execute operations sequentially
        for operation in operations:
            if operation["type"] == "database":
                result = await orchestrator.execute_database_operation(
                    operation["db_name"],
                    operation["query"],
                    operation.get("params")
                )
                results.append(result)
            
            if not request.concurrent:
                await asyncio.sleep(0.01)  # Small delay between operations
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful_ops = sum(1 for r in results if r.success)
    failed_ops = len(results) - successful_ops
    avg_time = sum(r.execution_time for r in results) / len(results) if results else 0
    
    return {
        "operation_type": "database",
        "total_time": total_time,
        "operations_performed": len(results),
        "successful_operations": successful_ops,
        "failed_operations": failed_ops,
        "throughput": len(results) / total_time if total_time > 0 else 0,
        "average_execution_time": avg_time,
        "concurrent": request.concurrent,
        "databases_used": ["postgresql", "sqlite", "redis"]
    }


@app.post("/demo/api")
async def demo_api_operations(request: DemoOperationRequest):
    """Demo: Async API operations."""
    start_time = time.time()
    results = []
    
    # Generate demo API operations
    operations = []
    
    for i in range(request.operation_count):
        # GET request
        operations.append({
            "type": "api",
            "api_name": "demo_external_api",
            "method": "GET",
            "endpoint": f"/posts/{i + 1}",
            "params": None
        })
        
        # POST request
        post_data = {
            "title": f"Demo Post {i + 1}",
            "body": f"This is demo post content {i + 1}",
            "userId": random.randint(1, 10)
        }
        operations.append({
            "type": "api",
            "api_name": "demo_external_api",
            "method": "POST",
            "endpoint": "/posts",
            "data": post_data
        })
        
        # PUT request
        put_data = {
            "id": i + 1,
            "title": f"Updated Post {i + 1}",
            "body": f"Updated content {i + 1}",
            "userId": random.randint(1, 10)
        }
        operations.append({
            "type": "api",
            "api_name": "demo_external_api",
            "method": "PUT",
            "endpoint": f"/posts/{i + 1}",
            "data": put_data
        })
    
    if request.concurrent:
        # Execute operations concurrently
        batch_results = await orchestrator.execute_batch_operations(operations)
        results.extend(batch_results)
    else:
        # Execute operations sequentially
        for operation in operations:
            result = await orchestrator.execute_api_operation(
                operation["api_name"],
                operation["method"],
                operation["endpoint"],
                operation.get("data"),
                operation.get("params")
            )
            results.append(result)
            
            if not request.concurrent:
                await asyncio.sleep(0.05)  # Small delay between API calls
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful_ops = sum(1 for r in results if r.success)
    failed_ops = len(results) - successful_ops
    avg_time = sum(r.execution_time for r in results) / len(results) if results else 0
    
    return {
        "operation_type": "api",
        "total_time": total_time,
        "operations_performed": len(results),
        "successful_operations": successful_ops,
        "failed_operations": failed_ops,
        "throughput": len(results) / total_time if total_time > 0 else 0,
        "average_execution_time": avg_time,
        "concurrent": request.concurrent,
        "api_endpoints_used": ["/posts", "/users", "/comments"]
    }


@app.post("/demo/batch")
async def demo_batch_operations(request: DemoOperationRequest):
    """Demo: Mixed batch operations (database + API)."""
    start_time = time.time()
    
    # Generate mixed operations
    operations = []
    
    for i in range(request.operation_count):
        # Database operations
        user_data = {
            "id": f"batch_user_{i}",
            "name": f"Batch User {i}",
            "email": f"batch{i}@example.com",
            "age": random.randint(18, 65)
        }
        
        # PostgreSQL insert
        operations.append({
            "type": "database",
            "db_name": "demo_postgres",
            "query": """
                INSERT INTO users (id, name, email, age)
                VALUES ($1, $2, $3, $4)
            """,
            "params": {
                "1": user_data["id"],
                "2": user_data["name"],
                "3": user_data["email"],
                "4": user_data["age"]
            }
        })
        
        # Redis cache
        operations.append({
            "type": "database",
            "db_name": "demo_redis",
            "query": f"user:{user_data['id']}",
            "params": {
                "operation": "set",
                "key": f"user:{user_data['id']}",
                "value": json.dumps(user_data)
            }
        })
        
        # API call to external service
        operations.append({
            "type": "api",
            "api_name": "demo_external_api",
            "method": "GET",
            "endpoint": f"/users/{random.randint(1, 10)}"
        })
        
        # API call to internal service
        operations.append({
            "type": "api",
            "api_name": "demo_internal_api",
            "method": "POST",
            "endpoint": "/user/activity",
            "data": {
                "user_id": user_data["id"],
                "action": "login",
                "timestamp": int(time.time())
            }
        })
    
    # Execute all operations concurrently
    results = await orchestrator.execute_batch_operations(operations)
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful_ops = sum(1 for r in results if r.success)
    failed_ops = len(results) - successful_ops
    avg_time = sum(r.execution_time for r in results) / len(results) if results else 0
    
    # Categorize results
    db_ops = [r for r in results if r.operation_context.operation_type in [
        OperationType.DATABASE_READ, OperationType.DATABASE_WRITE, OperationType.CACHE_GET, OperationType.CACHE_SET
    ]]
    api_ops = [r for r in results if r.operation_context.operation_type in [
        OperationType.API_GET, OperationType.API_POST, OperationType.API_PUT, OperationType.API_DELETE
    ]]
    
    return {
        "operation_type": "batch",
        "total_time": total_time,
        "operations_performed": len(results),
        "successful_operations": successful_ops,
        "failed_operations": failed_ops,
        "throughput": len(results) / total_time if total_time > 0 else 0,
        "average_execution_time": avg_time,
        "concurrent": True,
        "breakdown": {
            "database_operations": len(db_ops),
            "api_operations": len(api_ops),
            "database_success_rate": sum(1 for r in db_ops if r.success) / len(db_ops) if db_ops else 0,
            "api_success_rate": sum(1 for r in api_ops if r.success) / len(api_ops) if api_ops else 0
        }
    }


@app.post("/demo/retry")
async def demo_retry_mechanism():
    """Demo: Retry mechanism for failed operations."""
    results = []
    
    # Simulate operations that might fail
    async def failing_database_operation():
        """Simulate a database operation that might fail."""
        if random.random() < 0.7:  # 70% chance of failure
            raise Exception("Simulated database error")
        
        return OperationResult(
            success=True,
            data={"message": "Database operation successful"},
            execution_time=0.1,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.DATABASE_READ
            )
        )
    
    async def failing_api_operation():
        """Simulate an API operation that might fail."""
        if random.random() < 0.6:  # 60% chance of failure
            raise Exception("Simulated API error")
        
        return OperationResult(
            success=True,
            data={"message": "API operation successful"},
            execution_time=0.2,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.API_GET
            )
        )
    
    # Execute operations with retry
    for i in range(5):
        result = await execute_with_retry(
            failing_database_operation,
            max_retries=3,
            delay=0.5
        )
        results.append(result)
    
    for i in range(5):
        result = await execute_with_retry(
            failing_api_operation,
            max_retries=2,
            delay=1.0
        )
        results.append(result)
    
    successful_ops = sum(1 for r in results if r.success)
    failed_ops = len(results) - successful_ops
    
    return {
        "retry_demo": True,
        "operations_performed": len(results),
        "successful_operations": successful_ops,
        "failed_operations": failed_ops,
        "success_rate": successful_ops / len(results) if results else 0,
        "results": [
            {
                "operation_id": r.operation_context.operation_id,
                "success": r.success,
                "error": r.error,
                "execution_time": r.execution_time
            }
            for r in results
        ]
    }


@app.get("/demo/context-managers")
async def demo_context_managers():
    """Demo: Context managers for resource management."""
    results = []
    
    # Database context manager demo
    async with get_database_connection("demo_postgres") as db_manager:
        # Execute multiple operations with the same connection
        for i in range(3):
            result = await db_manager.execute_query(
                "SELECT COUNT(*) as user_count FROM users WHERE age > $1",
                {"1": 25}
            )
            results.append(result)
    
    # API context manager demo
    async with get_api_session("demo_external_api") as api_manager:
        # Execute multiple API calls with the same session
        for i in range(3):
            result = await api_manager.get(f"/posts/{i + 1}")
            results.append(result)
    
    successful_ops = sum(1 for r in results if r.success)
    
    return {
        "context_manager_demo": True,
        "operations_performed": len(results),
        "successful_operations": successful_ops,
        "success_rate": successful_ops / len(results) if results else 0,
        "context_managers_used": ["database_connection", "api_session"]
    }


@app.post("/demo/performance")
async def demo_performance_comparison(request: DemoOperationRequest):
    """Demo: Performance comparison between different operation types."""
    comparisons = []
    
    # Test database operations
    db_start = time.time()
    db_result = await demo_database_operations(request)
    db_time = time.time() - db_start
    
    comparisons.append(PerformanceComparison(
        operation_type="database",
        total_time=db_time,
        operations_performed=db_result["operations_performed"],
        throughput=db_result["throughput"],
        concurrent=request.concurrent,
        average_time=db_result["average_execution_time"]
    ))
    
    # Test API operations
    api_start = time.time()
    api_result = await demo_api_operations(request)
    api_time = time.time() - api_start
    
    comparisons.append(PerformanceComparison(
        operation_type="api",
        total_time=api_time,
        operations_performed=api_result["operations_performed"],
        throughput=api_result["throughput"],
        concurrent=request.concurrent,
        average_time=api_result["average_execution_time"]
    ))
    
    # Test batch operations
    batch_start = time.time()
    batch_result = await demo_batch_operations(request)
    batch_time = time.time() - batch_start
    
    comparisons.append(PerformanceComparison(
        operation_type="batch",
        total_time=batch_time,
        operations_performed=batch_result["operations_performed"],
        throughput=batch_result["throughput"],
        concurrent=True,
        average_time=batch_result["average_execution_time"]
    ))
    
    # Sort by throughput
    comparisons.sort(key=lambda x: x.throughput, reverse=True)
    
    return {
        "performance_comparison": True,
        "test_configuration": request.dict(),
        "results": [comp.dict() for comp in comparisons],
        "performance_ranking": [
            {
                "rank": i + 1,
                "operation_type": comp.operation_type,
                "throughput": comp.throughput,
                "average_time": comp.average_time
            }
            for i, comp in enumerate(comparisons)
        ],
        "best_performer": comparisons[0].operation_type if comparisons else None
    }


@app.get("/demo/stats")
async def get_demo_stats():
    """Get comprehensive statistics for all operations."""
    return orchestrator.get_performance_stats()


@app.get("/demo/health")
async def get_demo_health():
    """Get health status of all managers."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "managers": {
            "database_managers": len(orchestrator.database_managers),
            "api_managers": len(orchestrator.api_managers)
        },
        "performance": orchestrator.performance_stats,
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
    }
    
    return health_status


# Demo functions

async def demonstrate_database_operations():
    """Demonstrate various database operations."""
    print("\n=== Database Operations Demo ===")
    
    # PostgreSQL operations
    print("   📊 PostgreSQL Operations:")
    result = await orchestrator.execute_database_operation(
        "demo_postgres",
        "SELECT COUNT(*) as user_count FROM users",
        None
    )
    print(f"   - User count query: {result.success} ({result.execution_time:.3f}s)")
    
    # SQLite operations
    print("   💾 SQLite Operations:")
    result = await orchestrator.execute_database_operation(
        "demo_sqlite",
        "SELECT COUNT(*) as cache_entries FROM user_cache",
        None
    )
    print(f"   - Cache entries query: {result.success} ({result.execution_time:.3f}s)")
    
    # Redis operations
    print("   🔴 Redis Operations:")
    result = await orchestrator.execute_database_operation(
        "demo_redis",
        "user:demo_123",
        {"operation": "set", "key": "user:demo_123", "value": '{"name": "Demo User"}'}
    )
    print(f"   - Cache set operation: {result.success} ({result.execution_time:.3f}s)")


async def demonstrate_api_operations():
    """Demonstrate various API operations."""
    print("\n=== API Operations Demo ===")
    
    # GET request
    print("   🌐 GET Request:")
    result = await orchestrator.execute_api_operation(
        "demo_external_api",
        "GET",
        "/posts/1"
    )
    print(f"   - Get post: {result.success} ({result.execution_time:.3f}s)")
    
    # POST request
    print("   📝 POST Request:")
    post_data = {
        "title": "Demo Post",
        "body": "This is a demo post",
        "userId": 1
    }
    result = await orchestrator.execute_api_operation(
        "demo_external_api",
        "POST",
        "/posts",
        post_data
    )
    print(f"   - Create post: {result.success} ({result.execution_time:.3f}s)")
    
    # PUT request
    print("   ✏️ PUT Request:")
    put_data = {
        "id": 1,
        "title": "Updated Demo Post",
        "body": "Updated content",
        "userId": 1
    }
    result = await orchestrator.execute_api_operation(
        "demo_external_api",
        "PUT",
        "/posts/1",
        put_data
    )
    print(f"   - Update post: {result.success} ({result.execution_time:.3f}s)")


async def demonstrate_concurrent_operations():
    """Demonstrate concurrent operation execution."""
    print("\n=== Concurrent Operations Demo ===")
    
    # Generate operations
    operations = []
    
    # Database operations
    for i in range(5):
        operations.append({
            "type": "database",
            "db_name": "demo_postgres",
            "query": "SELECT $1 as test_value",
            "params": {"1": f"value_{i}"}
        })
    
    # API operations
    for i in range(5):
        operations.append({
            "type": "api",
            "api_name": "demo_external_api",
            "method": "GET",
            "endpoint": f"/posts/{i + 1}"
        })
    
    # Execute concurrently
    start_time = time.time()
    results = await orchestrator.execute_batch_operations(operations)
    total_time = time.time() - start_time
    
    successful_ops = sum(1 for r in results if r.success)
    
    print(f"   ⚡ Concurrent execution: {len(results)} operations in {total_time:.3f}s")
    print(f"   ✅ Successful: {successful_ops}/{len(results)}")
    print(f"   📈 Throughput: {len(results)/total_time:.2f} ops/sec")


async def demonstrate_retry_mechanism():
    """Demonstrate retry mechanism."""
    print("\n=== Retry Mechanism Demo ===")
    
    async def failing_operation():
        """Simulate a failing operation."""
        if random.random() < 0.8:  # 80% chance of failure
            raise Exception("Simulated failure")
        
        return OperationResult(
            success=True,
            data={"message": "Success after retries"},
            execution_time=0.1,
            operation_context=OperationContext(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType.DATABASE_READ
            )
        )
    
    # Execute with retry
    result = await execute_with_retry(
        failing_operation,
        max_retries=3,
        delay=0.5
    )
    
    print(f"   🔄 Retry mechanism: {result.success}")
    if not result.success:
        print(f"   ❌ Final error: {result.error}")


async def demonstrate_context_managers():
    """Demonstrate context managers."""
    print("\n=== Context Managers Demo ===")
    
    # Database context manager
    print("   🗄️ Database Context Manager:")
    async with get_database_connection("demo_postgres") as db_manager:
        result = await db_manager.execute_query("SELECT 1 as test")
        print(f"   - Connection test: {result.success}")
    
    # API context manager
    print("   🌐 API Context Manager:")
    async with get_api_session("demo_external_api") as api_manager:
        result = await api_manager.get("/posts/1")
        print(f"   - Session test: {result.success}")


async def run_comprehensive_demo():
    """Run comprehensive async operations demo."""
    print("🚀 Starting Async Database and API Operations Demo")
    print("=" * 70)
    
    try:
        await demonstrate_database_operations()
        await demonstrate_api_operations()
        await demonstrate_concurrent_operations()
        await demonstrate_retry_mechanism()
        await demonstrate_context_managers()
        
        print("\n" + "=" * 70)
        print("✅ All async operations demos completed successfully!")
        
        print("\n📋 Next Steps:")
        print("1. Run the FastAPI app: uvicorn async_operations_demo:app --reload")
        print("2. Test database operations: POST /demo/database")
        print("3. Test API operations: POST /demo/api")
        print("4. Test batch operations: POST /demo/batch")
        print("5. Check performance: POST /demo/performance")
        print("6. View statistics: GET /demo/stats")
        print("7. Test retry mechanism: POST /demo/retry")
        print("8. Check health: GET /demo/health")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_demo()) 