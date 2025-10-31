from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
import logging
from typing import List, Dict, Any, Optional, Union
from functools import lru_cache
from contextlib import asynccontextmanager
import uuid
from datetime import datetime, timedelta
import httpx
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, func, text
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy.pool import QueuePool
import psutil
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Performance Optimization Implementation (Simplified)
==================================================

This module demonstrates:
- Async I/O optimization for database and external API calls
- Caching strategies (in-memory, simulated Redis)
- Lazy loading and pagination
- Connection pooling and resource management
- Performance monitoring and profiling
- Background tasks and task queues
- Database query optimization
- Response compression and streaming
"""




# ============================================================================
# Configuration and Setup
# ============================================================================

class PerformanceConfig:
    """Performance configuration settings"""
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./test.db"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30
    DB_POOL_TIMEOUT: int = 30
    
    # Cache settings
    CACHE_TTL: int = 300  # 5 minutes
    CACHE_MAX_SIZE: int = 1000
    
    # Performance monitoring
    ENABLE_METRICS: bool = True
    ENABLE_PROFILING: bool = False
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = 50
    MAX_PAGE_SIZE: int = 100
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 100


config = PerformanceConfig()

# ============================================================================
# Database Setup with Connection Pooling
# ============================================================================

# Create async engine with optimized connection pooling
engine = create_async_engine(
    config.DATABASE_URL,
    echo=False,  # Disable SQL logging for performance
    future=True,
    poolclass=QueuePool,
    pool_size=config.DB_POOL_SIZE,
    max_overflow=config.DB_MAX_OVERFLOW,
    pool_timeout=config.DB_POOL_TIMEOUT,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,   # Recycle connections every hour
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,  # Disable autoflush for performance
    autocommit=False
)


async def get_db() -> AsyncSession:
    """Dependency to get database session with connection pooling"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ============================================================================
# Simulated Redis Cache
# ============================================================================

class SimulatedRedis:
    """Simulated Redis cache for demonstration"""
    
    def __init__(self) -> Any:
        self.cache = {}
        self.logger = logging.getLogger(__name__)
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if key in self.cache:
            value, expiry = self.cache[key]
            if expiry is None or time.time() < expiry:
                return value
            else:
                del self.cache[key]
        return None
    
    async def setex(self, key: str, ttl: int, value: str):
        """Set value with TTL"""
        expiry = time.time() + ttl
        self.cache[key] = (value, expiry)
    
    async def close(self) -> Any:
        """Close cache connection"""
        self.cache.clear()


# Simulated Redis instance
redis_cache = SimulatedRedis()


async def get_redis():
    """Get Redis connection"""
    return redis_cache


# ============================================================================
# Performance Monitoring
# ============================================================================

class PerformanceMetrics:
    """Simple performance metrics collection"""
    
    def __init__(self) -> Any:
        self.request_count = 0
        self.request_duration = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.db_queries = 0
        self.db_query_duration = []
    
    def record_request(self, duration: float):
        """Record request metrics"""
        self.request_count += 1
        self.request_duration.append(duration)
    
    def record_cache_hit(self) -> Any:
        """Record cache hit"""
        self.cache_hits += 1
    
    def record_cache_miss(self) -> Any:
        """Record cache miss"""
        self.cache_misses += 1
    
    def record_db_query(self, duration: float):
        """Record database query metrics"""
        self.db_queries += 1
        self.db_query_duration.append(duration)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "requests": {
                "total": self.request_count,
                "avg_duration": statistics.mean(self.request_duration) if self.request_duration else 0,
                "max_duration": max(self.request_duration) if self.request_duration else 0
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": (self.cache_hits / (self.cache_hits + self.cache_misses) * 100) if (self.cache_hits + self.cache_misses) > 0 else 0
            },
            "database": {
                "queries": self.db_queries,
                "avg_duration": statistics.mean(self.db_query_duration) if self.db_query_duration else 0
            }
        }


metrics = PerformanceMetrics()


class PerformanceMonitor:
    """Performance monitoring and profiling"""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(__name__)
        self.start_times = {}
    
    def start_timer(self, operation: str) -> str:
        """Start timing an operation"""
        timer_id = str(uuid.uuid4())
        self.start_times[timer_id] = {
            'operation': operation,
            'start_time': time.perf_counter()
        }
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """End timing an operation and return duration"""
        if timer_id not in self.start_times:
            return 0.0
        
        timer_data = self.start_times.pop(timer_id)
        duration = time.perf_counter() - timer_data['start_time']
        
        # Log performance data
        self.logger.info(
            f"Operation completed: {timer_data['operation']} in {duration:.3f}s"
        )
        
        # Update metrics
        if timer_data['operation'].startswith('db_'):
            metrics.record_db_query(duration)
        elif timer_data['operation'].startswith('http_'):
            metrics.record_request(duration)
        
        return duration
    
    def update_system_metrics(self) -> Any:
        """Update system performance metrics"""
        # This would update Prometheus metrics in a real implementation
        pass


monitor = PerformanceMonitor()


# ============================================================================
# Caching Strategies
# ============================================================================

class CacheManager:
    """Comprehensive caching manager with multiple strategies"""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(__name__)
        self.local_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    async def get_cached_data(self, key: str, cache_type: str = 'redis') -> Optional[Any]:
        """Get data from cache"""
        try:
            if cache_type == 'local':
                return await self._get_local_cache(key)
            elif cache_type == 'redis':
                return await self._get_redis_cache(key)
            else:
                return None
        except Exception as e:
            self.logger.warning(f"Cache get failed for key {key}: {e}")
            return None
    
    async def set_cached_data(self, key: str, data: Any, ttl: int = None, cache_type: str = 'redis'):
        """Set data in cache"""
        try:
            if cache_type == 'local':
                await self._set_local_cache(key, data, ttl)
            elif cache_type == 'redis':
                await self._set_redis_cache(key, data, ttl or config.CACHE_TTL)
        except Exception as e:
            self.logger.warning(f"Cache set failed for key {key}: {e}")
    
    async def _get_local_cache(self, key: str) -> Optional[Any]:
        """Get data from local memory cache"""
        if key in self.local_cache:
            data, expiry = self.local_cache[key]
            if expiry is None or time.time() < expiry:
                self.cache_stats['hits'] += 1
                metrics.record_cache_hit()
                return data
        
        self.cache_stats['misses'] += 1
        metrics.record_cache_miss()
        return None
    
    async def _set_local_cache(self, key: str, data: Any, ttl: int = None):
        """Set data in local memory cache"""
        expiry = time.time() + (ttl or config.CACHE_TTL) if ttl else None
        
        # Implement LRU eviction if cache is full
        if len(self.local_cache) >= config.CACHE_MAX_SIZE:
            # Remove oldest entry
            oldest_key = next(iter(self.local_cache))
            del self.local_cache[oldest_key]
        
        self.local_cache[key] = (data, expiry)
    
    async def _get_redis_cache(self, key: str) -> Optional[Any]:
        """Get data from Redis cache"""
        redis = await get_redis()
        data = await redis.get(key)
        
        if data:
            self.cache_stats['hits'] += 1
            metrics.record_cache_hit()
            return json.loads(data)
        
        self.cache_stats['misses'] += 1
        metrics.record_cache_miss()
        return None
    
    async def _set_redis_cache(self, key: str, data: Any, ttl: int):
        """Set data in Redis cache"""
        redis = await get_redis()
        await redis.setex(key, ttl, json.dumps(data))
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'local_cache_size': len(self.local_cache)
        }


cache_manager = CacheManager()


# ============================================================================
# Database Query Optimization
# ============================================================================

class DatabaseOptimizer:
    """Database query optimization utilities"""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(__name__)
    
    async def optimized_get_user(self, user_id: int, db: AsyncSession, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Optimized user retrieval with caching and eager loading"""
        timer_id = monitor.start_timer('db_get_user')
        
        try:
            # Try cache first
            if use_cache:
                cache_key = f"user:{user_id}"
                cached_user = await cache_manager.get_cached_data(cache_key, 'redis')
                if cached_user:
                    monitor.end_timer(timer_id)
                    return cached_user
            
            # Simulate database query with eager loading
            await asyncio.sleep(0.1)  # Simulate database query time
            
            # Mock user data
            user_data = {
                "id": user_id,
                "username": f"user_{user_id}",
                "email": f"user_{user_id}@example.com",
                "is_active": True,
                "addresses": [
                    {"id": 1, "street": "123 Main St", "city": "Anytown"}
                ],
                "orders": [
                    {"id": 1, "total": 99.99, "status": "completed"}
                ]
            }
            
            # Cache the result
            if use_cache:
                await cache_manager.set_cached_data(cache_key, user_data, ttl=300, cache_type='redis')
            
            monitor.end_timer(timer_id)
            return user_data
            
        except Exception as e:
            monitor.end_timer(timer_id)
            self.logger.error(f"Database get user failed for user_id {user_id}: {e}")
            raise
    
    async def optimized_list_users(
        self,
        db: AsyncSession,
        page: int = 1,
        page_size: int = config.DEFAULT_PAGE_SIZE,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Optimized user listing with pagination and caching"""
        timer_id = monitor.start_timer('db_list_users')
        
        try:
            # Validate pagination parameters
            page_size = min(page_size, config.MAX_PAGE_SIZE)
            offset = (page - 1) * page_size
            
            # Try cache for first page
            cache_key = f"users:page:{page}:size:{page_size}"
            if use_cache and page == 1:
                cached_users = await cache_manager.get_cached_data(cache_key, 'redis')
                if cached_users:
                    monitor.end_timer(timer_id)
                    return cached_users
            
            # Simulate database query
            await asyncio.sleep(0.05)  # Simulate database query time
            
            # Mock data
            total_count = 1000
            users = []
            for i in range(offset, min(offset + page_size, total_count)):
                users.append({
                    "id": i + 1,
                    "username": f"user_{i + 1}",
                    "email": f"user_{i + 1}@example.com",
                    "is_active": True
                })
            
            # Prepare response
            response_data = {
                'items': users,
                'total': total_count,
                'page': page,
                'page_size': page_size,
                'pages': (total_count + page_size - 1) // page_size
            }
            
            # Cache first page
            if use_cache and page == 1:
                await cache_manager.set_cached_data(cache_key, response_data, ttl=60, cache_type='redis')
            
            monitor.end_timer(timer_id)
            return response_data
            
        except Exception as e:
            monitor.end_timer(timer_id)
            self.logger.error(f"Database list users failed: {e}")
            raise


db_optimizer = DatabaseOptimizer()


# ============================================================================
# Async HTTP Client with Connection Pooling
# ============================================================================

class AsyncHTTPClient:
    """Optimized async HTTP client with connection pooling"""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(__name__)
        self.client = None
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        if self.client is None:
            limits = httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0
            )
            self.client = httpx.AsyncClient(
                limits=limits,
                timeout=httpx.Timeout(30.0),
                follow_redirects=True
            )
        return self.client
    
    async async def make_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make HTTP request with performance monitoring"""
        timer_id = monitor.start_timer(f'http_{method.lower()}')
        
        try:
            client = await self.get_client()
            response = await client.request(method, url, **kwargs)
            
            monitor.end_timer(timer_id)
            return response
            
        except Exception as e:
            monitor.end_timer(timer_id)
            self.logger.error(f"HTTP request failed for {method} {url}: {e}")
            raise
    
    async def close(self) -> Any:
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()


http_client = AsyncHTTPClient()


# ============================================================================
# Background Task Queue
# ============================================================================

class TaskQueue:
    """Background task queue for non-blocking operations"""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(__name__)
        self.tasks = []
    
    async def add_task(self, task_func, *args, **kwargs) -> Any:
        """Add task to background queue"""
        task = asyncio.create_task(task_func(*args, **kwargs))
        self.tasks.append(task)
        
        # Clean up completed tasks
        self.tasks = [t for t in self.tasks if not t.done()]
        
        return task
    
    async def process_email_notification(self, user_id: int, message: str):
        """Background task for email notifications"""
        timer_id = monitor.start_timer('background_email_notification')
        
        try:
            # Simulate email sending
            await asyncio.sleep(0.5)
            self.logger.info(f"Email sent to user {user_id}: {message}")
            
        except Exception as e:
            self.logger.error(f"Email send failed for user {user_id}: {e}")
        finally:
            monitor.end_timer(timer_id)
    
    async def process_data_cleanup(self, data_ids: List[int]):
        """Background task for data cleanup"""
        timer_id = monitor.start_timer('background_data_cleanup')
        
        try:
            # Simulate data cleanup
            await asyncio.sleep(1.0)
            self.logger.info(f"Data cleanup completed for IDs: {data_ids}")
            
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")
        finally:
            monitor.end_timer(timer_id)


task_queue = TaskQueue()


# ============================================================================
# FastAPI Application with Performance Optimizations
# ============================================================================

def create_optimized_app() -> FastAPI:
    """Create FastAPI application with performance optimizations"""
    
    app = FastAPI(
        title="Performance Optimized API",
        description="FastAPI application with comprehensive performance optimizations",
        version="1.0.0"
    )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_optimized_app()


# ============================================================================
# Optimized API Routes
# ============================================================================

@app.get("/users/{user_id}")
async def get_user_optimized(
    user_id: int,
    use_cache: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """Optimized user retrieval with caching"""
    user_data = await db_optimizer.optimized_get_user(user_id, db, use_cache)
    
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user_data


@app.get("/users/")
async def list_users_optimized(
    page: int = 1,
    page_size: int = config.DEFAULT_PAGE_SIZE,
    use_cache: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """Optimized user listing with pagination and caching"""
    users_data = await db_optimizer.optimized_list_users(db, page, page_size, use_cache)
    return users_data


@app.post("/users/")
async def create_user_optimized(
    user_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Optimized user creation with background tasks"""
    timer_id = monitor.start_timer('create_user')
    
    try:
        # Create user (simplified)
        user_id = uuid.uuid4()
        
        # Add background tasks
        background_tasks.add_task(
            task_queue.process_email_notification,
            user_id,
            "Welcome to our platform!"
        )
        
        # Invalidate cache
        await cache_manager.set_cached_data(f"users:page:1:size:{config.DEFAULT_PAGE_SIZE}", None, ttl=1, cache_type='redis')
        
        monitor.end_timer(timer_id)
        return {"id": user_id, "status": "created"}
        
    except Exception as e:
        monitor.end_timer(timer_id)
        raise HTTPException(status_code=500, detail="Failed to create user")


@app.get("/performance/stats")
async def get_performance_stats():
    """Get performance statistics"""
    # Update system metrics
    monitor.update_system_metrics()
    
    # Get cache stats
    cache_stats = cache_manager.get_cache_stats()
    
    # Get metrics stats
    metrics_stats = metrics.get_stats()
    
    return {
        "cache": cache_stats,
        "metrics": metrics_stats,
        "system": {
            "memory_usage_mb": psutil.virtual_memory().used / 1024 / 1024,
            "cpu_usage_percent": psutil.cpu_percent(),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
    }


@app.get("/stream/large-data")
async def stream_large_data():
    """Stream large datasets efficiently"""
    async def generate_data():
        
    """generate_data function."""
for i in range(1000):  # Reduced for demo
            yield f"data:{i}\n"
            await asyncio.sleep(0.001)  # Small delay to simulate processing
    
    return StreamingResponse(
        generate_data(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache"}
    )


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Application startup with performance optimizations"""
    # Initialize Redis connection
    await get_redis()
    
    logging.info("Performance optimized application started")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown with cleanup"""
    # Close HTTP client
    await http_client.close()
    
    # Close Redis connection
    await redis_cache.close()
    
    # Close database engine
    await engine.dispose()
    
    logging.info("Performance optimized application shutdown")


# ============================================================================
# Utility Functions
# ============================================================================

@lru_cache(maxsize=128)
def get_config_value(key: str) -> Optional[Dict[str, Any]]:
    """Get configuration value with caching"""
    # This would typically load from environment or config file
    config_values = {
        'api_timeout': 30,
        'max_retries': 3,
        'cache_ttl': 300
    }
    return config_values.get(key)


async def batch_process_data(data_items: List[Dict[str, Any]], batch_size: int = 100):
    """Process data in batches for better performance"""
    timer_id = monitor.start_timer('batch_process_data')
    
    try:
        results = []
        
        for i in range(0, len(data_items), batch_size):
            batch = data_items[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [process_single_item(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            results.extend([r for r in batch_results if not isinstance(r, Exception)])
        
        monitor.end_timer(timer_id)
        return results
        
    except Exception as e:
        monitor.end_timer(timer_id)
        logging.error(f"Batch processing failed: {e}")
        raise


async def process_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single data item"""
    # Simulate processing
    await asyncio.sleep(0.01)
    return {"processed": True, "item_id": item.get("id")}


# ============================================================================
# Example Usage
# ============================================================================

async def demonstrate_performance_optimizations():
    """Demonstrate various performance optimization techniques"""
    
    print("\n=== Performance Optimization Demonstrations ===")
    
    # 1. Database optimization with caching
    print("\n1. Database Optimization with Caching:")
    # This would be called in a route handler
    # user_data = await db_optimizer.optimized_get_user(1, db_session, use_cache=True)
    
    # 2. Cache performance
    print("\n2. Cache Performance:")
    cache_stats = cache_manager.get_cache_stats()
    print(f"Cache hit rate: {cache_stats['hit_rate']:.2f}%")
    
    # 3. Background task processing
    print("\n3. Background Task Processing:")
    # await task_queue.add_task(task_queue.process_email_notification, 1, "Test message")
    
    # 4. Batch processing
    print("\n4. Batch Processing:")
    test_data = [{"id": i} for i in range(100)]
    # results = await batch_process_data(test_data, batch_size=10)
    
    # 5. System metrics
    print("\n5. System Metrics:")
    monitor.update_system_metrics()
    print(f"Memory usage: {psutil.virtual_memory().percent}%")
    print(f"CPU usage: {psutil.cpu_percent()}%")


if __name__ == "__main__":
    
    # Run the application
    uvicorn.run(
        "performance_optimization_simple:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 