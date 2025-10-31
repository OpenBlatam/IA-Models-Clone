from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import threading
from typing import List, Dict, Any, Optional, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps
import hashlib
from fastapi import FastAPI, Request, Response, HTTPException, status, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, func, text
import httpx
import redis.asyncio as redis
from celery import Celery
import aiofiles
import aiohttp
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Non-Blocking Operations Implementation
====================================

This module demonstrates:
- Async/await patterns for non-blocking operations
- Background tasks for long-running operations
- Connection pooling for database operations
- Non-blocking I/O operations
- Task queues and job processing
- Caching strategies to reduce blocking
- Streaming responses for large data
- Circuit breaker patterns
- Rate limiting and throttling
- Performance optimization techniques
"""




# ============================================================================
# CONFIGURATION AND SETTINGS
# ============================================================================

@dataclass
class NonBlockingConfig:
    """Configuration for non-blocking operations."""
    
    # Database settings
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost/db"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30
    
    # Redis settings
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_POOL_SIZE: int = 10
    
    # HTTP client settings
    HTTP_TIMEOUT: int = 30
    HTTP_MAX_CONNECTIONS: int = 100
    
    # Task queue settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # Thread pool settings
    MAX_WORKERS: int = 10
    
    # Cache settings
    CACHE_TTL: int = 300  # 5 minutes
    CACHE_MAX_SIZE: int = 1000
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds


# ============================================================================
# ASYNC DATABASE MANAGER
# ============================================================================

class AsyncDatabaseManager:
    """Async database manager with connection pooling."""
    
    def __init__(self, database_url: str, pool_size: int, max_overflow: int):
        
    """__init__ function."""
self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.engine = None
        self.session_factory = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> Any:
        """Initialize async database connection pool."""
        if self.engine is None:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                future=True,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            self.logger.info("Async database connection pool initialized")
    
    async def get_session(self) -> AsyncSession:
        """Get async database session."""
        if self.session_factory is None:
            await self.initialize()
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute async database query."""
        async with self.session_factory() as session:
            try:
                result = await session.execute(text(query), params or {})
                return [dict(row._mapping) for row in result]
            except Exception as e:
                self.logger.error(f"Database query error: {e}")
                raise
    
    async def close(self) -> Any:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            self.logger.info("Database connections closed")


# ============================================================================
# ASYNC HTTP CLIENT MANAGER
# ============================================================================

class AsyncHTTPClientManager:
    """Async HTTP client manager with connection pooling."""
    
    def __init__(self, timeout: int, max_connections: int):
        
    """__init__ function."""
self.timeout = timeout
        self.max_connections = max_connections
        self.client = None
        self.logger = logging.getLogger(__name__)
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get async HTTP client with connection pooling."""
        if self.client is None:
            limits = httpx.Limits(
                max_keepalive_connections=self.max_connections,
                max_connections=self.max_connections * 2,
                keepalive_expiry=30.0
            )
            self.client = httpx.AsyncClient(
                limits=limits,
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True
            )
            self.logger.info("Async HTTP client initialized")
        return self.client
    
    async async def make_request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Make async HTTP request."""
        client = await self.get_client()
        try:
            response = await client.request(method, url, **kwargs)
            return response
        except Exception as e:
            self.logger.error(f"HTTP request error: {e}")
            raise
    
    async def close(self) -> Any:
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.logger.info("HTTP client closed")


# ============================================================================
# ASYNC CACHE MANAGER
# ============================================================================

class AsyncCacheManager:
    """Async cache manager with Redis."""
    
    def __init__(self, redis_url: str, pool_size: int, ttl: int, max_size: int):
        
    """__init__ function."""
self.redis_url = redis_url
        self.pool_size = pool_size
        self.ttl = ttl
        self.max_size = max_size
        self.redis_pool = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> Any:
        """Initialize Redis connection pool."""
        if self.redis_pool is None:
            self.redis_pool = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=self.pool_size
            )
            self.logger.info("Async Redis cache initialized")
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        if self.redis_pool is None:
            await self.initialize()
        try:
            return await self.redis_pool.get(key)
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        """Set value in cache."""
        if self.redis_pool is None:
            await self.initialize()
        try:
            await self.redis_pool.setex(key, ttl or self.ttl, value)
            return True
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if self.redis_pool is None:
            await self.initialize()
        try:
            await self.redis_pool.delete(key)
            return True
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False
    
    async def close(self) -> Any:
        """Close Redis connections."""
        if self.redis_pool:
            await self.redis_pool.close()
            self.logger.info("Redis cache closed")


# ============================================================================
# TASK QUEUE MANAGER
# ============================================================================

class TaskQueueManager:
    """Task queue manager with Celery."""
    
    def __init__(self, broker_url: str, result_backend: str):
        
    """__init__ function."""
self.broker_url = broker_url
        self.result_backend = result_backend
        self.celery_app = None
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> Any:
        """Initialize Celery application."""
        if self.celery_app is None:
            self.celery_app = Celery(
                'non_blocking_tasks',
                broker=self.broker_url,
                backend=self.result_backend
            )
            self.celery_app.conf.update(
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
                timezone='UTC',
                enable_utc=True,
                task_track_started=True,
                task_time_limit=30 * 60,  # 30 minutes
                task_soft_time_limit=25 * 60,  # 25 minutes
            )
            self.logger.info("Celery task queue initialized")
    
    def get_celery_app(self) -> Celery:
        """Get Celery application."""
        if self.celery_app is None:
            self.initialize()
        return self.celery_app
    
    def submit_task(self, task_name: str, *args, **kwargs) -> str:
        """Submit task to queue."""
        celery_app = self.get_celery_app()
        task = celery_app.send_task(task_name, args=args, kwargs=kwargs)
        return task.id
    
    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """Get task result."""
        celery_app = self.get_celery_app()
        task = celery_app.AsyncResult(task_id)
        return {
            'task_id': task_id,
            'status': task.status,
            'result': task.result if task.ready() else None,
            'ready': task.ready()
        }


# ============================================================================
# THREAD POOL MANAGER
# ============================================================================

class ThreadPoolManager:
    """Thread pool manager for CPU-bound operations."""
    
    def __init__(self, max_workers: int):
        
    """__init__ function."""
self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)
    
    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        """Run function in thread pool."""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self.thread_pool, 
                lambda: func(*args, **kwargs)
            )
            return result
        except Exception as e:
            self.logger.error(f"Thread pool execution error: {e}")
            raise
    
    async def run_in_process(self, func: Callable, *args, **kwargs) -> Any:
        """Run function in process pool."""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                self.process_pool, 
                lambda: func(*args, **kwargs)
            )
            return result
        except Exception as e:
            self.logger.error(f"Process pool execution error: {e}")
            raise
    
    def shutdown(self) -> Any:
        """Shutdown thread and process pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        self.logger.info("Thread and process pools shutdown")


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """Rate limiter for API endpoints."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        
    """__init__ function."""
self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        
        with self.lock:
            if client_id not in self.requests:
                self.requests[client_id] = []
            
            # Remove old requests outside the window
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window_seconds
            ]
            
            # Check if under limit
            if len(self.requests[client_id]) < self.max_requests:
                self.requests[client_id].append(now)
                return True
            
            return False
    
    async def get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = time.time()
        
        with self.lock:
            if client_id not in self.requests:
                return self.max_requests
            
            # Remove old requests
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if now - req_time < self.window_seconds
            ]
            
            return max(0, self.max_requests - len(self.requests[client_id]))


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        
    """__init__ function."""
self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self) -> Any:
        """Handle successful call."""
        with self.lock:
            self.failure_count = 0
            self.state = "CLOSED"
    
    def _on_failure(self) -> Any:
        """Handle failed call."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserCreate(BaseModel):
    """User creation model."""
    
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=100)


class UserResponse(BaseModel):
    """User response model."""
    
    id: int
    username: str
    email: str
    full_name: Optional[str]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class TaskRequest(BaseModel):
    """Task request model."""
    
    task_type: str = Field(..., description="Type of task to execute")
    data: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(1, ge=1, le=10, description="Task priority")


class TaskResponse(BaseModel):
    """Task response model."""
    
    task_id: str
    status: str
    message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FileUploadResponse(BaseModel):
    """File upload response model."""
    
    filename: str
    size: int
    upload_time: float
    status: str


# ============================================================================
# DEPENDENCY PROVIDERS
# ============================================================================

def get_config() -> NonBlockingConfig:
    """Get configuration."""
    return NonBlockingConfig()


async def get_db_manager(config: NonBlockingConfig = Depends(get_config)) -> AsyncDatabaseManager:
    """Get database manager."""
    manager = AsyncDatabaseManager(
        database_url=config.DATABASE_URL,
        pool_size=config.DB_POOL_SIZE,
        max_overflow=config.DB_MAX_OVERFLOW
    )
    await manager.initialize()
    return manager


async def get_http_client(config: NonBlockingConfig = Depends(get_config)) -> AsyncHTTPClientManager:
    """Get HTTP client manager."""
    return AsyncHTTPClientManager(
        timeout=config.HTTP_TIMEOUT,
        max_connections=config.HTTP_MAX_CONNECTIONS
    )


async def get_cache_manager(config: NonBlockingConfig = Depends(get_config)) -> AsyncCacheManager:
    """Get cache manager."""
    manager = AsyncCacheManager(
        redis_url=config.REDIS_URL,
        pool_size=config.REDIS_POOL_SIZE,
        ttl=config.CACHE_TTL,
        max_size=config.CACHE_MAX_SIZE
    )
    await manager.initialize()
    return manager


def get_task_queue(config: NonBlockingConfig = Depends(get_config)) -> TaskQueueManager:
    """Get task queue manager."""
    manager = TaskQueueManager(
        broker_url=config.CELERY_BROKER_URL,
        result_backend=config.CELERY_RESULT_BACKEND
    )
    return manager


def get_thread_pool(config: NonBlockingConfig = Depends(get_config)) -> ThreadPoolManager:
    """Get thread pool manager."""
    return ThreadPoolManager(max_workers=config.MAX_WORKERS)


def get_rate_limiter(config: NonBlockingConfig = Depends(get_config)) -> RateLimiter:
    """Get rate limiter."""
    return RateLimiter(
        max_requests=config.RATE_LIMIT_REQUESTS,
        window_seconds=config.RATE_LIMIT_WINDOW
    )


# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def send_email_background(email: str, subject: str, content: str):
    """Background task to send email."""
    # Simulate email sending
    await asyncio.sleep(2)
    logging.getLogger("background_task").info(f"Email sent to {email}: {subject}")


async def process_file_background(filename: str, file_content: bytes):
    """Background task to process file."""
    # Simulate file processing
    await asyncio.sleep(5)
    logging.getLogger("background_task").info(f"File processed: {filename}")


async def generate_report_background(user_id: int, report_type: str):
    """Background task to generate report."""
    # Simulate report generation
    await asyncio.sleep(10)
    logging.getLogger("background_task").info(f"Report generated for user {user_id}: {report_type}")


# ============================================================================
# CELERY TASKS
# ============================================================================

def get_celery_app():
    """Get Celery app for tasks."""
    config = NonBlockingConfig()
    return TaskQueueManager(
        broker_url=config.CELERY_BROKER_URL,
        result_backend=config.CELERY_RESULT_BACKEND
    ).get_celery_app()


celery_app = get_celery_app()


@celery_app.task
def heavy_computation_task(data: Dict[str, Any]) -> Dict[str, Any]:
    """Heavy computation task."""
    # Simulate heavy computation
    time.sleep(10)
    return {"result": "computation_completed", "data": data}


@celery_app.task
def data_processing_task(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Data processing task."""
    # Simulate data processing
    time.sleep(15)
    return {"processed_items": len(data), "status": "completed"}


@celery_app.task
def report_generation_task(user_id: int, report_type: str) -> Dict[str, Any]:
    """Report generation task."""
    # Simulate report generation
    time.sleep(20)
    return {"user_id": user_id, "report_type": report_type, "status": "generated"}


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application with non-blocking operations."""
    
    app = FastAPI(
        title="Non-Blocking Operations Demo",
        version="1.0.0",
        description="Demonstration of non-blocking operations in FastAPI"
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


# ============================================================================
# NON-BLOCKING API ROUTES
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - non-blocking."""
    return {"message": "Non-Blocking Operations Demo"}


@app.get("/users/")
async def get_users(
    db_manager: AsyncDatabaseManager = Depends(get_db_manager),
    cache_manager: AsyncCacheManager = Depends(get_cache_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    request: Request = None
):
    """Get users with caching and rate limiting."""
    
    # Rate limiting
    client_ip = request.client.host if request and request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # Try cache first
    cache_key = f"users:list"
    cached_data = await cache_manager.get(cache_key)
    if cached_data:
        return JSONResponse(
            content=json.loads(cached_data),
            headers={"X-Cache": "HIT"}
        )
    
    # Database query (non-blocking)
    try:
        users_data = await db_manager.execute_query(
            "SELECT id, username, email, full_name, created_at FROM users LIMIT 100"
        )
        
        # Cache the result
        await cache_manager.set(cache_key, json.dumps(users_data))
        
        return JSONResponse(
            content=users_data,
            headers={"X-Cache": "MISS"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )


@app.post("/users/", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    db_manager: AsyncDatabaseManager = Depends(get_db_manager),
    cache_manager: AsyncCacheManager = Depends(get_cache_manager)
):
    """Create user with background tasks."""
    
    # Non-blocking database operation
    try:
        # Simulate user creation
        user_dict = user_data.model_dump()
        user_dict.update({
            "id": 999,  # Mock ID
            "created_at": datetime.utcnow()
        })
        
        # Add background tasks
        background_tasks.add_task(
            send_email_background,
            user_data.email,
            "Welcome!",
            f"Welcome to our platform, {user_data.username}!"
        )
        
        background_tasks.add_task(
            generate_report_background,
            user_dict["id"],
            "user_creation"
        )
        
        # Invalidate cache
        await cache_manager.delete("users:list")
        
        return UserResponse(**user_dict)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"User creation failed: {str(e)}"
        )


@app.post("/tasks/", response_model=TaskResponse)
async def create_task(
    task_request: TaskRequest,
    task_queue: TaskQueueManager = Depends(get_task_queue)
):
    """Create background task."""
    
    # Submit task to queue (non-blocking)
    task_id = task_queue.submit_task(
        "heavy_computation_task",
        task_request.data
    )
    
    return TaskResponse(
        task_id=task_id,
        status="submitted",
        message=f"Task {task_request.task_type} submitted successfully"
    )


@app.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    task_queue: TaskQueueManager = Depends(get_task_queue)
):
    """Get task status."""
    
    result = task_queue.get_task_result(task_id)
    return result


@app.post("/upload/")
async def upload_file(
    background_tasks: BackgroundTasks,
    thread_pool: ThreadPoolManager = Depends(get_thread_pool)
):
    """Upload file with background processing."""
    
    # Simulate file upload
    filename = f"file_{int(time.time())}.txt"
    file_content = b"Sample file content"
    
    # Process file in background
    background_tasks.add_task(
        process_file_background,
        filename,
        file_content
    )
    
    return FileUploadResponse(
        filename=filename,
        size=len(file_content),
        upload_time=time.time(),
        status="uploaded"
    )


@app.get("/external-api/")
async def call_external_api(
    http_client: AsyncHTTPClientManager = Depends(get_http_client)
):
    """Call external API with circuit breaker."""
    
    # Create circuit breaker instance
    circuit_breaker = CircuitBreaker()
    
    async def make_external_request():
        """Make external API request."""
        response = await http_client.make_request(
            "GET",
            "https://httpbin.org/delay/2"  # Simulate slow external API
        )
        return response.json()
    
    try:
        result = await circuit_breaker.call(make_external_request)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"External API error: {str(e)}"
        )


@app.get("/stream/")
async def stream_data() -> StreamingResponse:
    """Stream large dataset."""
    
    async def generate_data():
        """Generate streaming data."""
        for i in range(1000):
            yield f"data:{i}\n"
            await asyncio.sleep(0.01)  # Non-blocking delay
    
    return StreamingResponse(
        generate_data(),
        media_type="text/plain",
        headers={"X-Streaming": "true"}
    )


@app.get("/cpu-intensive/")
async def cpu_intensive_operation(
    thread_pool: ThreadPoolManager = Depends(get_thread_pool)
):
    """CPU-intensive operation in thread pool."""
    
    def heavy_calculation():
        """Heavy CPU calculation."""
        result = 0
        for i in range(10000000):
            result += i * i
        return result
    
    # Run in thread pool (non-blocking)
    result = await thread_pool.run_in_thread(heavy_calculation)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    return {"result": result, "message": "CPU-intensive operation completed"}


@app.get("/file-operations/")
async def file_operations(
    thread_pool: ThreadPoolManager = Depends(get_thread_pool)
):
    """File operations in thread pool."""
    
    def read_large_file():
        """Read large file."""
        # Simulate reading large file
        time.sleep(2)
        return "Large file content"
    
    # Run in thread pool (non-blocking)
    content = await thread_pool.run_in_thread(read_large_file)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    return {"content": content[:100] + "...", "message": "File operation completed"}


@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    
    await websocket.accept()
    
    try:
        while True:
            # Non-blocking message handling
            data = await websocket.receive_text()
            
            # Process message
            response = {"message": f"Received: {data}", "timestamp": datetime.utcnow().isoformat()}
            
            # Send response
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logging.getLogger("websocket").info("WebSocket disconnected")


@app.get("/health")
async def health_check(
    db_manager: AsyncDatabaseManager = Depends(get_db_manager),
    cache_manager: AsyncCacheManager = Depends(get_cache_manager)
):
    """Health check with non-blocking operations."""
    
    # Check database health
    db_healthy = True
    try:
        await db_manager.execute_query("SELECT 1")
    except Exception:
        db_healthy = False
    
    # Check cache health
    cache_healthy = True
    try:
        await cache_manager.set("health_check", "ok", 10)
        await cache_manager.delete("health_check")
    except Exception:
        cache_healthy = False
    
    overall_health = "healthy" if db_healthy and cache_healthy else "unhealthy"
    
    return {
        "status": overall_health,
        "database": "healthy" if db_healthy else "unhealthy",
        "cache": "healthy" if cache_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Middleware to monitor performance."""
    
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Add performance headers
    response.headers["X-Response-Time"] = f"{response_time:.4f}s"
    response.headers["X-Request-ID"] = f"req_{int(start_time * 1000)}"
    
    # Log slow requests
    if response_time > 1.0:
        logging.getLogger("performance").warning(
            f"Slow request: {request.url.path} took {response_time:.3f}s"
        )
    
    return response


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def demonstrate_non_blocking_operations():
    """Demonstrate non-blocking operations."""
    
    print("\n=== Non-Blocking Operations Demonstrations ===")
    
    # 1. Database operations
    print("\n1. Async Database Operations:")
    config = NonBlockingConfig()
    db_manager = AsyncDatabaseManager(
        database_url=config.DATABASE_URL,
        pool_size=config.DB_POOL_SIZE,
        max_overflow=config.DB_MAX_OVERFLOW
    )
    await db_manager.initialize()
    print("   - Database manager initialized with connection pooling")
    
    # 2. HTTP client operations
    print("\n2. Async HTTP Client Operations:")
    http_client = AsyncHTTPClientManager(
        timeout=config.HTTP_TIMEOUT,
        max_connections=config.HTTP_MAX_CONNECTIONS
    )
    print("   - HTTP client initialized with connection pooling")
    
    # 3. Cache operations
    print("\n3. Async Cache Operations:")
    cache_manager = AsyncCacheManager(
        redis_url=config.REDIS_URL,
        pool_size=config.REDIS_POOL_SIZE,
        ttl=config.CACHE_TTL,
        max_size=config.CACHE_MAX_SIZE
    )
    await cache_manager.initialize()
    print("   - Cache manager initialized")
    
    # 4. Task queue operations
    print("\n4. Task Queue Operations:")
    task_queue = TaskQueueManager(
        broker_url=config.CELERY_BROKER_URL,
        result_backend=config.CELERY_RESULT_BACKEND
    )
    print("   - Task queue manager initialized")
    
    # 5. Thread pool operations
    print("\n5. Thread Pool Operations:")
    thread_pool = ThreadPoolManager(max_workers=config.MAX_WORKERS)
    print("   - Thread pool manager initialized")
    
    # 6. Rate limiting
    print("\n6. Rate Limiting:")
    rate_limiter = RateLimiter(
        max_requests=config.RATE_LIMIT_REQUESTS,
        window_seconds=config.RATE_LIMIT_WINDOW
    )
    print("   - Rate limiter initialized")
    
    # 7. Circuit breaker
    print("\n7. Circuit Breaker:")
    circuit_breaker = CircuitBreaker()
    print("   - Circuit breaker initialized")
    
    print("\n8. Non-blocking patterns demonstrated:")
    print("   - Async/await for I/O operations")
    print("   - Connection pooling for databases and HTTP clients")
    print("   - Background tasks for long-running operations")
    print("   - Task queues for heavy computations")
    print("   - Thread pools for CPU-bound operations")
    print("   - Caching to reduce blocking operations")
    print("   - Rate limiting to prevent overload")
    print("   - Circuit breaker for fault tolerance")
    print("   - Streaming responses for large data")
    print("   - WebSocket for real-time communication")


if __name__ == "__main__":
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application
    uvicorn.run(
        "non_blocking_operations_implementation:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 