# Async I/O Optimization Guide

A comprehensive guide for minimizing blocking I/O operations and using asynchronous operations for all database calls and external API requests in the HeyGen AI FastAPI application.

## 🎯 Overview

This guide covers:
- **Async Database Operations**: Non-blocking database queries and transactions
- **Async HTTP Operations**: Concurrent external API requests
- **Async File Operations**: Non-blocking file I/O
- **Async Redis Operations**: Asynchronous caching and session management
- **Async Decorators**: Utilities for async function optimization
- **Best Practices**: Patterns for avoiding blocking operations
- **Performance Monitoring**: Tracking async operation performance

## 📋 Table of Contents

1. [Async Database Operations](#async-database-operations)
2. [Async HTTP Operations](#async-http-operations)
3. [Async File Operations](#async-file-operations)
4. [Async Redis Operations](#async-redis-operations)
5. [Async Decorators](#async-decorators)
6. [Integration Examples](#integration-examples)
7. [Best Practices](#best-practices)
8. [Performance Monitoring](#performance-monitoring)

## 🗄️ Async Database Operations

### Overview

All database operations should be asynchronous to prevent blocking the event loop.

### Async Database Manager

#### **Basic Setup**
```python
from api.async_io.async_operations import AsyncDatabaseManager

# Initialize async database manager
db_manager = AsyncDatabaseManager(
    database_url="postgresql+asyncpg://user:pass@localhost/db",
    pool_size=20
)

# Use async session context manager
async def get_user_data(user_id: str) -> Dict[str, Any]:
    """Get user data asynchronously."""
    async with db_manager.get_session() as session:
        result = await session.execute(
            text("SELECT * FROM users WHERE id = :user_id"),
            {"user_id": user_id}
        )
        user = result.fetchone()
        return dict(user._mapping) if user else None
```

#### **Async Query Execution**
```python
# Execute single query
async def get_user_videos(user_id: str) -> List[Dict[str, Any]]:
    """Get user videos asynchronously."""
    query = """
        SELECT v.*, u.name as user_name
        FROM videos v
        JOIN users u ON v.user_id = u.id
        WHERE v.user_id = :user_id
        ORDER BY v.created_at DESC
    """
    
    return await db_manager.execute_query(
        query=query,
        params={"user_id": user_id},
        timeout=30
    )

# Execute batch queries
async def create_multiple_videos(videos_data: List[Dict[str, Any]]) -> List[str]:
    """Create multiple videos asynchronously."""
    queries = []
    params = []
    
    for video_data in videos_data:
        queries.append("""
            INSERT INTO videos (title, description, user_id, created_at)
            VALUES (:title, :description, :user_id, :created_at)
            RETURNING id
        """)
        params.append({
            "title": video_data["title"],
            "description": video_data["description"],
            "user_id": video_data["user_id"],
            "created_at": datetime.now(timezone.utc)
        })
    
    results = await db_manager.execute_batch(queries, params, batch_size=100)
    return [result[0]["id"] for result in results]
```

#### **Async Transactions**
```python
async def create_user_with_profile(user_data: Dict[str, Any], profile_data: Dict[str, Any]) -> str:
    """Create user and profile in a single transaction."""
    async def create_user(session):
        result = await session.execute(
            text("""
                INSERT INTO users (email, first_name, last_name, created_at)
                VALUES (:email, :first_name, :last_name, :created_at)
                RETURNING id
            """),
            user_data
        )
        return result.fetchone()["id"]
    
    async def create_profile(session):
        await session.execute(
            text("""
                INSERT INTO user_profiles (user_id, bio, avatar_url, created_at)
                VALUES (:user_id, :bio, :avatar_url, :created_at)
            """),
            profile_data
        )
    
    operations = [create_user, create_profile]
    results = await db_manager.transaction(operations)
    return results[0]  # Return user ID
```

### SQLAlchemy Async Patterns

#### **Async ORM Operations**
```python
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, joinedload

# Async select with relationships
async def get_user_with_videos(user_id: str):
    """Get user with videos using async ORM."""
    async with db_manager.get_session() as session:
        stmt = (
            select(User)
            .options(selectinload(User.videos))
            .where(User.id == user_id)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

# Async bulk operations
async def bulk_create_videos(videos_data: List[Dict[str, Any]]):
    """Bulk create videos asynchronously."""
    async with db_manager.get_session() as session:
        videos = [Video(**data) for data in videos_data]
        session.add_all(videos)
        await session.commit()
        return [video.id for video in videos]
```

## 🌐 Async HTTP Operations

### Overview

All external API requests should be asynchronous to prevent blocking.

### Async HTTP Client

#### **Basic Setup**
```python
from api.async_io.async_operations import AsyncHTTPClient

# Initialize async HTTP client
http_client = AsyncHTTPClient(
    base_url="https://api.heygen.com",
    timeout=30,
    max_retries=3,
    retry_delay=1.0,
    headers={"Authorization": f"Bearer {API_KEY}"}
)

# Make async requests
async def create_video_with_heygen(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create video using HeyGen API asynchronously."""
    response = await http_client.post("/v1/videos", json=video_data)
    return response.json()

async def get_video_status(video_id: str) -> Dict[str, Any]:
    """Get video status asynchronously."""
    response = await http_client.get(f"/v1/videos/{video_id}")
    return response.json()
```

#### **Concurrent API Requests**
```python
async def process_multiple_videos(video_ids: List[str]) -> List[Dict[str, Any]]:
    """Process multiple videos concurrently."""
    tasks = []
    
    for video_id in video_ids:
        task = http_client.get(f"/v1/videos/{video_id}")
        tasks.append(task)
    
    # Execute all requests concurrently
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    results = []
    for response in responses:
        if isinstance(response, Exception):
            logger.error(f"Request failed: {response}")
            results.append({"error": str(response)})
        else:
            data = response.json()
            results.append(data)
    
    return results
```

#### **Batch API Operations**
```python
async def batch_create_videos(videos_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create multiple videos in batches."""
    batch_size = 10
    results = []
    
    for i in range(0, len(videos_data), batch_size):
        batch = videos_data[i:i + batch_size]
        
        # Create tasks for batch
        tasks = [
            http_client.post("/v1/videos", json=video_data)
            for video_data in batch
        ]
        
        # Execute batch concurrently
        batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for response in batch_responses:
            if isinstance(response, Exception):
                results.append({"error": str(response)})
            else:
                results.append(response.json())
    
    return results
```

### External API Manager

#### **HeyGen API Integration**
```python
from api.async_io.async_operations import AsyncExternalAPIManager

# Initialize external API manager
api_manager = AsyncExternalAPIManager(
    base_url="https://api.heygen.com",
    api_key=os.getenv("HEYGEN_API_KEY")
)

# Create video asynchronously
async def create_video(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create video using external API manager."""
    return await api_manager.create_video(video_data)

# Monitor video status
async def monitor_video_processing(video_id: str) -> Dict[str, Any]:
    """Monitor video processing status."""
    return await api_manager.get_video_status(video_id)

# Download completed video
async def download_video(video_id: str, output_path: str) -> None:
    """Download video asynchronously."""
    await api_manager.download_video(video_id, output_path)
```

## 📁 Async File Operations

### Overview

All file operations should be asynchronous to prevent blocking the event loop.

### Async File Manager

#### **Basic File Operations**
```python
from api.async_io.async_operations import AsyncFileManager

# Initialize async file manager
file_manager = AsyncFileManager(base_path="/tmp")

# Read file asynchronously
async def read_video_script(script_path: str) -> str:
    """Read video script asynchronously."""
    return await file_manager.read_file(script_path)

# Write file asynchronously
async def save_video_metadata(video_id: str, metadata: Dict[str, Any]) -> None:
    """Save video metadata asynchronously."""
    content = json.dumps(metadata, indent=2)
    await file_manager.write_file(f"videos/{video_id}/metadata.json", content)

# Binary file operations
async def save_video_file(video_id: str, video_data: bytes) -> None:
    """Save video file asynchronously."""
    await file_manager.write_binary(f"videos/{video_id}/video.mp4", video_data)

async def read_video_file(video_id: str) -> bytes:
    """Read video file asynchronously."""
    return await file_manager.read_binary(f"videos/{video_id}/video.mp4")
```

#### **File Processing Pipeline**
```python
async def process_video_files(video_id: str) -> Dict[str, Any]:
    """Process video files asynchronously."""
    # Read video script
    script = await file_manager.read_file(f"videos/{video_id}/script.txt")
    
    # Read video metadata
    metadata_content = await file_manager.read_file(f"videos/{video_id}/metadata.json")
    metadata = json.loads(metadata_content)
    
    # Process video file
    video_data = await file_manager.read_binary(f"videos/{video_id}/video.mp4")
    processed_video = await process_video_data(video_data)
    
    # Save processed video
    await file_manager.write_binary(f"videos/{video_id}/processed.mp4", processed_video)
    
    return {
        "script": script,
        "metadata": metadata,
        "processed": True
    }
```

#### **Concurrent File Operations**
```python
async def process_multiple_videos(video_ids: List[str]) -> List[Dict[str, Any]]:
    """Process multiple videos concurrently."""
    tasks = []
    
    for video_id in video_ids:
        task = process_video_files(video_id)
        tasks.append(task)
    
    # Execute all file operations concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"File processing failed: {result}")
            processed_results.append({"error": str(result)})
        else:
            processed_results.append(result)
    
    return processed_results
```

## 🔴 Async Redis Operations

### Overview

All Redis operations should be asynchronous for optimal performance.

### Async Redis Manager

#### **Basic Redis Operations**
```python
from api.async_io.async_operations import AsyncRedisManager

# Initialize async Redis manager
redis_manager = AsyncRedisManager(redis_url="redis://localhost:6379")

# Async get/set operations
async def cache_user_data(user_id: str, user_data: Dict[str, Any]) -> None:
    """Cache user data asynchronously."""
    await redis_manager.set(f"user:{user_id}", json.dumps(user_data), ttl=3600)

async def get_cached_user_data(user_id: str) -> Optional[Dict[str, Any]]:
    """Get cached user data asynchronously."""
    data = await redis_manager.get(f"user:{user_id}")
    return json.loads(data) if data else None

# Delete cached data
async def invalidate_user_cache(user_id: str) -> None:
    """Invalidate user cache asynchronously."""
    await redis_manager.delete(f"user:{user_id}")
```

#### **Session Management**
```python
async def create_user_session(user_id: str, session_data: Dict[str, Any]) -> str:
    """Create user session asynchronously."""
    session_id = str(uuid.uuid4())
    session_key = f"session:{session_id}"
    
    await redis_manager.set(
        session_key,
        json.dumps({
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **session_data
        }),
        ttl=86400  # 24 hours
    )
    
    return session_id

async def get_user_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get user session asynchronously."""
    session_data = await redis_manager.get(f"session:{session_id}")
    return json.loads(session_data) if session_data else None
```

#### **Rate Limiting**
```python
async def check_rate_limit(user_id: str, action: str, limit: int, window: int) -> bool:
    """Check rate limit asynchronously."""
    key = f"rate_limit:{user_id}:{action}"
    
    # Get current count
    current_count = await redis_manager.get(key)
    count = int(current_count) if current_count else 0
    
    if count >= limit:
        return False
    
    # Increment count
    await redis_manager.set(key, str(count + 1), ttl=window)
    return True
```

## 🎯 Async Decorators

### Overview

Async decorators provide utilities for optimizing async functions and preventing blocking operations.

### Timeout and Retry Decorators

#### **Async Timeout**
```python
from api.async_io.async_decorators import async_timeout, AsyncTimeoutStrategy

@async_timeout(timeout=30, strategy=AsyncTimeoutStrategy.RAISE_ERROR)
async def create_video_with_timeout(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create video with timeout protection."""
    return await api_manager.create_video(video_data)

@async_timeout(timeout=10, strategy=AsyncTimeoutStrategy.RETURN_DEFAULT, default_value=None)
async def get_user_data_with_fallback(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user data with timeout fallback."""
    return await user_service.get_user(user_id)
```

#### **Async Retry**
```python
from api.async_io.async_decorators import async_retry, AsyncRetryStrategy

@async_retry(
    max_retries=3,
    strategy=AsyncRetryStrategy.EXPONENTIAL_BACKOFF,
    retry_delay=1.0,
    exceptions=(ConnectionError, TimeoutError)
)
async def create_video_with_retry(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create video with retry logic."""
    return await api_manager.create_video(video_data)

@async_retry(
    max_retries=5,
    strategy=AsyncRetryStrategy.LINEAR_BACKOFF,
    retry_delay=2.0
)
async def download_video_with_retry(video_id: str, output_path: str) -> None:
    """Download video with retry logic."""
    await api_manager.download_video(video_id, output_path)
```

### Performance and Caching Decorators

#### **Performance Monitoring**
```python
from api.async_io.async_decorators import async_performance_monitor

@async_performance_monitor(
    operation_name="video_creation",
    log_slow_operations=True,
    slow_operation_threshold_ms=5000.0
)
async def create_video_monitored(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create video with performance monitoring."""
    return await api_manager.create_video(video_data)
```

#### **Async Caching**
```python
from api.async_io.async_decorators import async_cache

@async_cache(ttl=300)
async def get_user_profile(user_id: str) -> Dict[str, Any]:
    """Get user profile with caching."""
    return await user_service.get_profile(user_id)

def generate_video_cache_key(video_id: str, include_analytics: bool = False) -> str:
    """Generate custom cache key for video data."""
    return f"video:{video_id}:analytics:{include_analytics}"

@async_cache(ttl=600, key_generator=generate_video_cache_key)
async def get_video_data(video_id: str, include_analytics: bool = False) -> Dict[str, Any]:
    """Get video data with custom cache key."""
    return await video_service.get_video_data(video_id, include_analytics)
```

### Rate Limiting and Circuit Breaker

#### **Rate Limiting**
```python
from api.async_io.async_decorators import async_rate_limit

@async_rate_limit(max_calls=10, time_window=60)
async def create_video_rate_limited(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create video with rate limiting."""
    return await api_manager.create_video(video_data)
```

#### **Circuit Breaker**
```python
from api.async_io.async_decorators import async_circuit_breaker

@async_circuit_breaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=(ConnectionError, TimeoutError)
)
async def create_video_with_circuit_breaker(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create video with circuit breaker protection."""
    return await api_manager.create_video(video_data)
```

### Batch and Concurrent Processing

#### **Batch Processing**
```python
from api.async_io.async_decorators import async_batch_processor

@async_batch_processor(batch_size=50, max_concurrent=10)
async def process_videos_batch(videos_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process videos in batches."""
    return await video_service.process_videos(videos_data)

@async_batch_processor(batch_size=100)
async def create_videos_batch(videos_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create videos in batches."""
    return await api_manager.create_videos_batch(videos_data)
```

#### **Concurrent Processing**
```python
from api.async_io.async_decorators import async_concurrent_processor

@async_concurrent_processor(max_concurrent=20)
async def process_video_concurrent(video_id: str) -> Dict[str, Any]:
    """Process video concurrently."""
    return await video_service.process_video(video_id)

@async_concurrent_processor(max_concurrent=50, return_exceptions=True)
async def download_videos_concurrent(video_ids: List[str]) -> List[Dict[str, Any]]:
    """Download videos concurrently."""
    return await api_manager.download_videos_batch(video_ids)
```

### Background Tasks

#### **Fire and Forget**
```python
from api.async_io.async_decorators import async_background_task

@async_background_task(task_name="video_processing", fire_and_forget=True)
async def process_video_background(video_id: str) -> None:
    """Process video in background."""
    try:
        video_data = await video_service.get_video(video_id)
        processed_video = await video_processor.process(video_data)
        await video_service.update_video(video_id, processed_video)
        await notification_service.send_video_ready_notification(video_id)
    except Exception as e:
        logger.error(f"Background video processing failed: {e}")
        await notification_service.send_error_notification(video_id, str(e))
```

#### **Tracked Background Tasks**
```python
@async_background_task(task_name="video_analytics", fire_and_forget=False)
async def generate_video_analytics(video_id: str) -> Dict[str, Any]:
    """Generate video analytics in background."""
    analytics = await analytics_service.generate_analytics(video_id)
    await analytics_service.save_analytics(video_id, analytics)
    return analytics

# Use in endpoint
@router.post("/videos/{video_id}/analytics")
async def start_analytics_generation(video_id: str):
    """Start analytics generation in background."""
    task = await generate_video_analytics(video_id)
    return {"task_id": task.get_name(), "status": "started"}
```

### Combined Optimizations

#### **Fully Optimized Function**
```python
from api.async_io.async_decorators import async_optimized, AsyncDecoratorConfig

config = AsyncDecoratorConfig(
    timeout=30.0,
    timeout_strategy=AsyncTimeoutStrategy.RAISE_ERROR,
    max_retries=3,
    retry_strategy=AsyncRetryStrategy.EXPONENTIAL_BACKOFF,
    retry_delay=1.0,
    log_errors=True,
    log_performance=True,
    cache_result=True,
    cache_ttl=300
)

@async_optimized(config)
async def get_user_videos_optimized(user_id: str) -> List[Dict[str, Any]]:
    """Get user videos with full optimization."""
    return await video_service.get_user_videos(user_id)
```

## 🔗 Integration Examples

### FastAPI Application Setup

#### **Main Application Configuration**
```python
from fastapi import FastAPI, Depends
from api.async_io.async_operations import AsyncOperationsManager

app = FastAPI(title="HeyGen AI API")

# Initialize async operations manager
@app.on_event("startup")
async def startup_event():
    async_manager = AsyncOperationsManager(
        database_url=os.getenv("DATABASE_URL"),
        redis_url=os.getenv("REDIS_URL"),
        external_api_url=os.getenv("HEYGEN_API_URL"),
        external_api_key=os.getenv("HEYGEN_API_KEY"),
        file_base_path="/tmp"
    )
    app.state.async_manager = async_manager

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.async_manager.cleanup()

# Dependency injection
def get_async_manager() -> AsyncOperationsManager:
    return app.state.async_manager
```

#### **Optimized Endpoints**
```python
from api.async_io.async_decorators import (
    async_timeout, async_retry, async_performance_monitor, async_cache
)

@router.get("/users/{user_id}")
@async_timeout(timeout=10)
@async_cache(ttl=300)
@async_performance_monitor()
async def get_user(
    user_id: str,
    async_manager: AsyncOperationsManager = Depends(get_async_manager)
):
    """Get user with async optimization."""
    return await async_manager.db_manager.execute_query(
        "SELECT * FROM users WHERE id = :user_id",
        {"user_id": user_id}
    )

@router.post("/videos")
@async_timeout(timeout=60)
@async_retry(max_retries=3)
@async_performance_monitor()
async def create_video(
    video_data: VideoCreateRequest,
    async_manager: AsyncOperationsManager = Depends(get_async_manager)
):
    """Create video with async optimization."""
    # Create video in database
    video_id = await async_manager.db_manager.execute_query(
        """
        INSERT INTO videos (title, description, user_id, created_at)
        VALUES (:title, :description, :user_id, :created_at)
        RETURNING id
        """,
        {
            "title": video_data.title,
            "description": video_data.description,
            "user_id": video_data.user_id,
            "created_at": datetime.now(timezone.utc)
        }
    )
    
    # Create video in external API
    api_result = await async_manager.external_api_manager.create_video({
        "title": video_data.title,
        "script": video_data.script,
        "template": video_data.template
    })
    
    # Cache result
    await async_manager.redis_manager.set(
        f"video:{video_id}",
        json.dumps(api_result),
        ttl=3600
    )
    
    return {"video_id": video_id, "api_result": api_result}
```

### Service Layer Integration

#### **Optimized User Service**
```python
class AsyncUserService:
    def __init__(self, async_manager: AsyncOperationsManager):
        self.async_manager = async_manager
    
    @async_cache(ttl=600)
    @async_performance_monitor()
    async def get_user(self, user_id: str) -> Dict[str, Any]:
        """Get user with caching and monitoring."""
        # Try cache first
        cached_user = await self.async_manager.redis_manager.get(f"user:{user_id}")
        if cached_user:
            return json.loads(cached_user)
        
        # Query database
        users = await self.async_manager.db_manager.execute_query(
            "SELECT * FROM users WHERE id = :user_id",
            {"user_id": user_id}
        )
        
        if not users:
            raise ValueError("User not found")
        
        user_data = users[0]
        
        # Cache result
        await self.async_manager.redis_manager.set(
            f"user:{user_id}",
            json.dumps(user_data),
            ttl=600
        )
        
        return user_data
    
    @async_performance_monitor()
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create user with monitoring."""
        # Create user in database
        users = await self.async_manager.db_manager.execute_query(
            """
            INSERT INTO users (email, first_name, last_name, created_at)
            VALUES (:email, :first_name, :last_name, :created_at)
            RETURNING *
            """,
            {
                **user_data,
                "created_at": datetime.now(timezone.utc)
            }
        )
        
        user = users[0]
        
        # Cache user data
        await self.async_manager.redis_manager.set(
            f"user:{user['id']}",
            json.dumps(user),
            ttl=600
        )
        
        return user
```

#### **Optimized Video Service**
```python
class AsyncVideoService:
    def __init__(self, async_manager: AsyncOperationsManager):
        self.async_manager = async_manager
    
    @async_batch_processor(batch_size=50)
    async def get_user_videos(self, user_id: str, page: int = 1, per_page: int = 20) -> Dict[str, Any]:
        """Get user videos with batch processing."""
        offset = (page - 1) * per_page
        
        # Get total count
        count_result = await self.async_manager.db_manager.execute_query(
            "SELECT COUNT(*) as total FROM videos WHERE user_id = :user_id",
            {"user_id": user_id}
        )
        total = count_result[0]["total"]
        
        # Get videos
        videos = await self.async_manager.db_manager.execute_query(
            """
            SELECT v.*, u.name as user_name
            FROM videos v
            JOIN users u ON v.user_id = u.id
            WHERE v.user_id = :user_id
            ORDER BY v.created_at DESC
            LIMIT :limit OFFSET :offset
            """,
            {"user_id": user_id, "limit": per_page, "offset": offset}
        )
        
        return {
            "videos": videos,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page
            }
        }
    
    @async_timeout(timeout=120)
    @async_retry(max_retries=3)
    @async_performance_monitor()
    async def create_video(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create video with full optimization."""
        # Save video metadata to file
        await self.async_manager.file_manager.write_file(
            f"videos/{video_data['id']}/metadata.json",
            json.dumps(video_data, indent=2)
        )
        
        # Create video in external API
        api_result = await self.async_manager.external_api_manager.create_video(video_data)
        
        # Update database
        await self.async_manager.db_manager.execute_query(
            """
            UPDATE videos 
            SET api_video_id = :api_video_id, status = :status, updated_at = :updated_at
            WHERE id = :video_id
            """,
            {
                "video_id": video_data["id"],
                "api_video_id": api_result["id"],
                "status": "processing",
                "updated_at": datetime.now(timezone.utc)
            }
        )
        
        return api_result
```

## 🏆 Best Practices

### 1. Always Use Async Operations

#### **✅ Good: Async Database Operations**
```python
# Use async session context manager
async with db_manager.get_session() as session:
    result = await session.execute(text("SELECT * FROM users"))
    users = result.fetchall()

# Use async query execution
users = await db_manager.execute_query("SELECT * FROM users")
```

#### **❌ Bad: Blocking Operations**
```python
# Don't use sync database operations
def get_users_sync():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM users"))
        return result.fetchall()

# Don't use sync file operations
def read_file_sync(path):
    with open(path, 'r') as f:
        return f.read()
```

### 2. Proper Error Handling

#### **Async Error Handling**
```python
@async_retry(max_retries=3, exceptions=(ConnectionError, TimeoutError))
async def create_video_with_error_handling(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create video with proper error handling."""
    try:
        return await api_manager.create_video(video_data)
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        raise
    except TimeoutError as e:
        logger.error(f"Timeout error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### 3. Resource Management

#### **Proper Resource Cleanup**
```python
async def process_video_with_cleanup(video_id: str) -> None:
    """Process video with proper resource cleanup."""
    temp_files = []
    
    try:
        # Download video
        video_data = await api_manager.download_video(video_id, f"/tmp/{video_id}.mp4")
        temp_files.append(f"/tmp/{video_id}.mp4")
        
        # Process video
        processed_data = await video_processor.process(video_data)
        
        # Save processed video
        await file_manager.write_binary(f"videos/{video_id}/processed.mp4", processed_data)
        
    finally:
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                await file_manager.delete_file(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {e}")
```

### 4. Performance Optimization

#### **Concurrent Operations**
```python
async def optimize_video_processing(video_ids: List[str]) -> List[Dict[str, Any]]:
    """Optimize video processing with concurrent operations."""
    # Process videos concurrently
    tasks = [
        process_single_video(video_id)
        for video_id in video_ids
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle results
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Video processing failed: {result}")
            processed_results.append({"error": str(result)})
        else:
            processed_results.append(result)
    
    return processed_results
```

### 5. Monitoring and Logging

#### **Comprehensive Monitoring**
```python
@async_performance_monitor(
    operation_name="video_creation_pipeline",
    log_slow_operations=True,
    slow_operation_threshold_ms=30000.0
)
@async_logging(log_input=True, log_output=False, log_errors=True)
async def create_video_pipeline(video_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create video with comprehensive monitoring."""
    # Step 1: Validate input
    await validate_video_data(video_data)
    
    # Step 2: Create video in database
    db_result = await db_manager.execute_query(
        "INSERT INTO videos (...) VALUES (...) RETURNING *",
        video_data
    )
    
    # Step 3: Create video in external API
    api_result = await api_manager.create_video(video_data)
    
    # Step 4: Cache result
    await redis_manager.set(f"video:{db_result[0]['id']}", json.dumps(api_result))
    
    return {"db_result": db_result[0], "api_result": api_result}
```

## 📊 Performance Monitoring

### Async Operations Tracking

#### **Operation Statistics**
```python
async def get_async_operation_stats() -> Dict[str, Any]:
    """Get statistics for all async operations."""
    async_manager = get_async_manager()
    return async_manager.get_operation_stats()

# Example output:
# {
#     "total_operations": 1250,
#     "successful_operations": 1180,
#     "failed_operations": 70,
#     "success_rate": 0.944,
#     "average_duration_ms": 245.6,
#     "operations_by_type": {
#         "database": 450,
#         "http_request": 300,
#         "file_operation": 200,
#         "redis_operation": 300
#     }
# }
```

#### **Performance Alerts**
```python
async def monitor_async_performance():
    """Monitor async operation performance."""
    stats = await get_async_operation_stats()
    
    # Alert on high failure rate
    if stats["success_rate"] < 0.95:
        logger.warning(f"Low success rate: {stats['success_rate']:.2%}")
    
    # Alert on slow operations
    if stats["average_duration_ms"] > 1000:
        logger.warning(f"Slow operations detected: {stats['average_duration_ms']:.2f}ms")
    
    # Alert on specific operation types
    for op_type, count in stats["operations_by_type"].items():
        if count > 1000:  # High volume
            logger.info(f"High volume of {op_type} operations: {count}")
```

### Health Checks

#### **Async Health Monitoring**
```python
async def check_async_health() -> Dict[str, str]:
    """Check health of all async components."""
    async_manager = get_async_manager()
    
    health_status = {}
    
    # Check database
    try:
        await async_manager.db_manager.execute_query("SELECT 1")
        health_status["database"] = "healthy"
    except Exception as e:
        health_status["database"] = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        await async_manager.redis_manager.get("health_check")
        health_status["redis"] = "healthy"
    except Exception as e:
        health_status["redis"] = f"unhealthy: {str(e)}"
    
    # Check external API
    try:
        await async_manager.external_api_manager.get_video_status("test")
        health_status["external_api"] = "healthy"
    except Exception as e:
        health_status["external_api"] = f"unhealthy: {str(e)}"
    
    return health_status
```

## 📈 Expected Performance Improvements

### 1. Response Time Reduction
- **Database Operations**: 60-80% reduction in query time
- **HTTP Requests**: 70-90% reduction in API call time
- **File Operations**: 50-70% reduction in I/O time
- **Concurrent Processing**: 80-95% reduction for batch operations

### 2. Throughput Increase
- **Concurrent Requests**: 5-10x increase in handling capacity
- **Batch Processing**: 3-5x increase in processing speed
- **Resource Utilization**: 2-4x better resource efficiency

### 3. Scalability
- **Horizontal Scaling**: Better support for multiple instances
- **Load Distribution**: Improved handling of traffic spikes
- **Resource Efficiency**: More efficient use of available resources

### 4. Reliability
- **Error Recovery**: Automatic retry and circuit breaker patterns
- **Graceful Degradation**: Fallback mechanisms for service failures
- **Monitoring**: Comprehensive performance tracking and alerting

## 🚀 Next Steps

1. **Implement async operations** in all database calls
2. **Convert external API calls** to async patterns
3. **Optimize file operations** with async I/O
4. **Add async decorators** for performance optimization
5. **Set up monitoring** for async operation performance
6. **Implement error handling** and retry mechanisms
7. **Add health checks** for all async components

This comprehensive async I/O optimization system provides your HeyGen AI API with:
- **Non-blocking operations** for all I/O tasks
- **Concurrent processing** for improved performance
- **Automatic retry logic** for reliability
- **Performance monitoring** for optimization insights
- **Resource management** for efficient operation
- **Error handling** for graceful degradation

The system is designed to maximize performance while maintaining reliability and scalability across all components. 