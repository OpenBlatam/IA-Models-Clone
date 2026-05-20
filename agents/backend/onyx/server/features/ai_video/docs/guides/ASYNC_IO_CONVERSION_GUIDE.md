# 🚀 ASYNC I/O CONVERSION GUIDE

## Overview

This guide provides comprehensive patterns and examples for converting all blocking I/O operations to asynchronous operations in the AI Video system. The goal is to ensure zero blocking operations for maximum performance and scalability.

## Table of Contents

1. [Core Principles](#core-principles)
2. [Database Operations](#database-operations)
3. [HTTP/API Operations](#httpapi-operations)
4. [File I/O Operations](#file-io-operations)
5. [Redis/Cache Operations](#rediscache-operations)
6. [Third-Party Library Integration](#third-party-library-integration)
7. [Performance Optimization](#performance-optimization)
8. [Best Practices](#best-practices)
9. [Migration Checklist](#migration-checklist)

## Core Principles

### ✅ Always Use Async For:
- Database queries and transactions
- HTTP/API requests
- File read/write operations
- Network operations
- External service calls
- Cache operations
- Long-running computations

### ❌ Never Use Sync For:
- I/O operations in async context
- Blocking calls in event loop
- Synchronous database drivers
- Synchronous HTTP clients
- Synchronous file operations

## Database Operations

### SQLAlchemy Async Patterns

```python
# ✅ GOOD: Async SQLAlchemy
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Initialize async engine
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/db",
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)

async_session = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Async database operations
async def get_video_by_id(video_id: str) -> Optional[Video]:
    async with async_session() as session:
        result = await session.execute(
            select(Video).where(Video.id == video_id)
        )
        return result.scalar_one_or_none()

async def create_video(video_data: Dict[str, Any]) -> Video:
    async with async_session() as session:
        video = Video(**video_data)
        session.add(video)
        await session.commit()
        await session.refresh(video)
        return video

async def update_video_status(video_id: str, status: str) -> bool:
    async with async_session() as session:
        result = await session.execute(
            update(Video)
            .where(Video.id == video_id)
            .values(status=status)
        )
        await session.commit()
        return result.rowcount > 0
```

### Raw SQL with Async Drivers

```python
# ✅ GOOD: Async PostgreSQL
import asyncpg

async def get_videos_async():
    conn = await asyncpg.connect(
        user='user',
        password='pass',
        database='db',
        host='localhost'
    )
    
    try:
        rows = await conn.fetch(
            'SELECT * FROM videos WHERE status = $1',
            'processing'
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()

# ✅ GOOD: Async MySQL
import aiomysql

async def get_user_videos_async(user_id: str):
    conn = await aiomysql.connect(
        host='localhost',
        user='user',
        password='pass',
        db='database'
    )
    
    try:
        async with conn.cursor() as cursor:
            await cursor.execute(
                'SELECT * FROM videos WHERE user_id = %s',
                (user_id,)
            )
            rows = await cursor.fetchall()
            return rows
    finally:
        conn.close()
        await conn.wait_closed()
```

### ❌ BAD: Synchronous Database Operations

```python
# ❌ BAD: Synchronous SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql://user:pass@localhost/db")
Session = sessionmaker(bind=engine)

def get_video_by_id_sync(video_id: str):  # Blocking!
    session = Session()
    try:
        return session.query(Video).filter_by(id=video_id).first()
    finally:
        session.close()

# ❌ BAD: Synchronous raw SQL
import psycopg2

def get_videos_sync():  # Blocking!
    conn = psycopg2.connect(
        dbname="db",
        user="user",
        password="pass",
        host="localhost"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM videos")
    return cursor.fetchall()
```

## HTTP/API Operations

### Async HTTP Clients

```python
# ✅ GOOD: aiohttp
import aiohttp

async def fetch_video_data_async(video_id: str) -> Dict[str, Any]:
    async with aiohttp.ClientSession() as session:
        async with session.get(f'/api/videos/{video_id}') as response:
            response.raise_for_status()
            return await response.json()

async def upload_video_async(video_data: bytes, url: str) -> bool:
    async with aiohttp.ClientSession() as session:
        async with session.post(url, data=video_data) as response:
            return response.status == 200

# ✅ GOOD: httpx
import httpx

async def call_external_api_async():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
        return response.json()

# ✅ GOOD: Multiple concurrent requests
async def fetch_multiple_videos_async(video_ids: List[str]) -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_video_data_async(vid, session) 
            for vid in video_ids
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
```

### ❌ BAD: Synchronous HTTP Operations

```python
# ❌ BAD: requests library
import requests

def fetch_video_data_sync(video_id: str):  # Blocking!
    response = requests.get(f'/api/videos/{video_id}')
    return response.json()

def upload_video_sync(video_data: bytes, url: str):  # Blocking!
    response = requests.post(url, data=video_data)
    return response.status_code == 200
```

## File I/O Operations

### Async File Operations

```python
# ✅ GOOD: aiofiles
import aiofiles

async def read_video_file_async(file_path: str) -> bytes:
    async with aiofiles.open(file_path, 'rb') as f:
        return await f.read()

async def write_video_file_async(file_path: str, data: bytes) -> bool:
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(data)
        return True
    except Exception as e:
        logger.error(f"Failed to write file: {e}")
        return False

async def append_log_async(log_file: str, message: str) -> None:
    async with aiofiles.open(log_file, 'a') as f:
        await f.write(f"{time.time()}: {message}\n")

# ✅ GOOD: Multiple file operations
async def process_video_files_async(file_paths: List[str]) -> List[bytes]:
    tasks = [read_video_file_async(path) for path in file_paths]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### ❌ BAD: Synchronous File Operations

```python
# ❌ BAD: Synchronous file operations
def read_video_file_sync(file_path: str) -> bytes:  # Blocking!
    with open(file_path, 'rb') as f:
        return f.read()

def write_video_file_sync(file_path: str, data: bytes) -> bool:  # Blocking!
    with open(file_path, 'wb') as f:
        f.write(data)
    return True
```

## Redis/Cache Operations

### Async Redis Operations

```python
# ✅ GOOD: aioredis
import aioredis

async def get_cached_video_async(video_id: str) -> Optional[Dict]:
    redis_client = aioredis.from_url("redis://localhost")
    try:
        data = await redis_client.get(f"video:{video_id}")
        return json.loads(data) if data else None
    finally:
        await redis_client.close()

async def set_cached_video_async(video_id: str, data: Dict, ttl: int = 3600):
    redis_client = aioredis.from_url("redis://localhost")
    try:
        await redis_client.setex(
            f"video:{video_id}",
            ttl,
            json.dumps(data)
        )
    finally:
        await redis_client.close()

# ✅ GOOD: Redis connection pooling
class AsyncRedisManager:
    def __init__(self):
        self.redis_pool = None
    
    async def initialize(self):
        self.redis_pool = aioredis.from_url(
            "redis://localhost",
            encoding="utf-8",
            decode_responses=True
        )
    
    async def get(self, key: str) -> Optional[str]:
        return await self.redis_pool.get(key)
    
    async def set(self, key: str, value: str, ttl: Optional[int] = None):
        if ttl:
            await self.redis_pool.setex(key, ttl, value)
        else:
            await self.redis_pool.set(key, value)
```

### ❌ BAD: Synchronous Redis Operations

```python
# ❌ BAD: Synchronous Redis
import redis

def get_cached_video_sync(video_id: str):  # Blocking!
    r = redis.Redis(host='localhost', port=6379, db=0)
    data = r.get(f"video:{video_id}")
    return json.loads(data) if data else None

def set_cached_video_sync(video_id: str, data: Dict):  # Blocking!
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.setex(f"video:{video_id}", 3600, json.dumps(data))
```

## Third-Party Library Integration

### Converting Sync Libraries to Async

```python
# ✅ GOOD: Converting sync libraries
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncLibraryWrapper:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def sync_to_async(self, func, *args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args, **kwargs)

# Example: Converting PIL operations
from PIL import Image
import io

async def process_image_async(image_data: bytes) -> bytes:
    def process_image_sync(data: bytes) -> bytes:
        img = Image.open(io.BytesIO(data))
        img = img.resize((512, 512))
        output = io.BytesIO()
        img.save(output, format='JPEG')
        return output.getvalue()
    
    return await AsyncLibraryWrapper().sync_to_async(
        process_image_sync, image_data
    )

# Example: Converting numpy operations
import numpy as np

async def process_array_async(data: np.ndarray) -> np.ndarray:
    def process_array_sync(arr: np.ndarray) -> np.ndarray:
        return np.mean(arr, axis=0)
    
    return await AsyncLibraryWrapper().sync_to_async(
        process_array_sync, data
    )
```

### CPU-Bound Operations

```python
# ✅ GOOD: CPU-bound operations in process pool
from concurrent.futures import ProcessPoolExecutor

async def cpu_intensive_operation_async(data: List[float]) -> float:
    def cpu_intensive_sync(numbers: List[float]) -> float:
        # CPU-intensive computation
        return sum(x ** 2 for x in numbers) / len(numbers)
    
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as executor:
        return await loop.run_in_executor(executor, cpu_intensive_sync, data)
```

## Performance Optimization

### Connection Pooling

```python
# ✅ GOOD: Connection pooling for all services
class AsyncConnectionManager:
    def __init__(self):
        self.db_pool = None
        self.redis_pool = None
        self.http_session = None
    
    async def initialize(self):
        # Database pool
        self.db_pool = await asyncpg.create_pool(
            user='user',
            password='pass',
            database='db',
            host='localhost',
            min_size=5,
            max_size=20
        )
        
        # Redis pool
        self.redis_pool = aioredis.from_url(
            "redis://localhost",
            encoding="utf-8",
            decode_responses=True
        )
        
        # HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30
        )
        self.http_session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector
        )
    
    async def cleanup(self):
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_pool:
            await self.redis_pool.close()
        if self.http_session:
            await self.http_session.close()
```

### Batch Operations

```python
# ✅ GOOD: Batch database operations
async def batch_create_videos_async(videos: List[Dict]) -> List[Video]:
    async with async_session() as session:
        video_objects = [Video(**video_data) for video_data in videos]
        session.add_all(video_objects)
        await session.commit()
        
        # Refresh all objects
        for video in video_objects:
            await session.refresh(video)
        
        return video_objects

# ✅ GOOD: Batch HTTP requests
async def batch_fetch_videos_async(video_ids: List[str]) -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        tasks = []
        for video_id in video_ids:
            task = session.get(f'/api/videos/{video_id}')
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for response in responses:
            if isinstance(response, Exception):
                results.append(None)
            else:
                data = await response.json()
                results.append(data)
        
        return results
```

### Error Handling and Retries

```python
# ✅ GOOD: Robust async operations with retries
async def robust_async_operation(operation: Callable, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            delay = 2 ** attempt  # Exponential backoff
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
            await asyncio.sleep(delay)

# Usage example
async def fetch_video_with_retry(video_id: str) -> Dict:
    async def fetch_operation():
        async with aiohttp.ClientSession() as session:
            async with session.get(f'/api/videos/{video_id}') as response:
                response.raise_for_status()
                return await response.json()
    
    return await robust_async_operation(fetch_operation)
```

## Best Practices

### 1. Always Use Async Context Managers

```python
# ✅ GOOD: Proper resource management
async def process_video_async(video_id: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(f'/api/videos/{video_id}') as response:
            data = await response.json()
    
    async with aiofiles.open(f'output_{video_id}.json', 'w') as f:
        await f.write(json.dumps(data))
```

### 2. Avoid Blocking in Async Functions

```python
# ❌ BAD: Blocking in async function
async def bad_async_function():
    time.sleep(1)  # Blocking!
    requests.get('http://api.example.com')  # Blocking!

# ✅ GOOD: Non-blocking async function
async def good_async_function():
    await asyncio.sleep(1)  # Non-blocking
    async with aiohttp.ClientSession() as session:
        async with session.get('http://api.example.com') as response:
            return await response.json()
```

### 3. Use Proper Exception Handling

```python
# ✅ GOOD: Proper async exception handling
async def safe_async_operation():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://api.example.com') as response:
                response.raise_for_status()
                return await response.json()
    except aiohttp.ClientError as e:
        logger.error(f"HTTP error: {e}")
        raise
    except asyncio.TimeoutError:
        logger.error("Request timeout")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
```

### 4. Optimize for Concurrency

```python
# ✅ GOOD: Concurrent operations
async def process_multiple_videos_async(video_ids: List[str]):
    tasks = []
    for video_id in video_ids:
        task = process_single_video_async(video_id)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]
```

## Migration Checklist

### Database Operations
- [ ] Replace `psycopg2` with `asyncpg`
- [ ] Replace `pymysql` with `aiomysql`
- [ ] Convert SQLAlchemy to async version
- [ ] Update all database queries to use async patterns
- [ ] Implement connection pooling

### HTTP Operations
- [ ] Replace `requests` with `aiohttp` or `httpx`
- [ ] Convert all API calls to async
- [ ] Implement proper session management
- [ ] Add timeout handling
- [ ] Implement retry mechanisms

### File Operations
- [ ] Replace `open()` with `aiofiles.open()`
- [ ] Convert all file read/write operations
- [ ] Implement async file processing
- [ ] Add proper error handling

### Cache Operations
- [ ] Replace `redis` with `aioredis`
- [ ] Convert all cache operations to async
- [ ] Implement connection pooling
- [ ] Add proper error handling

### Third-Party Libraries
- [ ] Identify all blocking operations
- [ ] Wrap sync operations with `asyncio.to_thread()`
- [ ] Use `ProcessPoolExecutor` for CPU-bound operations
- [ ] Test all conversions thoroughly

### Testing
- [ ] Test all async operations
- [ ] Verify no blocking operations remain
- [ ] Test error handling and retries
- [ ] Performance testing
- [ ] Load testing

## Performance Monitoring

```python
# ✅ GOOD: Async operation monitoring
import time
import asyncio

class AsyncOperationMonitor:
    def __init__(self):
        self.operation_times = {}
    
    async def monitor_operation(self, operation_name: str, operation: Callable):
        start_time = time.time()
        try:
            result = await operation()
            duration = time.time() - start_time
            
            if operation_name not in self.operation_times:
                self.operation_times[operation_name] = []
            
            self.operation_times[operation_name].append(duration)
            
            logger.info(f"Operation {operation_name} completed in {duration:.3f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Operation {operation_name} failed after {duration:.3f}s: {e}")
            raise
    
    def get_stats(self):
        stats = {}
        for operation, times in self.operation_times.items():
            if times:
                stats[operation] = {
                    'count': len(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        return stats

# Usage
monitor = AsyncOperationMonitor()

async def monitored_operation():
    return await monitor.monitor_operation(
        "fetch_video",
        lambda: fetch_video_async("video_123")
    )
```

## Conclusion

By following this guide and converting all blocking I/O operations to asynchronous operations, your AI Video system will achieve:

1. **Maximum Concurrency** - Handle thousands of concurrent operations
2. **Better Resource Utilization** - Efficient use of system resources
3. **Improved Responsiveness** - No blocking operations in the event loop
4. **Scalability** - System can handle increased load without performance degradation
5. **Reliability** - Proper error handling and retry mechanisms

Remember: **Every I/O operation should be asynchronous in an async application.** 