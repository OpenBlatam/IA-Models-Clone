# Async Database and External API Operations Guide

## Overview

This guide covers the dedicated async functions for database and external API operations in the Video-OpusClip system. These modules provide optimized, non-blocking operations with connection pooling, caching, rate limiting, and comprehensive error handling.

## Table of Contents

1. [Database Operations](#database-operations)
2. [External API Operations](#external-api-operations)
3. [Connection Pooling](#connection-pooling)
4. [Caching Strategies](#caching-strategies)
5. [Rate Limiting](#rate-limiting)
6. [Error Handling](#error-handling)
7. [Batch Operations](#batch-operations)
8. [Transaction Management](#transaction-management)
9. [Metrics and Monitoring](#metrics-and-monitoring)
10. [Integration Examples](#integration-examples)
11. [Best Practices](#best-practices)
12. [Performance Optimization](#performance-optimization)

## Database Operations

### Core Components

The async database system provides:

- **Connection Pooling**: Efficient resource management
- **Query Optimization**: Caching and query planning
- **Transaction Management**: ACID compliance
- **Batch Operations**: High-performance bulk operations
- **Metrics Collection**: Performance monitoring

### Supported Databases

| Database | Driver | Features |
|----------|--------|----------|
| PostgreSQL | asyncpg | Full async support, JSONB, advanced features |
| MySQL | aiomysql | Connection pooling, prepared statements |
| SQLite | aiosqlite | Lightweight, file-based |
| Redis | aioredis | Caching, pub/sub, data structures |

### Basic Usage

```python
from async_database import (
    create_database_config,
    create_database_pool,
    create_async_database_operations,
    DatabaseType
)

# Create configuration
config = create_database_config(
    host="localhost",
    port=5432,
    database="video_opusclip",
    username="postgres",
    password="password",
    max_connections=20
)

# Create pool
pool = create_database_pool(config, DatabaseType.POSTGRESQL)
await pool.initialize()

# Create operations
db_ops = create_async_database_operations(pool)

# Execute query
result = await db_ops.execute_query(
    "SELECT * FROM videos WHERE status = $1",
    ["pending"],
    query_type=QueryType.SELECT,
    cache_key="videos:pending",
    cache_ttl=300
)
```

### Video-Specific Operations

```python
from async_database import create_async_video_database

# Create video database operations
video_db = create_async_video_database(db_ops)

# Create video record
video_id = await video_db.create_video_record({
    "url": "https://youtube.com/watch?v=example",
    "title": "Amazing Video",
    "duration": 180,
    "status": "pending"
})

# Get video by ID (with caching)
video = await video_db.get_video_by_id(video_id)

# Update video status
success = await video_db.update_video_status(video_id, "processing")

# Create clip record
clip_id = await video_db.create_clip_record({
    "video_id": video_id,
    "start_time": 0,
    "end_time": 30,
    "duration": 30,
    "caption": "Amazing clip!",
    "effects": ["fade_in", "fade_out"],
    "file_path": "/path/to/clip.mp4"
})

# Get clips for video
clips = await video_db.get_clips_by_video_id(video_id)
```

### Batch Operations

```python
from async_database import create_async_batch_database_operations

# Create batch operations
batch_ops = create_async_batch_database_operations(db_ops)

# Batch insert videos
videos = [
    {"url": "video1.mp4", "title": "Video 1"},
    {"url": "video2.mp4", "title": "Video 2"},
    {"url": "video3.mp4", "title": "Video 3"}
]

video_ids = await batch_ops.batch_insert_videos(videos)

# Batch insert clips
clips = [
    {"video_id": video_ids[0], "start_time": 0, "end_time": 30},
    {"video_id": video_ids[1], "start_time": 0, "end_time": 30},
    {"video_id": video_ids[2], "start_time": 0, "end_time": 30}
]

clip_ids = await batch_ops.batch_insert_clips(clips)

# Batch update statuses
updates = [
    (video_ids[0], "processing"),
    (video_ids[1], "processing"),
    (video_ids[2], "processing")
]

updated_count = await batch_ops.batch_update_video_status(updates)
```

### Transaction Management

```python
from async_database import create_async_transaction_manager

# Create transaction manager
tx_manager = create_async_transaction_manager(db_ops)

# Execute operations in transaction
async def create_video_with_clips(video_data, clips_data):
    async with tx_manager.transaction() as connection:
        # Create video
        video_id = await video_db.create_video_record(video_data)
        
        # Create clips
        for clip_data in clips_data:
            clip_data["video_id"] = video_id
            await video_db.create_clip_record(clip_data)
        
        return video_id

# Execute transaction
video_id = await create_video_with_clips(video_data, clips_data)
```

## External API Operations

### Core Components

The async external API system provides:

- **HTTP Client Pooling**: Efficient connection management
- **Rate Limiting**: API quota management
- **Caching**: Response caching with TTL
- **Retry Logic**: Automatic retry with exponential backoff
- **Metrics Collection**: Performance monitoring

### Supported APIs

| API | Features | Use Case |
|-----|----------|----------|
| YouTube | Video info, captions, search | Video metadata extraction |
| OpenAI | Text generation, analysis | Content optimization |
| Stability AI | Image generation | Thumbnail creation |
| ElevenLabs | Text-to-speech | Video narration |

### Basic Usage

```python
from async_external_apis import (
    create_api_config,
    create_async_http_client,
    create_async_external_api_operations
)

# Create configuration
config = create_api_config(
    base_url="https://api.youtube.com/v3",
    api_key="your_api_key",
    timeout=30.0,
    max_retries=3,
    rate_limit_per_minute=60
)

# Create HTTP client
http_client = create_async_http_client(config)
await http_client.initialize()

# Create API operations
api_ops = create_async_external_api_operations(http_client)

# Make request
result = await api_ops.make_request(
    HTTPMethod.GET,
    "videos",
    params={"id": "video_id", "part": "snippet"},
    cache_key="youtube:video:video_id",
    use_cache=True
)
```

### YouTube API Operations

```python
from async_external_apis import create_async_youtube_api

# Create YouTube API operations
youtube_api = create_async_youtube_api(api_ops)

# Get video information
video_info = await youtube_api.get_video_info("dQw4w9WgXcQ")

# Get video captions
captions = await youtube_api.get_video_captions("dQw4w9WgXcQ", "en")

# Search for videos
search_results = await youtube_api.search_videos("machine learning", max_results=10)

# Get channel videos
channel_videos = await youtube_api.get_channel_videos("channel_id", max_results=50)
```

### OpenAI API Operations

```python
from async_external_apis import create_async_openai_api

# Create OpenAI API operations
openai_api = create_async_openai_api(api_ops)

# Generate text
response = await openai_api.generate_text(
    prompt="Write a viral caption for a cooking video",
    model="gpt-3.5-turbo",
    max_tokens=1000,
    temperature=0.7
)

# Generate captions
captions = await openai_api.generate_captions(
    audio_text="Today we're making the most amazing pasta dish...",
    style="casual",
    language="en"
)

# Analyze video content
analysis = await openai_api.analyze_video_content(
    video_description="Learn to cook amazing pasta in 5 minutes",
    video_title="5-Minute Pasta Recipe"
)
```

### Stability AI API Operations

```python
from async_external_apis import create_async_stability_ai_api

# Create Stability AI API operations
stability_api = create_async_stability_ai_api(api_ops)

# Generate image
image_result = await stability_api.generate_image(
    prompt="A beautiful sunset over mountains, digital art style",
    width=1024,
    height=1024,
    steps=30
)

# Generate video thumbnail
thumbnail = await stability_api.generate_video_thumbnail(
    video_title="Amazing Cooking Tutorial",
    video_description="Learn to cook delicious pasta in minutes"
)
```

### ElevenLabs API Operations

```python
from async_external_apis import create_async_elevenlabs_api

# Create ElevenLabs API operations
elevenlabs_api = create_async_elevenlabs_api(api_ops)

# Convert text to speech
audio_result = await elevenlabs_api.text_to_speech(
    text="Welcome to our amazing cooking tutorial!",
    voice_id="21m00Tcm4TlvDq8ikWAM"
)

# Get available voices
voices = await elevenlabs_api.get_available_voices()

# Generate video narration
narration = await elevenlabs_api.generate_video_narration(
    script="Today we're making the most delicious pasta...",
    voice_id="21m00Tcm4TlvDq8ikWAM"
)
```

### Batch API Operations

```python
from async_external_apis import create_async_batch_api_operations

# Create batch API operations
batch_api = create_async_batch_api_operations(api_ops)

# Batch get video information
video_ids = ["video1", "video2", "video3", "video4"]
video_infos = await batch_api.batch_get_video_info(video_ids)

# Batch generate captions
audio_texts = [
    "Amazing cooking tutorial...",
    "Learn to code in Python...",
    "Travel vlog from Japan..."
]

captions = await batch_api.batch_generate_captions(audio_texts, style="casual")
```

## Connection Pooling

### Database Connection Pooling

```python
# PostgreSQL connection pool
postgres_pool = PostgreSQLPool(config)
await postgres_pool.initialize()

# Use connection from pool
async with postgres_pool.get_connection() as connection:
    result = await connection.fetch("SELECT * FROM videos")

# MySQL connection pool
mysql_pool = MySQLPool(config)
await mysql_pool.initialize()

# SQLite connection (file-based)
sqlite_pool = SQLitePool(config)
await sqlite_pool.initialize()

# Redis connection pool
redis_pool = RedisPool(config)
await redis_pool.initialize()
```

### HTTP Connection Pooling

```python
# HTTP client with connection pooling
http_client = AsyncHTTPClient(config)
await http_client.initialize()

# Connection pool settings
connector = aiohttp.TCPConnector(
    limit=50,                    # Total connections
    limit_per_host=10,           # Connections per host
    ttl_dns_cache=300,           # DNS cache TTL
    use_dns_cache=True,          # Enable DNS caching
    ssl=True                     # SSL verification
)

# Use connection from pool
async with http_client.session.get("https://api.example.com/data") as response:
    data = await response.json()
```

## Caching Strategies

### Database Query Caching

```python
# Execute query with caching
result = await db_ops.execute_query(
    "SELECT * FROM videos WHERE status = $1",
    ["pending"],
    query_type=QueryType.SELECT,
    cache_key="videos:pending",
    cache_ttl=300  # 5 minutes
)

# Cache invalidation
await video_db._invalidate_video_cache(video_id)
```

### API Response Caching

```python
# API request with caching
result = await api_ops.make_request(
    HTTPMethod.GET,
    "videos",
    params={"id": video_id},
    cache_key=f"youtube:video:{video_id}",
    use_cache=True
)

# Cache management
await http_client.cache.clear()
```

### Multi-Level Caching

```python
class MultiLevelCache:
    def __init__(self):
        self.l1_cache = APICache(max_size=100, ttl=60)    # Fast, small
        self.l2_cache = APICache(max_size=1000, ttl=3600) # Slower, larger
    
    async def get(self, key: str):
        # Check L1 cache first
        result = await self.l1_cache.get(key)
        if result:
            return result
        
        # Check L2 cache
        result = await self.l2_cache.get(key)
        if result:
            # Populate L1 cache
            await self.l1_cache.set(key, result)
            return result
        
        return None
```

## Rate Limiting

### API Rate Limiting

```python
# Rate limiter configuration
rate_limiter = RateLimiter(
    requests_per_minute=60,
    requests_per_hour=1000
)

# Automatic rate limiting in HTTP client
http_client = AsyncHTTPClient(config)
await http_client.initialize()

# Rate limiting is automatic
for i in range(100):
    result = await http_client.get("endpoint")  # Rate limited automatically
```

### Custom Rate Limiting

```python
class CustomRateLimiter:
    def __init__(self, requests_per_second: int = 10):
        self.requests_per_second = requests_per_second
        self.last_request_time = 0
        self._lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        async with self._lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            min_interval = 1.0 / self.requests_per_second
            
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()
```

## Error Handling

### Database Error Handling

```python
# Execute query with error handling
try:
    result = await db_ops.execute_query(
        "SELECT * FROM videos WHERE id = $1",
        [video_id],
        query_type=QueryType.SELECT
    )
except Exception as e:
    logger.error(f"Database query failed: {e}")
    # Fallback or retry logic
    result = await fallback_query(video_id)
```

### API Error Handling

```python
# API request with retry logic
try:
    result = await api_ops.make_request(
        HTTPMethod.GET,
        "videos",
        retry_on_error=True,
        max_retries=3
    )
except Exception as e:
    logger.error(f"API request failed: {e}")
    # Circuit breaker or fallback
    result = await fallback_api_call()
```

### Circuit Breaker Pattern

```python
from async_flows import CircuitBreaker

# Circuit breaker for external services
circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)

async def reliable_api_call():
    return await circuit_breaker.call(unreliable_api_call)

# Use circuit breaker
try:
    result = await reliable_api_call()
except Exception as e:
    logger.warning(f"Circuit breaker open: {e}")
    result = await fallback_service()
```

## Batch Operations

### Database Batch Operations

```python
# Batch insert with PostgreSQL
async def batch_insert_videos_postgresql(videos: List[Dict]):
    query = """
    INSERT INTO videos (url, title, duration, status, created_at, updated_at)
    VALUES ($1, $2, $3, $4, $5, $6)
    RETURNING id
    """
    
    values = []
    for video in videos:
        values.append((
            video["url"],
            video["title"],
            video["duration"],
            video["status"],
            datetime.now(),
            datetime.now()
        ))
    
    async with pool.get_connection() as connection:
        result = await connection.fetch(query, *values)
        return [row['id'] for row in result]
```

### API Batch Operations

```python
# Batch API requests
async def batch_api_requests(endpoints: List[str]):
    tasks = []
    for endpoint in endpoints:
        task = api_ops.make_request(HTTPMethod.GET, endpoint)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions
    successful_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"API request failed: {result}")
        else:
            successful_results.append(result)
    
    return successful_results
```

## Transaction Management

### Database Transactions

```python
# Transaction with rollback
async def process_video_transaction(video_data, clips_data):
    async with tx_manager.transaction() as connection:
        try:
            # Create video
            video_id = await video_db.create_video_record(video_data)
            
            # Create clips
            for clip_data in clips_data:
                clip_data["video_id"] = video_id
                await video_db.create_clip_record(clip_data)
            
            # Update status
            await video_db.update_video_status(video_id, "completed")
            
            return video_id
            
        except Exception as e:
            # Transaction will automatically rollback
            logger.error(f"Transaction failed: {e}")
            raise
```

### Distributed Transactions

```python
# Multi-database transaction
async def distributed_transaction():
    async with asyncio.gather(
        tx_manager1.transaction(),
        tx_manager2.transaction()
    ) as (conn1, conn2):
        try:
            # Operations on database 1
            result1 = await operation1(conn1)
            
            # Operations on database 2
            result2 = await operation2(conn2)
            
            return result1, result2
            
        except Exception as e:
            # Both transactions will rollback
            raise
```

## Metrics and Monitoring

### Database Metrics

```python
# Get database metrics
metrics = get_query_metrics(db_ops)

print(f"Queries executed: {metrics['queries_executed']}")
print(f"Average execution time: {metrics['avg_execution_time']:.3f}s")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
```

### API Metrics

```python
# Get API metrics
metrics = get_api_metrics(api_ops)

print(f"Requests made: {metrics['requests_made']}")
print(f"Success rate: {metrics['successful_requests'] / metrics['requests_made']:.2%}")
print(f"Average response time: {metrics['total_response_time'] / metrics['requests_made']:.3f}s")
```

### Custom Metrics Collection

```python
# Custom metrics collector
metrics_collector = APIMetricsCollector()

async def monitored_api_call(endpoint: str):
    start_time = time.perf_counter()
    
    try:
        result = await api_ops.make_request(HTTPMethod.GET, endpoint)
        
        response_time = time.perf_counter() - start_time
        await metrics_collector.record_request(
            endpoint, "GET", True, response_time
        )
        
        return result
        
    except Exception as e:
        response_time = time.perf_counter() - start_time
        await metrics_collector.record_request(
            endpoint, "GET", False, response_time, type(e).__name__
        )
        raise

# Get metrics
metrics = await metrics_collector.get_metrics()
```

## Integration Examples

### Complete Video Processing Pipeline

```python
async def complete_video_processing_pipeline(video_url: str):
    # Initialize components
    db_ops = await setup_database_connection(DatabaseType.POSTGRESQL)
    api_ops = await setup_external_api(APIType.YOUTUBE, "https://api.youtube.com/v3")
    
    try:
        # Extract video ID from URL
        video_id = extract_video_id(video_url)
        
        # Get video info from YouTube API
        youtube_api = create_async_youtube_api(api_ops)
        video_info = await youtube_api.get_video_info(video_id)
        
        # Create video record in database
        video_db = create_async_video_database(db_ops)
        db_video_id = await video_db.create_video_record({
            "url": video_url,
            "title": video_info["snippet"]["title"],
            "duration": parse_duration(video_info["contentDetails"]["duration"]),
            "status": "downloading"
        })
        
        # Download video (simulated)
        await download_video(video_url)
        await video_db.update_video_status(db_video_id, "processing")
        
        # Generate captions using OpenAI
        openai_api = create_async_openai_api(api_ops)
        captions = await openai_api.generate_captions(
            video_info["snippet"]["description"],
            style="engaging"
        )
        
        # Create clip records
        for i, caption in enumerate(captions):
            await video_db.create_clip_record({
                "video_id": db_video_id,
                "start_time": i * 30,
                "end_time": (i + 1) * 30,
                "duration": 30,
                "caption": caption,
                "effects": ["fade_in", "fade_out"]
            })
        
        # Update final status
        await video_db.update_video_status(db_video_id, "completed")
        
        return db_video_id
        
    finally:
        # Cleanup
        await close_database_connection(db_ops)
        await close_external_api(api_ops)
```

### Batch Processing System

```python
async def batch_video_processing_system(video_urls: List[str]):
    # Initialize components
    db_ops = await setup_database_connection(DatabaseType.POSTGRESQL)
    api_ops = await setup_external_api(APIType.YOUTUBE, "https://api.youtube.com/v3")
    
    try:
        # Create batch operations
        batch_db = create_async_batch_database_operations(db_ops)
        batch_api = create_async_batch_api_operations(api_ops)
        
        # Extract video IDs
        video_ids = [extract_video_id(url) for url in video_urls]
        
        # Batch get video information
        video_infos = await batch_api.batch_get_video_info(video_ids)
        
        # Prepare video data for batch insert
        video_data = []
        for url, info in zip(video_urls, video_infos):
            video_data.append({
                "url": url,
                "title": info["snippet"]["title"],
                "duration": parse_duration(info["contentDetails"]["duration"]),
                "status": "pending"
            })
        
        # Batch insert videos
        db_video_ids = await batch_db.batch_insert_videos(video_data)
        
        # Process videos in parallel
        tasks = []
        for db_video_id, video_info in zip(db_video_ids, video_infos):
            task = process_single_video(db_video_id, video_info, api_ops, db_ops)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        successful = 0
        failed = 0
        for result in results:
            if isinstance(result, Exception):
                failed += 1
                logger.error(f"Video processing failed: {result}")
            else:
                successful += 1
        
        logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
        
        return successful, failed
        
    finally:
        # Cleanup
        await close_database_connection(db_ops)
        await close_external_api(api_ops)
```

## Best Practices

### 1. Connection Management

```python
# Always use context managers
async with pool.get_connection() as connection:
    result = await connection.fetch(query)

# Initialize and cleanup properly
async def main():
    db_ops = await setup_database_connection(DatabaseType.POSTGRESQL)
    try:
        # Use database operations
        pass
    finally:
        await close_database_connection(db_ops)
```

### 2. Error Handling

```python
# Use retry logic for transient failures
@async_retry(max_attempts=3, delay=1.0)
async def reliable_operation():
    return await potentially_failing_operation()

# Use circuit breakers for external services
circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
result = await circuit_breaker.call(external_api_call)
```

### 3. Caching Strategy

```python
# Cache frequently accessed data
result = await db_ops.execute_query(
    query, params, cache_key="frequent_query", cache_ttl=300
)

# Invalidate cache on updates
await video_db.update_video_status(video_id, "completed")
await video_db._invalidate_video_cache(video_id)
```

### 4. Batch Operations

```python
# Use batch operations for better performance
video_ids = await batch_db.batch_insert_videos(video_data)

# Process in chunks to avoid memory issues
chunk_size = 100
for i in range(0, len(items), chunk_size):
    chunk = items[i:i + chunk_size]
    await process_chunk(chunk)
```

### 5. Monitoring

```python
# Collect metrics for performance monitoring
metrics = await metrics_collector.get_metrics()
logger.info("API performance", **metrics)

# Set up alerts for failures
if metrics["error_rate"] > 0.05:
    send_alert("High error rate detected")
```

## Performance Optimization

### 1. Connection Pooling

```python
# Optimize connection pool settings
config = create_database_config(
    max_connections=50,
    min_connections=10,
    timeout=30.0
)

# HTTP connection pooling
http_config = create_api_config(
    max_connections=100,
    max_connections_per_host=20
)
```

### 2. Query Optimization

```python
# Use prepared statements
query = "SELECT * FROM videos WHERE status = $1 AND created_at > $2"
result = await db_ops.execute_query(query, ["pending", yesterday])

# Use indexes
await db_ops.execute_query(
    "CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status)"
)
```

### 3. Caching Optimization

```python
# Multi-level caching
l1_cache = APICache(max_size=100, ttl=60)    # Fast, small
l2_cache = APICache(max_size=1000, ttl=3600) # Slower, larger

# Cache warming
async def warm_cache():
    popular_videos = await video_db.get_popular_videos()
    for video in popular_videos:
        await cache.set(f"video:{video['id']}", video)
```

### 4. Batch Processing

```python
# Optimal batch sizes
optimal_batch_size = 50  # For database operations
optimal_api_batch_size = 10  # For API calls

# Parallel processing
tasks = [process_item(item) for item in items]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

## Quick Start Commands

### Installation
```bash
# Install dependencies
pip install asyncpg aiomysql aiosqlite aioredis aiohttp

# Run database setup
python -c "
import asyncio
from async_database import setup_database_connection, create_async_database_setup
async def setup():
    db_ops = await setup_database_connection(DatabaseType.POSTGRESQL)
    setup = create_async_database_setup(db_ops)
    await setup.create_tables()
    print('Database setup completed')
asyncio.run(setup())
"
```

### Basic Usage
```python
# Database operations
from async_database import setup_database_connection, create_async_video_database

db_ops = await setup_database_connection(DatabaseType.POSTGRESQL)
video_db = create_async_video_database(db_ops)

video_id = await video_db.create_video_record({"url": "video.mp4"})
video = await video_db.get_video_by_id(video_id)

# API operations
from async_external_apis import setup_external_api, create_async_youtube_api

api_ops = await setup_external_api(APIType.YOUTUBE, "https://api.youtube.com/v3")
youtube_api = create_async_youtube_api(api_ops)

video_info = await youtube_api.get_video_info("video_id")
```

## Conclusion

The async database and external API operations system provides:

- **High Performance**: Non-blocking operations with connection pooling
- **Reliability**: Comprehensive error handling and retry logic
- **Scalability**: Batch operations and parallel processing
- **Efficiency**: Caching and rate limiting
- **Observability**: Metrics collection and monitoring
- **Flexibility**: Support for multiple databases and APIs

This system enables the Video-OpusClip platform to efficiently handle large-scale video processing workloads with excellent performance and reliability. 