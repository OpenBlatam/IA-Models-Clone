# Async and Non-Blocking Flows Guide

## Overview

This guide covers the comprehensive async and non-blocking flows implementation for the Video-OpusClip system. The system provides advanced asynchronous processing patterns, event-driven architecture, and optimized concurrency for high-performance video processing.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Async Flow Patterns](#async-flow-patterns)
3. [Event-Driven Architecture](#event-driven-architecture)
4. [Connection Pooling](#connection-pooling)
5. [Caching Strategies](#caching-strategies)
6. [Circuit Breaker Pattern](#circuit-breaker-pattern)
7. [Task Queues and Priority](#task-queues-and-priority)
8. [Workflow Engine](#workflow-engine)
9. [Monitoring and Metrics](#monitoring-and-metrics)
10. [Best Practices](#best-practices)
11. [Performance Optimization](#performance-optimization)
12. [Error Handling](#error-handling)
13. [Integration Examples](#integration-examples)

## Architecture Overview

### Core Components

```
async_flows.py
├── AsyncFlowConfig          # Configuration management
├── AsyncEventLoopManager    # Event loop optimization
├── PriorityTaskQueue        # Priority-based task processing
├── AsyncFlowPattern         # Flow pattern implementations
├── AsyncConnectionPool      # Connection pooling
├── AsyncCache              # Caching with TTL
├── CircuitBreaker          # Fault tolerance
├── AsyncEventBus           # Event-driven architecture
├── AsyncWorkflowEngine     # Complex workflow orchestration
├── AsyncMetricsCollector   # Performance monitoring
└── AsyncFlowManager        # Main orchestrator
```

### Key Features

- **Non-blocking I/O**: All operations are non-blocking
- **Event-driven architecture**: Loose coupling through events
- **Connection pooling**: Efficient resource management
- **Circuit breaker**: Fault tolerance and resilience
- **Priority queues**: Task prioritization
- **Caching**: Performance optimization
- **Metrics collection**: Real-time monitoring
- **Workflow orchestration**: Complex process management

## Async Flow Patterns

### 1. Sequential Flow

Execute tasks one after another:

```python
from async_flows import SequentialFlow, create_async_flow_config

config = create_async_flow_config(max_concurrent_tasks=10)
sequential_flow = SequentialFlow(config)

async def task1():
    await asyncio.sleep(1)
    return "Task 1 completed"

async def task2():
    await asyncio.sleep(1)
    return "Task 2 completed"

tasks = [task1, task2]
results = await sequential_flow.execute(tasks)
# Results: ["Task 1 completed", "Task 2 completed"]
```

### 2. Parallel Flow

Execute tasks concurrently:

```python
from async_flows import ParallelFlow

parallel_flow = ParallelFlow(config)

async def download_video(url):
    # Simulate video download
    await asyncio.sleep(2)
    return f"Downloaded {url}"

urls = ["video1.mp4", "video2.mp4", "video3.mp4"]
download_tasks = [lambda url=url: download_video(url) for url in urls]

results = await parallel_flow.execute(download_tasks)
# All videos downloaded concurrently
```

### 3. Streaming Flow

Process data streams:

```python
from async_flows import StreamingFlow

streaming_flow = StreamingFlow(config)

async def video_url_stream():
    urls = ["video1.mp4", "video2.mp4", "video3.mp4"]
    for url in urls:
        yield url
        await asyncio.sleep(0.1)

async def process_video(url):
    # Process video
    await asyncio.sleep(1)
    return f"Processed {url}"

async for result in streaming_flow.execute(video_url_stream(), process_video):
    print(result)
```

### 4. Pipeline Flow

Process data through multiple stages:

```python
from async_flows import PipelineFlow

pipeline_flow = PipelineFlow(config)

async def download_stage(url):
    return f"Downloaded {url}"

async def process_stage(downloaded_data):
    return f"Processed {downloaded_data}"

async def encode_stage(processed_data):
    return f"Encoded {processed_data}"

stages = [download_stage, process_stage, encode_stage]
result = await pipeline_flow.execute("video.mp4", stages)
```

### 5. Fan-Out Flow

Distribute data to multiple processors:

```python
from async_flows import FanOutFlow

fan_out_flow = FanOutFlow(config)

async def processor1(data):
    return f"Processor 1: {data}"

async def processor2(data):
    return f"Processor 2: {data}"

async def processor3(data):
    return f"Processor 3: {data}"

processors = [processor1, processor2, processor3]
results = await fan_out_flow.execute("shared_data", processors)
```

### 6. Fan-In Flow

Aggregate multiple data streams:

```python
from async_flows import FanInFlow

fan_in_flow = FanInFlow(config)

async def stream1():
    for i in range(3):
        yield f"Stream 1: {i}"
        await asyncio.sleep(0.1)

async def stream2():
    for i in range(3):
        yield f"Stream 2: {i}"
        await asyncio.sleep(0.1)

async def aggregate(items):
    return f"Aggregated: {items}"

streams = [stream1(), stream2()]
result = await fan_in_flow.execute(streams, aggregate)
```

## Event-Driven Architecture

### Event Bus

```python
from async_flows import create_async_event_bus

event_bus = create_async_event_bus()

# Subscribe to events
async def handle_video_processed(data):
    print(f"Video processed: {data}")

async def handle_error(data):
    print(f"Error occurred: {data}")

await event_bus.subscribe("video_processed", handle_video_processed)
await event_bus.subscribe("error", handle_error)

# Publish events
await event_bus.publish("video_processed", {"url": "video.mp4", "status": "completed"})
await event_bus.publish("error", {"message": "Download failed"})
```

### Event-Driven Video Processing

```python
class EventDrivenVideoProcessor:
    def __init__(self, event_bus):
        self.event_bus = event_bus
    
    async def process_video(self, url):
        try:
            # Download video
            await self.event_bus.publish("download_started", {"url": url})
            video_data = await self.download_video(url)
            await self.event_bus.publish("download_completed", {"url": url})
            
            # Process video
            await self.event_bus.publish("processing_started", {"url": url})
            result = await self.process_video_data(video_data)
            await self.event_bus.publish("processing_completed", {"url": url, "result": result})
            
            return result
            
        except Exception as e:
            await self.event_bus.publish("error", {"url": url, "error": str(e)})
            raise
```

## Connection Pooling

### HTTP Connection Pool

```python
from async_flows import HTTPConnectionPool

# Create connection pool
http_pool = HTTPConnectionPool("https://api.example.com", max_connections=50)

async def download_video(url):
    async with http_pool.get_connection() as session:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            else:
                raise Exception(f"HTTP {response.status}")

# Use in batch processing
urls = ["video1.mp4", "video2.mp4", "video3.mp4"]
tasks = [download_video(url) for url in urls]
results = await asyncio.gather(*tasks)
```

### Custom Connection Pool

```python
from async_flows import AsyncConnectionPool

class DatabaseConnectionPool(AsyncConnectionPool):
    def __init__(self, connection_string, max_connections=20):
        super().__init__(max_connections)
        self.connection_string = connection_string
    
    async def _create_connection(self):
        # Create database connection
        return await create_db_connection(self.connection_string)
    
    async def _release_connection(self, connection):
        # Close database connection
        await connection.close()

# Usage
db_pool = DatabaseConnectionPool("postgresql://localhost/db")
async with db_pool.get_connection() as conn:
    result = await conn.execute("SELECT * FROM videos")
```

## Caching Strategies

### Async Cache with TTL

```python
from async_flows import AsyncCache

# Create cache with 1 hour TTL
cache = AsyncCache(max_size=1000, ttl=3600)

async def get_video_metadata(url):
    # Check cache first
    cached = await cache.get(url)
    if cached:
        return cached
    
    # Fetch from API
    metadata = await fetch_video_metadata(url)
    
    # Cache result
    await cache.set(url, metadata)
    return metadata

# Batch cache operations
async def cache_multiple_videos(video_data):
    for url, data in video_data.items():
        await cache.set(url, data)
```

### Cache with Custom Serialization

```python
import pickle
import gzip

class CompressedCache(AsyncCache):
    async def set(self, key: str, value: Any):
        # Compress data before caching
        compressed = gzip.compress(pickle.dumps(value))
        await super().set(key, compressed)
    
    async def get(self, key: str) -> Optional[Any]:
        compressed = await super().get(key)
        if compressed:
            return pickle.loads(gzip.decompress(compressed))
        return None
```

## Circuit Breaker Pattern

### Basic Usage

```python
from async_flows import CircuitBreaker

circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)

async def unreliable_api_call():
    # Simulate unreliable API
    if random.random() < 0.3:
        raise Exception("API Error")
    return "Success"

# Use circuit breaker
try:
    result = await circuit_breaker.call(unreliable_api_call)
    print(result)
except Exception as e:
    print(f"Circuit breaker: {e}")
```

### Circuit Breaker with Video Processing

```python
class ResilientVideoProcessor:
    def __init__(self):
        self.download_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
        self.process_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
    
    async def process_video(self, url):
        # Download with circuit breaker
        video_data = await self.download_circuit_breaker.call(
            self.download_video, url
        )
        
        # Process with circuit breaker
        result = await self.process_circuit_breaker.call(
            self.process_video_data, video_data
        )
        
        return result
```

## Task Queues and Priority

### Priority Task Queue

```python
from async_flows import PriorityTaskQueue, AsyncTask, TaskPriority

queue = PriorityTaskQueue(maxsize=1000)

# Create tasks with different priorities
high_priority_task = AsyncTask(
    func=process_urgent_video,
    args=("urgent_video.mp4",),
    priority=TaskPriority.HIGH,
    timeout=30.0
)

normal_priority_task = AsyncTask(
    func=process_normal_video,
    args=("normal_video.mp4",),
    priority=TaskPriority.NORMAL,
    timeout=60.0
)

# Add tasks to queue
await queue.put(high_priority_task)
await queue.put(normal_priority_task)

# Process tasks
await queue.process_tasks(max_workers=5)
```

### Custom Task Processing

```python
class VideoTaskProcessor:
    def __init__(self):
        self.queue = PriorityTaskQueue()
        self.workers = []
    
    async def start_workers(self, num_workers=5):
        self.workers = [
            asyncio.create_task(self._worker(i))
            for i in range(num_workers)
        ]
    
    async def _worker(self, worker_id):
        while True:
            try:
                task = await self.queue.get()
                print(f"Worker {worker_id} processing task: {task.task_id}")
                
                result = await self._execute_task(task)
                print(f"Worker {worker_id} completed task: {task.task_id}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
    
    async def _execute_task(self, task):
        # Custom task execution logic
        return await task.func(*task.args, **task.kwargs)
```

## Workflow Engine

### Simple Workflow

```python
from async_flows import AsyncWorkflowEngine, WorkflowStep

# Create workflow engine
workflow_engine = AsyncWorkflowEngine(config)

# Define workflow steps
download_step = WorkflowStep(
    name="download",
    func=download_video,
    timeout=30.0
)

process_step = WorkflowStep(
    name="process",
    func=process_video,
    dependencies=["download"],
    timeout=60.0
)

encode_step = WorkflowStep(
    name="encode",
    func=encode_video,
    dependencies=["process"],
    timeout=120.0
)

# Add steps to workflow
workflow_engine.add_step(download_step)
workflow_engine.add_step(process_step)
workflow_engine.add_step(encode_step)

# Execute workflow
results = await workflow_engine.execute_workflow("video_url")
```

### Complex Workflow with Dependencies

```python
# Define complex workflow
workflow_steps = [
    WorkflowStep("validate", validate_video_url),
    WorkflowStep("download", download_video, dependencies=["validate"]),
    WorkflowStep("extract_audio", extract_audio, dependencies=["download"]),
    WorkflowStep("generate_captions", generate_captions, dependencies=["extract_audio"]),
    WorkflowStep("create_clips", create_clips, dependencies=["generate_captions"]),
    WorkflowStep("add_effects", add_effects, dependencies=["create_clips"]),
    WorkflowStep("upload", upload_video, dependencies=["add_effects"]),
    WorkflowStep("notify", send_notification, dependencies=["upload"])
]

# Add all steps
for step in workflow_steps:
    workflow_engine.add_step(step)

# Execute with event monitoring
async def workflow_monitor(event_type, data):
    print(f"Workflow event: {event_type} - {data}")

await workflow_engine.event_bus.subscribe("step_completed", workflow_monitor)
await workflow_engine.event_bus.subscribe("step_failed", workflow_monitor)

results = await workflow_engine.execute_workflow("video_url")
```

## Monitoring and Metrics

### Metrics Collection

```python
from async_flows import create_async_metrics_collector

metrics_collector = create_async_metrics_collector()

async def monitored_task():
    start_time = time.time()
    try:
        result = await process_video("video.mp4")
        duration = time.time() - start_time
        await metrics_collector.record_task_execution("process_video", duration, True)
        return result
    except Exception as e:
        duration = time.time() - start_time
        await metrics_collector.record_task_execution("process_video", duration, False)
        raise

# Get metrics
metrics = await metrics_collector.get_metrics()
print(f"Success rate: {metrics['successful_tasks'] / (metrics['successful_tasks'] + metrics['failed_tasks'])}")
print(f"Average execution time: {metrics['avg_execution_time']}")
```

### Real-time Monitoring

```python
class VideoProcessingMonitor:
    def __init__(self, metrics_collector, event_bus):
        self.metrics_collector = metrics_collector
        self.event_bus = event_bus
        self.monitoring_task = None
    
    async def start_monitoring(self):
        self.monitoring_task = asyncio.create_task(self._monitor_loop())
    
    async def _monitor_loop(self):
        while True:
            await asyncio.sleep(10)  # Monitor every 10 seconds
            
            metrics = await self.metrics_collector.get_metrics()
            
            # Check for issues
            if metrics['failed_tasks'] > 10:
                await self.event_bus.publish("high_failure_rate", metrics)
            
            if metrics['avg_execution_time'] > 30:
                await self.event_bus.publish("slow_processing", metrics)
            
            # Log metrics
            logger.info("Processing metrics", **metrics)
    
    async def stop_monitoring(self):
        if self.monitoring_task:
            self.monitoring_task.cancel()
```

## Best Practices

### 1. Resource Management

```python
# Always use context managers for resources
async with http_pool.get_connection() as session:
    async with session.get(url) as response:
        data = await response.read()

# Clean up resources properly
async def cleanup_resources():
    await http_pool.close()
    await cache.clear()
    await event_bus.unsubscribe_all()
```

### 2. Error Handling

```python
# Use circuit breakers for external services
async def reliable_api_call():
    return await circuit_breaker.call(unreliable_api_call)

# Implement retry logic
@async_retry(max_attempts=3, delay=1.0)
async def robust_operation():
    return await potentially_failing_operation()

# Handle timeouts
async def timeout_safe_operation():
    try:
        return await asyncio.wait_for(slow_operation(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Operation timed out")
        return fallback_result()
```

### 3. Concurrency Control

```python
# Use semaphores to limit concurrency
semaphore = asyncio.Semaphore(10)

async def controlled_concurrent_operation():
    async with semaphore:
        return await heavy_operation()

# Batch operations for efficiency
async def batch_process(items, batch_size=100):
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        tasks = [process_item(item) for item in batch]
        results = await asyncio.gather(*tasks)
        yield from results
```

### 4. Memory Management

```python
# Use streaming for large datasets
async def process_large_dataset():
    async for item in data_stream():
        result = await process_item(item)
        yield result
        # Memory is automatically freed

# Clear caches periodically
async def cache_maintenance():
    while True:
        await asyncio.sleep(3600)  # Every hour
        await cache.clear_expired()
```

## Performance Optimization

### 1. Event Loop Optimization

```python
# Use uvloop for better performance
import uvloop
uvloop.install()

# Configure event loop
loop = asyncio.get_event_loop()
loop.set_debug(False)
loop.slow_callback_duration = 0.1
```

### 2. Connection Pooling

```python
# Optimize connection pool settings
http_pool = HTTPConnectionPool(
    "https://api.example.com",
    max_connections=100,
    limit_per_host=50
)

# Use connection pooling for all HTTP requests
async def optimized_download(urls):
    async with http_pool.get_connection() as session:
        tasks = [session.get(url) for url in urls]
        responses = await asyncio.gather(*tasks)
        return [await resp.read() for resp in responses]
```

### 3. Caching Strategy

```python
# Implement multi-level caching
class MultiLevelCache:
    def __init__(self):
        self.l1_cache = AsyncCache(max_size=100, ttl=60)  # Fast, small
        self.l2_cache = AsyncCache(max_size=1000, ttl=3600)  # Slower, larger
    
    async def get(self, key):
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

### 4. Batch Processing

```python
# Process items in optimal batch sizes
async def optimized_batch_process(items, optimal_batch_size=50):
    semaphore = asyncio.Semaphore(10)  # Limit concurrent batches
    
    async def process_batch(batch):
        async with semaphore:
            return await asyncio.gather(*[process_item(item) for item in batch])
    
    # Split into batches
    batches = [items[i:i + optimal_batch_size] 
              for i in range(0, len(items), optimal_batch_size)]
    
    # Process batches concurrently
    batch_results = await asyncio.gather(*[process_batch(batch) for batch in batches])
    
    # Flatten results
    return [item for batch in batch_results for item in batch]
```

## Error Handling

### 1. Graceful Degradation

```python
class ResilientVideoProcessor:
    def __init__(self):
        self.primary_processor = PrimaryVideoProcessor()
        self.fallback_processor = FallbackVideoProcessor()
    
    async def process_video(self, url):
        try:
            return await self.primary_processor.process(url)
        except Exception as e:
            logger.warning(f"Primary processor failed: {e}")
            return await self.fallback_processor.process(url)
```

### 2. Circuit Breaker Integration

```python
class FaultTolerantService:
    def __init__(self):
        self.circuit_breakers = {
            'download': CircuitBreaker(failure_threshold=5, timeout=30),
            'process': CircuitBreaker(failure_threshold=3, timeout=60),
            'upload': CircuitBreaker(failure_threshold=2, timeout=45)
        }
    
    async def process_video(self, url):
        # Download with circuit breaker
        video_data = await self.circuit_breakers['download'].call(
            self.download_video, url
        )
        
        # Process with circuit breaker
        processed_data = await self.circuit_breakers['process'].call(
            self.process_video, video_data
        )
        
        # Upload with circuit breaker
        result = await self.circuit_breakers['upload'].call(
            self.upload_video, processed_data
        )
        
        return result
```

### 3. Retry with Exponential Backoff

```python
async def exponential_backoff_retry(func, max_attempts=5, base_delay=1.0):
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise e
            
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            await asyncio.sleep(delay)
```

## Integration Examples

### 1. Complete Video Processing Pipeline

```python
async def complete_video_pipeline():
    # Create async flow manager
    config = create_async_flow_config(
        max_concurrent_tasks=50,
        max_concurrent_connections=20,
        enable_metrics=True
    )
    
    flow_manager = create_async_flow_manager(config)
    
    try:
        # Start the manager
        await flow_manager.start()
        
        # Process videos
        video_urls = ["video1.mp4", "video2.mp4", "video3.mp4"]
        results = await flow_manager.video_processor.process_video_batch(video_urls)
        
        # Get metrics
        metrics = await flow_manager.metrics_collector.get_metrics()
        print(f"Processing completed: {metrics}")
        
        return results
        
    finally:
        await flow_manager.shutdown()
```

### 2. Event-Driven Video Processing

```python
async def event_driven_video_processing():
    event_bus = create_async_event_bus()
    
    # Setup event handlers
    async def handle_video_ready(data):
        print(f"Video ready for processing: {data}")
        # Trigger processing pipeline
    
    async def handle_processing_complete(data):
        print(f"Processing complete: {data}")
        # Trigger upload
    
    async def handle_error(data):
        print(f"Error occurred: {data}")
        # Trigger error recovery
    
    await event_bus.subscribe("video_ready", handle_video_ready)
    await event_bus.subscribe("processing_complete", handle_processing_complete)
    await event_bus.subscribe("error", handle_error)
    
    # Start processing
    await process_video_with_events("video.mp4", event_bus)
```

### 3. High-Performance Batch Processing

```python
async def high_performance_batch_processing():
    # Create optimized configuration
    config = create_async_flow_config(
        max_concurrent_tasks=100,
        chunk_size=50,
        use_uvloop=True,
        enable_circuit_breaker=True
    )
    
    processor = create_async_video_processor(config)
    
    # Generate large batch of video URLs
    video_urls = [f"video_{i}.mp4" for i in range(1000)]
    
    # Process in optimized batches
    results = []
    for i in range(0, len(video_urls), config.chunk_size):
        batch = video_urls[i:i + config.chunk_size]
        batch_results = await processor.process_video_batch(batch)
        results.extend(batch_results)
        
        # Progress reporting
        print(f"Processed {len(results)}/{len(video_urls)} videos")
    
    return results
```

### 4. Real-time Streaming Processing

```python
async def real_time_streaming_processing():
    # Create streaming processor
    config = create_async_flow_config(max_concurrent_tasks=20)
    processor = create_async_video_processor(config)
    
    # Create video URL stream
    async def video_url_stream():
        while True:
            # Get new video URLs from queue/API
            new_urls = await get_new_video_urls()
            for url in new_urls:
                yield url
            await asyncio.sleep(1)  # Check every second
    
    # Process streaming data
    async for result in processor.process_video_stream(video_url_stream()):
        print(f"Processed: {result}")
        
        # Send to next stage (e.g., upload, notification)
        await upload_processed_video(result)
        await send_notification(result)
```

## Quick Start

### Basic Setup

```python
from async_flows import (
    create_async_flow_config,
    create_async_video_processor,
    create_async_flow_manager
)

# Create configuration
config = create_async_flow_config(
    max_concurrent_tasks=50,
    max_concurrent_connections=20,
    enable_metrics=True
)

# Create video processor
processor = create_async_video_processor(config)

# Process videos
video_urls = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = await processor.process_video_batch(video_urls)
```

### Advanced Setup

```python
# Create complete async flow manager
flow_manager = create_async_flow_manager(config)

# Start the manager
await flow_manager.start()

# Use different flow patterns
results = await flow_manager.video_processor.process_video_pipeline("video.mp4")

# Monitor performance
metrics = await flow_manager.metrics_collector.get_metrics()

# Shutdown gracefully
await flow_manager.shutdown()
```

## Conclusion

The async and non-blocking flows system provides a comprehensive solution for high-performance video processing with:

- **Scalability**: Handle thousands of concurrent operations
- **Reliability**: Circuit breakers, retry logic, and error handling
- **Performance**: Optimized event loops, connection pooling, and caching
- **Flexibility**: Multiple flow patterns for different use cases
- **Monitoring**: Real-time metrics and event-driven architecture
- **Maintainability**: Clean separation of concerns and modular design

This system enables the Video-OpusClip platform to process large volumes of video content efficiently while maintaining high availability and performance. 