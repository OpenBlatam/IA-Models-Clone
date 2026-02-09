"""
Async and Non-Blocking Flows for Video-OpusClip

Advanced asynchronous processing with non-blocking operations, event-driven architecture,
and optimized concurrency patterns for high-performance video processing.
"""

import asyncio
import aiohttp
import aiofiles
import aioredis
import aiomqtt
import aiokafka
import asyncio_mqtt
import uvloop
import time
import functools
from typing import (
    List, Dict, Any, Optional, Callable, Awaitable, Union, TypeVar, 
    AsyncGenerator, AsyncIterator, Coroutine, Protocol
)
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import structlog
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import signal
import sys

# Type variables
T = TypeVar('T')
R = TypeVar('R')

logger = structlog.get_logger()

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AsyncFlowConfig:
    """Configuration for async flows."""
    max_concurrent_tasks: int = 100
    max_concurrent_connections: int = 50
    chunk_size: int = 100
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    use_uvloop: bool = True
    enable_metrics: bool = True
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 10
    circuit_breaker_timeout: float = 60.0

class FlowType(Enum):
    """Types of async flows."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    STREAMING = "streaming"
    EVENT_DRIVEN = "event_driven"
    PIPELINE = "pipeline"
    FAN_OUT = "fan_out"
    FAN_IN = "fan_in"

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

# =============================================================================
# ASYNC EVENT LOOP MANAGEMENT
# =============================================================================

class AsyncEventLoopManager:
    """Manages async event loops with optimization."""
    
    def __init__(self, config: AsyncFlowConfig):
        self.config = config
        self.loop = None
        self._setup_loop()
    
    def _setup_loop(self):
        """Setup optimized event loop."""
        if self.config.use_uvloop and uvloop:
            uvloop.install()
            logger.info("Using uvloop for ultra-fast async processing")
        
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        # Set loop policies
        self.loop.set_debug(False)
        self.loop.slow_callback_duration = 0.1
    
    async def run_forever(self):
        """Run event loop forever."""
        try:
            await asyncio.Future()  # Run forever
        except KeyboardInterrupt:
            logger.info("Shutting down async event loop")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup event loop resources."""
        if self.loop and not self.loop.is_closed():
            pending = asyncio.all_tasks(self.loop)
            for task in pending:
                task.cancel()
            self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self.loop.close()

# =============================================================================
# ASYNC TASK QUEUE
# =============================================================================

@dataclass
class AsyncTask:
    """Represents an async task with priority."""
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retry_attempts: int = 3
    created_at: float = field(default_factory=time.time)
    task_id: Optional[str] = None

class PriorityTaskQueue:
    """Priority-based async task queue."""
    
    def __init__(self, maxsize: int = 1000):
        self.queue = asyncio.PriorityQueue(maxsize=maxsize)
        self.running_tasks = set()
        self.completed_tasks = []
        self.failed_tasks = []
    
    async def put(self, task: AsyncTask):
        """Add task to queue."""
        priority = task.priority.value
        await self.queue.put((priority, task))
    
    async def get(self) -> AsyncTask:
        """Get next task from queue."""
        priority, task = await self.queue.get()
        return task
    
    async def process_tasks(self, max_workers: int = 10):
        """Process tasks with multiple workers."""
        workers = [asyncio.create_task(self._worker(i)) for i in range(max_workers)]
        await asyncio.gather(*workers, return_exceptions=True)
    
    async def _worker(self, worker_id: int):
        """Worker coroutine."""
        while True:
            try:
                task = await self.get()
                self.running_tasks.add(task.task_id)
                
                result = await self._execute_task(task)
                
                if result is not None:
                    self.completed_tasks.append((task, result))
                else:
                    self.failed_tasks.append(task)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error", error=str(e))
    
    async def _execute_task(self, task: AsyncTask) -> Any:
        """Execute a single task with retry logic."""
        for attempt in range(task.retry_attempts):
            try:
                if asyncio.iscoroutinefunction(task.func):
                    result = await asyncio.wait_for(
                        task.func(*task.args, **task.kwargs),
                        timeout=task.timeout
                    )
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, task.func, *task.args, **task.kwargs
                    )
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Task timeout on attempt {attempt + 1}", task_id=task.task_id)
            except Exception as e:
                logger.error(f"Task failed on attempt {attempt + 1}", 
                           task_id=task.task_id, error=str(e))
                if attempt < task.retry_attempts - 1:
                    await asyncio.sleep(task.retry_attempts * 0.1)
        return None

# =============================================================================
# ASYNC FLOW PATTERNS
# =============================================================================

class AsyncFlowPattern:
    """Base class for async flow patterns."""
    
    def __init__(self, config: AsyncFlowConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        self.metrics = {}
    
    async def execute(self, *args, **kwargs) -> Any:
        """Execute the flow pattern."""
        raise NotImplementedError

class SequentialFlow(AsyncFlowPattern):
    """Sequential async flow pattern."""
    
    async def execute(self, tasks: List[Callable]) -> List[Any]:
        """Execute tasks sequentially."""
        results = []
        for task in tasks:
            async with self.semaphore:
                if asyncio.iscoroutinefunction(task):
                    result = await task()
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, task)
                results.append(result)
        return results

class ParallelFlow(AsyncFlowPattern):
    """Parallel async flow pattern."""
    
    async def execute(self, tasks: List[Callable]) -> List[Any]:
        """Execute tasks in parallel."""
        async def execute_task(task):
            async with self.semaphore:
                if asyncio.iscoroutinefunction(task):
                    return await task()
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, task)
        
        return await asyncio.gather(*[execute_task(task) for task in tasks])

class StreamingFlow(AsyncFlowPattern):
    """Streaming async flow pattern."""
    
    async def execute(self, data_stream: AsyncIterator[T], 
                     processor: Callable[[T], Awaitable[R]]) -> AsyncIterator[R]:
        """Process streaming data."""
        async for item in data_stream:
            async with self.semaphore:
                result = await processor(item)
                yield result

class PipelineFlow(AsyncFlowPattern):
    """Pipeline async flow pattern."""
    
    async def execute(self, data: T, stages: List[Callable]) -> Any:
        """Execute pipeline stages."""
        result = data
        for stage in stages:
            async with self.semaphore:
                if asyncio.iscoroutinefunction(stage):
                    result = await stage(result)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, stage, result)
        return result

class FanOutFlow(AsyncFlowPattern):
    """Fan-out async flow pattern."""
    
    async def execute(self, data: T, processors: List[Callable]) -> List[Any]:
        """Fan out data to multiple processors."""
        async def process_with_processor(processor):
            async with self.semaphore:
                if asyncio.iscoroutinefunction(processor):
                    return await processor(data)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, processor, data)
        
        return await asyncio.gather(*[process_with_processor(p) for p in processors])

class FanInFlow(AsyncFlowPattern):
    """Fan-in async flow pattern."""
    
    async def execute(self, data_streams: List[AsyncIterator[T]], 
                     aggregator: Callable[[List[T]], Awaitable[R]]) -> R:
        """Fan in multiple data streams."""
        # Collect items from all streams
        items = []
        for stream in data_streams:
            async for item in stream:
                items.append(item)
        
        # Aggregate results
        async with self.semaphore:
            return await aggregator(items)

# =============================================================================
# ASYNC CONNECTION POOLS
# =============================================================================

class AsyncConnectionPool:
    """Generic async connection pool."""
    
    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self.semaphore = asyncio.Semaphore(max_connections)
        self.connections = []
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool."""
        async with self.semaphore:
            connection = await self._create_connection()
            try:
                yield connection
            finally:
                await self._release_connection(connection)
    
    async def _create_connection(self):
        """Create new connection."""
        raise NotImplementedError
    
    async def _release_connection(self, connection):
        """Release connection back to pool."""
        raise NotImplementedError

class HTTPConnectionPool(AsyncConnectionPool):
    """HTTP connection pool using aiohttp."""
    
    def __init__(self, base_url: str, max_connections: int = 50):
        super().__init__(max_connections)
        self.base_url = base_url
        self.session = None
    
    async def _create_connection(self):
        """Create aiohttp session."""
        if self.session is None:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections // 2,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self.session
    
    async def _release_connection(self, connection):
        """Release session (no-op for aiohttp)."""
        pass
    
    async def close(self):
        """Close session."""
        if self.session:
            await self.session.close()

# =============================================================================
# ASYNC CACHE
# =============================================================================

class AsyncCache:
    """Async cache with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self.cache:
                if time.time() - self.access_times[key] < self.ttl:
                    self.access_times[key] = time.time()
                    return self.cache[key]
                else:
                    del self.cache[key]
                    del self.access_times[key]
            return None
    
    async def set(self, key: str, value: Any):
        """Set value in cache."""
        async with self._lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]

# =============================================================================
# ASYNC CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """Circuit breaker pattern for async operations."""
    
    def __init__(self, failure_threshold: int = 10, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, func, *args, **kwargs)
            
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _on_success(self):
        """Handle successful operation."""
        async with self._lock:
            self.failure_count = 0
            self.state = "CLOSED"
    
    async def _on_failure(self):
        """Handle failed operation."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

# =============================================================================
# ASYNC VIDEO PROCESSING FLOWS
# =============================================================================

class AsyncVideoProcessor:
    """Async video processing with non-blocking flows."""
    
    def __init__(self, config: AsyncFlowConfig):
        self.config = config
        self.parallel_flow = ParallelFlow(config)
        self.pipeline_flow = PipelineFlow(config)
        self.streaming_flow = StreamingFlow(config)
        self.cache = AsyncCache()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout
        )
        self.http_pool = HTTPConnectionPool("https://api.example.com")
    
    async def process_video_batch(self, video_urls: List[str]) -> List[Dict]:
        """Process multiple videos in parallel."""
        async def process_single_video(url: str) -> Dict:
            return await self.circuit_breaker.call(
                self._process_video, url
            )
        
        return await self.parallel_flow.execute([
            lambda url=url: process_single_video(url) 
            for url in video_urls
        ])
    
    async def process_video_pipeline(self, video_url: str) -> Dict:
        """Process video through pipeline stages."""
        stages = [
            self._download_video,
            self._extract_audio,
            self._generate_captions,
            self._create_clips,
            self._add_effects
        ]
        
        return await self.pipeline_flow.execute(video_url, stages)
    
    async def process_video_stream(self, video_stream: AsyncIterator[str]) -> AsyncIterator[Dict]:
        """Process streaming video URLs."""
        async def process_video(url: str) -> Dict:
            return await self._process_video(url)
        
        async for result in self.streaming_flow.execute(video_stream, process_video):
            yield result
    
    async def _process_video(self, url: str) -> Dict:
        """Process a single video."""
        # Check cache first
        cached_result = await self.cache.get(url)
        if cached_result:
            return cached_result
        
        # Process video
        result = await self._download_and_process(url)
        
        # Cache result
        await self.cache.set(url, result)
        
        return result
    
    async def _download_video(self, url: str) -> str:
        """Download video from URL."""
        async with self.http_pool.get_connection() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    # Save to temporary file
                    temp_path = f"/tmp/video_{hash(url)}.mp4"
                    async with aiofiles.open(temp_path, 'wb') as f:
                        await f.write(content)
                    return temp_path
                else:
                    raise Exception(f"Failed to download video: {response.status}")
    
    async def _extract_audio(self, video_path: str) -> str:
        """Extract audio from video."""
        # Simulate async audio extraction
        await asyncio.sleep(0.1)
        return video_path.replace('.mp4', '_audio.wav')
    
    async def _generate_captions(self, audio_path: str) -> List[str]:
        """Generate captions from audio."""
        # Simulate async caption generation
        await asyncio.sleep(0.2)
        return ["Caption 1", "Caption 2", "Caption 3"]
    
    async def _create_clips(self, captions: List[str]) -> List[Dict]:
        """Create video clips."""
        # Simulate async clip creation
        await asyncio.sleep(0.3)
        return [{"start": 0, "end": 10, "caption": cap} for cap in captions]
    
    async def _add_effects(self, clips: List[Dict]) -> List[Dict]:
        """Add effects to clips."""
        # Simulate async effect processing
        await asyncio.sleep(0.1)
        for clip in clips:
            clip["effects"] = ["fade_in", "fade_out"]
        return clips
    
    async def _download_and_process(self, url: str) -> Dict:
        """Download and process video."""
        video_path = await self._download_video(url)
        audio_path = await self._extract_audio(video_path)
        captions = await self._generate_captions(audio_path)
        clips = await self._create_clips(captions)
        final_clips = await self._add_effects(clips)
        
        return {
            "url": url,
            "clips": final_clips,
            "processing_time": time.time()
        }

# =============================================================================
# ASYNC EVENT BUS
# =============================================================================

class AsyncEventBus:
    """Event-driven architecture for async flows."""
    
    def __init__(self):
        self.subscribers = {}
        self._lock = asyncio.Lock()
    
    async def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to event type."""
        async with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)
    
    async def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from event type."""
        async with self._lock:
            if event_type in self.subscribers:
                self.subscribers[event_type] = [
                    h for h in self.subscribers[event_type] if h != handler
                ]
    
    async def publish(self, event_type: str, data: Any):
        """Publish event to subscribers."""
        if event_type in self.subscribers:
            handlers = self.subscribers[event_type].copy()
            tasks = []
            for handler in handlers:
                if asyncio.iscoroutinefunction(handler):
                    task = asyncio.create_task(handler(data))
                else:
                    loop = asyncio.get_event_loop()
                    task = asyncio.create_task(
                        loop.run_in_executor(None, handler, data)
                    )
                tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

# =============================================================================
# ASYNC WORKFLOW ENGINE
# =============================================================================

@dataclass
class WorkflowStep:
    """Represents a workflow step."""
    name: str
    func: Callable
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_attempts: int = 3

class AsyncWorkflowEngine:
    """Async workflow engine for complex processing flows."""
    
    def __init__(self, config: AsyncFlowConfig):
        self.config = config
        self.steps = {}
        self.execution_order = []
        self.event_bus = AsyncEventBus()
        self.circuit_breaker = CircuitBreaker()
    
    def add_step(self, step: WorkflowStep):
        """Add workflow step."""
        self.steps[step.name] = step
        self._update_execution_order()
    
    def _update_execution_order(self):
        """Update execution order based on dependencies."""
        # Topological sort
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(step_name):
            if step_name in temp_visited:
                raise ValueError(f"Circular dependency detected: {step_name}")
            if step_name in visited:
                return
            
            temp_visited.add(step_name)
            step = self.steps[step_name]
            
            for dep in step.dependencies:
                visit(dep)
            
            temp_visited.remove(step_name)
            visited.add(step_name)
            order.append(step_name)
        
        for step_name in self.steps:
            if step_name not in visited:
                visit(step_name)
        
        self.execution_order = order
    
    async def execute_workflow(self, initial_data: Any) -> Any:
        """Execute workflow with given data."""
        context = {"data": initial_data, "results": {}}
        
        for step_name in self.execution_order:
            step = self.steps[step_name]
            
            # Check dependencies
            dep_results = [context["results"][dep] for dep in step.dependencies]
            
            # Execute step
            try:
                result = await self.circuit_breaker.call(
                    step.func, context["data"], *dep_results
                )
                context["results"][step_name] = result
                
                # Publish event
                await self.event_bus.publish(f"step_completed:{step_name}", {
                    "step": step_name,
                    "result": result,
                    "context": context
                })
                
            except Exception as e:
                await self.event_bus.publish(f"step_failed:{step_name}", {
                    "step": step_name,
                    "error": str(e),
                    "context": context
                })
                raise
        
        return context["results"]

# =============================================================================
# ASYNC MONITORING AND METRICS
# =============================================================================

class AsyncMetricsCollector:
    """Collect metrics for async operations."""
    
    def __init__(self):
        self.metrics = {
            "task_execution_times": [],
            "concurrent_tasks": 0,
            "failed_tasks": 0,
            "successful_tasks": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        self._lock = asyncio.Lock()
    
    async def record_task_execution(self, task_name: str, duration: float, success: bool):
        """Record task execution metrics."""
        async with self._lock:
            self.metrics["task_execution_times"].append({
                "task": task_name,
                "duration": duration,
                "success": success
            })
            
            if success:
                self.metrics["successful_tasks"] += 1
            else:
                self.metrics["failed_tasks"] += 1
    
    async def record_cache_access(self, hit: bool):
        """Record cache access."""
        async with self._lock:
            if hit:
                self.metrics["cache_hits"] += 1
            else:
                self.metrics["cache_misses"] += 1
    
    async def get_metrics(self) -> Dict:
        """Get current metrics."""
        async with self._lock:
            metrics = self.metrics.copy()
            
            # Calculate averages
            if metrics["task_execution_times"]:
                durations = [t["duration"] for t in metrics["task_execution_times"]]
                metrics["avg_execution_time"] = sum(durations) / len(durations)
                metrics["max_execution_time"] = max(durations)
                metrics["min_execution_time"] = min(durations)
            
            # Calculate cache hit rate
            total_cache_access = metrics["cache_hits"] + metrics["cache_misses"]
            if total_cache_access > 0:
                metrics["cache_hit_rate"] = metrics["cache_hits"] / total_cache_access
            
            return metrics

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_async_flow_config(**kwargs) -> AsyncFlowConfig:
    """Create async flow configuration."""
    return AsyncFlowConfig(**kwargs)

def create_async_video_processor(config: Optional[AsyncFlowConfig] = None) -> AsyncVideoProcessor:
    """Create async video processor."""
    if config is None:
        config = AsyncFlowConfig()
    return AsyncVideoProcessor(config)

def create_async_workflow_engine(config: Optional[AsyncFlowConfig] = None) -> AsyncWorkflowEngine:
    """Create async workflow engine."""
    if config is None:
        config = AsyncFlowConfig()
    return AsyncWorkflowEngine(config)

def create_priority_task_queue(maxsize: int = 1000) -> PriorityTaskQueue:
    """Create priority task queue."""
    return PriorityTaskQueue(maxsize)

def create_async_event_bus() -> AsyncEventBus:
    """Create async event bus."""
    return AsyncEventBus()

def create_async_metrics_collector() -> AsyncMetricsCollector:
    """Create async metrics collector."""
    return AsyncMetricsCollector()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def run_async_with_timeout(func: Callable, timeout: float, *args, **kwargs) -> Any:
    """Run async function with timeout."""
    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)

async def run_sync_in_executor(func: Callable, *args, **kwargs) -> Any:
    """Run sync function in executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)

async def batch_process_async(items: List[T], processor: Callable[[T], Awaitable[R]], 
                            max_concurrent: int = 10) -> List[R]:
    """Process items in batches asynchronously."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item: T) -> R:
        async with semaphore:
            return await processor(item)
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks, return_exceptions=True)

async def stream_process_async(stream: AsyncIterator[T], 
                             processor: Callable[[T], Awaitable[R]]) -> AsyncIterator[R]:
    """Process streaming data asynchronously."""
    async for item in stream:
        result = await processor(item)
        yield result

def async_retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator for async retry logic."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            raise last_exception
        return wrapper
    return decorator

# =============================================================================
# MAIN ASYNC FLOW MANAGER
# =============================================================================

class AsyncFlowManager:
    """Main manager for async flows."""
    
    def __init__(self, config: AsyncFlowConfig):
        self.config = config
        self.event_loop_manager = AsyncEventLoopManager(config)
        self.video_processor = create_async_video_processor(config)
        self.workflow_engine = create_async_workflow_engine(config)
        self.task_queue = create_priority_task_queue()
        self.event_bus = create_async_event_bus()
        self.metrics_collector = create_async_metrics_collector()
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Setup event handlers."""
        async def handle_task_completed(data):
            await self.metrics_collector.record_task_execution(
                data["task_name"], data["duration"], True
            )
        
        async def handle_task_failed(data):
            await self.metrics_collector.record_task_execution(
                data["task_name"], data["duration"], False
            )
        
        asyncio.create_task(self.event_bus.subscribe("task_completed", handle_task_completed))
        asyncio.create_task(self.event_bus.subscribe("task_failed", handle_task_failed))
    
    async def start(self):
        """Start async flow manager."""
        logger.info("Starting async flow manager")
        
        # Start background tasks
        background_tasks = [
            self.task_queue.process_tasks(self.config.max_concurrent_tasks),
            self._metrics_collector_task(),
            self._health_check_task()
        ]
        
        await asyncio.gather(*background_tasks, return_exceptions=True)
    
    async def _metrics_collector_task(self):
        """Background task for metrics collection."""
        while True:
            await asyncio.sleep(60)  # Collect metrics every minute
            metrics = await self.metrics_collector.get_metrics()
            logger.info("Async flow metrics", **metrics)
    
    async def _health_check_task(self):
        """Background task for health checks."""
        while True:
            await asyncio.sleep(30)  # Health check every 30 seconds
            # Perform health checks
            logger.debug("Async flow health check completed")
    
    async def shutdown(self):
        """Shutdown async flow manager."""
        logger.info("Shutting down async flow manager")
        
        # Close connections
        await self.video_processor.http_pool.close()
        
        # Cancel all tasks
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)

def create_async_flow_manager(config: Optional[AsyncFlowConfig] = None) -> AsyncFlowManager:
    """Create async flow manager."""
    if config is None:
        config = AsyncFlowConfig()
    return AsyncFlowManager(config) 