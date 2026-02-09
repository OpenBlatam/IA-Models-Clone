"""
Async and Non-Blocking Flows Examples

Comprehensive examples demonstrating async patterns, event-driven architecture,
and non-blocking operations for Video-OpusClip system.
"""

import asyncio
import time
import random
import json
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import structlog

# Import async flows components
from async_flows import (
    create_async_flow_config,
    create_async_video_processor,
    create_async_workflow_engine,
    create_priority_task_queue,
    create_async_event_bus,
    create_async_metrics_collector,
    create_async_flow_manager,
    AsyncTask,
    TaskPriority,
    WorkflowStep,
    SequentialFlow,
    ParallelFlow,
    StreamingFlow,
    PipelineFlow,
    FanOutFlow,
    FanInFlow,
    async_retry,
    batch_process_async,
    stream_process_async,
    run_async_with_timeout,
    run_sync_in_executor
)

logger = structlog.get_logger()

# =============================================================================
# EXAMPLE 1: VIDEO DOWNLOAD PIPELINE
# =============================================================================

async def example_video_download_pipeline():
    """Example of video download pipeline with async flows."""
    print("=== Example 1: Video Download Pipeline ===")
    
    config = create_async_flow_config(max_concurrent_tasks=10)
    pipeline_flow = PipelineFlow(config)
    
    # Pipeline stages
    async def validate_url(url: str) -> str:
        """Validate video URL."""
        await asyncio.sleep(0.1)
        if not url.startswith("http"):
            raise ValueError("Invalid URL")
        return url
    
    async def download_video(url: str) -> Dict:
        """Download video from URL."""
        await asyncio.sleep(0.2)
        return {
            "url": url,
            "file_path": f"/tmp/video_{hash(url)}.mp4",
            "size": random.randint(1000000, 10000000),
            "duration": random.randint(60, 600)
        }
    
    async def extract_metadata(video_data: Dict) -> Dict:
        """Extract video metadata."""
        await asyncio.sleep(0.1)
        video_data["metadata"] = {
            "format": "mp4",
            "codec": "h264",
            "resolution": "1920x1080",
            "fps": 30
        }
        return video_data
    
    async def generate_thumbnail(video_data: Dict) -> Dict:
        """Generate video thumbnail."""
        await asyncio.sleep(0.15)
        video_data["thumbnail"] = f"/tmp/thumb_{hash(video_data['url'])}.jpg"
        return video_data
    
    # Execute pipeline
    stages = [validate_url, download_video, extract_metadata, generate_thumbnail]
    
    video_urls = [
        "https://youtube.com/watch?v=video1",
        "https://youtube.com/watch?v=video2",
        "https://youtube.com/watch?v=video3"
    ]
    
    results = []
    for url in video_urls:
        try:
            result = await pipeline_flow.execute(url, stages)
            results.append(result)
            print(f"   ✅ Pipeline completed for {url}")
        except Exception as e:
            print(f"   ❌ Pipeline failed for {url}: {e}")
    
    print(f"   Total processed: {len(results)}/{len(video_urls)}")
    print()

# =============================================================================
# EXAMPLE 2: BATCH VIDEO PROCESSING
# =============================================================================

async def example_batch_video_processing():
    """Example of batch video processing with parallel flows."""
    print("=== Example 2: Batch Video Processing ===")
    
    config = create_async_flow_config(max_concurrent_tasks=20)
    parallel_flow = ParallelFlow(config)
    
    # Video processing tasks
    async def process_video_quality_high(video_id: str) -> Dict:
        """Process video with high quality settings."""
        await asyncio.sleep(0.3)
        return {
            "video_id": video_id,
            "quality": "high",
            "resolution": "4K",
            "bitrate": "50Mbps",
            "processing_time": 0.3
        }
    
    async def process_video_quality_medium(video_id: str) -> Dict:
        """Process video with medium quality settings."""
        await asyncio.sleep(0.2)
        return {
            "video_id": video_id,
            "quality": "medium",
            "resolution": "1080p",
            "bitrate": "20Mbps",
            "processing_time": 0.2
        }
    
    async def process_video_quality_low(video_id: str) -> Dict:
        """Process video with low quality settings."""
        await asyncio.sleep(0.1)
        return {
            "video_id": video_id,
            "quality": "low",
            "resolution": "720p",
            "bitrate": "8Mbps",
            "processing_time": 0.1
        }
    
    # Generate video IDs
    video_ids = [f"video_{i}" for i in range(15)]
    
    # Create processing tasks for different qualities
    high_quality_tasks = [lambda vid=vid: process_video_quality_high(vid) for vid in video_ids[:5]]
    medium_quality_tasks = [lambda vid=vid: process_video_quality_medium(vid) for vid in video_ids[5:10]]
    low_quality_tasks = [lambda vid=vid: process_video_quality_low(vid) for vid in video_ids[10:]]
    
    print(f"Processing {len(video_ids)} videos in parallel...")
    
    # Process all qualities in parallel
    start_time = time.perf_counter()
    
    high_results, medium_results, low_results = await asyncio.gather(
        parallel_flow.execute(high_quality_tasks),
        parallel_flow.execute(medium_quality_tasks),
        parallel_flow.execute(low_quality_tasks)
    )
    
    duration = time.perf_counter() - start_time
    
    print(f"✅ Batch processing completed in {duration:.3f}s")
    print(f"   High quality: {len(high_results)} videos")
    print(f"   Medium quality: {len(medium_results)} videos")
    print(f"   Low quality: {len(low_results)} videos")
    
    # Calculate statistics
    all_results = high_results + medium_results + low_results
    avg_processing_time = sum(r["processing_time"] for r in all_results) / len(all_results)
    print(f"   Average processing time: {avg_processing_time:.3f}s")
    print()

# =============================================================================
# EXAMPLE 3: STREAMING VIDEO PROCESSING
# =============================================================================

async def example_streaming_video_processing():
    """Example of streaming video processing."""
    print("=== Example 3: Streaming Video Processing ===")
    
    config = create_async_flow_config(max_concurrent_tasks=5)
    streaming_flow = StreamingFlow(config)
    
    # Simulate video stream
    async def video_stream() -> AsyncIterator[str]:
        """Generate stream of video URLs."""
        video_urls = [
            "https://stream1.com/video1.mp4",
            "https://stream2.com/video2.mp4",
            "https://stream3.com/video3.mp4",
            "https://stream4.com/video4.mp4",
            "https://stream5.com/video5.mp4"
        ]
        
        for url in video_urls:
            yield url
            await asyncio.sleep(0.2)  # Simulate stream delay
    
    # Video processor
    async def process_streaming_video(url: str) -> Dict:
        """Process streaming video."""
        await asyncio.sleep(0.3)
        return {
            "url": url,
            "processed_at": time.time(),
            "status": "completed",
            "clips_generated": random.randint(1, 5)
        }
    
    print("🔄 Processing streaming videos...")
    
    start_time = time.perf_counter()
    processed_count = 0
    
    async for result in streaming_flow.execute(video_stream(), process_streaming_video):
        processed_count += 1
        print(f"   📹 Processed: {result['url']} -> {result['clips_generated']} clips")
    
    duration = time.perf_counter() - start_time
    
    print(f"✅ Streaming processing completed in {duration:.3f}s")
    print(f"   Processed {processed_count} videos")
    print()

# =============================================================================
# EXAMPLE 4: EVENT-DRIVEN VIDEO WORKFLOW
# =============================================================================

async def example_event_driven_video_workflow():
    """Example of event-driven video workflow."""
    print("=== Example 4: Event-Driven Video Workflow ===")
    
    event_bus = create_async_event_bus()
    
    # Event handlers
    async def handle_video_uploaded(data):
        print(f"   📤 Video uploaded: {data['video_id']}")
        # Trigger processing
        await event_bus.publish("start_processing", data)
    
    async def handle_processing_started(data):
        print(f"   🔄 Processing started: {data['video_id']}")
        # Simulate processing
        await asyncio.sleep(0.2)
        await event_bus.publish("processing_completed", data)
    
    async def handle_processing_completed(data):
        print(f"   ✅ Processing completed: {data['video_id']}")
        # Trigger encoding
        await event_bus.publish("start_encoding", data)
    
    async def handle_encoding_started(data):
        print(f"   🎬 Encoding started: {data['video_id']}")
        # Simulate encoding
        await asyncio.sleep(0.3)
        await event_bus.publish("encoding_completed", data)
    
    async def handle_encoding_completed(data):
        print(f"   🎬 Encoding completed: {data['video_id']}")
        # Trigger upload
        await event_bus.publish("start_upload", data)
    
    async def handle_upload_started(data):
        print(f"   📤 Upload started: {data['video_id']}")
        # Simulate upload
        await asyncio.sleep(0.1)
        await event_bus.publish("upload_completed", data)
    
    async def handle_upload_completed(data):
        print(f"   ✅ Upload completed: {data['video_id']}")
        # Send notification
        await event_bus.publish("send_notification", data)
    
    async def handle_notification(data):
        print(f"   📧 Notification sent: {data['video_id']}")
    
    # Subscribe to events
    event_handlers = [
        ("video_uploaded", handle_video_uploaded),
        ("start_processing", handle_processing_started),
        ("processing_completed", handle_processing_completed),
        ("start_encoding", handle_encoding_started),
        ("encoding_completed", handle_encoding_completed),
        ("start_upload", handle_upload_started),
        ("upload_completed", handle_upload_completed),
        ("send_notification", handle_notification)
    ]
    
    for event_type, handler in event_handlers:
        await event_bus.subscribe(event_type, handler)
    
    print("✅ Event handlers registered")
    
    # Simulate video uploads
    video_ids = ["video_001", "video_002", "video_003"]
    
    print("🔄 Starting event-driven workflow...")
    start_time = time.perf_counter()
    
    # Trigger workflow for each video
    for video_id in video_ids:
        await event_bus.publish("video_uploaded", {"video_id": video_id})
        await asyncio.sleep(0.1)  # Small delay between videos
    
    # Wait for all workflows to complete
    await asyncio.sleep(2.0)
    
    duration = time.perf_counter() - start_time
    print(f"✅ Event-driven workflow completed in {duration:.3f}s")
    print()

# =============================================================================
# EXAMPLE 5: PRIORITY-BASED TASK PROCESSING
# =============================================================================

async def example_priority_based_task_processing():
    """Example of priority-based task processing."""
    print("=== Example 5: Priority-Based Task Processing ===")
    
    task_queue = create_priority_task_queue(maxsize=100)
    
    # Task definitions
    async def critical_video_processing(video_id: str) -> str:
        """Critical video processing task."""
        await asyncio.sleep(0.2)
        return f"Critical processing completed for {video_id}"
    
    async def normal_video_processing(video_id: str) -> str:
        """Normal video processing task."""
        await asyncio.sleep(0.3)
        return f"Normal processing completed for {video_id}"
    
    async def background_video_processing(video_id: str) -> str:
        """Background video processing task."""
        await asyncio.sleep(0.4)
        return f"Background processing completed for {video_id}"
    
    # Create tasks with different priorities
    tasks = []
    
    # Critical tasks (HIGH priority)
    for i in range(3):
        task = AsyncTask(
            func=lambda vid=f"critical_{i}": critical_video_processing(vid),
            priority=TaskPriority.HIGH,
            task_id=f"critical_{i}",
            timeout=60.0
        )
        tasks.append(task)
    
    # Normal tasks (NORMAL priority)
    for i in range(5):
        task = AsyncTask(
            func=lambda vid=f"normal_{i}": normal_video_processing(vid),
            priority=TaskPriority.NORMAL,
            task_id=f"normal_{i}",
            timeout=120.0
        )
        tasks.append(task)
    
    # Background tasks (LOW priority)
    for i in range(7):
        task = AsyncTask(
            func=lambda vid=f"background_{i}": background_video_processing(vid),
            priority=TaskPriority.LOW,
            task_id=f"background_{i}",
            timeout=300.0
        )
        tasks.append(task)
    
    # Add tasks to queue
    print(f"Adding {len(tasks)} tasks to priority queue...")
    for task in tasks:
        await task_queue.put(task)
    
    # Process tasks
    print("🔄 Processing tasks by priority...")
    start_time = time.perf_counter()
    
    # Start workers
    worker_tasks = []
    for i in range(3):  # 3 workers
        worker_task = asyncio.create_task(task_queue._worker(i))
        worker_tasks.append(worker_task)
    
    # Wait for all tasks to complete
    while len(task_queue.completed_tasks) < len(tasks):
        await asyncio.sleep(0.1)
    
    # Cancel workers
    for worker_task in worker_tasks:
        worker_task.cancel()
    
    await asyncio.gather(*worker_tasks, return_exceptions=True)
    
    duration = time.perf_counter() - start_time
    
    print(f"✅ Priority-based processing completed in {duration:.3f}s")
    print(f"   Completed tasks: {len(task_queue.completed_tasks)}")
    print(f"   Failed tasks: {len(task_queue.failed_tasks)}")
    
    # Show completion order
    print("   Completion order:")
    for i, (task, result) in enumerate(task_queue.completed_tasks[:10]):  # Show first 10
        print(f"     {i+1}. {task.task_id}: {result}")
    print()

# =============================================================================
# EXAMPLE 6: WORKFLOW ENGINE WITH DEPENDENCIES
# =============================================================================

async def example_workflow_engine_with_dependencies():
    """Example of workflow engine with complex dependencies."""
    print("=== Example 6: Workflow Engine with Dependencies ===")
    
    config = create_async_flow_config()
    workflow_engine = create_async_workflow_engine(config)
    
    # Workflow step functions
    async def validate_input(data):
        """Validate input data."""
        await asyncio.sleep(0.1)
        if not data.get("video_url"):
            raise ValueError("Missing video URL")
        return data
    
    async def download_video(data):
        """Download video."""
        await asyncio.sleep(0.2)
        data["video_file"] = f"/tmp/{hash(data['video_url'])}.mp4"
        return data
    
    async def extract_audio(data):
        """Extract audio from video."""
        await asyncio.sleep(0.15)
        data["audio_file"] = data["video_file"].replace(".mp4", ".wav")
        return data
    
    async def generate_captions(data):
        """Generate captions from audio."""
        await asyncio.sleep(0.25)
        data["captions"] = [
            {"start": 0, "end": 10, "text": "Caption 1"},
            {"start": 10, "end": 20, "text": "Caption 2"},
            {"start": 20, "end": 30, "text": "Caption 3"}
        ]
        return data
    
    async def analyze_content(data):
        """Analyze video content."""
        await asyncio.sleep(0.2)
        data["analysis"] = {
            "duration": 30,
            "scenes": 5,
            "objects_detected": ["person", "car", "building"],
            "sentiment": "positive"
        }
        return data
    
    async def create_clips(data):
        """Create video clips based on analysis."""
        await asyncio.sleep(0.3)
        data["clips"] = [
            {"start": 0, "end": 10, "type": "intro"},
            {"start": 10, "end": 20, "type": "main"},
            {"start": 20, "end": 30, "type": "outro"}
        ]
        return data
    
    async def add_effects(data):
        """Add effects to clips."""
        await asyncio.sleep(0.2)
        for clip in data["clips"]:
            clip["effects"] = ["fade_in", "fade_out"]
        return data
    
    async def encode_clips(data):
        """Encode final clips."""
        await asyncio.sleep(0.4)
        data["encoded_clips"] = [
            f"/output/clip_{i}.mp4" for i in range(len(data["clips"]))
        ]
        return data
    
    async def upload_results(data):
        """Upload results."""
        await asyncio.sleep(0.1)
        data["upload_urls"] = [
            f"https://cdn.example.com/clip_{i}.mp4" 
            for i in range(len(data["encoded_clips"]))
        ]
        return data
    
    # Create workflow steps with dependencies
    workflow_steps = [
        WorkflowStep("validate", validate_input, timeout=30.0),
        WorkflowStep("download", download_video, dependencies=["validate"], timeout=60.0),
        WorkflowStep("extract_audio", extract_audio, dependencies=["download"], timeout=45.0),
        WorkflowStep("generate_captions", generate_captions, dependencies=["extract_audio"], timeout=90.0),
        WorkflowStep("analyze_content", analyze_content, dependencies=["download"], timeout=60.0),
        WorkflowStep("create_clips", create_clips, dependencies=["analyze_content", "generate_captions"], timeout=120.0),
        WorkflowStep("add_effects", add_effects, dependencies=["create_clips"], timeout=60.0),
        WorkflowStep("encode_clips", encode_clips, dependencies=["add_effects"], timeout=180.0),
        WorkflowStep("upload_results", upload_results, dependencies=["encode_clips"], timeout=60.0)
    ]
    
    # Add steps to workflow
    for step in workflow_steps:
        workflow_engine.add_step(step)
    
    print(f"✅ Workflow created with {len(workflow_steps)} steps")
    print(f"   Execution order: {workflow_engine.execution_order}")
    
    # Execute workflow
    initial_data = {
        "video_url": "https://youtube.com/watch?v=example",
        "user_id": "user_123",
        "project_id": "project_456"
    }
    
    print("🔄 Executing workflow...")
    start_time = time.perf_counter()
    
    try:
        results = await workflow_engine.execute_workflow(initial_data)
        duration = time.perf_counter() - start_time
        
        print(f"✅ Workflow completed in {duration:.3f}s")
        print(f"   Generated {len(results['upload_results']['upload_urls'])} clips")
        print(f"   Final upload URLs: {results['upload_results']['upload_urls']}")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
    
    print()

# =============================================================================
# EXAMPLE 7: METRICS AND PERFORMANCE MONITORING
# =============================================================================

async def example_metrics_and_performance_monitoring():
    """Example of comprehensive metrics and performance monitoring."""
    print("=== Example 7: Metrics and Performance Monitoring ===")
    
    metrics_collector = create_async_metrics_collector()
    
    # Simulate various operations with different characteristics
    async def fast_operation(operation_name: str):
        """Fast operation (should succeed)."""
        start_time = time.perf_counter()
        await asyncio.sleep(0.05)
        duration = time.perf_counter() - start_time
        await metrics_collector.record_task_execution(operation_name, duration, True)
        return f"{operation_name} completed"
    
    async def slow_operation(operation_name: str):
        """Slow operation (should succeed)."""
        start_time = time.perf_counter()
        await asyncio.sleep(0.3)
        duration = time.perf_counter() - start_time
        await metrics_collector.record_task_execution(operation_name, duration, True)
        return f"{operation_name} completed"
    
    async def failing_operation(operation_name: str):
        """Operation that fails."""
        start_time = time.perf_counter()
        await asyncio.sleep(0.1)
        duration = time.perf_counter() - start_time
        await metrics_collector.record_task_execution(operation_name, duration, False)
        raise Exception(f"{operation_name} failed")
    
    async def intermittent_operation(operation_name: str):
        """Operation that sometimes fails."""
        start_time = time.perf_counter()
        await asyncio.sleep(0.15)
        duration = time.perf_counter() - start_time
        
        # 20% chance of failure
        if random.random() < 0.2:
            await metrics_collector.record_task_execution(operation_name, duration, False)
            raise Exception(f"{operation_name} failed")
        else:
            await metrics_collector.record_task_execution(operation_name, duration, True)
            return f"{operation_name} completed"
    
    # Execute various operations
    operations = [
        ("download_video", fast_operation),
        ("process_video", slow_operation),
        ("upload_video", fast_operation),
        ("generate_captions", slow_operation),
        ("add_effects", fast_operation),
        ("encode_video", slow_operation),
        ("broken_api_call", failing_operation),
        ("unreliable_service", intermittent_operation)
    ]
    
    print("🔄 Executing operations for metrics collection...")
    
    # Execute operations multiple times
    for _ in range(3):  # 3 rounds
        for op_name, op_func in operations:
            try:
                result = await op_func(op_name)
                print(f"   ✅ {result}")
            except Exception as e:
                print(f"   ❌ {op_name}: {e}")
    
    # Get comprehensive metrics
    metrics = await metrics_collector.get_metrics()
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Total operations: {metrics['successful_tasks'] + metrics['failed_tasks']}")
    print(f"   Successful: {metrics['successful_tasks']}")
    print(f"   Failed: {metrics['failed_tasks']}")
    print(f"   Success rate: {metrics['successful_tasks'] / (metrics['successful_tasks'] + metrics['failed_tasks']):.2%}")
    
    if 'avg_execution_time' in metrics:
        print(f"   Average execution time: {metrics['avg_execution_time']:.3f}s")
        print(f"   Min execution time: {metrics['min_execution_time']:.3f}s")
        print(f"   Max execution time: {metrics['max_execution_time']:.3f}s")
    
    # Analyze operation types
    operation_stats = {}
    for task_data in metrics['task_execution_times']:
        op_name = task_data['task']
        if op_name not in operation_stats:
            operation_stats[op_name] = {'count': 0, 'success': 0, 'total_time': 0}
        
        operation_stats[op_name]['count'] += 1
        operation_stats[op_name]['total_time'] += task_data['duration']
        if task_data['success']:
            operation_stats[op_name]['success'] += 1
    
    print(f"\n📈 Operation Statistics:")
    for op_name, stats in operation_stats.items():
        avg_time = stats['total_time'] / stats['count']
        success_rate = stats['success'] / stats['count']
        print(f"   {op_name}:")
        print(f"     Count: {stats['count']}")
        print(f"     Success rate: {success_rate:.2%}")
        print(f"     Avg time: {avg_time:.3f}s")
    
    print()

# =============================================================================
# EXAMPLE 8: RETRY AND RESILIENCE PATTERNS
# =============================================================================

async def example_retry_and_resilience_patterns():
    """Example of retry and resilience patterns."""
    print("=== Example 8: Retry and Resilience Patterns ===")
    
    # Simulate unreliable service
    call_counters = {}
    
    async def unreliable_service(service_name: str, failure_rate: float = 0.3):
        """Simulate unreliable service."""
        if service_name not in call_counters:
            call_counters[service_name] = 0
        
        call_counters[service_name] += 1
        
        await asyncio.sleep(0.1)
        
        if random.random() < failure_rate:
            raise Exception(f"{service_name} failed (attempt {call_counters[service_name]})")
        
        return f"{service_name} succeeded (attempt {call_counters[service_name]})"
    
    # Retry decorator example
    @async_retry(max_attempts=3, delay=0.1)
    async def retry_service_call(service_name: str):
        return await unreliable_service(service_name, failure_rate=0.5)
    
    print("🔄 Testing retry decorator...")
    
    services = ["api_service", "database_service", "file_service"]
    
    for service in services:
        try:
            result = await retry_service_call(service)
            print(f"   ✅ {result}")
        except Exception as e:
            print(f"   ❌ {service}: {e}")
    
    # Manual retry with exponential backoff
    async def exponential_backoff_retry(func, max_attempts=5, base_delay=0.1):
        """Manual retry with exponential backoff."""
        for attempt in range(max_attempts):
            try:
                return await func()
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise e
                
                delay = base_delay * (2 ** attempt)
                print(f"   ⏳ Attempt {attempt + 1} failed, retrying in {delay:.3f}s...")
                await asyncio.sleep(delay)
    
    print("\n🔄 Testing exponential backoff...")
    
    async def test_service():
        return await unreliable_service("backoff_test", failure_rate=0.7)
    
    try:
        result = await exponential_backoff_retry(test_service)
        print(f"   ✅ {result}")
    except Exception as e:
        print(f"   ❌ All retry attempts failed: {e}")
    
    # Circuit breaker pattern simulation
    print("\n🔄 Testing circuit breaker pattern...")
    
    from async_flows import CircuitBreaker
    
    circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=1.0)
    
    async def circuit_breaker_test():
        return await unreliable_service("circuit_test", failure_rate=0.8)
    
    for i in range(5):
        try:
            result = await circuit_breaker.call(circuit_breaker_test)
            print(f"   ✅ Call {i+1}: {result}")
        except Exception as e:
            print(f"   ❌ Call {i+1}: {e}")
    
    print()

# =============================================================================
# EXAMPLE 9: BATCH AND STREAMING PROCESSING
# =============================================================================

async def example_batch_and_streaming_processing():
    """Example of batch and streaming processing patterns."""
    print("=== Example 9: Batch and Streaming Processing ===")
    
    # Batch processing example
    async def process_video_batch(video_id: str) -> Dict:
        """Process video in batch."""
        await asyncio.sleep(0.1)
        return {
            "video_id": video_id,
            "processed_at": time.time(),
            "clips_generated": random.randint(1, 5),
            "processing_time": 0.1
        }
    
    video_ids = [f"batch_video_{i}" for i in range(20)]
    
    print("🔄 Batch processing...")
    start_time = time.perf_counter()
    
    batch_results = await batch_process_async(
        video_ids, 
        process_video_batch, 
        max_concurrent=5
    )
    
    duration = time.perf_counter() - start_time
    
    print(f"✅ Batch processing completed in {duration:.3f}s")
    print(f"   Processed {len(batch_results)} videos")
    print(f"   Average processing time: {sum(r['processing_time'] for r in batch_results) / len(batch_results):.3f}s")
    
    # Streaming processing example
    async def video_stream() -> AsyncIterator[str]:
        """Generate stream of video URLs."""
        for i in range(10):
            yield f"stream_video_{i}"
            await asyncio.sleep(0.05)  # Simulate stream delay
    
    async def process_streaming_video(video_id: str) -> Dict:
        """Process streaming video."""
        await asyncio.sleep(0.15)
        return {
            "video_id": video_id,
            "streamed_at": time.time(),
            "status": "processed"
        }
    
    print("\n🔄 Streaming processing...")
    start_time = time.perf_counter()
    
    stream_results = []
    async for result in stream_process_async(video_stream(), process_streaming_video):
        stream_results.append(result)
        print(f"   📹 Streamed: {result['video_id']}")
    
    duration = time.perf_counter() - start_time
    
    print(f"✅ Streaming processing completed in {duration:.3f}s")
    print(f"   Processed {len(stream_results)} videos")
    print()

# =============================================================================
# EXAMPLE 10: COMPLETE ASYNC FLOW INTEGRATION
# =============================================================================

async def example_complete_async_flow_integration():
    """Example of complete async flow integration."""
    print("=== Example 10: Complete Async Flow Integration ===")
    
    # Create comprehensive configuration
    config = create_async_flow_config(
        max_concurrent_tasks=30,
        max_concurrent_connections=15,
        chunk_size=50,
        timeout=60.0,
        retry_attempts=3,
        use_uvloop=True,
        enable_metrics=True,
        enable_circuit_breaker=True,
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=30.0
    )
    
    # Create flow manager
    flow_manager = create_async_flow_manager(config)
    
    print("🚀 Starting complete async flow integration...")
    
    # Start background tasks
    manager_task = asyncio.create_task(flow_manager.start())
    
    # Give manager time to start
    await asyncio.sleep(0.1)
    
    # Simulate complex video processing workflow
    video_requests = [
        {
            "video_id": f"video_{i}",
            "url": f"https://youtube.com/watch?v=video_{i}",
            "priority": TaskPriority.HIGH if i < 3 else TaskPriority.NORMAL,
            "quality": "high" if i < 5 else "medium"
        }
        for i in range(10)
    ]
    
    print(f"📹 Processing {len(video_requests)} video requests...")
    
    # Add tasks to queue with different priorities
    for request in video_requests:
        task = AsyncTask(
            func=lambda req=request: process_complex_video(req),
            priority=request["priority"],
            task_id=request["video_id"],
            timeout=120.0,
            retry_attempts=3
        )
        await flow_manager.task_queue.put(task)
    
    # Wait for processing
    await asyncio.sleep(2.0)
    
    # Get comprehensive metrics
    metrics = await flow_manager.metrics_collector.get_metrics()
    
    print(f"📊 Integration Results:")
    print(f"   Successful tasks: {metrics['successful_tasks']}")
    print(f"   Failed tasks: {metrics['failed_tasks']}")
    print(f"   Success rate: {metrics['successful_tasks'] / (metrics['successful_tasks'] + metrics['failed_tasks']):.2%}")
    
    if 'avg_execution_time' in metrics:
        print(f"   Average execution time: {metrics['avg_execution_time']:.3f}s")
    
    print(f"   Completed tasks: {len(flow_manager.task_queue.completed_tasks)}")
    print(f"   Failed tasks: {len(flow_manager.task_queue.failed_tasks)}")
    
    # Shutdown gracefully
    print("🛑 Shutting down async flow manager...")
    manager_task.cancel()
    
    try:
        await manager_task
    except asyncio.CancelledError:
        pass
    
    await flow_manager.shutdown()
    print("✅ Complete async flow integration finished")
    print()

# Helper function for complex video processing
async def process_complex_video(request: Dict) -> Dict:
    """Simulate complex video processing."""
    await asyncio.sleep(0.2)
    return {
        "video_id": request["video_id"],
        "status": "completed",
        "quality": request["quality"],
        "processing_time": 0.2,
        "clips_generated": random.randint(1, 5)
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Run all comprehensive async flow examples."""
    print("🚀 Async and Non-Blocking Flows - Comprehensive Examples")
    print("=" * 70)
    print()
    
    try:
        # Run all examples
        await example_video_download_pipeline()
        await example_batch_video_processing()
        await example_streaming_video_processing()
        await example_event_driven_video_workflow()
        await example_priority_based_task_processing()
        await example_workflow_engine_with_dependencies()
        await example_metrics_and_performance_monitoring()
        await example_retry_and_resilience_patterns()
        await example_batch_and_streaming_processing()
        await example_complete_async_flow_integration()
        
        print("🎉 All comprehensive examples completed successfully!")
        print()
        print("📚 Key Takeaways:")
        print("   ✅ Async flows provide non-blocking, high-performance processing")
        print("   ✅ Event-driven architecture enables loose coupling and scalability")
        print("   ✅ Priority queues ensure critical tasks are processed first")
        print("   ✅ Circuit breakers provide fault tolerance and resilience")
        print("   ✅ Comprehensive metrics enable performance monitoring")
        print("   ✅ Workflow engines handle complex dependencies efficiently")
        print("   ✅ Retry patterns ensure reliability in unreliable environments")
        print("   ✅ Batch and streaming processing optimize resource usage")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the comprehensive async examples
    asyncio.run(main()) 