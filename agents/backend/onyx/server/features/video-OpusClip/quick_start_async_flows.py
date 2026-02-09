#!/usr/bin/env python3
"""
Quick Start: Async and Non-Blocking Flows

This script demonstrates the key features of the async flows system
for Video-OpusClip with practical examples.
"""

import asyncio
import time
import random
from typing import List, Dict, Any
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
    stream_process_async
)

logger = structlog.get_logger()

# =============================================================================
# EXAMPLE 1: BASIC ASYNC FLOW CONFIGURATION
# =============================================================================

def example_basic_configuration():
    """Demonstrate basic async flow configuration."""
    print("=== Example 1: Basic Configuration ===")
    
    # Create configuration
    config = create_async_flow_config(
        max_concurrent_tasks=50,
        max_concurrent_connections=20,
        chunk_size=100,
        timeout=30.0,
        retry_attempts=3,
        use_uvloop=True,
        enable_metrics=True,
        enable_circuit_breaker=True
    )
    
    print(f"✅ Configuration created:")
    print(f"   Max concurrent tasks: {config.max_concurrent_tasks}")
    print(f"   Max connections: {config.max_concurrent_connections}")
    print(f"   Chunk size: {config.chunk_size}")
    print(f"   Timeout: {config.timeout}s")
    print(f"   Retry attempts: {config.retry_attempts}")
    print(f"   Use uvloop: {config.use_uvloop}")
    print(f"   Enable metrics: {config.enable_metrics}")
    print(f"   Circuit breaker: {config.enable_circuit_breaker}")
    print()

# =============================================================================
# EXAMPLE 2: FLOW PATTERNS DEMONSTRATION
# =============================================================================

async def example_flow_patterns():
    """Demonstrate different async flow patterns."""
    print("=== Example 2: Flow Patterns ===")
    
    config = create_async_flow_config(max_concurrent_tasks=10)
    
    # Simulate async tasks
    async def task1():
        await asyncio.sleep(0.1)
        return "Task 1 completed"
    
    async def task2():
        await asyncio.sleep(0.1)
        return "Task 2 completed"
    
    async def task3():
        await asyncio.sleep(0.1)
        return "Task 3 completed"
    
    tasks = [task1, task2, task3]
    
    # Sequential Flow
    print("🔄 Sequential Flow:")
    sequential_flow = SequentialFlow(config)
    start_time = time.perf_counter()
    results = await sequential_flow.execute(tasks)
    duration = time.perf_counter() - start_time
    print(f"   Duration: {duration:.3f}s")
    print(f"   Results: {results}")
    
    # Parallel Flow
    print("\n🔄 Parallel Flow:")
    parallel_flow = ParallelFlow(config)
    start_time = time.perf_counter()
    results = await parallel_flow.execute(tasks)
    duration = time.perf_counter() - start_time
    print(f"   Duration: {duration:.3f}s")
    print(f"   Results: {results}")
    
    # Pipeline Flow
    print("\n🔄 Pipeline Flow:")
    pipeline_flow = PipelineFlow(config)
    
    async def stage1(data):
        await asyncio.sleep(0.1)
        return f"Stage 1 processed: {data}"
    
    async def stage2(data):
        await asyncio.sleep(0.1)
        return f"Stage 2 processed: {data}"
    
    async def stage3(data):
        await asyncio.sleep(0.1)
        return f"Stage 3 processed: {data}"
    
    stages = [stage1, stage2, stage3]
    start_time = time.perf_counter()
    result = await pipeline_flow.execute("input_data", stages)
    duration = time.perf_counter() - start_time
    print(f"   Duration: {duration:.3f}s")
    print(f"   Result: {result}")
    
    # Fan-Out Flow
    print("\n🔄 Fan-Out Flow:")
    fan_out_flow = FanOutFlow(config)
    
    async def processor1(data):
        await asyncio.sleep(0.1)
        return f"Processor 1: {data}"
    
    async def processor2(data):
        await asyncio.sleep(0.1)
        return f"Processor 2: {data}"
    
    async def processor3(data):
        await asyncio.sleep(0.1)
        return f"Processor 3: {data}"
    
    processors = [processor1, processor2, processor3]
    start_time = time.perf_counter()
    results = await fan_out_flow.execute("shared_data", processors)
    duration = time.perf_counter() - start_time
    print(f"   Duration: {duration:.3f}s")
    print(f"   Results: {results}")
    print()

# =============================================================================
# EXAMPLE 3: PRIORITY TASK QUEUE
# =============================================================================

async def example_priority_task_queue():
    """Demonstrate priority task queue."""
    print("=== Example 3: Priority Task Queue ===")
    
    queue = create_priority_task_queue(maxsize=100)
    
    # Create tasks with different priorities
    high_priority_task = AsyncTask(
        func=lambda: "High priority task completed",
        priority=TaskPriority.HIGH,
        task_id="high_1"
    )
    
    normal_priority_task = AsyncTask(
        func=lambda: "Normal priority task completed",
        priority=TaskPriority.NORMAL,
        task_id="normal_1"
    )
    
    low_priority_task = AsyncTask(
        func=lambda: "Low priority task completed",
        priority=TaskPriority.LOW,
        task_id="low_1"
    )
    
    # Add tasks to queue
    await queue.put(high_priority_task)
    await queue.put(normal_priority_task)
    await queue.put(low_priority_task)
    
    print(f"✅ Added {queue.queue.qsize()} tasks to queue")
    
    # Process tasks
    async def process_single_task():
        task = await queue.get()
        result = await queue._execute_task(task)
        print(f"   Processed {task.task_id}: {result}")
        return result
    
    # Process all tasks
    results = await asyncio.gather(*[process_single_task() for _ in range(3)])
    print(f"   All tasks completed: {results}")
    print()

# =============================================================================
# EXAMPLE 4: EVENT-DRIVEN ARCHITECTURE
# =============================================================================

async def example_event_driven_architecture():
    """Demonstrate event-driven architecture."""
    print("=== Example 4: Event-Driven Architecture ===")
    
    event_bus = create_async_event_bus()
    
    # Event handlers
    async def handle_video_processed(data):
        print(f"   📹 Video processed: {data}")
    
    async def handle_error(data):
        print(f"   ❌ Error occurred: {data}")
    
    async def handle_progress(data):
        print(f"   📊 Progress update: {data}")
    
    # Subscribe to events
    await event_bus.subscribe("video_processed", handle_video_processed)
    await event_bus.subscribe("error", handle_error)
    await event_bus.subscribe("progress", handle_progress)
    
    print("✅ Event handlers registered")
    
    # Simulate video processing with events
    async def process_video_with_events(video_id: str):
        try:
            # Start processing
            await event_bus.publish("progress", {"video_id": video_id, "status": "started"})
            await asyncio.sleep(0.1)
            
            # Processing step
            await event_bus.publish("progress", {"video_id": video_id, "status": "processing"})
            await asyncio.sleep(0.1)
            
            # Complete
            await event_bus.publish("video_processed", {
                "video_id": video_id,
                "status": "completed",
                "duration": 0.2
            })
            
        except Exception as e:
            await event_bus.publish("error", {
                "video_id": video_id,
                "error": str(e)
            })
    
    # Process multiple videos
    video_ids = ["video_1", "video_2", "video_3"]
    await asyncio.gather(*[process_video_with_events(vid) for vid in video_ids])
    print()

# =============================================================================
# EXAMPLE 5: WORKFLOW ENGINE
# =============================================================================

async def example_workflow_engine():
    """Demonstrate workflow engine."""
    print("=== Example 5: Workflow Engine ===")
    
    config = create_async_flow_config()
    workflow_engine = create_async_workflow_engine(config)
    
    # Define workflow steps
    async def download_step(data):
        await asyncio.sleep(0.1)
        return f"Downloaded: {data}"
    
    async def process_step(data, download_result):
        await asyncio.sleep(0.1)
        return f"Processed: {download_result}"
    
    async def encode_step(data, process_result):
        await asyncio.sleep(0.1)
        return f"Encoded: {process_result}"
    
    # Create workflow steps
    steps = [
        WorkflowStep("download", download_step, timeout=30.0),
        WorkflowStep("process", process_step, dependencies=["download"], timeout=60.0),
        WorkflowStep("encode", encode_step, dependencies=["process"], timeout=120.0)
    ]
    
    # Add steps to workflow
    for step in steps:
        workflow_engine.add_step(step)
    
    print(f"✅ Workflow created with {len(steps)} steps")
    print(f"   Execution order: {workflow_engine.execution_order}")
    
    # Execute workflow
    start_time = time.perf_counter()
    results = await workflow_engine.execute_workflow("video_url")
    duration = time.perf_counter() - start_time
    
    print(f"   Duration: {duration:.3f}s")
    print(f"   Results: {results}")
    print()

# =============================================================================
# EXAMPLE 6: ASYNC VIDEO PROCESSOR
# =============================================================================

async def example_async_video_processor():
    """Demonstrate async video processor."""
    print("=== Example 6: Async Video Processor ===")
    
    config = create_async_flow_config(max_concurrent_tasks=5)
    processor = create_async_video_processor(config)
    
    # Simulate video URLs
    video_urls = [
        "https://example.com/video1.mp4",
        "https://example.com/video2.mp4",
        "https://example.com/video3.mp4"
    ]
    
    print(f"Processing {len(video_urls)} videos...")
    
    # Batch processing
    start_time = time.perf_counter()
    results = await processor.process_video_batch(video_urls)
    duration = time.perf_counter() - start_time
    
    print(f"✅ Batch processing completed in {duration:.3f}s")
    print(f"   Processed {len(results)} videos")
    for i, result in enumerate(results):
        print(f"   Video {i+1}: {result['url']} -> {len(result['clips'])} clips")
    
    # Pipeline processing
    print("\n🔄 Pipeline processing:")
    start_time = time.perf_counter()
    pipeline_result = await processor.process_video_pipeline(video_urls[0])
    duration = time.perf_counter() - start_time
    
    print(f"   Pipeline completed in {duration:.3f}s")
    print(f"   Result: {pipeline_result}")
    print()

# =============================================================================
# EXAMPLE 7: METRICS AND MONITORING
# =============================================================================

async def example_metrics_and_monitoring():
    """Demonstrate metrics collection and monitoring."""
    print("=== Example 7: Metrics and Monitoring ===")
    
    metrics_collector = create_async_metrics_collector()
    
    # Simulate monitored operations
    async def monitored_operation(operation_name: str, should_fail: bool = False):
        start_time = time.perf_counter()
        
        try:
            await asyncio.sleep(0.1)
            if should_fail:
                raise Exception("Simulated failure")
            
            duration = time.perf_counter() - start_time
            await metrics_collector.record_task_execution(operation_name, duration, True)
            return f"{operation_name} completed"
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            await metrics_collector.record_task_execution(operation_name, duration, False)
            raise e
    
    # Execute monitored operations
    operations = [
        ("download_video", False),
        ("process_video", False),
        ("upload_video", True),  # This will fail
        ("generate_captions", False),
        ("add_effects", False)
    ]
    
    for op_name, should_fail in operations:
        try:
            result = await monitored_operation(op_name, should_fail)
            print(f"   ✅ {result}")
        except Exception as e:
            print(f"   ❌ {op_name} failed: {e}")
    
    # Get metrics
    metrics = await metrics_collector.get_metrics()
    
    print(f"\n📊 Metrics Summary:")
    print(f"   Successful tasks: {metrics['successful_tasks']}")
    print(f"   Failed tasks: {metrics['failed_tasks']}")
    print(f"   Success rate: {metrics['successful_tasks'] / (metrics['successful_tasks'] + metrics['failed_tasks']):.2%}")
    if 'avg_execution_time' in metrics:
        print(f"   Average execution time: {metrics['avg_execution_time']:.3f}s")
    print()

# =============================================================================
# EXAMPLE 8: RETRY AND ERROR HANDLING
# =============================================================================

async def example_retry_and_error_handling():
    """Demonstrate retry logic and error handling."""
    print("=== Example 8: Retry and Error Handling ===")
    
    # Simulate unreliable operation
    call_count = 0
    
    async def unreliable_operation():
        nonlocal call_count
        call_count += 1
        
        if call_count < 3:  # Fail first 2 times
            raise Exception(f"Operation failed (attempt {call_count})")
        
        return "Operation succeeded"
    
    # Retry decorator
    @async_retry(max_attempts=3, delay=0.1)
    async def retry_operation():
        return await unreliable_operation()
    
    print("🔄 Testing retry logic...")
    start_time = time.perf_counter()
    
    try:
        result = await retry_operation()
        duration = time.perf_counter() - start_time
        print(f"   ✅ {result} after {call_count} attempts in {duration:.3f}s")
    except Exception as e:
        print(f"   ❌ All retry attempts failed: {e}")
    
    # Reset for next test
    call_count = 0
    
    # Manual retry with exponential backoff
    async def exponential_backoff_retry(func, max_attempts=3, base_delay=0.1):
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
    start_time = time.perf_counter()
    
    try:
        result = await exponential_backoff_retry(unreliable_operation)
        duration = time.perf_counter() - start_time
        print(f"   ✅ {result} after {call_count} attempts in {duration:.3f}s")
    except Exception as e:
        print(f"   ❌ All retry attempts failed: {e}")
    
    print()

# =============================================================================
# EXAMPLE 9: BATCH AND STREAMING PROCESSING
# =============================================================================

async def example_batch_and_streaming():
    """Demonstrate batch and streaming processing."""
    print("=== Example 9: Batch and Streaming Processing ===")
    
    # Batch processing
    async def process_item(item: str) -> str:
        await asyncio.sleep(0.05)  # Simulate processing
        return f"Processed: {item}"
    
    items = [f"item_{i}" for i in range(20)]
    
    print("🔄 Batch processing...")
    start_time = time.perf_counter()
    batch_results = await batch_process_async(items, process_item, max_concurrent=5)
    duration = time.perf_counter() - start_time
    
    print(f"   ✅ Batch processing completed in {duration:.3f}s")
    print(f"   Processed {len(batch_results)} items")
    
    # Streaming processing
    async def item_stream():
        for i in range(10):
            yield f"stream_item_{i}"
            await asyncio.sleep(0.01)  # Simulate stream delay
    
    print("\n🔄 Streaming processing...")
    start_time = time.perf_counter()
    
    stream_results = []
    async for result in stream_process_async(item_stream(), process_item):
        stream_results.append(result)
    
    duration = time.perf_counter() - start_time
    
    print(f"   ✅ Streaming processing completed in {duration:.3f}s")
    print(f"   Processed {len(stream_results)} items")
    print()

# =============================================================================
# EXAMPLE 10: COMPLETE ASYNC FLOW MANAGER
# =============================================================================

async def example_complete_async_flow_manager():
    """Demonstrate complete async flow manager."""
    print("=== Example 10: Complete Async Flow Manager ===")
    
    # Create configuration
    config = create_async_flow_config(
        max_concurrent_tasks=20,
        max_concurrent_connections=10,
        enable_metrics=True,
        enable_circuit_breaker=True
    )
    
    # Create flow manager
    flow_manager = create_async_flow_manager(config)
    
    print("🚀 Starting async flow manager...")
    
    # Start background tasks
    manager_task = asyncio.create_task(flow_manager.start())
    
    # Give manager time to start
    await asyncio.sleep(0.1)
    
    # Simulate video processing
    video_urls = [f"video_{i}.mp4" for i in range(5)]
    
    print(f"📹 Processing {len(video_urls)} videos...")
    
    # Add tasks to queue
    for i, url in enumerate(video_urls):
        task = AsyncTask(
            func=lambda u=url: f"Processed {u}",
            priority=TaskPriority.HIGH if i < 2 else TaskPriority.NORMAL,
            task_id=f"video_{i}",
            timeout=30.0
        )
        await flow_manager.task_queue.put(task)
    
    # Wait for processing
    await asyncio.sleep(1.0)
    
    # Get metrics
    metrics = await flow_manager.metrics_collector.get_metrics()
    
    print(f"📊 Processing metrics:")
    print(f"   Successful tasks: {metrics['successful_tasks']}")
    print(f"   Failed tasks: {metrics['failed_tasks']}")
    print(f"   Completed tasks: {len(flow_manager.task_queue.completed_tasks)}")
    
    # Shutdown manager
    print("🛑 Shutting down async flow manager...")
    manager_task.cancel()
    
    try:
        await manager_task
    except asyncio.CancelledError:
        pass
    
    await flow_manager.shutdown()
    print("✅ Async flow manager shutdown complete")
    print()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Run all async flow examples."""
    print("🚀 Async and Non-Blocking Flows - Quick Start Examples")
    print("=" * 60)
    print()
    
    try:
        # Run examples
        example_basic_configuration()
        await example_flow_patterns()
        await example_priority_task_queue()
        await example_event_driven_architecture()
        await example_workflow_engine()
        await example_async_video_processor()
        await example_metrics_and_monitoring()
        await example_retry_and_error_handling()
        await example_batch_and_streaming()
        await example_complete_async_flow_manager()
        
        print("🎉 All examples completed successfully!")
        print()
        print("📚 Next Steps:")
        print("   1. Review the ASYNC_FLOWS_GUIDE.md for detailed documentation")
        print("   2. Explore the async_flows.py module for implementation details")
        print("   3. Integrate async flows into your Video-OpusClip application")
        print("   4. Monitor performance with the built-in metrics collector")
        print("   5. Customize flow patterns for your specific use cases")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the async examples
    asyncio.run(main()) 