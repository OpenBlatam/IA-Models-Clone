from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import uuid
import json
from typing import Dict, List, Any, Optional
import random
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from async_non_blocking_flows import (
from typing import Any, List, Dict, Optional
import logging
"""
Asynchronous and Non-Blocking Flows Demo

This demo showcases:
- Event-driven architecture with async event streams
- Reactive programming patterns
- Async data processing pipelines
- Non-blocking API flows
- Background task orchestration
- Stream processing
- Real-time updates
- Performance comparisons
"""



    AsyncFlowOrchestrator, EventBus, AsyncDataPipeline, ReactiveStream,
    AsyncMessageQueue, AsyncEvent, FlowContext, EventType, FlowType,
    DataValidationProcessor, DataEnrichmentProcessor, DataTransformationProcessor,
    create_async_flow, run_async_flow
)


# Pydantic models for demo
class DemoData(BaseModel):
    """Demo data model."""
    id: str
    name: str
    type: str
    value: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FlowDemoRequest(BaseModel):
    """Request model for flow demonstrations."""
    flow_type: FlowType
    data_count: int = Field(10, description="Number of data items to process")
    delay: float = Field(0.1, description="Delay between items")
    concurrent: bool = Field(True, description="Process concurrently")


class PerformanceComparison(BaseModel):
    """Performance comparison result."""
    flow_type: FlowType
    total_time: float
    items_processed: int
    throughput: float
    concurrent: bool


class FlowExample(BaseModel):
    """Flow example configuration."""
    name: str
    description: str
    flow_type: FlowType
    endpoint: str
    benefits: List[str]
    use_cases: List[str]


# Demo FastAPI application
app = FastAPI(
    title="Async Non-Blocking Flows Demo",
    description="Comprehensive demo of asynchronous and non-blocking flow patterns",
    version="1.0.0"
)

# Global orchestrator
orchestrator = AsyncFlowOrchestrator()


@app.on_event("startup")
async def startup_event():
    """Initialize demo flows on startup."""
    # Create data processing pipeline
    pipeline = AsyncDataPipeline("demo_pipeline")
    pipeline.add_processor(DataValidationProcessor())
    pipeline.add_processor(DataEnrichmentProcessor())
    pipeline.add_processor(DataTransformationProcessor())
    
    # Create reactive stream
    stream = ReactiveStream("demo_stream")
    
    # Add stream transformers
    async def uppercase_transformer(data) -> Any:
        if isinstance(data, dict) and "name" in data:
            data["name"] = data["name"].upper()
        return data
    
    async def timestamp_transformer(data) -> Any:
        if isinstance(data, dict):
            data["processed_at"] = time.time()
        return data
    
    stream.add_transformer(uppercase_transformer)
    stream.add_transformer(timestamp_transformer)
    
    # Create message queue
    message_queue = AsyncMessageQueue("demo_queue")
    
    # Add consumers
    async def demo_consumer(message) -> Any:
        await asyncio.sleep(0.05)
        print(f"Demo consumer processed: {message}")
    
    async def analytics_consumer(message) -> Any:
        await asyncio.sleep(0.03)
        print(f"Analytics processed: {message}")
    
    message_queue.add_consumer(demo_consumer)
    message_queue.add_consumer(analytics_consumer)
    
    # Add flows to orchestrator
    orchestrator.add_pipeline("demo_pipeline", pipeline)
    orchestrator.add_stream("demo_stream", stream)
    orchestrator.add_message_queue("demo_queue", message_queue)
    
    # Start orchestrator
    await orchestrator.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup demo flows on shutdown."""
    await orchestrator.stop()


# Demo routes

@app.get("/")
async def root():
    """Root endpoint with demo information."""
    return {
        "message": "Async Non-Blocking Flows Demo",
        "version": "1.0.0",
        "endpoints": [
            "/demo/pipeline",
            "/demo/stream",
            "/demo/queue",
            "/demo/events",
            "/demo/performance",
            "/demo/websocket",
            "/demo/examples",
            "/demo/stats",
            "/demo/health"
        ],
        "flow_types": [
            {
                "type": "event_driven",
                "description": "Event-driven architecture with async event streams"
            },
            {
                "type": "reactive",
                "description": "Reactive programming with data transformation streams"
            },
            {
                "type": "pipeline",
                "description": "Async data processing pipelines with multiple stages"
            },
            {
                "type": "message_queue",
                "description": "Background task processing with message queues"
            }
        ]
    }


@app.post("/demo/pipeline")
async def demo_pipeline(request: FlowDemoRequest):
    """Demo: Data processing pipeline."""
    start_time = time.time()
    results = []
    
    # Generate demo data
    demo_data_items = [
        {
            "id": f"item_{i}",
            "name": f"Product {i}",
            "type": "product",
            "value": random.uniform(10.0, 1000.0),
            "metadata": {"category": random.choice(["A", "B", "C"])}
        }
        for i in range(request.data_count)
    ]
    
    pipeline = orchestrator.pipelines["demo_pipeline"]
    
    if request.concurrent:
        # Process concurrently
        tasks = []
        for data in demo_data_items:
            context = FlowContext(
                flow_id=str(uuid.uuid4()),
                user_id="demo_user",
                session_id="demo_session"
            )
            task = pipeline.feed_data(data, context)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Collect results
        async for result in pipeline.get_output():
            results.append(result)
            if len(results) >= request.data_count:
                break
    else:
        # Process sequentially
        for data in demo_data_items:
            context = FlowContext(
                flow_id=str(uuid.uuid4()),
                user_id="demo_user",
                session_id="demo_session"
            )
            await pipeline.feed_data(data, context)
            
            async for result in pipeline.get_output():
                results.append(result)
                break  # Get one result at a time
            
            if request.delay > 0:
                await asyncio.sleep(request.delay)
    
    total_time = time.time() - start_time
    
    return {
        "flow_type": "pipeline",
        "total_time": total_time,
        "items_processed": len(results),
        "throughput": len(results) / total_time if total_time > 0 else 0,
        "concurrent": request.concurrent,
        "results": results[:5],  # Show first 5 results
        "pipeline_stats": pipeline.stats
    }


@app.post("/demo/stream")
async def demo_stream(request: FlowDemoRequest):
    """Demo: Reactive stream processing."""
    start_time = time.time()
    results = []
    
    # Generate demo data
    demo_data_items = [
        {
            "id": f"stream_{i}",
            "name": f"Stream Item {i}",
            "type": "stream_data",
            "value": random.uniform(1.0, 100.0),
            "metadata": {"source": "demo"}
        }
        for i in range(request.data_count)
    ]
    
    stream = orchestrator.streams["demo_stream"]
    
    # Subscribe to stream output
    output_queue = asyncio.Queue()
    
    async def stream_subscriber(data) -> Any:
        await output_queue.put(data)
    
    stream.subscribe(stream_subscriber)
    
    if request.concurrent:
        # Emit concurrently
        tasks = [stream.emit(data) for data in demo_data_items]
        await asyncio.gather(*tasks)
        
        # Collect results
        for _ in range(request.data_count):
            try:
                result = await asyncio.wait_for(output_queue.get(), timeout=2.0)
                results.append(result)
            except asyncio.TimeoutError:
                break
    else:
        # Emit sequentially
        for data in demo_data_items:
            await stream.emit(data)
            
            try:
                result = await asyncio.wait_for(output_queue.get(), timeout=1.0)
                results.append(result)
            except asyncio.TimeoutError:
                results.append(data)  # Use original data if no transformation
            
            if request.delay > 0:
                await asyncio.sleep(request.delay)
    
    total_time = time.time() - start_time
    
    return {
        "flow_type": "reactive_stream",
        "total_time": total_time,
        "items_processed": len(results),
        "throughput": len(results) / total_time if total_time > 0 else 0,
        "concurrent": request.concurrent,
        "results": results[:5],  # Show first 5 results
        "transformations_applied": ["uppercase", "timestamp"]
    }


@app.post("/demo/queue")
async def demo_queue(request: FlowDemoRequest):
    """Demo: Message queue processing."""
    start_time = time.time()
    
    # Generate demo data
    demo_data_items = [
        {
            "id": f"queue_{i}",
            "name": f"Queue Item {i}",
            "type": "queue_data",
            "value": random.uniform(1.0, 100.0),
            "priority": random.choice(["high", "medium", "low"])
        }
        for i in range(request.data_count)
    ]
    
    queue = orchestrator.message_queues["demo_queue"]
    
    if request.concurrent:
        # Send messages concurrently
        tasks = [queue.send_message(data) for data in demo_data_items]
        await asyncio.gather(*tasks)
    else:
        # Send messages sequentially
        for data in demo_data_items:
            await queue.send_message(data)
            if request.delay > 0:
                await asyncio.sleep(request.delay)
    
    total_time = time.time() - start_time
    
    return {
        "flow_type": "message_queue",
        "total_time": total_time,
        "messages_sent": len(demo_data_items),
        "throughput": len(demo_data_items) / total_time if total_time > 0 else 0,
        "concurrent": request.concurrent,
        "queue_stats": queue.stats,
        "consumers": len(queue.consumers)
    }


@app.get("/demo/stream-data")
async def stream_data_demo():
    """Demo: Real-time data streaming."""
    async def generate_stream():
        
    """generate_stream function."""
stream = orchestrator.streams["demo_stream"]
        
        # Subscribe to stream output
        output_queue = asyncio.Queue()
        
        async def stream_subscriber(data) -> Any:
            await output_queue.put(data)
        
        stream.subscribe(stream_subscriber)
        
        # Generate and emit data
        for i in range(20):
            data = {
                "id": f"stream_{i}",
                "name": f"Real-time Item {i}",
                "type": "stream_data",
                "value": random.uniform(1.0, 100.0),
                "timestamp": time.time()
            }
            
            await stream.emit(data)
            
            # Get processed data
            try:
                processed_data = await asyncio.wait_for(output_queue.get(), timeout=1.0)
                yield f"data: {json.dumps(processed_data)}\n\n"
            except asyncio.TimeoutError:
                yield f"data: {json.dumps(data)}\n\n"
            
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@app.post("/demo/events")
async def demo_events():
    """Demo: Event-driven architecture."""
    events_published = []
    
    # Publish various events
    event_types = [
        (EventType.DATA_PROCESSED, {"processed_items": 100, "processing_time": 2.5}),
        (EventType.USER_ACTION, {"user_id": "demo_user", "action": "data_export"}),
        (EventType.SYSTEM_UPDATE, {"component": "pipeline", "status": "healthy"}),
        (EventType.CACHE_UPDATED, {"cache_key": "products", "items_updated": 50}),
        (EventType.TASK_COMPLETED, {"task_id": "task_123", "result": "success"})
    ]
    
    for event_type, data in event_types:
        event = AsyncEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            data=data,
            source="demo",
            correlation_id="demo_correlation"
        )
        
        await orchestrator.publish_event(event)
        events_published.append({
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "data": event.data,
            "timestamp": event.timestamp
        })
        
        await asyncio.sleep(0.1)  # Small delay between events
    
    return {
        "events_published": len(events_published),
        "events": events_published,
        "event_bus_stats": {
            "subscribers": {et.value: len(orchestrator.event_bus.subscribers[et]) for et in EventType},
            "events_processed": len(orchestrator.event_bus.event_history)
        }
    }


@app.post("/demo/performance")
async def demo_performance_comparison(request: FlowDemoRequest):
    """Demo: Performance comparison between flow types."""
    results = []
    
    # Test pipeline performance
    pipeline_result = await demo_pipeline(request)
    results.append(PerformanceComparison(
        flow_type=FlowType.PIPELINE,
        total_time=pipeline_result["total_time"],
        items_processed=pipeline_result["items_processed"],
        throughput=pipeline_result["throughput"],
        concurrent=request.concurrent
    ))
    
    # Test stream performance
    stream_result = await demo_stream(request)
    results.append(PerformanceComparison(
        flow_type=FlowType.REACTIVE,
        total_time=stream_result["total_time"],
        items_processed=stream_result["items_processed"],
        throughput=stream_result["throughput"],
        concurrent=request.concurrent
    ))
    
    # Test queue performance
    queue_result = await demo_queue(request)
    results.append(PerformanceComparison(
        flow_type=FlowType.MESSAGE_QUEUE,
        total_time=queue_result["total_time"],
        items_processed=queue_result["messages_sent"],
        throughput=queue_result["throughput"],
        concurrent=request.concurrent
    ))
    
    # Calculate performance rankings
    sorted_results = sorted(results, key=lambda x: x.throughput, reverse=True)
    
    return {
        "test_configuration": request.dict(),
        "results": [result.dict() for result in results],
        "performance_ranking": [
            {
                "rank": i + 1,
                "flow_type": result.flow_type.value,
                "throughput": result.throughput,
                "total_time": result.total_time
            }
            for i, result in enumerate(sorted_results)
        ],
        "best_performer": sorted_results[0].flow_type.value if sorted_results else None
    }


@app.websocket("/demo/websocket")
async def websocket_demo(websocket: WebSocket):
    """Demo: WebSocket for real-time flow updates."""
    await websocket.accept()
    
    # Subscribe to events
    async def event_handler(event: AsyncEvent):
        
    """event_handler function."""
await websocket.send_text(json.dumps({
            "type": "event",
            "event_type": event.event_type.value,
            "data": event.data,
            "timestamp": event.timestamp
        }))
    
    # Subscribe to all event types
    for event_type in EventType:
        orchestrator.subscribe_to_event(event_type, event_handler)
    
    try:
        while True:
            # Receive commands from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("command") == "publish_event":
                # Publish event
                event = AsyncEvent(
                    event_id=str(uuid.uuid4()),
                    event_type=EventType(message["event_type"]),
                    timestamp=time.time(),
                    data=message["data"],
                    source="websocket"
                )
                
                await orchestrator.publish_event(event)
                
                await websocket.send_text(json.dumps({
                    "type": "event_published",
                    "event_id": event.event_id
                }))
            
            elif message.get("command") == "emit_stream":
                # Emit to stream
                stream = orchestrator.streams["demo_stream"]
                await stream.emit(message["data"])
                
                await websocket.send_text(json.dumps({
                    "type": "stream_emitted",
                    "data": message["data"]
                }))
            
            elif message.get("command") == "send_queue":
                # Send to queue
                queue = orchestrator.message_queues["demo_queue"]
                await queue.send_message(message["data"])
                
                await websocket.send_text(json.dumps({
                    "type": "message_queued",
                    "data": message["data"]
                }))
    
    except WebSocketDisconnect:
        # Unsubscribe from events
        for event_type in EventType:
            orchestrator.event_bus.unsubscribe(event_type, event_handler)


@app.get("/demo/examples")
async def get_flow_examples():
    """Get examples of different flow types."""
    examples = [
        FlowExample(
            name="Data Processing Pipeline",
            description="Multi-stage data processing with validation, enrichment, and transformation",
            flow_type=FlowType.PIPELINE,
            endpoint="/demo/pipeline",
            benefits=[
                "Sequential processing with error handling",
                "Data validation and enrichment",
                "Modular processor architecture",
                "Pipeline statistics and monitoring"
            ],
            use_cases=[
                "ETL (Extract, Transform, Load) processes",
                "Data validation and cleaning",
                "Multi-step data processing",
                "Batch data operations"
            ]
        ),
        FlowExample(
            name="Reactive Stream",
            description="Real-time data transformation with reactive programming patterns",
            flow_type=FlowType.REACTIVE,
            endpoint="/demo/stream",
            benefits=[
                "Real-time data transformation",
                "Reactive programming patterns",
                "Multiple subscribers support",
                "Non-blocking data flow"
            ],
            use_cases=[
                "Real-time data processing",
                "Live data transformation",
                "Event streaming",
                "Reactive UI updates"
            ]
        ),
        FlowExample(
            name="Message Queue",
            description="Background task processing with asynchronous message queues",
            flow_type=FlowType.MESSAGE_QUEUE,
            endpoint="/demo/queue",
            benefits=[
                "Background task processing",
                "Multiple consumer support",
                "Task distribution and load balancing",
                "Reliable message delivery"
            ],
            use_cases=[
                "Background job processing",
                "Task distribution",
                "Asynchronous notifications",
                "Batch processing"
            ]
        ),
        FlowExample(
            name="Event-Driven Architecture",
            description="Event-driven system with async event bus and subscribers",
            flow_type=FlowType.EVENT_DRIVEN,
            endpoint="/demo/events",
            benefits=[
                "Loose coupling between components",
                "Event-driven architecture",
                "Multiple event subscribers",
                "Asynchronous event processing"
            ],
            use_cases=[
                "Microservices communication",
                "Event sourcing",
                "System monitoring",
                "Audit logging"
            ]
        )
    ]
    
    return {
        "examples": [example.dict() for example in examples],
        "total_examples": len(examples)
    }


@app.get("/demo/stats")
async def get_demo_stats():
    """Get comprehensive statistics for all flows."""
    return orchestrator.get_flow_stats()


@app.get("/demo/health")
async def get_demo_health():
    """Get health status of all demo flows."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "flows": {
            "pipelines": len(orchestrator.pipelines),
            "streams": len(orchestrator.streams),
            "message_queues": len(orchestrator.message_queues)
        },
        "event_bus": {
            "running": orchestrator.event_bus._running,
            "subscribers": sum(len(subscribers) for subscribers in orchestrator.event_bus.subscribers.values()),
            "events_processed": len(orchestrator.event_bus.event_history)
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
    }
    
    return health_status


# Demo functions

async def demonstrate_event_driven_architecture():
    """Demonstrate event-driven architecture."""
    print("\n=== Event-Driven Architecture Demo ===")
    
    # Create event handlers
    async def data_processed_handler(event: AsyncEvent):
        
    """data_processed_handler function."""
print(f"   📊 Data processed: {event.data}")
    
    async def user_action_handler(event: AsyncEvent):
        
    """user_action_handler function."""
print(f"   👤 User action: {event.data}")
    
    async def system_update_handler(event: AsyncEvent):
        
    """system_update_handler function."""
print(f"   🔧 System update: {event.data}")
    
    # Subscribe to events
    orchestrator.subscribe_to_event(EventType.DATA_PROCESSED, data_processed_handler)
    orchestrator.subscribe_to_event(EventType.USER_ACTION, user_action_handler)
    orchestrator.subscribe_to_event(EventType.SYSTEM_UPDATE, system_update_handler)
    
    # Publish events
    events = [
        (EventType.DATA_PROCESSED, {"items": 100, "time": 2.5}),
        (EventType.USER_ACTION, {"user": "demo", "action": "export"}),
        (EventType.SYSTEM_UPDATE, {"component": "pipeline", "status": "healthy"})
    ]
    
    for event_type, data in events:
        event = AsyncEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            data=data,
            source="demo"
        )
        await orchestrator.publish_event(event)
        await asyncio.sleep(0.1)
    
    print("✅ Event-driven architecture demo completed!")


async def demonstrate_reactive_streams():
    """Demonstrate reactive stream processing."""
    print("\n=== Reactive Streams Demo ===")
    
    stream = orchestrator.streams["demo_stream"]
    
    # Subscribe to stream output
    output_queue = asyncio.Queue()
    
    async def stream_subscriber(data) -> Any:
        await output_queue.put(data)
        print(f"   🔄 Stream processed: {data.get('name', 'Unknown')}")
    
    stream.subscribe(stream_subscriber)
    
    # Emit data to stream
    for i in range(5):
        data = {
            "id": f"item_{i}",
            "name": f"Stream Item {i}",
            "type": "demo",
            "value": random.uniform(1.0, 100.0)
        }
        
        await stream.emit(data)
        await asyncio.sleep(0.2)
    
    # Collect results
    results = []
    for _ in range(5):
        try:
            result = await asyncio.wait_for(output_queue.get(), timeout=1.0)
            results.append(result)
        except asyncio.TimeoutError:
            break
    
    print(f"   📈 Processed {len(results)} items through reactive stream")
    print("✅ Reactive streams demo completed!")


async def demonstrate_data_pipeline():
    """Demonstrate data processing pipeline."""
    print("\n=== Data Processing Pipeline Demo ===")
    
    pipeline = orchestrator.pipelines["demo_pipeline"]
    
    # Create test data
    test_data = {
        "id": "pipeline_test",
        "name": "Pipeline Test Item",
        "type": "test",
        "value": 42.0,
        "metadata": {"test": True}
    }
    
    context = FlowContext(
        flow_id=str(uuid.uuid4()),
        user_id="demo_user",
        session_id="demo_session"
    )
    
    # Process through pipeline
    await pipeline.feed_data(test_data, context)
    
    # Get result
    async for result in pipeline.get_output():
        print(f"   🔄 Pipeline input: {test_data}")
        print(f"   ✅ Pipeline output: {result}")
        break
    
    print("✅ Data processing pipeline demo completed!")


async def demonstrate_message_queue():
    """Demonstrate message queue processing."""
    print("\n=== Message Queue Demo ===")
    
    queue = orchestrator.message_queues["demo_queue"]
    
    # Send messages
    messages = [
        {"id": f"msg_{i}", "content": f"Message {i}", "priority": "high"}
        for i in range(5)
    ]
    
    for message in messages:
        await queue.send_message(message)
        print(f"   📤 Sent message: {message['id']}")
        await asyncio.sleep(0.1)
    
    # Wait for processing
    await asyncio.sleep(1.0)
    
    print(f"   📊 Queue stats: {queue.stats}")
    print("✅ Message queue demo completed!")


async def demonstrate_performance_comparison():
    """Demonstrate performance comparison between flow types."""
    print("\n=== Performance Comparison Demo ===")
    
    # Test configuration
    test_config = FlowDemoRequest(
        flow_type=FlowType.PIPELINE,  # Will be overridden
        data_count=10,
        delay=0.0,
        concurrent=True
    )
    
    flow_types = [FlowType.PIPELINE, FlowType.REACTIVE, FlowType.MESSAGE_QUEUE]
    results = []
    
    for flow_type in flow_types:
        test_config.flow_type = flow_type
        
        start_time = time.time()
        
        if flow_type == FlowType.PIPELINE:
            result = await demo_pipeline(test_config)
        elif flow_type == FlowType.REACTIVE:
            result = await demo_stream(test_config)
        else:
            result = await demo_queue(test_config)
        
        total_time = time.time() - start_time
        
        results.append({
            "flow_type": flow_type.value,
            "total_time": total_time,
            "throughput": result.get("throughput", 0)
        })
    
    # Sort by throughput
    results.sort(key=lambda x: x["throughput"], reverse=True)
    
    print("   📊 Performance Results:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['flow_type']}: {result['throughput']:.2f} items/sec")
    
    print("✅ Performance comparison demo completed!")


async def run_comprehensive_demo():
    """Run comprehensive async flows demo."""
    print("🚀 Starting Async Non-Blocking Flows Demo")
    print("=" * 60)
    
    try:
        await demonstrate_event_driven_architecture()
        await demonstrate_reactive_streams()
        await demonstrate_data_pipeline()
        await demonstrate_message_queue()
        await demonstrate_performance_comparison()
        
        print("\n" + "=" * 60)
        print("✅ All async flows demos completed successfully!")
        
        print("\n📋 Next Steps:")
        print("1. Run the FastAPI app: uvicorn async_flows_demo:app --reload")
        print("2. Test endpoints: http://localhost:8000/docs")
        print("3. Try pipeline processing: POST /demo/pipeline")
        print("4. Test reactive streams: POST /demo/stream")
        print("5. Check performance: POST /demo/performance")
        print("6. View real-time updates: GET /demo/stream-data")
        print("7. Connect via WebSocket: ws://localhost:8000/demo/websocket")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive demo
    asyncio.run(run_comprehensive_demo()) 