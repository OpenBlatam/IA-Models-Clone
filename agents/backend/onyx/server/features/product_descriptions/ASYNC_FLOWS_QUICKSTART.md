# Async Non-Blocking Flows - Quick Start Guide

## 🚀 Getting Started

This guide will help you get up and running with the Async Non-Blocking Flows system in minutes.

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements-async-flows.txt
```

2. **Run the demo**:
```bash
python async_flows_demo.py
```

3. **Start the FastAPI server**:
```bash
uvicorn async_non_blocking_flows:app --reload
```

## Quick Examples

### 1. Basic Event-Driven Flow

```python
from async_non_blocking_flows import EventBus, AsyncEvent, EventType
import asyncio

async def main():
    # Create event bus
    event_bus = EventBus()
    await event_bus.start()
    
    # Subscribe to events
    async def handle_data_processed(event):
        print(f"Data processed: {event.data}")
    
    event_bus.subscribe(EventType.DATA_PROCESSED, handle_data_processed)
    
    # Publish event
    event = AsyncEvent(
        event_id="123",
        event_type=EventType.DATA_PROCESSED,
        timestamp=time.time(),
        data={"items": 100},
        source="demo"
    )
    
    await event_bus.publish(event)
    await asyncio.sleep(0.1)
    await event_bus.stop()

asyncio.run(main())
```

### 2. Data Processing Pipeline

```python
from async_non_blocking_flows import AsyncDataPipeline, FlowContext
from async_non_blocking_flows import DataValidationProcessor, DataEnrichmentProcessor

async def main():
    # Create pipeline
    pipeline = AsyncDataPipeline("my_pipeline")
    pipeline.add_processor(DataValidationProcessor())
    pipeline.add_processor(DataEnrichmentProcessor())
    
    await pipeline.start()
    
    # Process data
    data = {"id": "1", "name": "Product", "type": "item"}
    context = FlowContext(flow_id="flow_123")
    
    await pipeline.feed_data(data, context)
    
    # Get results
    async for result in pipeline.get_output():
        print(f"Processed: {result}")
        break
    
    await pipeline.stop()

asyncio.run(main())
```

### 3. Reactive Stream

```python
from async_non_blocking_flows import ReactiveStream

async def main():
    # Create stream
    stream = ReactiveStream("my_stream")
    
    # Add transformer
    async def uppercase_transformer(data):
        if "name" in data:
            data["name"] = data["name"].upper()
        return data
    
    stream.add_transformer(uppercase_transformer)
    
    # Subscribe to output
    async def subscriber(data):
        print(f"Received: {data}")
    
    stream.subscribe(subscriber)
    
    await stream.start()
    
    # Emit data
    await stream.emit({"name": "test item"})
    await asyncio.sleep(0.1)
    
    await stream.stop()

asyncio.run(main())
```

### 4. Message Queue

```python
from async_non_blocking_flows import AsyncMessageQueue

async def main():
    # Create queue
    queue = AsyncMessageQueue("my_queue")
    
    # Add consumer
    async def consumer(message):
        print(f"Processing: {message}")
    
    queue.add_consumer(consumer)
    
    await queue.start()
    
    # Send messages
    await queue.send_message({"task": "process_data"})
    await queue.send_message({"task": "send_notification"})
    
    await asyncio.sleep(0.2)
    await queue.stop()

asyncio.run(main())
```

### 5. Complete Orchestrator

```python
from async_non_blocking_flows import AsyncFlowOrchestrator, AsyncDataPipeline, ReactiveStream

async def main():
    # Create orchestrator
    orchestrator = AsyncFlowOrchestrator()
    
    # Add flows
    pipeline = AsyncDataPipeline("data_pipeline")
    stream = ReactiveStream("data_stream")
    
    orchestrator.add_pipeline("pipeline", pipeline)
    orchestrator.add_stream("stream", stream)
    
    # Start everything
    await orchestrator.start()
    
    # Use flows
    data = {"id": "1", "name": "test"}
    context = FlowContext(flow_id="test")
    
    await pipeline.feed_data(data, context)
    await stream.emit(data)
    
    await asyncio.sleep(0.2)
    await orchestrator.stop()

asyncio.run(main())
```

## API Endpoints

Once the server is running, you can access these endpoints:

### Process Data
```bash
curl -X POST "http://localhost:8000/flows/process" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {"id": "1", "name": "Product", "type": "item"},
    "flow_type": "pipeline",
    "user_id": "user123"
  }'
```

### Stream Data
```bash
curl "http://localhost:8000/flows/stream"
```

### Publish Event
```bash
curl -X POST "http://localhost:8000/flows/events" \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "data_processed",
    "data": {"processed": true},
    "source": "api"
  }'
```

### Get Stats
```bash
curl "http://localhost:8000/flows/stats"
```

### Health Check
```bash
curl "http://localhost:8000/flows/health"
```

## Demo Endpoints

The demo server provides additional endpoints:

### Pipeline Demo
```bash
curl -X POST "http://localhost:8000/demo/pipeline" \
  -H "Content-Type: application/json" \
  -d '{
    "flow_type": "pipeline",
    "data_count": 10,
    "concurrent": true
  }'
```

### Stream Demo
```bash
curl -X POST "http://localhost:8000/demo/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "flow_type": "reactive",
    "data_count": 10,
    "concurrent": true
  }'
```

### Performance Comparison
```bash
curl -X POST "http://localhost:8000/demo/performance" \
  -H "Content-Type: application/json" \
  -d '{
    "flow_type": "pipeline",
    "data_count": 100,
    "concurrent": true
  }'
```

## WebSocket Connection

Connect to the WebSocket for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/flows/websocket');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Received:', data);
};

// Send commands
ws.send(JSON.stringify({
    command: 'publish_event',
    event_type: 'data_processed',
    data: {processed: true}
}));
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest test_async_flows.py -v

# Run specific test categories
pytest test_async_flows.py::TestEventBus -v
pytest test_async_flows.py::TestAsyncDataPipeline -v
pytest test_async_flows.py::TestReactiveStream -v
pytest test_async_flows.py::TestAsyncMessageQueue -v

# Run performance tests
pytest test_async_flows.py::TestPerformance -v

# Run with coverage
pytest test_async_flows.py --cov=async_non_blocking_flows --cov-report=html
```

## Common Patterns

### 1. Error Handling

```python
async def robust_processor(data, context):
    try:
        result = await process_data(data)
        return result
    except Exception as e:
        # Log error and continue
        print(f"Error processing {data}: {e}")
        return None
```

### 2. Concurrent Processing

```python
# Process multiple items concurrently
tasks = [process_item(item) for item in items]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

### 3. Batch Processing

```python
async def batch_processor(batch):
    # Process batch of items
    return await process_batch_concurrently(batch)
```

### 4. Resource Management

```python
async with AsyncFlowOrchestrator() as orchestrator:
    await orchestrator.start()
    # Use orchestrator
    await orchestrator.stop()
```

## Performance Tips

1. **Use concurrent processing** for multiple items
2. **Batch operations** when possible
3. **Monitor performance** with built-in stats
4. **Handle errors gracefully** to prevent crashes
5. **Use appropriate flow types** for your use case

## Next Steps

1. **Explore the demo**: Run `python async_flows_demo.py`
2. **Read the documentation**: Check `ASYNC_FLOWS_SUMMARY.md`
3. **Run tests**: Execute the test suite
4. **Build your own flows**: Start with simple examples
5. **Monitor performance**: Use the built-in monitoring tools

## Support

- **Documentation**: `ASYNC_FLOWS_SUMMARY.md`
- **Examples**: `async_flows_demo.py`
- **Tests**: `test_async_flows.py`
- **API Docs**: `http://localhost:8000/docs` (when server is running)

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
2. **Async context errors**: Ensure you're in an async context
3. **Performance issues**: Check if you're using concurrent processing
4. **Memory issues**: Monitor queue sizes and processing rates

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Use the built-in stats:

```python
stats = orchestrator.get_flow_stats()
print(f"Pipeline stats: {stats['pipelines']}")
print(f"Queue stats: {stats['message_queues']}")
```

Happy coding! 🚀 