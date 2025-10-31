from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import time
import uuid
import pytest
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from async_non_blocking_flows import (
from typing import Any, List, Dict, Optional
import logging
"""
Test Suite for Async Non-Blocking Flows

This test suite covers:
- Unit tests for all flow components
- Integration tests for flow orchestration
- Performance tests and benchmarks
- Error handling and edge cases
- Async flow patterns and best practices
"""


    AsyncFlowOrchestrator, EventBus, AsyncDataPipeline, ReactiveStream,
    AsyncMessageQueue, AsyncEvent, FlowContext, EventType, FlowType,
    DataValidationProcessor, DataEnrichmentProcessor, DataTransformationProcessor,
    AsyncFlowProcessor, create_async_flow, run_async_flow
)


# Test data fixtures
@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {
        "id": "test_001",
        "name": "Test Product",
        "type": "product",
        "value": 99.99,
        "metadata": {"category": "electronics"}
    }


@pytest.fixture
def sample_context():
    """Sample flow context for testing."""
    return FlowContext(
        flow_id=str(uuid.uuid4()),
        user_id="test_user",
        session_id="test_session",
        request_id=str(uuid.uuid4())
    )


@pytest.fixture
def sample_event():
    """Sample async event for testing."""
    return AsyncEvent(
        event_id=str(uuid.uuid4()),
        event_type=EventType.DATA_PROCESSED,
        timestamp=time.time(),
        data={"processed_items": 10},
        source="test"
    )


# Unit Tests

class TestEventBus:
    """Test cases for EventBus."""
    
    @pytest.mark.asyncio
    async def test_event_bus_creation(self) -> Any:
        """Test EventBus creation."""
        event_bus = EventBus()
        assert event_bus.subscribers == {}
        assert len(event_bus.event_history) == 0
        assert not event_bus._running
    
    @pytest.mark.asyncio
    async def test_event_bus_start_stop(self) -> Any:
        """Test EventBus start and stop."""
        event_bus = EventBus()
        
        await event_bus.start()
        assert event_bus._running
        assert event_bus._processing_task is not None
        
        await event_bus.stop()
        assert not event_bus._running
    
    @pytest.mark.asyncio
    async def test_event_publishing(self) -> Any:
        """Test event publishing."""
        event_bus = EventBus()
        await event_bus.start()
        
        event = AsyncEvent(
            event_id="test_001",
            event_type=EventType.DATA_PROCESSED,
            timestamp=time.time(),
            data={"test": True},
            source="test"
        )
        
        await event_bus.publish(event)
        await asyncio.sleep(0.1)  # Allow processing
        
        assert len(event_bus.event_history) == 1
        assert event_bus.event_history[0].event_id == "test_001"
        
        await event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_event_subscription(self) -> Any:
        """Test event subscription and handling."""
        event_bus = EventBus()
        await event_bus.start()
        
        received_events = []
        
        async def event_handler(event: AsyncEvent):
            
    """event_handler function."""
received_events.append(event)
        
        event_bus.subscribe(EventType.DATA_PROCESSED, event_handler)
        
        event = AsyncEvent(
            event_id="test_002",
            event_type=EventType.DATA_PROCESSED,
            timestamp=time.time(),
            data={"test": True},
            source="test"
        )
        
        await event_bus.publish(event)
        await asyncio.sleep(0.1)  # Allow processing
        
        assert len(received_events) == 1
        assert received_events[0].event_id == "test_002"
        
        await event_bus.stop()
    
    @pytest.mark.asyncio
    async def test_event_unsubscription(self) -> Any:
        """Test event unsubscription."""
        event_bus = EventBus()
        await event_bus.start()
        
        received_events = []
        
        async def event_handler(event: AsyncEvent):
            
    """event_handler function."""
received_events.append(event)
        
        event_bus.subscribe(EventType.DATA_PROCESSED, event_handler)
        event_bus.unsubscribe(EventType.DATA_PROCESSED, event_handler)
        
        event = AsyncEvent(
            event_id="test_003",
            event_type=EventType.DATA_PROCESSED,
            timestamp=time.time(),
            data={"test": True},
            source="test"
        )
        
        await event_bus.publish(event)
        await asyncio.sleep(0.1)  # Allow processing
        
        assert len(received_events) == 0
        
        await event_bus.stop()


class TestAsyncDataPipeline:
    """Test cases for AsyncDataPipeline."""
    
    @pytest.mark.asyncio
    async def test_pipeline_creation(self) -> Any:
        """Test AsyncDataPipeline creation."""
        pipeline = AsyncDataPipeline("test_pipeline")
        assert pipeline.name == "test_pipeline"
        assert len(pipeline.processors) == 0
        assert pipeline.stats["processed"] == 0
        assert pipeline.stats["errors"] == 0
    
    @pytest.mark.asyncio
    async def test_pipeline_processor_addition(self) -> Any:
        """Test adding processors to pipeline."""
        pipeline = AsyncDataPipeline("test_pipeline")
        processor = DataValidationProcessor()
        
        pipeline.add_processor(processor)
        assert len(pipeline.processors) == 1
        assert pipeline.processors[0] == processor
    
    @pytest.mark.asyncio
    async def test_pipeline_start_stop(self) -> Any:
        """Test pipeline start and stop."""
        pipeline = AsyncDataPipeline("test_pipeline")
        
        await pipeline.start()
        assert pipeline._running
        assert pipeline._processing_task is not None
        assert pipeline.stats["start_time"] is not None
        
        await pipeline.stop()
        assert not pipeline._running
        assert pipeline.stats["end_time"] is not None
    
    @pytest.mark.asyncio
    async def test_pipeline_data_processing(self, sample_data, sample_context) -> Any:
        """Test data processing through pipeline."""
        pipeline = AsyncDataPipeline("test_pipeline")
        pipeline.add_processor(DataValidationProcessor())
        pipeline.add_processor(DataEnrichmentProcessor())
        
        await pipeline.start()
        
        # Feed data
        await pipeline.feed_data(sample_data, sample_context)
        
        # Get output
        results = []
        async for result in pipeline.get_output():
            results.append(result)
            break  # Get first result
        
        await pipeline.stop()
        
        assert len(results) == 1
        result = results[0]
        assert result["validated"] is True
        assert result["enriched"] is True
        assert "validation_timestamp" in result
        assert "enrichment_timestamp" in result
    
    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, sample_context) -> Any:
        """Test pipeline error handling."""
        pipeline = AsyncDataPipeline("test_pipeline")
        pipeline.add_processor(DataValidationProcessor())
        
        await pipeline.start()
        
        # Feed invalid data
        invalid_data = {"name": "Invalid"}  # Missing required fields
        
        await pipeline.feed_data(invalid_data, sample_context)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        await pipeline.stop()
        
        # Should have recorded an error
        assert pipeline.stats["errors"] > 0


class TestReactiveStream:
    """Test cases for ReactiveStream."""
    
    @pytest.mark.asyncio
    async def test_stream_creation(self) -> Any:
        """Test ReactiveStream creation."""
        stream = ReactiveStream("test_stream")
        assert stream.name == "test_stream"
        assert len(stream.transformers) == 0
        assert len(stream.subscribers) == 0
    
    @pytest.mark.asyncio
    async def test_stream_transformer_addition(self) -> Any:
        """Test adding transformers to stream."""
        stream = ReactiveStream("test_stream")
        
        async def test_transformer(data) -> Any:
            return data
        
        stream.add_transformer(test_transformer)
        assert len(stream.transformers) == 1
    
    @pytest.mark.asyncio
    async def test_stream_subscription(self) -> Any:
        """Test stream subscription."""
        stream = ReactiveStream("test_stream")
        
        received_data = []
        
        async def test_subscriber(data) -> Any:
            received_data.append(data)
        
        stream.subscribe(test_subscriber)
        assert len(stream.subscribers) == 1
    
    @pytest.mark.asyncio
    async def test_stream_data_processing(self) -> Any:
        """Test data processing through stream."""
        stream = ReactiveStream("test_stream")
        
        # Add transformer
        async def uppercase_transformer(data) -> Any:
            if isinstance(data, dict) and "name" in data:
                data["name"] = data["name"].upper()
            return data
        
        stream.add_transformer(uppercase_transformer)
        
        # Add subscriber
        received_data = []
        
        async def test_subscriber(data) -> Any:
            received_data.append(data)
        
        stream.subscribe(test_subscriber)
        
        await stream.start()
        
        # Emit data
        test_data = {"id": "1", "name": "test item"}
        await stream.emit(test_data)
        
        await asyncio.sleep(0.1)  # Allow processing
        
        await stream.stop()
        
        assert len(received_data) == 1
        assert received_data[0]["name"] == "TEST ITEM"


class TestAsyncMessageQueue:
    """Test cases for AsyncMessageQueue."""
    
    @pytest.mark.asyncio
    async def test_queue_creation(self) -> Any:
        """Test AsyncMessageQueue creation."""
        queue = AsyncMessageQueue("test_queue")
        assert queue.name == "test_queue"
        assert len(queue.consumers) == 0
        assert queue.stats["messages_sent"] == 0
        assert queue.stats["messages_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_queue_consumer_addition(self) -> Any:
        """Test adding consumers to queue."""
        queue = AsyncMessageQueue("test_queue")
        
        async def test_consumer(message) -> Any:
            pass
        
        queue.add_consumer(test_consumer)
        assert len(queue.consumers) == 1
    
    @pytest.mark.asyncio
    async def test_queue_message_processing(self) -> Any:
        """Test message processing through queue."""
        queue = AsyncMessageQueue("test_queue")
        
        processed_messages = []
        
        async def test_consumer(message) -> Any:
            processed_messages.append(message)
        
        queue.add_consumer(test_consumer)
        
        await queue.start()
        
        # Send messages
        test_messages = [
            {"id": "1", "content": "message 1"},
            {"id": "2", "content": "message 2"}
        ]
        
        for message in test_messages:
            await queue.send_message(message)
        
        await asyncio.sleep(0.2)  # Allow processing
        
        await queue.stop()
        
        assert len(processed_messages) == 2
        assert queue.stats["messages_sent"] == 2
        assert queue.stats["messages_processed"] == 2


class TestAsyncFlowOrchestrator:
    """Test cases for AsyncFlowOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_creation(self) -> Any:
        """Test AsyncFlowOrchestrator creation."""
        orchestrator = AsyncFlowOrchestrator()
        assert len(orchestrator.flows) == 0
        assert len(orchestrator.pipelines) == 0
        assert len(orchestrator.streams) == 0
        assert len(orchestrator.message_queues) == 0
    
    @pytest.mark.asyncio
    async def test_orchestrator_flow_management(self) -> Any:
        """Test adding flows to orchestrator."""
        orchestrator = AsyncFlowOrchestrator()
        
        # Add pipeline
        pipeline = AsyncDataPipeline("test_pipeline")
        orchestrator.add_pipeline("test_pipeline", pipeline)
        assert "test_pipeline" in orchestrator.pipelines
        assert "test_pipeline" in orchestrator.flows
        
        # Add stream
        stream = ReactiveStream("test_stream")
        orchestrator.add_stream("test_stream", stream)
        assert "test_stream" in orchestrator.streams
        assert "test_stream" in orchestrator.flows
        
        # Add queue
        queue = AsyncMessageQueue("test_queue")
        orchestrator.add_message_queue("test_queue", queue)
        assert "test_queue" in orchestrator.message_queues
        assert "test_queue" in orchestrator.flows
    
    @pytest.mark.asyncio
    async def test_orchestrator_start_stop(self) -> Any:
        """Test orchestrator start and stop."""
        orchestrator = AsyncFlowOrchestrator()
        
        # Add flows
        pipeline = AsyncDataPipeline("test_pipeline")
        stream = ReactiveStream("test_stream")
        queue = AsyncMessageQueue("test_queue")
        
        orchestrator.add_pipeline("test_pipeline", pipeline)
        orchestrator.add_stream("test_stream", stream)
        orchestrator.add_message_queue("test_queue", queue)
        
        await orchestrator.start()
        assert orchestrator.event_bus._running
        assert pipeline._running
        assert stream._running
        assert queue._running
        
        await orchestrator.stop()
        assert not orchestrator.event_bus._running
        assert not pipeline._running
        assert not stream._running
        assert not queue._running
    
    @pytest.mark.asyncio
    async def test_orchestrator_event_publishing(self, sample_event) -> Any:
        """Test event publishing through orchestrator."""
        orchestrator = AsyncFlowOrchestrator()
        await orchestrator.start()
        
        await orchestrator.publish_event(sample_event)
        await asyncio.sleep(0.1)  # Allow processing
        
        assert len(orchestrator.event_bus.event_history) == 1
        
        await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_orchestrator_stats(self) -> Any:
        """Test orchestrator statistics."""
        orchestrator = AsyncFlowOrchestrator()
        
        # Add flows
        pipeline = AsyncDataPipeline("test_pipeline")
        stream = ReactiveStream("test_stream")
        queue = AsyncMessageQueue("test_queue")
        
        orchestrator.add_pipeline("test_pipeline", pipeline)
        orchestrator.add_stream("test_stream", stream)
        orchestrator.add_message_queue("test_queue", queue)
        
        stats = orchestrator.get_flow_stats()
        
        assert "pipelines" in stats
        assert "streams" in stats
        assert "message_queues" in stats
        assert "event_bus" in stats


class TestDataProcessors:
    """Test cases for data processors."""
    
    @pytest.mark.asyncio
    async def test_validation_processor(self, sample_data, sample_context) -> Any:
        """Test DataValidationProcessor."""
        processor = DataValidationProcessor()
        
        # Test can_process
        assert await processor.can_process(sample_data) is True
        assert await processor.can_process("invalid") is False
        
        # Test processing
        result = await processor.process(sample_data, sample_context)
        assert result["validated"] is True
        assert "validation_timestamp" in result
    
    @pytest.mark.asyncio
    async def test_validation_processor_error(self, sample_context) -> Any:
        """Test DataValidationProcessor error handling."""
        processor = DataValidationProcessor()
        
        invalid_data = {"name": "Invalid"}  # Missing required fields
        
        with pytest.raises(ValueError):
            await processor.process(invalid_data, sample_context)
    
    @pytest.mark.asyncio
    async def test_enrichment_processor(self, sample_data, sample_context) -> Any:
        """Test DataEnrichmentProcessor."""
        processor = DataEnrichmentProcessor()
        
        # Add validation flag
        sample_data["validated"] = True
        
        # Test can_process
        assert await processor.can_process(sample_data) is True
        assert await processor.can_process({"name": "test"}) is False
        
        # Test processing
        result = await processor.process(sample_data, sample_context)
        assert result["enriched"] is True
        assert "enrichment_timestamp" in result
        assert "flow_context" in result
    
    @pytest.mark.asyncio
    async def test_transformation_processor(self, sample_data, sample_context) -> Any:
        """Test DataTransformationProcessor."""
        processor = DataTransformationProcessor()
        
        # Add required flags
        sample_data["validated"] = True
        sample_data["enriched"] = True
        
        # Test can_process
        assert await processor.can_process(sample_data) is True
        assert await processor.can_process({"name": "test"}) is False
        
        # Test processing
        result = await processor.process(sample_data, sample_context)
        assert result["transformed"] is True
        assert result["name"] == "TEST PRODUCT"  # Should be uppercase
        assert "transformation_timestamp" in result
        assert "original_data" in result


# Integration Tests

class TestFlowIntegration:
    """Integration tests for flow components."""
    
    @pytest.mark.asyncio
    async def test_pipeline_with_processors(self, sample_data, sample_context) -> Any:
        """Test complete pipeline with all processors."""
        pipeline = AsyncDataPipeline("integration_pipeline")
        pipeline.add_processor(DataValidationProcessor())
        pipeline.add_processor(DataEnrichmentProcessor())
        pipeline.add_processor(DataTransformationProcessor())
        
        await pipeline.start()
        
        await pipeline.feed_data(sample_data, sample_context)
        
        results = []
        async for result in pipeline.get_output():
            results.append(result)
            break
        
        await pipeline.stop()
        
        assert len(results) == 1
        result = results[0]
        
        # Check all processing stages
        assert result["validated"] is True
        assert result["enriched"] is True
        assert result["transformed"] is True
        assert result["name"] == "TEST PRODUCT"
        assert "validation_timestamp" in result
        assert "enrichment_timestamp" in result
        assert "transformation_timestamp" in result
        assert "flow_context" in result
        assert "original_data" in result
    
    @pytest.mark.asyncio
    async def test_event_driven_pipeline(self, sample_data, sample_context) -> Any:
        """Test event-driven pipeline integration."""
        orchestrator = AsyncFlowOrchestrator()
        
        # Create pipeline
        pipeline = AsyncDataPipeline("event_pipeline")
        pipeline.add_processor(DataValidationProcessor())
        pipeline.add_processor(DataEnrichmentProcessor())
        
        orchestrator.add_pipeline("event_pipeline", pipeline)
        
        # Subscribe to events
        events_received = []
        
        async def event_handler(event: AsyncEvent):
            
    """event_handler function."""
events_received.append(event)
        
        orchestrator.subscribe_to_event(EventType.DATA_PROCESSED, event_handler)
        
        await orchestrator.start()
        
        # Process data
        await pipeline.feed_data(sample_data, sample_context)
        
        # Publish event
        event = AsyncEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.DATA_PROCESSED,
            timestamp=time.time(),
            data={"processed": True},
            source="test"
        )
        
        await orchestrator.publish_event(event)
        
        await asyncio.sleep(0.2)  # Allow processing
        
        await orchestrator.stop()
        
        assert len(events_received) == 1
    
    @pytest.mark.asyncio
    async def test_stream_with_queue(self) -> Any:
        """Test stream integration with message queue."""
        orchestrator = AsyncFlowOrchestrator()
        
        # Create stream
        stream = ReactiveStream("integration_stream")
        
        # Add transformer
        async def test_transformer(data) -> Any:
            data["transformed"] = True
            return data
        
        stream.add_transformer(test_transformer)
        
        # Create queue
        queue = AsyncMessageQueue("integration_queue")
        
        processed_messages = []
        
        async def queue_consumer(message) -> Any:
            processed_messages.append(message)
        
        queue.add_consumer(queue_consumer)
        
        orchestrator.add_stream("integration_stream", stream)
        orchestrator.add_message_queue("integration_queue", queue)
        
        await orchestrator.start()
        
        # Subscribe stream to queue
        async def stream_to_queue(data) -> Any:
            await queue.send_message(data)
        
        stream.subscribe(stream_to_queue)
        
        # Emit data
        test_data = {"id": "1", "name": "test"}
        await stream.emit(test_data)
        
        await asyncio.sleep(0.2)  # Allow processing
        
        await orchestrator.stop()
        
        assert len(processed_messages) == 1
        assert processed_messages[0]["transformed"] is True


# Performance Tests

class TestPerformance:
    """Performance tests for async flows."""
    
    @pytest.mark.asyncio
    async def test_pipeline_performance(self) -> Any:
        """Test pipeline performance with multiple items."""
        pipeline = AsyncDataPipeline("performance_pipeline")
        pipeline.add_processor(DataValidationProcessor())
        pipeline.add_processor(DataEnrichmentProcessor())
        
        await pipeline.start()
        
        # Generate test data
        test_data_items = [
            {
                "id": f"item_{i}",
                "name": f"Product {i}",
                "type": "product",
                "value": i * 10.0
            }
            for i in range(100)
        ]
        
        start_time = time.time()
        
        # Process data concurrently
        tasks = []
        for data in test_data_items:
            context = FlowContext(flow_id=str(uuid.uuid4()))
            task = pipeline.feed_data(data, context)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Collect results
        results = []
        async for result in pipeline.get_output():
            results.append(result)
            if len(results) >= 100:
                break
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        await pipeline.stop()
        
        assert len(results) == 100
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert pipeline.stats["processed"] == 100
    
    @pytest.mark.asyncio
    async def test_stream_performance(self) -> Any:
        """Test stream performance with multiple subscribers."""
        stream = ReactiveStream("performance_stream")
        
        # Add transformer
        async def test_transformer(data) -> Any:
            data["processed"] = True
            return data
        
        stream.add_transformer(test_transformer)
        
        # Add multiple subscribers
        subscriber_results = [[] for _ in range(5)]
        
        for i in range(5):
            async def subscriber(data, index=i) -> Any:
                subscriber_results[index].append(data)
            
            stream.subscribe(subscriber)
        
        await stream.start()
        
        # Emit data
        test_data_items = [
            {"id": f"item_{i}", "name": f"Item {i}"}
            for i in range(50)
        ]
        
        start_time = time.time()
        
        for data in test_data_items:
            await stream.emit(data)
        
        await asyncio.sleep(0.5)  # Allow processing
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        await stream.stop()
        
        # Check all subscribers received data
        for results in subscriber_results:
            assert len(results) == 50
        
        assert processing_time < 3.0  # Should complete within 3 seconds
    
    @pytest.mark.asyncio
    async def test_queue_performance(self) -> Any:
        """Test message queue performance."""
        queue = AsyncMessageQueue("performance_queue")
        
        processed_messages = []
        
        async def consumer(message) -> Any:
            processed_messages.append(message)
            await asyncio.sleep(0.001)  # Simulate processing time
        
        queue.add_consumer(consumer)
        
        await queue.start(num_workers=5)
        
        # Send messages
        test_messages = [
            {"id": f"msg_{i}", "content": f"Message {i}"}
            for i in range(200)
        ]
        
        start_time = time.time()
        
        for message in test_messages:
            await queue.send_message(message)
        
        await asyncio.sleep(1.0)  # Allow processing
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        await queue.stop()
        
        assert len(processed_messages) == 200
        assert processing_time < 10.0  # Should complete within 10 seconds
        assert queue.stats["messages_processed"] == 200


# Error Handling Tests

class TestErrorHandling:
    """Error handling tests for async flows."""
    
    @pytest.mark.asyncio
    async def test_pipeline_processor_error(self, sample_context) -> Any:
        """Test pipeline error handling with processor failures."""
        pipeline = AsyncDataPipeline("error_pipeline")
        
        # Create failing processor
        class FailingProcessor(AsyncFlowProcessor):
            async def can_process(self, data) -> Any:
                return True
            
            async def process(self, data, context) -> Any:
                raise Exception("Processor failed")
        
        pipeline.add_processor(FailingProcessor())
        
        await pipeline.start()
        
        test_data = {"id": "1", "name": "test"}
        await pipeline.feed_data(test_data, sample_context)
        
        await asyncio.sleep(0.1)  # Allow processing
        
        await pipeline.stop()
        
        assert pipeline.stats["errors"] > 0
    
    @pytest.mark.asyncio
    async def test_stream_transformer_error(self) -> Any:
        """Test stream error handling with transformer failures."""
        stream = ReactiveStream("error_stream")
        
        # Create failing transformer
        async def failing_transformer(data) -> Any:
            raise Exception("Transformer failed")
        
        stream.add_transformer(failing_transformer)
        
        received_data = []
        
        async def subscriber(data) -> Any:
            received_data.append(data)
        
        stream.subscribe(subscriber)
        
        await stream.start()
        
        test_data = {"id": "1", "name": "test"}
        await stream.emit(test_data)
        
        await asyncio.sleep(0.1)  # Allow processing
        
        await stream.stop()
        
        # Should not receive data due to transformer error
        assert len(received_data) == 0
    
    @pytest.mark.asyncio
    async def test_queue_consumer_error(self) -> Any:
        """Test message queue error handling with consumer failures."""
        queue = AsyncMessageQueue("error_queue")
        
        # Create failing consumer
        async def failing_consumer(message) -> Any:
            raise Exception("Consumer failed")
        
        queue.add_consumer(failing_consumer)
        
        await queue.start()
        
        test_message = {"id": "1", "content": "test"}
        await queue.send_message(test_message)
        
        await asyncio.sleep(0.1)  # Allow processing
        
        await queue.stop()
        
        assert queue.stats["errors"] > 0


# Utility Function Tests

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    @pytest.mark.asyncio
    async def test_create_async_flow(self) -> Any:
        """Test create_async_flow function."""
        # Test pipeline creation
        pipeline = await create_async_flow(FlowType.PIPELINE, "test_pipeline")
        assert isinstance(pipeline, AsyncDataPipeline)
        assert pipeline.name == "test_pipeline"
        
        # Test stream creation
        stream = await create_async_flow(FlowType.REACTIVE, "test_stream")
        assert isinstance(stream, ReactiveStream)
        assert stream.name == "test_stream"
        
        # Test queue creation
        queue = await create_async_flow(FlowType.MESSAGE_QUEUE, "test_queue")
        assert isinstance(queue, AsyncMessageQueue)
        assert queue.name == "test_queue"
        
        # Test invalid flow type
        with pytest.raises(ValueError):
            await create_async_flow(FlowType.EVENT_DRIVEN, "test")
    
    @pytest.mark.asyncio
    async def test_run_async_flow(self, sample_data, sample_context) -> Any:
        """Test run_async_flow function."""
        # Test with pipeline
        pipeline = AsyncDataPipeline("test_pipeline")
        pipeline.add_processor(DataValidationProcessor())
        
        result = await run_async_flow(pipeline, sample_data, sample_context)
        assert result["validated"] is True
        
        # Test with stream
        stream = ReactiveStream("test_stream")
        result = await run_async_flow(stream, sample_data, sample_context)
        assert result == sample_data
        
        # Test with queue
        queue = AsyncMessageQueue("test_queue")
        result = await run_async_flow(queue, sample_data, sample_context)
        assert result["status"] == "queued"


# Benchmark Tests

class TestBenchmarks:
    """Benchmark tests for performance comparison."""
    
    @pytest.mark.asyncio
    async def test_benchmark_pipeline_vs_stream(self) -> Any:
        """Benchmark pipeline vs stream performance."""
        # Setup pipeline
        pipeline = AsyncDataPipeline("benchmark_pipeline")
        pipeline.add_processor(DataValidationProcessor())
        pipeline.add_processor(DataEnrichmentProcessor())
        
        # Setup stream
        stream = ReactiveStream("benchmark_stream")
        
        async def stream_transformer(data) -> Any:
            data["validated"] = True
            data["enriched"] = True
            return data
        
        stream.add_transformer(stream_transformer)
        
        # Test data
        test_data_items = [
            {
                "id": f"item_{i}",
                "name": f"Product {i}",
                "type": "product"
            }
            for i in range(50)
        ]
        
        # Benchmark pipeline
        await pipeline.start()
        pipeline_start = time.time()
        
        for data in test_data_items:
            context = FlowContext(flow_id=str(uuid.uuid4()))
            await pipeline.feed_data(data, context)
        
        pipeline_results = []
        async for result in pipeline.get_output():
            pipeline_results.append(result)
            if len(pipeline_results) >= 50:
                break
        
        pipeline_time = time.time() - pipeline_start
        await pipeline.stop()
        
        # Benchmark stream
        await stream.start()
        stream_start = time.time()
        
        stream_results = []
        
        async def stream_subscriber(data) -> Any:
            stream_results.append(data)
        
        stream.subscribe(stream_subscriber)
        
        for data in test_data_items:
            await stream.emit(data)
        
        await asyncio.sleep(0.5)  # Allow processing
        stream_time = time.time() - stream_start
        await stream.stop()
        
        # Compare results
        assert len(pipeline_results) == 50
        assert len(stream_results) == 50
        
        print(f"Pipeline time: {pipeline_time:.3f}s")
        print(f"Stream time: {stream_time:.3f}s")
        
        # Both should be reasonably fast
        assert pipeline_time < 10.0
        assert stream_time < 10.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 