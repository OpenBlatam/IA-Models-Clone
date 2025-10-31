from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional, Union, Callable, AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
import weakref
from fastapi import FastAPI, Request, Response, HTTPException, status, BackgroundTasks, Depends, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, func, text
import httpx
import redis.asyncio as redis
from celery import Celery
import aiofiles
import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text
import httpx
from fastapi import Depends
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from typing import List, Optional, Dict, Any
from typing import Any, List, Dict, Optional
"""
Asynchronous and Non-Blocking Flows Implementation
=================================================

This module demonstrates:
- Advanced async/await patterns and flows
- Event-driven architecture with async event loops
- Reactive programming with async streams
- Non-blocking data processing pipelines
- Async context managers and resource management
- Concurrent task orchestration
- Async generators and iterators
- Event sourcing with async patterns
- CQRS with async command/query separation
- Saga pattern with async compensation
- Async message queues and pub/sub
- Reactive streams and backpressure handling
"""




# ============================================================================
# ASYNC FLOW PATTERNS
# ============================================================================

class FlowState(Enum):
    """Flow state enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AsyncFlow:
    """Represents an asynchronous flow."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    state: FlowState = FlowState.PENDING
    steps: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def add_step(self, step_name: str, step_func: Callable, **kwargs):
        """Add a step to the flow."""
        self.steps.append({
            "name": step_name,
            "func": step_func,
            "kwargs": kwargs,
            "state": FlowState.PENDING,
            "result": None,
            "error": None
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


# ============================================================================
# ASYNC FLOW MANAGER
# ============================================================================

class AsyncFlowManager:
    """Manages asynchronous flows and their execution."""
    
    def __init__(self) -> Any:
        self.flows: Dict[str, AsyncFlow] = {}
        self.running_flows: Dict[str, asyncio.Task] = {}
        self.flow_results: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    async def create_flow(self, name: str) -> AsyncFlow:
        """Create a new async flow."""
        flow = AsyncFlow(name=name)
        self.flows[flow.id] = flow
        self.logger.info(f"Created flow: {flow.id} - {name}")
        return flow
    
    async def execute_flow(self, flow_id: str) -> AsyncFlow:
        """Execute an async flow."""
        if flow_id not in self.flows:
            raise ValueError(f"Flow {flow_id} not found")
        
        flow = self.flows[flow_id]
        if flow.state != FlowState.PENDING:
            raise ValueError(f"Flow {flow_id} is not in pending state")
        
        # Create task for flow execution
        task = asyncio.create_task(self._execute_flow_steps(flow))
        self.running_flows[flow_id] = task
        
        flow.state = FlowState.RUNNING
        flow.started_at = datetime.utcnow()
        
        self.logger.info(f"Started flow execution: {flow_id}")
        return flow
    
    async def _execute_flow_steps(self, flow: AsyncFlow):
        """Execute flow steps asynchronously."""
        try:
            for i, step in enumerate(flow.steps):
                flow.current_step = i
                step["state"] = FlowState.RUNNING
                
                self.logger.info(f"Executing step {i+1}/{len(flow.steps)}: {step['name']}")
                
                try:
                    # Execute step function
                    if asyncio.iscoroutinefunction(step["func"]):
                        result = await step["func"](**step["kwargs"])
                    else:
                        result = step["func"](**step["kwargs"])
                    
                    step["result"] = result
                    step["state"] = FlowState.COMPLETED
                    
                    self.logger.info(f"Step {step['name']} completed successfully")
                    
                except Exception as e:
                    step["error"] = str(e)
                    step["state"] = FlowState.FAILED
                    flow.error = f"Step {step['name']} failed: {str(e)}"
                    flow.state = FlowState.FAILED
                    flow.completed_at = datetime.utcnow()
                    
                    self.logger.error(f"Step {step['name']} failed: {str(e)}")
                    return
            
            # All steps completed successfully
            flow.state = FlowState.COMPLETED
            flow.completed_at = datetime.utcnow()
            flow.result = {"steps_completed": len(flow.steps)}
            
            self.logger.info(f"Flow {flow.id} completed successfully")
            
        except Exception as e:
            flow.error = str(e)
            flow.state = FlowState.FAILED
            flow.completed_at = datetime.utcnow()
            self.logger.error(f"Flow {flow.id} failed: {str(e)}")
        
        finally:
            # Clean up running task
            if flow.id in self.running_flows:
                del self.running_flows[flow.id]
    
    async def get_flow_status(self, flow_id: str) -> Optional[AsyncFlow]:
        """Get flow status."""
        return self.flows.get(flow_id)
    
    async def cancel_flow(self, flow_id: str) -> bool:
        """Cancel a running flow."""
        if flow_id in self.running_flows:
            task = self.running_flows[flow_id]
            task.cancel()
            
            flow = self.flows[flow_id]
            flow.state = FlowState.CANCELLED
            flow.completed_at = datetime.utcnow()
            
            del self.running_flows[flow_id]
            self.logger.info(f"Flow {flow_id} cancelled")
            return True
        
        return False
    
    async def list_flows(self) -> List[AsyncFlow]:
        """List all flows."""
        return list(self.flows.values())


# ============================================================================
# ASYNC EVENT SYSTEM
# ============================================================================

@dataclass
class AsyncEvent:
    """Represents an asynchronous event."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None


class AsyncEventBus:
    """Asynchronous event bus for event-driven architecture."""
    
    def __init__(self) -> Any:
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[AsyncEvent] = []
        self.logger = logging.getLogger(__name__)
    
    async def publish(self, event: AsyncEvent):
        """Publish an event asynchronously."""
        self.event_history.append(event)
        
        # Get subscribers for this event type
        subscribers = self.subscribers.get(event.type, [])
        
        # Publish to all subscribers concurrently
        if subscribers:
            tasks = [subscriber(event) for subscriber in subscribers]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info(f"Published event: {event.type} - {event.id}")
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type."""
        self.subscribers[event_type].append(handler)
        self.logger.info(f"Subscribed to event type: {event_type}")
    
    async def get_event_history(self, event_type: Optional[str] = None) -> List[AsyncEvent]:
        """Get event history."""
        if event_type:
            return [event for event in self.event_history if event.type == event_type]
        return self.event_history


# ============================================================================
# ASYNC STREAM PROCESSING
# ============================================================================

class AsyncStreamProcessor:
    """Processes data streams asynchronously."""
    
    def __init__(self, buffer_size: int = 1000):
        
    """__init__ function."""
self.buffer_size = buffer_size
        self.processors: List[Callable] = []
        self.logger = logging.getLogger(__name__)
    
    def add_processor(self, processor: Callable):
        """Add a processor to the pipeline."""
        self.processors.append(processor)
    
    async def process_stream(self, data_stream: AsyncIterator[Any]) -> AsyncIterator[Any]:
        """Process a data stream asynchronously."""
        buffer = deque(maxlen=self.buffer_size)
        
        async for item in data_stream:
            # Add to buffer
            buffer.append(item)
            
            # Process through pipeline
            processed_item = item
            for processor in self.processors:
                if asyncio.iscoroutinefunction(processor):
                    processed_item = await processor(processed_item)
                else:
                    processed_item = processor(processed_item)
            
            yield processed_item
    
    async def batch_process(self, data_stream: AsyncIterator[Any], batch_size: int = 100) -> AsyncIterator[List[Any]]:
        """Process data in batches."""
        batch = []
        
        async for item in data_stream:
            batch.append(item)
            
            if len(batch) >= batch_size:
                # Process batch
                processed_batch = []
                for processor in self.processors:
                    if asyncio.iscoroutinefunction(processor):
                        processed_batch = await processor(batch)
                    else:
                        processed_batch = processor(batch)
                    batch = processed_batch
                
                yield batch
                batch = []
        
        # Process remaining items
        if batch:
            processed_batch = []
            for processor in self.processors:
                if asyncio.iscoroutinefunction(processor):
                    processed_batch = await processor(batch)
                else:
                    processed_batch = processor(batch)
                batch = processed_batch
            
            yield batch


# ============================================================================
# ASYNC CONTEXT MANAGERS
# ============================================================================

@asynccontextmanager
async def async_resource_manager(resource_name: str):
    """Async context manager for resource management."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Acquiring resource: {resource_name}")
        # Simulate resource acquisition
        await asyncio.sleep(0.1)
        yield resource_name
    finally:
        logger.info(f"Releasing resource: {resource_name}")
        # Simulate resource release
        await asyncio.sleep(0.1)


@asynccontextmanager
async def async_transaction_manager():
    """Async context manager for database transactions."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting transaction")
        # Simulate transaction start
        await asyncio.sleep(0.1)
        yield
        logger.info("Committing transaction")
        # Simulate transaction commit
        await asyncio.sleep(0.1)
    except Exception as e:
        logger.error(f"Rolling back transaction: {str(e)}")
        # Simulate transaction rollback
        await asyncio.sleep(0.1)
        raise


# ============================================================================
# ASYNC GENERATORS
# ============================================================================

async def async_data_generator(start: int, end: int, delay: float = 0.1) -> AsyncGenerator[int, None]:
    """Generate data asynchronously."""
    for i in range(start, end):
        await asyncio.sleep(delay)
        yield i


async def async_file_reader(file_path: str, chunk_size: int = 1024) -> AsyncGenerator[bytes, None]:
    """Read file asynchronously in chunks."""
    async with aiofiles.open(file_path, 'rb') as file:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        while True:
            chunk = await file.read(chunk_size)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            if not chunk:
                break
            yield chunk


async async def async_api_poller(url: str, interval: float = 1.0) -> AsyncGenerator[Dict[str, Any], None]:
    """Poll API endpoint asynchronously."""
    async with httpx.AsyncClient() as client:
        while True:
            try:
                response = await client.get(url)
                yield response.json()
            except Exception as e:
                yield {"error": str(e)}
            
            await asyncio.sleep(interval)


# ============================================================================
# ASYNC COMMAND/QUERY SEPARATION (CQRS)
# ============================================================================

class AsyncCommand:
    """Base class for async commands."""
    
    def __init__(self, command_id: str = None):
        
    """__init__ function."""
self.command_id = command_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow()


class AsyncQuery:
    """Base class for async queries."""
    
    def __init__(self, query_id: str = None):
        
    """__init__ function."""
self.query_id = query_id or str(uuid.uuid4())
        self.timestamp = datetime.utcnow()


class AsyncCommandHandler:
    """Handles async commands."""
    
    def __init__(self) -> Any:
        self.handlers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_handler(self, command_type: str, handler: Callable):
        """Register a command handler."""
        self.handlers[command_type] = handler
    
    async def handle(self, command: AsyncCommand) -> Any:
        """Handle a command asynchronously."""
        command_type = type(command).__name__
        
        if command_type not in self.handlers:
            raise ValueError(f"No handler registered for command type: {command_type}")
        
        handler = self.handlers[command_type]
        
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(command)
            else:
                result = handler(command)
            
            self.logger.info(f"Command {command_type} handled successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Command {command_type} failed: {str(e)}")
            raise


class AsyncQueryHandler:
    """Handles async queries."""
    
    def __init__(self) -> Any:
        self.handlers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_handler(self, query_type: str, handler: Callable):
        """Register a query handler."""
        self.handlers[query_type] = handler
    
    async def handle(self, query: AsyncQuery) -> Any:
        """Handle a query asynchronously."""
        query_type = type(query).__name__
        
        if query_type not in self.handlers:
            raise ValueError(f"No handler registered for query type: {query_type}")
        
        handler = self.handlers[query_type]
        
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(query)
            else:
                result = handler(query)
            
            self.logger.info(f"Query {query_type} handled successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Query {query_type} failed: {str(e)}")
            raise


# ============================================================================
# ASYNC SAGA PATTERN
# ============================================================================

class AsyncSagaStep:
    """Represents a step in an async saga."""
    
    def __init__(self, name: str, action: Callable, compensation: Callable):
        
    """__init__ function."""
self.name = name
        self.action = action
        self.compensation = compensation
        self.completed = False
        self.compensated = False


class AsyncSaga:
    """Implements the saga pattern for distributed transactions."""
    
    def __init__(self, saga_id: str = None):
        
    """__init__ function."""
self.saga_id = saga_id or str(uuid.uuid4())
        self.steps: List[AsyncSagaStep] = []
        self.current_step = 0
        self.completed_steps: List[AsyncSagaStep] = []
        self.logger = logging.getLogger(__name__)
    
    def add_step(self, name: str, action: Callable, compensation: Callable):
        """Add a step to the saga."""
        step = AsyncSagaStep(name, action, compensation)
        self.steps.append(step)
    
    async def execute(self) -> bool:
        """Execute the saga."""
        self.logger.info(f"Starting saga execution: {self.saga_id}")
        
        try:
            for i, step in enumerate(self.steps):
                self.current_step = i
                self.logger.info(f"Executing saga step: {step.name}")
                
                try:
                    # Execute action
                    if asyncio.iscoroutinefunction(step.action):
                        await step.action()
                    else:
                        step.action()
                    
                    step.completed = True
                    self.completed_steps.append(step)
                    self.logger.info(f"Saga step completed: {step.name}")
                    
                except Exception as e:
                    self.logger.error(f"Saga step failed: {step.name} - {str(e)}")
                    await self._compensate()
                    return False
            
            self.logger.info(f"Saga completed successfully: {self.saga_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Saga execution failed: {self.saga_id} - {str(e)}")
            await self._compensate()
            return False
    
    async def _compensate(self) -> Any:
        """Compensate for completed steps."""
        self.logger.info(f"Starting saga compensation: {self.saga_id}")
        
        for step in reversed(self.completed_steps):
            try:
                self.logger.info(f"Compensating saga step: {step.name}")
                
                if asyncio.iscoroutinefunction(step.compensation):
                    await step.compensation()
                else:
                    step.compensation()
                
                step.compensated = True
                self.logger.info(f"Saga step compensated: {step.name}")
                
            except Exception as e:
                self.logger.error(f"Saga compensation failed: {step.name} - {str(e)}")


# ============================================================================
# ASYNC MESSAGE QUEUE
# ============================================================================

@dataclass
class AsyncMessage:
    """Represents an async message."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    priority: int = 0


class AsyncMessageQueue:
    """Asynchronous message queue implementation."""
    
    def __init__(self) -> Any:
        self.queues: Dict[str, deque] = defaultdict(deque)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.processing = False
        self.logger = logging.getLogger(__name__)
    
    async def publish(self, topic: str, payload: Dict[str, Any], priority: int = 0):
        """Publish a message to a topic."""
        message = AsyncMessage(topic=topic, payload=payload, priority=priority)
        self.queues[topic].append(message)
        self.logger.info(f"Published message to topic {topic}: {message.id}")
    
    def subscribe(self, topic: str, handler: Callable):
        """Subscribe to a topic."""
        self.subscribers[topic].append(handler)
        self.logger.info(f"Subscribed to topic: {topic}")
    
    async def start_processing(self) -> Any:
        """Start processing messages."""
        self.processing = True
        self.logger.info("Started message queue processing")
        
        while self.processing:
            for topic, queue in self.queues.items():
                if queue and topic in self.subscribers:
                    message = queue.popleft()
                    handlers = self.subscribers[topic]
                    
                    # Process with all handlers concurrently
                    tasks = [handler(message) for handler in handlers]
                    await asyncio.gather(*tasks, return_exceptions=True)
            
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
    
    async def stop_processing(self) -> Any:
        """Stop processing messages."""
        self.processing = False
        self.logger.info("Stopped message queue processing")


# ============================================================================
# REACTIVE STREAMS
# ============================================================================

class AsyncReactiveStream:
    """Implements reactive streams with backpressure handling."""
    
    def __init__(self, buffer_size: int = 1000):
        
    """__init__ function."""
self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.subscribers: List[Callable] = []
        self.processing = False
        self.logger = logging.getLogger(__name__)
    
    async def emit(self, data: Any):
        """Emit data to the stream."""
        if len(self.buffer) >= self.buffer_size:
            # Backpressure: wait for space
            while len(self.buffer) >= self.buffer_size:
                await asyncio.sleep(0.01)
        
        self.buffer.append(data)
        self.logger.debug(f"Emitted data to stream: {data}")
    
    def subscribe(self, handler: Callable):
        """Subscribe to the stream."""
        self.subscribers.append(handler)
        self.logger.info("Added subscriber to reactive stream")
    
    async def start_processing(self) -> Any:
        """Start processing the stream."""
        self.processing = True
        self.logger.info("Started reactive stream processing")
        
        while self.processing:
            if self.buffer:
                data = self.buffer.popleft()
                
                # Process with all subscribers concurrently
                tasks = [subscriber(data) for subscriber in self.subscribers]
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                await asyncio.sleep(0.01)
    
    async def stop_processing(self) -> Any:
        """Stop processing the stream."""
        self.processing = False
        self.logger.info("Stopped reactive stream processing")


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class FlowCreateRequest(BaseModel):
    """Request to create a flow."""
    
    name: str = Field(..., description="Flow name")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Flow steps")


class FlowResponse(BaseModel):
    """Flow response model."""
    
    id: str
    name: str
    state: str
    current_step: int
    total_steps: int
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    
    model_config = ConfigDict(from_attributes=True)


class EventPublishRequest(BaseModel):
    """Request to publish an event."""
    
    type: str = Field(..., description="Event type")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event data")
    source: str = Field(..., description="Event source")
    correlation_id: Optional[str] = Field(None, description="Correlation ID")


class MessagePublishRequest(BaseModel):
    """Request to publish a message."""
    
    topic: str = Field(..., description="Message topic")
    payload: Dict[str, Any] = Field(..., description="Message payload")
    priority: int = Field(0, ge=0, le=10, description="Message priority")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

def create_app() -> FastAPI:
    """Create FastAPI application with async flows."""
    
    app = FastAPI(
        title="Async Non-Blocking Flows Demo",
        version="1.0.0",
        description="Demonstration of asynchronous and non-blocking flows"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()

# Initialize managers
flow_manager = AsyncFlowManager()
event_bus = AsyncEventBus()
message_queue = AsyncMessageQueue()
reactive_stream = AsyncReactiveStream()

# Start background processing
@app.on_event("startup")
async def startup_event():
    """Start background processing on startup."""
    asyncio.create_task(message_queue.start_processing())
    asyncio.create_task(reactive_stream.start_processing())


@app.on_event("shutdown")
async def shutdown_event():
    """Stop background processing on shutdown."""
    await message_queue.stop_processing()
    await reactive_stream.stop_processing()


# ============================================================================
# ASYNC FLOW API ROUTES
# ============================================================================

@app.post("/flows/", response_model=FlowResponse)
async def create_flow(request: FlowCreateRequest):
    """Create a new async flow."""
    
    flow = await flow_manager.create_flow(request.name)
    
    # Add steps to flow
    for step in request.steps:
        flow.add_step(
            step_name=step["name"],
            step_func=lambda **kwargs: asyncio.sleep(1),  # Mock step function
            **step.get("kwargs", {})
        )
    
    return FlowResponse(**flow.to_dict())


@app.post("/flows/{flow_id}/execute")
async def execute_flow(flow_id: str):
    """Execute an async flow."""
    
    try:
        flow = await flow_manager.execute_flow(flow_id)
        return {"message": f"Flow {flow_id} execution started", "flow": flow.to_dict()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/flows/{flow_id}", response_model=FlowResponse)
async def get_flow_status(flow_id: str):
    """Get flow status."""
    
    flow = await flow_manager.get_flow_status(flow_id)
    if not flow:
        raise HTTPException(status_code=404, detail="Flow not found")
    
    return FlowResponse(**flow.to_dict())


@app.delete("/flows/{flow_id}")
async def cancel_flow(flow_id: str):
    """Cancel a running flow."""
    
    cancelled = await flow_manager.cancel_flow(flow_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Flow not found or not running")
    
    return {"message": f"Flow {flow_id} cancelled"}


@app.get("/flows/")
async def list_flows():
    """List all flows."""
    
    flows = await flow_manager.list_flows()
    return {"flows": [flow.to_dict() for flow in flows]}


# ============================================================================
# ASYNC EVENT API ROUTES
# ============================================================================

@app.post("/events/")
async def publish_event(request: EventPublishRequest):
    """Publish an event."""
    
    event = AsyncEvent(
        type=request.type,
        data=request.data,
        source=request.source,
        correlation_id=request.correlation_id
    )
    
    await event_bus.publish(event)
    return {"message": "Event published", "event_id": event.id}


@app.get("/events/")
async def get_events(event_type: Optional[str] = None):
    """Get event history."""
    
    events = await event_bus.get_event_history(event_type)
    return {"events": [{"id": e.id, "type": e.type, "data": e.data, "timestamp": e.timestamp.isoformat()} for e in events]}


# ============================================================================
# ASYNC MESSAGE QUEUE API ROUTES
# ============================================================================

@app.post("/messages/")
async def publish_message(request: MessagePublishRequest):
    """Publish a message."""
    
    await message_queue.publish(
        topic=request.topic,
        payload=request.payload,
        priority=request.priority
    )
    
    return {"message": "Message published"}


# ============================================================================
# ASYNC STREAMING API ROUTES
# ============================================================================

@app.get("/stream/data/")
async def stream_data() -> StreamingResponse:
    """Stream data asynchronously."""
    
    async def generate_data():
        """Generate streaming data."""
        for i in range(100):
            yield f"data:{i}\n"
            await asyncio.sleep(0.1)
    
    return StreamingResponse(
        generate_data(),
        media_type="text/plain",
        headers={"X-Streaming": "true"}
    )


@app.get("/stream/numbers/")
async def stream_numbers(start: int = 0, end: int = 100) -> StreamingResponse:
    """Stream numbers asynchronously."""
    
    async def generate_numbers():
        """Generate streaming numbers."""
        async for num in async_data_generator(start, end):
            yield f"number:{num}\n"
    
    return StreamingResponse(
        generate_numbers(),
        media_type="text/plain",
        headers={"X-Streaming": "true"}
    )


# ============================================================================
# ASYNC CONTEXT MANAGER API ROUTES
# ============================================================================

@app.get("/resource/")
async def use_resource():
    """Demonstrate async context manager usage."""
    
    async with async_resource_manager("database_connection"):
        # Simulate work with resource
        await asyncio.sleep(0.5)
        return {"message": "Resource used successfully"}


@app.post("/transaction/")
async def perform_transaction():
    """Demonstrate async transaction usage."""
    
    async with async_transaction_manager():
        # Simulate transaction work
        await asyncio.sleep(0.5)
        return {"message": "Transaction completed successfully"}


# ============================================================================
# ASYNC SAGA API ROUTES
# ============================================================================

@app.post("/saga/order/")
async def create_order_saga():
    """Create an order using saga pattern."""
    
    # Define saga steps
    async def reserve_inventory():
        
    """reserve_inventory function."""
await asyncio.sleep(1)
        return {"inventory_reserved": True}
    
    async def process_payment():
        
    """process_payment function."""
await asyncio.sleep(1)
        return {"payment_processed": True}
    
    async def create_order():
        
    """create_order function."""
await asyncio.sleep(1)
        return {"order_created": True}
    
    # Compensation actions
    async def release_inventory():
        
    """release_inventory function."""
await asyncio.sleep(0.5)
        return {"inventory_released": True}
    
    async def refund_payment():
        
    """refund_payment function."""
await asyncio.sleep(0.5)
        return {"payment_refunded": True}
    
    async def cancel_order():
        
    """cancel_order function."""
await asyncio.sleep(0.5)
        return {"order_cancelled": True}
    
    # Create and execute saga
    saga = AsyncSaga()
    saga.add_step("reserve_inventory", reserve_inventory, release_inventory)
    saga.add_step("process_payment", process_payment, refund_payment)
    saga.add_step("create_order", create_order, cancel_order)
    
    success = await saga.execute()
    
    return {
        "saga_id": saga.saga_id,
        "success": success,
        "message": "Order saga completed" if success else "Order saga failed"
    }


# ============================================================================
# REACTIVE STREAM API ROUTES
# ============================================================================

@app.post("/reactive/emit/")
async def emit_to_reactive_stream(data: Dict[str, Any]):
    """Emit data to reactive stream."""
    
    await reactive_stream.emit(data)
    return {"message": "Data emitted to reactive stream"}


@app.get("/reactive/subscribe/")
async def subscribe_to_reactive_stream():
    """Subscribe to reactive stream."""
    
    async def handle_data(data: Any):
        """Handle stream data."""
        print(f"Received data: {data}")
    
    reactive_stream.subscribe(handle_data)
    return {"message": "Subscribed to reactive stream"}


# ============================================================================
# WEBSOCKET FOR REAL-TIME COMMUNICATION
# ============================================================================

@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process message asynchronously
            if message.get("type") == "subscribe":
                # Subscribe to events
                async def event_handler(event: AsyncEvent):
                    
    """event_handler function."""
await websocket.send_text(json.dumps({
                        "type": "event",
                        "data": event.data
                    }))
                
                event_bus.subscribe(message.get("topic", "default"), event_handler)
                
                await websocket.send_text(json.dumps({
                    "type": "subscribed",
                    "topic": message.get("topic", "default")
                }))
            
            elif message.get("type") == "publish":
                # Publish event
                event = AsyncEvent(
                    type=message.get("event_type", "default"),
                    data=message.get("data", {}),
                    source="websocket"
                )
                await event_bus.publish(event)
                
                await websocket.send_text(json.dumps({
                    "type": "published",
                    "event_id": event.id
                }))
            
    except Exception as e:
        logging.getLogger("websocket").error(f"WebSocket error: {str(e)}")


# ============================================================================
# DEDICATED ASYNC FUNCTIONS FOR DATABASE AND EXTERNAL API OPERATIONS
# ============================================================================

# --- Async Database Utilities ---

class AsyncDB:
    """Async database utility with connection pooling."""
    def __init__(self, db_url: str):
        
    """__init__ function."""
self.engine = create_async_engine(db_url, echo=False, future=True)
        self.session_factory = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async async def fetch_users(self) -> list:
        async with self.session_factory() as session:
            result = await session.execute(text("SELECT id, username, email FROM users LIMIT 10"))
            return [dict(row._mapping) for row in result]

    async def add_user(self, username: str, email: str) -> int:
        async with self.session_factory() as session:
            await session.execute(text("INSERT INTO users (username, email) VALUES (:username, :email)"), {"username": username, "email": email})
            await session.commit()
            return 1

# --- Async External API Utilities ---

class AsyncAPIClient:
    """Async HTTP client utility with connection pooling."""
    def __init__(self, base_url: str, timeout: int = 10):
        
    """__init__ function."""
self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def get_resource(self, endpoint: str) -> dict:
        response = await self.client.get(endpoint)
        response.raise_for_status()
        return response.json()

    async def post_resource(self, endpoint: str, data: dict) -> dict:
        response = await self.client.post(endpoint, json=data)
        response.raise_for_status()
        return response.json()

# --- FastAPI Dependency Providers ---

def get_async_db() -> AsyncDB:
    return AsyncDB(db_url="sqlite+aiosqlite:///./test.db")

async def get_api_client() -> AsyncAPIClient:
    return AsyncAPIClient(base_url="https://jsonplaceholder.typicode.com")

# --- Example FastAPI Endpoints Using Dedicated Async Functions ---
@app.get("/dedicated/users/")
async def list_users(db: AsyncDB = Depends(get_async_db)):
    users = await db.fetch_users()
    return {"users": users}

@app.post("/dedicated/users/")
async def create_user(username: str, email: str, db: AsyncDB = Depends(get_async_db)):
    await db.add_user(username, email)
    return {"message": "User created"}

@app.get("/dedicated/external/")
async def get_external_resource(api: AsyncAPIClient = Depends(get_api_client)):
    data = await api.get_resource("/todos/1")
    return {"external_data": data}

# --- Example Usage in Async Flows ---
async def async_flow_with_db_and_api(db: AsyncDB, api: AsyncAPIClient):
    
    """async_flow_with_db_and_api function."""
users = await db.fetch_users()
    external = await api.get_resource("/todos/1")
    return {"users": users, "external": external}


# ============================================================================
# CLEAR ROUTE AND DEPENDENCY STRUCTURE FOR READABILITY AND MAINTAINABILITY
# ============================================================================

"""
Best Practices for Route and Dependency Organization:
1. Separate routes by domain/feature
2. Use APIRouter for modular organization
3. Group dependencies by functionality
4. Implement clear dependency hierarchies
5. Use dependency factories for complex dependencies
6. Separate business logic from route handlers
7. Implement consistent error handling
8. Use type hints and Pydantic models
"""

# --- Modular Route Organization ---

# OAuth2 scheme for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User-related routes
user_router = APIRouter(prefix="/users", tags=["users"])

# Product-related routes  
product_router = APIRouter(prefix="/products", tags=["products"])

# Order-related routes
order_router = APIRouter(prefix="/orders", tags=["orders"])

# Analytics routes
analytics_router = APIRouter(prefix="/analytics", tags=["analytics"])

# --- Dependency Organization by Functionality ---

# Authentication Dependencies
class AuthDependencies:
    """Centralized authentication dependencies."""
    
    @staticmethod
    async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
        """Get current authenticated user."""
        # Implementation here
        return {"user_id": 1, "username": "test_user", "is_active": True, "role": "admin"}
    
    @staticmethod
    async def get_current_active_user(
        current_user: Dict[str, Any] = Depends(get_current_user)
    ) -> Dict[str, Any]:
        """Get current active user."""
        if not current_user.get("is_active", True):
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user
    
    @staticmethod
    async def require_admin_role(
        current_user: Dict[str, Any] = Depends(get_current_active_user)
    ) -> Dict[str, Any]:
        """Require admin role for access."""
        if current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")
        return current_user

# Helper functions for dependencies
async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Get current authenticated user."""
    return await AuthDependencies.get_current_user(token)

async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current active user."""
    return await AuthDependencies.get_current_active_user(current_user)

async def require_admin_role(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Require admin role for access."""
    return await AuthDependencies.require_admin_role(current_user)

# Database Dependencies
class DatabaseDependencies:
    """Centralized database dependencies."""
    
    @staticmethod
    async def get_user_db() -> AsyncDB:
        """Get user database connection."""
        return AsyncDB("sqlite+aiosqlite:///./users.db")
    
    @staticmethod
    async def get_product_db() -> AsyncDB:
        """Get product database connection."""
        return AsyncDB("sqlite+aiosqlite:///./products.db")
    
    @staticmethod
    async def get_order_db() -> AsyncDB:
        """Get order database connection."""
        return AsyncDB("sqlite+aiosqlite:///./orders.db")

# External API Dependencies
class ExternalAPIDependencies:
    """Centralized external API dependencies."""
    
    @staticmethod
    async async def get_payment_api() -> AsyncAPIClient:
        """Get payment API client."""
        return AsyncAPIClient("https://api.payments.com")
    
    @staticmethod
    async async def get_notification_api() -> AsyncAPIClient:
        """Get notification API client."""
        return AsyncAPIClient("https://api.notifications.com")
    
    @staticmethod
    async async def get_analytics_api() -> AsyncAPIClient:
        """Get analytics API client."""
        return AsyncAPIClient("https://api.analytics.com")

# Cache Dependencies
class CacheDependencies:
    """Centralized cache dependencies."""
    
    @staticmethod
    async def get_user_cache() -> redis.Redis:
        """Get user cache connection."""
        return redis.Redis(host="localhost", port=6379, db=0)
    
    @staticmethod
    async def get_product_cache() -> redis.Redis:
        """Get product cache connection."""
        return redis.Redis(host="localhost", port=6379, db=1)

# --- Pydantic Models for Request/Response ---

class UserCreate(BaseModel):
    """User creation model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=8)

class UserResponse(BaseModel):
    """User response model."""
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class ProductCreate(BaseModel):
    """Product creation model."""
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    description: Optional[str] = None

class ProductResponse(BaseModel):
    """Product response model."""
    id: int
    name: str
    price: float
    description: Optional[str]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

class OrderCreate(BaseModel):
    """Order creation model."""
    user_id: int
    product_ids: List[int] = Field(..., min_items=1)
    shipping_address: str

class OrderResponse(BaseModel):
    """Order response model."""
    id: int
    user_id: int
    products: List[ProductResponse]
    total_amount: float
    status: str
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# --- Service Layer (Business Logic) ---

class UserService:
    """User business logic service."""
    
    def __init__(
        self,
        db: AsyncDB,
        cache: redis.Redis,
        notification_api: AsyncAPIClient
    ):
        
    """__init__ function."""
self.db = db
        self.cache = cache
        self.notification_api = notification_api
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user with business logic."""
        # Check if user exists
        existing_user = await self.db.fetch_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(status_code=409, detail="User already exists")
        
        # Create user
        user_id = await self.db.add_user(user_data.username, user_data.email)
        
        # Send welcome notification
        await self.notification_api.post_resource("/notifications", {
            "user_id": user_id,
            "type": "welcome",
            "message": f"Welcome {user_data.username}!"
        })
        
        # Cache user data
        await self.cache.setex(f"user:{user_id}", 3600, user_data.model_dump_json())
        
        return UserResponse(
            id=user_id,
            username=user_data.username,
            email=user_data.email,
            is_active=True,
            created_at=datetime.utcnow()
        )
    
    async def get_user(self, user_id: int) -> Optional[UserResponse]:
        """Get user by ID with caching."""
        # Check cache first
        cached_user = await self.cache.get(f"user:{user_id}")
        if cached_user:
            return UserResponse.model_validate_json(cached_user)
        
        # Fetch from database
        user = await self.db.fetch_user_by_id(user_id)
        if user:
            user_response = UserResponse(**user)
            await self.cache.setex(f"user:{user_id}", 3600, user_response.model_dump_json())
            return user_response
        
        return None

class ProductService:
    """Product business logic service."""
    
    def __init__(self, db: AsyncDB, cache: redis.Redis):
        
    """__init__ function."""
self.db = db
        self.cache = cache
    
    async def create_product(self, product_data: ProductCreate) -> ProductResponse:
        """Create a new product."""
        product_id = await self.db.add_product(
            product_data.name,
            product_data.price,
            product_data.description
        )
        
        return ProductResponse(
            id=product_id,
            name=product_data.name,
            price=product_data.price,
            description=product_data.description,
            created_at=datetime.utcnow()
        )
    
    async def get_product(self, product_id: int) -> Optional[ProductResponse]:
        """Get product by ID."""
        product = await self.db.fetch_product_by_id(product_id)
        return ProductResponse(**product) if product else None

class OrderService:
    """Order business logic service."""
    
    def __init__(
        self,
        db: AsyncDB,
        payment_api: AsyncAPIClient,
        notification_api: AsyncAPIClient
    ):
        
    """__init__ function."""
self.db = db
        self.payment_api = payment_api
        self.notification_api = notification_api
    
    async def create_order(self, order_data: OrderCreate) -> OrderResponse:
        """Create a new order with payment processing."""
        # Validate products exist
        products = []
        total_amount = 0.0
        
        for product_id in order_data.product_ids:
            product = await self.db.fetch_product_by_id(product_id)
            if not product:
                raise HTTPException(
                    status_code=404,
                    detail=f"Product {product_id} not found"
                )
            products.append(product)
            total_amount += product["price"]
        
        # Process payment
        payment_result = await self.payment_api.post_resource("/payments", {
            "amount": total_amount,
            "user_id": order_data.user_id
        })
        
        if payment_result.get("status") != "success":
            raise HTTPException(
                status_code=400,
                detail="Payment processing failed"
            )
        
        # Create order
        order_id = await self.db.add_order(
            order_data.user_id,
            order_data.product_ids,
            total_amount,
            order_data.shipping_address
        )
        
        # Send order confirmation
        await self.notification_api.post_resource("/notifications", {
            "user_id": order_data.user_id,
            "type": "order_confirmation",
            "order_id": order_id
        })
        
        return OrderResponse(
            id=order_id,
            user_id=order_data.user_id,
            products=[ProductResponse(**p) for p in products],
            total_amount=total_amount,
            status="confirmed",
            created_at=datetime.utcnow()
        )

# --- Dependency Factories ---

def create_user_service(
    db: AsyncDB = Depends(DatabaseDependencies.get_user_db),
    cache: redis.Redis = Depends(CacheDependencies.get_user_cache),
    notification_api: AsyncAPIClient = Depends(ExternalAPIDependencies.get_notification_api)
) -> UserService:
    """Factory for creating UserService with dependencies."""
    return UserService(db, cache, notification_api)

def create_product_service(
    db: AsyncDB = Depends(DatabaseDependencies.get_product_db),
    cache: redis.Redis = Depends(CacheDependencies.get_product_cache)
) -> ProductService:
    """Factory for creating ProductService with dependencies."""
    return ProductService(db, cache)

def create_order_service(
    db: AsyncDB = Depends(DatabaseDependencies.get_order_db),
    payment_api: AsyncAPIClient = Depends(ExternalAPIDependencies.get_payment_api),
    notification_api: AsyncAPIClient = Depends(ExternalAPIDependencies.get_notification_api)
) -> OrderService:
    """Factory for creating OrderService with dependencies."""
    return OrderService(db, payment_api, notification_api)

# --- Route Handlers with Clear Structure ---

# User Routes
@user_router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(create_user_service),
    current_user: Dict[str, Any] = Depends(require_admin_role)
):
    """Create a new user (admin only)."""
    return await user_service.create_user(user_data)

@user_router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    user_service: UserService = Depends(create_user_service),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Get user by ID."""
    user = await user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@user_router.get("/", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    user_service: UserService = Depends(create_user_service),
    current_user: Dict[str, Any] = Depends(require_admin_role)
):
    """List all users (admin only)."""
    users = await user_service.get_users(skip=skip, limit=limit)
    return users

# Product Routes
@product_router.post("/", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(
    product_data: ProductCreate,
    product_service: ProductService = Depends(create_product_service),
    current_user: Dict[str, Any] = Depends(require_admin_role)
):
    """Create a new product (admin only)."""
    return await product_service.create_product(product_data)

@product_router.get("/{product_id}", response_model=ProductResponse)
async def get_product(
    product_id: int,
    product_service: ProductService = Depends(create_product_service)
):
    """Get product by ID (public access)."""
    product = await product_service.get_product(product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product

@product_router.get("/", response_model=List[ProductResponse])
async def list_products(
    skip: int = 0,
    limit: int = 100,
    product_service: ProductService = Depends(create_product_service)
):
    """List all products (public access)."""
    products = await product_service.get_products(skip=skip, limit=limit)
    return products

# Order Routes
@order_router.post("/", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def create_order(
    order_data: OrderCreate,
    order_service: OrderService = Depends(create_order_service),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Create a new order."""
    return await order_service.create_order(order_data)

@order_router.get("/{order_id}", response_model=OrderResponse)
async def get_order(
    order_id: int,
    order_service: OrderService = Depends(create_order_service),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Get order by ID."""
    order = await order_service.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order

@order_router.get("/", response_model=List[OrderResponse])
async def list_user_orders(
    user_id: int,
    skip: int = 0,
    limit: int = 100,
    order_service: OrderService = Depends(create_order_service),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """List orders for a specific user."""
    orders = await order_service.get_user_orders(user_id, skip=skip, limit=limit)
    return orders

# Analytics Routes
@analytics_router.get("/dashboard")
async def get_dashboard_analytics(
    analytics_api: AsyncAPIClient = Depends(ExternalAPIDependencies.get_analytics_api),
    current_user: Dict[str, Any] = Depends(require_admin_role)
):
    """Get dashboard analytics (admin only)."""
    analytics = await analytics_api.get_resource("/dashboard")
    return analytics

@analytics_router.get("/user/{user_id}")
async def get_user_analytics(
    user_id: int,
    analytics_api: AsyncAPIClient = Depends(ExternalAPIDependencies.get_analytics_api),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Get user-specific analytics."""
    analytics = await analytics_api.get_resource(f"/user/{user_id}")
    return analytics

# --- Main Application with Organized Routes ---

def create_organized_app() -> FastAPI:
    """Create FastAPI app with organized routes and dependencies."""
    app = FastAPI(
        title="Organized FastAPI Application",
        description="Demonstrates clear route and dependency structure",
        version="1.0.0"
    )
    
    # Include routers in logical order
    app.include_router(user_router, prefix="/api/v1")
    app.include_router(product_router, prefix="/api/v1")
    app.include_router(order_router, prefix="/api/v1")
    app.include_router(analytics_router, prefix="/api/v1")
    
    return app

# --- Example Usage in Async Flows ---

async def complex_business_flow(
    user_service: UserService,
    product_service: ProductService,
    order_service: OrderService,
    user_data: UserCreate,
    product_data: ProductCreate,
    order_data: OrderCreate
) -> Dict[str, Any]:
    """Complex business flow demonstrating organized dependencies."""
    
    # Create user
    user = await user_service.create_user(user_data)
    
    # Create product
    product = await product_service.create_product(product_data)
    
    # Create order with the new user and product
    order_data.user_id = user.id
    order_data.product_ids = [product.id]
    order = await order_service.create_order(order_data)
    
    return {
        "user": user,
        "product": product,
        "order": order,
        "flow_completed": True
    }

# --- Error Handling Middleware ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Centralized HTTP exception handling."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Centralized general exception handling."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    ) 