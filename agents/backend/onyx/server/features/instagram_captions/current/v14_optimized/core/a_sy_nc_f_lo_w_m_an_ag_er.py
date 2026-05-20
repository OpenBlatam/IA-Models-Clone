from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import weakref
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic, Union, AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
from functools import wraps
import threading
from collections import deque
import heapq
from concurrent.futures import ThreadPoolExecutor
import signal
import contextvars
from typing import Any, List, Dict, Optional
"""
Async Flow Manager for Instagram Captions API v14.0

Comprehensive async and non-blocking flow management:
- Async pipelines and workflows
- Event-driven processing
- Reactive patterns
- Non-blocking I/O operations
- Async streams and backpressure
- Flow control and rate limiting
- Async state machines
"""


logger = logging.getLogger(__name__)

# Type variables
T = TypeVar('T')
U = TypeVar('U')
V = TypeVar('V')

# Context variables for flow tracking
flow_context = contextvars.ContextVar('flow_context', default={})
request_context = contextvars.ContextVar('request_context', default={})


class FlowType(Enum):
    """Types of async flows"""
    PIPELINE = "pipeline"
    STREAM = "stream"
    EVENT_DRIVEN = "event_driven"
    REACTIVE = "reactive"
    STATE_MACHINE = "state_machine"
    WORKFLOW = "workflow"


class FlowState(Enum):
    """States of async flows"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FlowConfig:
    """Configuration for async flows"""
    max_concurrent_flows: int = 100
    max_queue_size: int = 1000
    timeout: float = 30.0
    enable_backpressure: bool = True
    backpressure_threshold: int = 100
    enable_flow_tracking: bool = True
    enable_performance_monitoring: bool = True


@dataclass
class FlowMetrics:
    """Metrics for flow tracking"""
    total_flows: int = 0
    active_flows: int = 0
    completed_flows: int = 0
    failed_flows: int = 0
    average_duration: float = 0.0
    throughput_per_second: float = 0.0
    queue_size: int = 0


class AsyncPipeline:
    """Async pipeline for processing data through multiple stages"""
    
    def __init__(self, name: str = "pipeline"):
        
    """__init__ function."""
self.name = name
        self.stages: List[Callable] = []
        self.stage_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.metrics = FlowMetrics()
    
    def add_stage(self, stage_func: Callable, config: Optional[Dict[str, Any]] = None):
        """Add a stage to the pipeline"""
        stage_name = f"stage_{len(self.stages)}"
        self.stages.append(stage_func)
        self.stage_configs[stage_name] = config or {}
        return self
    
    async def process(self, data: T, context: Optional[Dict[str, Any]] = None) -> U:
        """Process data through all pipeline stages"""
        start_time = time.time()
        current_data = data
        context = context or {}
        
        try:
            self.metrics.total_flows += 1
            self.metrics.active_flows += 1
            
            for i, stage_func in enumerate(self.stages):
                stage_name = f"stage_{i}"
                stage_config = self.stage_configs.get(stage_name, {})
                
                # Execute stage with timeout and retry logic
                current_data = await self._execute_stage(
                    stage_func, current_data, context, stage_config
                )
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics.completed_flows += 1
            self.metrics.average_duration = (
                (self.metrics.average_duration * (self.metrics.completed_flows - 1) + duration) /
                self.metrics.completed_flows
            )
            
            return current_data
            
        except Exception as e:
            self.metrics.failed_flows += 1
            logger.error(f"Pipeline {self.name} failed: {e}")
            raise
        finally:
            self.metrics.active_flows -= 1
    
    async def _execute_stage(
        self, 
        stage_func: Callable, 
        data: Any, 
        context: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Any:
        """Execute a single pipeline stage"""
        timeout = config.get('timeout', 10.0)
        retries = config.get('retries', 0)
        
        for attempt in range(retries + 1):
            try:
                if asyncio.iscoroutinefunction(stage_func):
                    return await asyncio.wait_for(stage_func(data, context), timeout=timeout)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    return await asyncio.wait_for(
                        loop.run_in_executor(None, stage_func, data, context),
                        timeout=timeout
                    )
            except asyncio.TimeoutError:
                if attempt == retries:
                    raise Exception(f"Stage timeout after {timeout}s")
                await asyncio.sleep(0.1 * (attempt + 1))
            except Exception as e:
                if attempt == retries:
                    raise e
                await asyncio.sleep(0.1 * (attempt + 1))


class AsyncStream:
    """Async stream with backpressure control"""
    
    def __init__(self, max_buffer_size: int = 1000):
        
    """__init__ function."""
self.max_buffer_size = max_buffer_size
        self.buffer = deque()
        self.consumers: List[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self._producer_task = None
        self._consumer_tasks: List[asyncio.Task] = []
    
    async def produce(self, item: T) -> bool:
        """Produce an item to the stream with backpressure control"""
        async with self._lock:
            if len(self.buffer) >= self.max_buffer_size:
                return False  # Backpressure: buffer full
            
            self.buffer.append(item)
            
            # Notify consumers
            for consumer_queue in self.consumers:
                try:
                    consumer_queue.put_nowait(item)
                except asyncio.QueueFull:
                    # Consumer is slow, skip this consumer
                    continue
            
            return True
    
    async def consume(self) -> AsyncGenerator[T, None]:
        """Consume items from the stream"""
        consumer_queue = asyncio.Queue(maxsize=self.max_buffer_size)
        
        async with self._lock:
            self.consumers.append(consumer_queue)
        
        try:
            while True:
                try:
                    item = await asyncio.wait_for(consumer_queue.get(), timeout=1.0)
                    yield item
                    consumer_queue.task_done()
                except asyncio.TimeoutError:
                    # Check if stream is still active
                    continue
        finally:
            async with self._lock:
                if consumer_queue in self.consumers:
                    self.consumers.remove(consumer_queue)


class EventBus:
    """Event-driven processing with async event bus"""
    
    def __init__(self) -> Any:
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_queue = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._processor_task = None
    
    async def start(self) -> Any:
        """Start the event processor"""
        if self._processor_task is None:
            self._processor_task = asyncio.create_task(self._process_events())
    
    async def stop(self) -> Any:
        """Stop the event processor"""
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None
    
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        async with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        async with self._lock:
            if event_type in self.subscribers:
                try:
                    self.subscribers[event_type].remove(handler)
                except ValueError:
                    pass
    
    async def publish(self, event_type: str, event_data: Any):
        """Publish an event"""
        await self.event_queue.put((event_type, event_data))
    
    async def _process_events(self) -> Any:
        """Process events from the queue"""
        while True:
            try:
                event_type, event_data = await self.event_queue.get()
                
                # Get handlers for this event type
                handlers = []
                async with self._lock:
                    handlers = self.subscribers.get(event_type, []).copy()
                
                # Execute handlers concurrently
                if handlers:
                    tasks = []
                    for handler in handlers:
                        if asyncio.iscoroutinefunction(handler):
                            task = asyncio.create_task(handler(event_data))
                        else:
                            # Run sync handler in thread pool
                            loop = asyncio.get_event_loop()
                            task = asyncio.create_task(
                                loop.run_in_executor(None, handler, event_data)
                            )
                        tasks.append(task)
                    
                    # Wait for all handlers to complete
                    await asyncio.gather(*tasks, return_exceptions=True)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}")


class ReactiveFlow:
    """Reactive flow with automatic dependency management"""
    
    def __init__(self) -> Any:
        self.dependencies: Dict[str, List[str]] = {}
        self.computations: Dict[str, Callable] = {}
        self.cache: Dict[str, Any] = {}
        self.dirty_flags: Dict[str, bool] = {}
        self._lock = asyncio.Lock()
    
    def add_computation(self, name: str, func: Callable, deps: List[str]):
        """Add a reactive computation"""
        async with self._lock:
            self.computations[name] = func
            self.dependencies[name] = deps
            self.dirty_flags[name] = True
    
    async def invalidate(self, name: str):
        """Invalidate a computation and its dependents"""
        async with self._lock:
            self.dirty_flags[name] = True
            # Mark dependents as dirty
            for comp_name, deps in self.dependencies.items():
                if name in deps:
                    self.dirty_flags[comp_name] = True
    
    async def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a computation result, computing if necessary"""
        async with self._lock:
            if name not in self.computations:
                raise ValueError(f"Computation {name} not found")
            
            if self.dirty_flags.get(name, True):
                # Compute dependencies first
                deps = self.dependencies[name]
                dep_results = []
                for dep in deps:
                    dep_result = await self.get(dep)
                    dep_results.append(dep_result)
                
                # Execute computation
                func = self.computations[name]
                if asyncio.iscoroutinefunction(func):
                    result = await func(*dep_results)
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, func, *dep_results)
                
                self.cache[name] = result
                self.dirty_flags[name] = False
            
            return self.cache[name]


class AsyncStateMachine:
    """Async state machine with non-blocking transitions"""
    
    def __init__(self, initial_state: str):
        
    """__init__ function."""
self.current_state = initial_state
        self.states: Dict[str, Dict[str, Any]] = {}
        self.transitions: Dict[str, Dict[str, str]] = {}
        self.state_handlers: Dict[str, Callable] = {}
        self.transition_handlers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
        self._state_history: List[str] = [initial_state]
    
    def add_state(self, state: str, config: Optional[Dict[str, Any]] = None):
        """Add a state to the state machine"""
        self.states[state] = config or {}
        return self
    
    def add_transition(self, from_state: str, to_state: str, trigger: str):
        """Add a transition between states"""
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        self.transitions[from_state][trigger] = to_state
        return self
    
    def set_state_handler(self, state: str, handler: Callable):
        """Set handler for entering a state"""
        self.state_handlers[state] = handler
        return self
    
    def set_transition_handler(self, transition_key: str, handler: Callable):
        """Set handler for a transition"""
        self.transition_handlers[transition_key] = handler
        return self
    
    async def trigger(self, trigger: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Trigger a state transition"""
        context = context or {}
        
        async with self._lock:
            current_state = self.current_state
            
            if current_state not in self.transitions:
                return False
            
            if trigger not in self.transitions[current_state]:
                return False
            
            new_state = self.transitions[current_state][trigger]
            transition_key = f"{current_state}->{new_state}"
            
            # Execute transition handler
            if transition_key in self.transition_handlers:
                handler = self.transition_handlers[transition_key]
                if asyncio.iscoroutinefunction(handler):
                    await handler(current_state, new_state, context)
                else:
                    # Run sync handler in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler, current_state, new_state, context)
            
            # Update state
            self.current_state = new_state
            self._state_history.append(new_state)
            
            # Execute state handler
            if new_state in self.state_handlers:
                handler = self.state_handlers[new_state]
                if asyncio.iscoroutinefunction(handler):
                    await handler(context)
                else:
                    # Run sync handler in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler, context)
            
            return True
    
    def get_current_state(self) -> str:
        """Get current state"""
        return self.current_state
    
    def get_state_history(self) -> List[str]:
        """Get state transition history"""
        return self._state_history.copy()


class AsyncFlowManager:
    """Main async flow manager"""
    
    def __init__(self, config: FlowConfig):
        
    """__init__ function."""
self.config = config
        self.pipelines: Dict[str, AsyncPipeline] = {}
        self.streams: Dict[str, AsyncStream] = {}
        self.event_bus = EventBus()
        self.reactive_flows: Dict[str, ReactiveFlow] = {}
        self.state_machines: Dict[str, AsyncStateMachine] = {}
        self.flow_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.metrics = FlowMetrics()
        self._lock = asyncio.Lock()
        self._processor_task = None
        self._monitor_task = None
    
    async def start(self) -> Any:
        """Start the flow manager"""
        await self.event_bus.start()
        self._processor_task = asyncio.create_task(self._process_flows())
        self._monitor_task = asyncio.create_task(self._monitor_flows())
    
    async def stop(self) -> Any:
        """Stop the flow manager"""
        await self.event_bus.stop()
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def create_pipeline(self, name: str) -> AsyncPipeline:
        """Create a new async pipeline"""
        async with self._lock:
            pipeline = AsyncPipeline(name)
            self.pipelines[name] = pipeline
            return pipeline
    
    async def create_stream(self, name: str, max_buffer_size: int = 1000) -> AsyncStream:
        """Create a new async stream"""
        async with self._lock:
            stream = AsyncStream(max_buffer_size)
            self.streams[name] = stream
            return stream
    
    async def create_reactive_flow(self, name: str) -> ReactiveFlow:
        """Create a new reactive flow"""
        async with self._lock:
            flow = ReactiveFlow()
            self.reactive_flows[name] = flow
            return flow
    
    async def create_state_machine(self, name: str, initial_state: str) -> AsyncStateMachine:
        """Create a new async state machine"""
        async with self._lock:
            state_machine = AsyncStateMachine(initial_state)
            self.state_machines[name] = state_machine
            return state_machine
    
    async def execute_flow(self, flow_type: FlowType, flow_name: str, data: Any) -> Any:
        """Execute a flow with the given type and name"""
        try:
            self.metrics.total_flows += 1
            self.metrics.active_flows += 1
            
            if flow_type == FlowType.PIPELINE:
                if flow_name in self.pipelines:
                    return await self.pipelines[flow_name].process(data)
                else:
                    raise ValueError(f"Pipeline {flow_name} not found")
            
            elif flow_type == FlowType.STREAM:
                if flow_name in self.streams:
                    # For streams, we return the stream object
                    return self.streams[flow_name]
                else:
                    raise ValueError(f"Stream {flow_name} not found")
            
            elif flow_type == FlowType.REACTIVE:
                if flow_name in self.reactive_flows:
                    # For reactive flows, we need to specify which computation to get
                    if isinstance(data, str):
                        return await self.reactive_flows[flow_name].get(data)
                    else:
                        raise ValueError("Reactive flow requires computation name as data")
                else:
                    raise ValueError(f"Reactive flow {flow_name} not found")
            
            else:
                raise ValueError(f"Unsupported flow type: {flow_type}")
        
        except Exception as e:
            self.metrics.failed_flows += 1
            logger.error(f"Flow execution failed: {e}")
            raise
        finally:
            self.metrics.active_flows -= 1
    
    async def _process_flows(self) -> Any:
        """Process flows from the queue"""
        while True:
            try:
                flow_data = await self.flow_queue.get()
                # Process flow data
                await self._handle_flow_data(flow_data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flow processing error: {e}")
    
    async def _handle_flow_data(self, flow_data: Any):
        """Handle flow data processing"""
        # Implementation depends on specific flow data structure
        pass
    
    async def _monitor_flows(self) -> Any:
        """Monitor flow metrics"""
        while True:
            try:
                await asyncio.sleep(1.0)
                
                # Update metrics
                self.metrics.queue_size = self.flow_queue.qsize()
                
                # Calculate throughput
                if self.metrics.total_flows > 0:
                    self.metrics.throughput_per_second = (
                        self.metrics.completed_flows / 
                        max(1, time.time() - self._start_time)
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Flow monitoring error: {e}")


# Global flow manager instance
flow_manager: Optional[AsyncFlowManager] = None


async def get_flow_manager() -> AsyncFlowManager:
    """Get the global flow manager instance"""
    global flow_manager
    if flow_manager is None:
        config = FlowConfig()
        flow_manager = AsyncFlowManager(config)
        await flow_manager.start()
    return flow_manager


# Decorators for easy flow usage
def async_pipeline(name: str):
    """Decorator to create an async pipeline"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            manager = await get_flow_manager()
            pipeline = await manager.create_pipeline(name)
            return await pipeline.process(func(*args, **kwargs))
        return wrapper
    return decorator


def async_stream(name: str, max_buffer_size: int = 1000):
    """Decorator to create an async stream"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            manager = await get_flow_manager()
            stream = await manager.create_stream(name, max_buffer_size)
            return stream
        return wrapper
    return decorator


def reactive_computation(name: str, dependencies: List[str]):
    """Decorator to create a reactive computation"""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            manager = await get_flow_manager()
            flow = await manager.create_reactive_flow(name)
            flow.add_computation(name, func, dependencies)
            return await flow.get(name)
        return wrapper
    return decorator


# Context managers for flow control
@asynccontextmanager
async def flow_context(flow_name: str, flow_type: FlowType):
    """Context manager for flow execution"""
    manager = await get_flow_manager()
    try:
        yield manager
    finally:
        pass


@asynccontextmanager
async def non_blocking_operation():
    """Context manager for non-blocking operations"""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        if duration > 0.1:  # Log slow operations
            logger.warning(f"Non-blocking operation took {duration:.3f}s")


# Utility functions for non-blocking operations
async def non_blocking_call(func: Callable, *args, **kwargs) -> Any:
    """Execute a function in a non-blocking manner"""
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    else:
        # Run sync function in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)


async def non_blocking_batch(items: List[Any], processor: Callable, max_concurrent: int = 10) -> List[Any]:
    """Process items in batches with non-blocking operations"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_item(item) -> Any:
        async with semaphore:
            return await non_blocking_call(processor, item)
    
    tasks = [process_item(item) for item in items]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def non_blocking_stream(producer: Callable, consumer: Callable, buffer_size: int = 100):
    """Create a non-blocking stream between producer and consumer"""
    queue = asyncio.Queue(maxsize=buffer_size)
    
    # Producer task
    async def produce():
        
    """produce function."""
try:
            while True:
                item = await non_blocking_call(producer)
                await queue.put(item)
        except Exception as e:
            logger.error(f"Producer error: {e}")
    
    # Consumer task
    async def consume():
        
    """consume function."""
try:
            while True:
                item = await queue.get()
                await non_blocking_call(consumer, item)
                queue.task_done()
        except Exception as e:
            logger.error(f"Consumer error: {e}")
    
    # Start both tasks
    producer_task = asyncio.create_task(produce())
    consumer_task = asyncio.create_task(consume())
    
    try:
        await asyncio.gather(producer_task, consumer_task)
    except asyncio.CancelledError:
        producer_task.cancel()
        consumer_task.cancel()
        await asyncio.gather(producer_task, consumer_task, return_exceptions=True) 