"""
Real-Time Processing Engine
Ultra-modular Facebook Posts System v8.0

Advanced real-time processing capabilities:
- Sub-millisecond response times
- Real-time data streaming
- Event-driven architecture
- Stream processing
- Real-time analytics
- Live content generation
- Instant optimization
- Real-time monitoring
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Callable, Awaitable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import websockets
from websockets.server import WebSocketServerProtocol
from collections import deque
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from queue import PriorityQueue, Queue
import structlog

logger = structlog.get_logger(__name__)

class ProcessingPriority(Enum):
    """Processing priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class EventType(Enum):
    """Event types for real-time processing"""
    CONTENT_GENERATION = "content_generation"
    CONTENT_OPTIMIZATION = "content_optimization"
    ANALYTICS_UPDATE = "analytics_update"
    PERFORMANCE_METRIC = "performance_metric"
    USER_INTERACTION = "user_interaction"
    SYSTEM_ALERT = "system_alert"
    NEURAL_SIGNAL = "neural_signal"
    HOLOGRAPHIC_INTERACTION = "holographic_interaction"
    QUANTUM_RESULT = "quantum_result"
    EDGE_UPDATE = "edge_update"

class ProcessingStatus(Enum):
    """Processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingTask:
    """Real-time processing task"""
    task_id: str
    event_type: EventType
    priority: ProcessingPriority
    data: Dict[str, Any]
    callback: Optional[Callable] = None
    timeout: float = 5.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@dataclass
class StreamEvent:
    """Real-time stream event"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    priority: ProcessingPriority = ProcessingPriority.NORMAL

@dataclass
class ProcessingMetrics:
    """Real-time processing metrics"""
    tasks_processed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_processing_time: float = 0.0
    total_processing_time: float = 0.0
    queue_size: int = 0
    active_workers: int = 0
    throughput_per_second: float = 0.0
    error_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0

class RealTimeProcessingEngine:
    """Advanced real-time processing engine for instant responses"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_running = False
        self.is_initialized = False
        
        # Processing configuration
        self.max_workers = self.config.get("max_workers", 100)
        self.max_queue_size = self.config.get("max_queue_size", 10000)
        self.processing_timeout = self.config.get("processing_timeout", 5.0)
        self.batch_size = self.config.get("batch_size", 100)
        self.enable_priority_queue = self.config.get("enable_priority_queue", True)
        self.enable_stream_processing = self.config.get("enable_stream_processing", True)
        
        # Processing queues and workers
        self.task_queue = PriorityQueue(maxsize=self.max_queue_size)
        self.stream_queue = Queue(maxsize=self.max_queue_size)
        self.completed_tasks = deque(maxlen=10000)
        self.failed_tasks = deque(maxlen=1000)
        
        # Worker management
        self.workers = []
        self.worker_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.stream_workers = []
        self.stream_pool = ThreadPoolExecutor(max_workers=20)
        
        # Event handlers
        self.event_handlers = {}
        self.stream_handlers = {}
        
        # Real-time metrics
        self.metrics = ProcessingMetrics()
        self.metrics_history = deque(maxlen=1000)
        
        # WebSocket connections for real-time updates
        self.websocket_connections = set()
        
        # Processing locks
        self.processing_lock = asyncio.Lock()
        self.metrics_lock = asyncio.Lock()
        
        # Performance tracking
        self.latency_samples = deque(maxlen=10000)
        self.throughput_samples = deque(maxlen=1000)
        
    async def initialize(self) -> bool:
        """Initialize real-time processing engine"""
        try:
            logger.info("Initializing Real-Time Processing Engine...")
            
            # Initialize event handlers
            await self._initialize_event_handlers()
            
            # Initialize stream processors
            await self._initialize_stream_processors()
            
            # Initialize metrics collection
            await self._initialize_metrics_collection()
            
            # Initialize WebSocket server
            await self._initialize_websocket_server()
            
            self.is_initialized = True
            logger.info("✓ Real-Time Processing Engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Real-Time Processing Engine: {e}")
            return False
    
    async def start(self) -> bool:
        """Start real-time processing engine"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            logger.info("Starting Real-Time Processing Engine...")
            
            # Start task processors
            for i in range(self.max_workers):
                worker = asyncio.create_task(self._process_tasks())
                self.workers.append(worker)
            
            # Start stream processors
            for i in range(20):
                worker = asyncio.create_task(self._process_streams())
                self.stream_workers.append(worker)
            
            # Start metrics collection
            self.metrics_task = asyncio.create_task(self._collect_metrics())
            
            # Start WebSocket server
            self.websocket_task = asyncio.create_task(self._run_websocket_server())
            
            self.is_running = True
            logger.info("✓ Real-Time Processing Engine started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Real-Time Processing Engine: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop real-time processing engine"""
        try:
            logger.info("Stopping Real-Time Processing Engine...")
            
            self.is_running = False
            
            # Cancel all workers
            for worker in self.workers:
                worker.cancel()
            
            for worker in self.stream_workers:
                worker.cancel()
            
            # Cancel background tasks
            if hasattr(self, 'metrics_task'):
                self.metrics_task.cancel()
            
            if hasattr(self, 'websocket_task'):
                self.websocket_task.cancel()
            
            # Close WebSocket connections
            for connection in self.websocket_connections:
                await connection.close()
            
            # Shutdown thread pools
            self.worker_pool.shutdown(wait=True)
            self.stream_pool.shutdown(wait=True)
            
            logger.info("✓ Real-Time Processing Engine stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Real-Time Processing Engine: {e}")
            return False
    
    async def _initialize_event_handlers(self) -> None:
        """Initialize event handlers for different event types"""
        logger.info("Initializing event handlers...")
        
        # Content generation handler
        self.event_handlers[EventType.CONTENT_GENERATION] = self._handle_content_generation
        
        # Content optimization handler
        self.event_handlers[EventType.CONTENT_OPTIMIZATION] = self._handle_content_optimization
        
        # Analytics update handler
        self.event_handlers[EventType.ANALYTICS_UPDATE] = self._handle_analytics_update
        
        # Performance metric handler
        self.event_handlers[EventType.PERFORMANCE_METRIC] = self._handle_performance_metric
        
        # User interaction handler
        self.event_handlers[EventType.USER_INTERACTION] = self._handle_user_interaction
        
        # System alert handler
        self.event_handlers[EventType.SYSTEM_ALERT] = self._handle_system_alert
        
        # Neural signal handler
        self.event_handlers[EventType.NEURAL_SIGNAL] = self._handle_neural_signal
        
        # Holographic interaction handler
        self.event_handlers[EventType.HOLOGRAPHIC_INTERACTION] = self._handle_holographic_interaction
        
        # Quantum result handler
        self.event_handlers[EventType.QUANTUM_RESULT] = self._handle_quantum_result
        
        # Edge update handler
        self.event_handlers[EventType.EDGE_UPDATE] = self._handle_edge_update
        
        logger.info("✓ Event handlers initialized")
    
    async def _initialize_stream_processors(self) -> None:
        """Initialize stream processors for real-time data"""
        logger.info("Initializing stream processors...")
        
        # Real-time analytics processor
        self.stream_handlers["analytics"] = self._process_analytics_stream
        
        # Performance monitoring processor
        self.stream_handlers["performance"] = self._process_performance_stream
        
        # User interaction processor
        self.stream_handlers["user_interaction"] = self._process_user_interaction_stream
        
        # Neural signal processor
        self.stream_handlers["neural"] = self._process_neural_stream
        
        # Holographic interaction processor
        self.stream_handlers["holographic"] = self._process_holographic_stream
        
        logger.info("✓ Stream processors initialized")
    
    async def _initialize_metrics_collection(self) -> None:
        """Initialize metrics collection system"""
        logger.info("Initializing metrics collection...")
        
        # Initialize performance tracking
        self.performance_tracker = {
            "start_time": time.time(),
            "last_update": time.time(),
            "sample_count": 0
        }
        
        logger.info("✓ Metrics collection initialized")
    
    async def _initialize_websocket_server(self) -> None:
        """Initialize WebSocket server for real-time updates"""
        logger.info("Initializing WebSocket server...")
        
        # WebSocket server will be started in the start method
        logger.info("✓ WebSocket server initialized")
    
    async def _process_tasks(self) -> None:
        """Process tasks from the priority queue"""
        while self.is_running:
            try:
                # Get task from priority queue
                if self.enable_priority_queue:
                    try:
                        priority, task = self.task_queue.get(timeout=1.0)
                    except:
                        continue
                else:
                    # Simple FIFO processing
                    try:
                        task = self.task_queue.get(timeout=1.0)
                    except:
                        continue
                
                # Process task
                await self._execute_task(task)
                
            except Exception as e:
                logger.error(f"Task processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_streams(self) -> None:
        """Process real-time streams"""
        while self.is_running:
            try:
                # Get stream event
                try:
                    event = self.stream_queue.get(timeout=1.0)
                except:
                    continue
                
                # Process stream event
                await self._execute_stream_event(event)
                
            except Exception as e:
                logger.error(f"Stream processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _execute_task(self, task: ProcessingTask) -> None:
        """Execute a processing task"""
        try:
            task.status = ProcessingStatus.PROCESSING
            task.started_at = datetime.now()
            
            start_time = time.time()
            
            # Get event handler
            handler = self.event_handlers.get(task.event_type)
            if not handler:
                raise ValueError(f"No handler for event type: {task.event_type}")
            
            # Execute handler
            result = await handler(task.data)
            
            # Update task
            task.status = ProcessingStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.latency_samples.append(processing_time)
            
            # Update metrics
            await self._update_task_metrics(task, processing_time)
            
            # Store completed task
            self.completed_tasks.append(task)
            
            # Execute callback if provided
            if task.callback:
                try:
                    await task.callback(task)
                except Exception as e:
                    logger.error(f"Task callback error: {e}")
            
            # Broadcast task completion
            await self._broadcast_task_completion(task)
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            await self._handle_task_failure(task, str(e))
    
    async def _execute_stream_event(self, event: StreamEvent) -> None:
        """Execute a stream event"""
        try:
            # Get stream handler
            handler = self.stream_handlers.get(event.source)
            if not handler:
                logger.warning(f"No handler for stream source: {event.source}")
                return
            
            # Execute handler
            await handler(event)
            
            # Update throughput metrics
            self.throughput_samples.append(time.time())
            
        except Exception as e:
            logger.error(f"Stream event execution error: {e}")
    
    async def _update_task_metrics(self, task: ProcessingTask, processing_time: float) -> None:
        """Update task processing metrics"""
        async with self.metrics_lock:
            self.metrics.tasks_processed += 1
            self.metrics.tasks_completed += 1
            self.metrics.total_processing_time += processing_time
            self.metrics.avg_processing_time = (
                self.metrics.total_processing_time / self.metrics.tasks_processed
            )
            self.metrics.queue_size = self.task_queue.qsize()
            self.metrics.active_workers = len([w for w in self.workers if not w.done()])
            
            # Calculate error rate
            if self.metrics.tasks_processed > 0:
                self.metrics.error_rate = (
                    self.metrics.tasks_failed / self.metrics.tasks_processed
                )
            
            # Calculate latency percentiles
            if self.latency_samples:
                sorted_latencies = sorted(self.latency_samples)
                n = len(sorted_latencies)
                self.metrics.latency_p50 = sorted_latencies[int(n * 0.5)]
                self.metrics.latency_p95 = sorted_latencies[int(n * 0.95)]
                self.metrics.latency_p99 = sorted_latencies[int(n * 0.99)]
            
            # Calculate throughput
            if self.throughput_samples:
                now = time.time()
                recent_samples = [t for t in self.throughput_samples if now - t < 1.0]
                self.metrics.throughput_per_second = len(recent_samples)
    
    async def _handle_task_failure(self, task: ProcessingTask, error: str) -> None:
        """Handle task failure"""
        task.status = ProcessingStatus.FAILED
        task.error = error
        task.completed_at = datetime.now()
        
        # Retry if possible
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = ProcessingStatus.PENDING
            task.started_at = None
            task.completed_at = None
            task.error = None
            
            # Re-queue task
            await self.submit_task(task)
        else:
            # Store failed task
            self.failed_tasks.append(task)
            
            # Update metrics
            async with self.metrics_lock:
                self.metrics.tasks_failed += 1
    
    async def _collect_metrics(self) -> None:
        """Collect and update metrics periodically"""
        while self.is_running:
            try:
                # Store metrics history
                self.metrics_history.append(asdict(self.metrics))
                
                # Log metrics
                logger.info(
                    "Real-time processing metrics",
                    **asdict(self.metrics)
                )
                
                # Broadcast metrics
                await self._broadcast_metrics()
                
                await asyncio.sleep(1.0)  # Update every second
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(1.0)
    
    async def _run_websocket_server(self) -> None:
        """Run WebSocket server for real-time updates"""
        try:
            async with websockets.serve(
                self._handle_websocket_connection,
                "localhost",
                8765
            ):
                logger.info("WebSocket server started on port 8765")
                await asyncio.Future()  # Run forever
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
    
    async def _handle_websocket_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle WebSocket connection"""
        self.websocket_connections.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.websocket_connections.discard(websocket)
    
    # Event handlers
    
    async def _handle_content_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content generation event"""
        try:
            # Simulate content generation
            content = {
                "id": str(uuid.uuid4()),
                "type": "post",
                "content": data.get("content", "Generated content"),
                "timestamp": datetime.now().isoformat(),
                "status": "generated"
            }
            
            return content
            
        except Exception as e:
            logger.error(f"Content generation error: {e}")
            raise
    
    async def _handle_content_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle content optimization event"""
        try:
            # Simulate content optimization
            optimization = {
                "id": str(uuid.uuid4()),
                "original_content": data.get("content", ""),
                "optimized_content": f"Optimized: {data.get('content', '')}",
                "optimization_score": np.random.uniform(0.7, 0.95),
                "timestamp": datetime.now().isoformat(),
                "status": "optimized"
            }
            
            return optimization
            
        except Exception as e:
            logger.error(f"Content optimization error: {e}")
            raise
    
    async def _handle_analytics_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analytics update event"""
        try:
            # Simulate analytics update
            analytics = {
                "id": str(uuid.uuid4()),
                "metrics": {
                    "views": np.random.randint(1000, 10000),
                    "likes": np.random.randint(100, 1000),
                    "shares": np.random.randint(10, 100),
                    "comments": np.random.randint(5, 50)
                },
                "timestamp": datetime.now().isoformat(),
                "status": "updated"
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Analytics update error: {e}")
            raise
    
    async def _handle_performance_metric(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance metric event"""
        try:
            # Simulate performance metric
            metric = {
                "id": str(uuid.uuid4()),
                "metric_type": data.get("metric_type", "cpu_usage"),
                "value": np.random.uniform(0.0, 100.0),
                "timestamp": datetime.now().isoformat(),
                "status": "recorded"
            }
            
            return metric
            
        except Exception as e:
            logger.error(f"Performance metric error: {e}")
            raise
    
    async def _handle_user_interaction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle user interaction event"""
        try:
            # Simulate user interaction
            interaction = {
                "id": str(uuid.uuid4()),
                "user_id": data.get("user_id", "unknown"),
                "interaction_type": data.get("interaction_type", "click"),
                "timestamp": datetime.now().isoformat(),
                "status": "processed"
            }
            
            return interaction
            
        except Exception as e:
            logger.error(f"User interaction error: {e}")
            raise
    
    async def _handle_system_alert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system alert event"""
        try:
            # Simulate system alert
            alert = {
                "id": str(uuid.uuid4()),
                "alert_type": data.get("alert_type", "warning"),
                "message": data.get("message", "System alert"),
                "severity": data.get("severity", "medium"),
                "timestamp": datetime.now().isoformat(),
                "status": "alerted"
            }
            
            return alert
            
        except Exception as e:
            logger.error(f"System alert error: {e}")
            raise
    
    async def _handle_neural_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle neural signal event"""
        try:
            # Simulate neural signal processing
            signal = {
                "id": str(uuid.uuid4()),
                "signal_type": data.get("signal_type", "eeg"),
                "amplitude": np.random.uniform(0.0, 1.0),
                "frequency": np.random.uniform(1.0, 100.0),
                "timestamp": datetime.now().isoformat(),
                "status": "processed"
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Neural signal error: {e}")
            raise
    
    async def _handle_holographic_interaction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle holographic interaction event"""
        try:
            # Simulate holographic interaction
            interaction = {
                "id": str(uuid.uuid4()),
                "interaction_type": data.get("interaction_type", "gesture"),
                "position": data.get("position", [0.0, 0.0, 0.0]),
                "timestamp": datetime.now().isoformat(),
                "status": "processed"
            }
            
            return interaction
            
        except Exception as e:
            logger.error(f"Holographic interaction error: {e}")
            raise
    
    async def _handle_quantum_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quantum result event"""
        try:
            # Simulate quantum result
            result = {
                "id": str(uuid.uuid4()),
                "algorithm": data.get("algorithm", "qaoa"),
                "result": np.random.uniform(0.0, 1.0),
                "confidence": np.random.uniform(0.8, 0.99),
                "timestamp": datetime.now().isoformat(),
                "status": "completed"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Quantum result error: {e}")
            raise
    
    async def _handle_edge_update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle edge update event"""
        try:
            # Simulate edge update
            update = {
                "id": str(uuid.uuid4()),
                "edge_location": data.get("location", "unknown"),
                "latency": np.random.uniform(1.0, 10.0),
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            }
            
            return update
            
        except Exception as e:
            logger.error(f"Edge update error: {e}")
            raise
    
    # Stream processors
    
    async def _process_analytics_stream(self, event: StreamEvent) -> None:
        """Process analytics stream"""
        try:
            # Process analytics data in real-time
            logger.debug(f"Processing analytics stream: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Analytics stream processing error: {e}")
    
    async def _process_performance_stream(self, event: StreamEvent) -> None:
        """Process performance stream"""
        try:
            # Process performance data in real-time
            logger.debug(f"Processing performance stream: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Performance stream processing error: {e}")
    
    async def _process_user_interaction_stream(self, event: StreamEvent) -> None:
        """Process user interaction stream"""
        try:
            # Process user interaction data in real-time
            logger.debug(f"Processing user interaction stream: {event.event_id}")
            
        except Exception as e:
            logger.error(f"User interaction stream processing error: {e}")
    
    async def _process_neural_stream(self, event: StreamEvent) -> None:
        """Process neural stream"""
        try:
            # Process neural data in real-time
            logger.debug(f"Processing neural stream: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Neural stream processing error: {e}")
    
    async def _process_holographic_stream(self, event: StreamEvent) -> None:
        """Process holographic stream"""
        try:
            # Process holographic data in real-time
            logger.debug(f"Processing holographic stream: {event.event_id}")
            
        except Exception as e:
            logger.error(f"Holographic stream processing error: {e}")
    
    # Broadcasting methods
    
    async def _broadcast_task_completion(self, task: ProcessingTask) -> None:
        """Broadcast task completion to WebSocket clients"""
        if self.websocket_connections:
            message = {
                "type": "task_completion",
                "task_id": task.task_id,
                "event_type": task.event_type.value,
                "status": task.status.value,
                "result": task.result,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.websocket_connections:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_connections -= disconnected_clients
    
    async def _broadcast_metrics(self) -> None:
        """Broadcast metrics to WebSocket clients"""
        if self.websocket_connections:
            message = {
                "type": "metrics_update",
                "metrics": asdict(self.metrics),
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.websocket_connections:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_connections -= disconnected_clients
    
    # Public API methods
    
    async def submit_task(self, task: ProcessingTask) -> str:
        """Submit a task for processing"""
        try:
            # Set task ID if not provided
            if not task.task_id:
                task.task_id = str(uuid.uuid4())
            
            # Set created timestamp
            if not task.created_at:
                task.created_at = datetime.now()
            
            # Add to priority queue
            if self.enable_priority_queue:
                self.task_queue.put((task.priority.value, task))
            else:
                self.task_queue.put(task)
            
            logger.info(f"Task submitted: {task.task_id}")
            return task.task_id
            
        except Exception as e:
            logger.error(f"Task submission error: {e}")
            raise
    
    async def submit_stream_event(self, event: StreamEvent) -> str:
        """Submit a stream event for processing"""
        try:
            # Set event ID if not provided
            if not event.event_id:
                event.event_id = str(uuid.uuid4())
            
            # Add to stream queue
            self.stream_queue.put(event)
            
            logger.debug(f"Stream event submitted: {event.event_id}")
            return event.event_id
            
        except Exception as e:
            logger.error(f"Stream event submission error: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        # Check completed tasks
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return asdict(task)
        
        # Check failed tasks
        for task in self.failed_tasks:
            if task.task_id == task_id:
                return asdict(task)
        
        return None
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics"""
        async with self.metrics_lock:
            return asdict(self.metrics)
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get processing engine health status"""
        return {
            "status": "healthy" if self.is_running else "unhealthy",
            "running": self.is_running,
            "workers_active": len([w for w in self.workers if not w.done()]),
            "queue_size": self.task_queue.qsize(),
            "stream_queue_size": self.stream_queue.qsize(),
            "websocket_connections": len(self.websocket_connections),
            "metrics": asdict(self.metrics)
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **asdict(self.metrics),
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "real_time_processing_engine": {
                "status": "running" if self.is_running else "stopped",
                "workers": len(self.workers),
                "stream_workers": len(self.stream_workers),
                "queue_sizes": {
                    "task_queue": self.task_queue.qsize(),
                    "stream_queue": self.stream_queue.qsize()
                },
                "metrics": asdict(self.metrics)
            },
            "timestamp": datetime.now().isoformat()
        }
