"""
Blaze AI Distributed Processing Module v7.5.0

Advanced distributed computing system providing horizontal scalability,
node management, task distribution, load balancing, and fault tolerance.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
import aiohttp
import aioredis
from pathlib import Path

from .base import BaseModule, ModuleConfig, ModuleStatus

logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND CONFIGURATION
# ============================================================================

class NodeStatus(Enum):
    """Node status enumeration."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"
    ADAPTIVE = "adaptive"

class FaultToleranceStrategy(Enum):
    """Fault tolerance strategies."""
    REPLICATION = "replication"
    CHECKPOINTING = "checkpointing"
    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"

@dataclass
class DistributedProcessingConfig(ModuleConfig):
    """Configuration for Distributed Processing module."""
    # Node management
    node_id: str = field(default_factory=lambda: f"node_{uuid.uuid4().hex[:8]}")
    node_name: str = "Blaze AI Node"
    node_capacity: int = 100  # Maximum concurrent tasks
    node_weight: float = 1.0  # Load balancing weight
    
    # Network settings
    discovery_port: int = 8888
    communication_port: int = 8889
    heartbeat_interval: float = 5.0  # seconds
    node_timeout: float = 30.0  # seconds
    
    # Task management
    max_task_retries: int = 3
    task_timeout: float = 300.0  # seconds
    batch_size: int = 10
    enable_checkpointing: bool = True
    checkpoint_interval: float = 60.0  # seconds
    
    # Load balancing
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE
    enable_auto_scaling: bool = True
    min_nodes: int = 1
    max_nodes: int = 100
    scaling_threshold: float = 0.8  # 80% capacity triggers scaling
    
    # Fault tolerance
    fault_tolerance_strategy: FaultToleranceStrategy = FaultToleranceStrategy.REPLICATION
    replication_factor: int = 2
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0  # seconds
    
    # Storage and persistence
    task_storage_path: str = "./distributed/tasks"
    node_storage_path: str = "./distributed/nodes"
    enable_persistence: bool = True
    backup_interval: float = 3600.0  # seconds

@dataclass
class NodeInfo:
    """Node information."""
    node_id: str
    node_name: str
    node_address: str
    node_port: int
    node_capacity: int
    node_weight: float
    node_status: NodeStatus
    current_load: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DistributedTask:
    """Distributed task definition."""
    task_id: str
    task_type: str
    task_data: Dict[str, Any]
    priority: TaskPriority
    status: TaskStatus
    source_node: str
    target_node: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 300.0
    result: Optional[Any] = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ClusterMetrics:
    """Cluster performance metrics."""
    total_nodes: int = 0
    active_nodes: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: float = 0.0
    cluster_load: float = 0.0
    network_latency: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

# ============================================================================
# NETWORK COMMUNICATION
# ============================================================================

class NetworkProtocol(ABC):
    """Abstract base class for network communication protocols."""
    
    @abstractmethod
    async def send_message(self, target: str, message: Dict[str, Any]) -> bool:
        """Send message to target node."""
        pass
    
    @abstractmethod
    async def broadcast_message(self, message: Dict[str, Any]) -> bool:
        """Broadcast message to all nodes."""
        pass
    
    @abstractmethod
    async def start_listening(self) -> bool:
        """Start listening for incoming messages."""
        pass
    
    @abstractmethod
    async def stop_listening(self) -> bool:
        """Stop listening for incoming messages."""
        pass

class HTTPProtocol(NetworkProtocol):
    """HTTP-based network communication protocol."""
    
    def __init__(self, config: DistributedProcessingConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.server: Optional[asyncio.Server] = None
        self.message_handlers: Dict[str, Callable] = {}
        
    async def send_message(self, target: str, message: Dict[str, Any]) -> bool:
        """Send HTTP message to target node."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            url = f"http://{target}:{self.config.communication_port}/message"
            async with self.session.post(url, json=message) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to send message to {target}: {e}")
            return False
    
    async def broadcast_message(self, message: Dict[str, Any]) -> bool:
        """Broadcast HTTP message to all known nodes."""
        # This would be implemented with node discovery
        return True
    
    async def start_listening(self) -> bool:
        """Start HTTP server for incoming messages."""
        try:
            self.server = await asyncio.start_server(
                self._handle_request,
                '0.0.0.0',
                self.config.communication_port
            )
            logger.info(f"HTTP server started on port {self.config.communication_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start HTTP server: {e}")
            return False
    
    async def stop_listening(self) -> bool:
        """Stop HTTP server."""
        try:
            if self.server:
                self.server.close()
                await self.server.wait_closed()
            if self.session:
                await self.session.close()
            return True
        except Exception as e:
            logger.error(f"Failed to stop HTTP server: {e}")
            return False
    
    async def _handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming HTTP requests."""
        try:
            # Parse HTTP request
            request_line = await reader.readline()
            headers = await self._parse_headers(reader)
            
            # Read body if present
            body = b""
            if 'content-length' in headers:
                content_length = int(headers['content-length'])
                body = await reader.read(content_length)
            
            # Process message
            if body:
                message = json.loads(body.decode())
                await self._process_message(message)
            
            # Send response
            response = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
            writer.write(response.encode())
            await writer.drain()
            
        except Exception as e:
            logger.error(f"Error handling HTTP request: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _parse_headers(self, reader: asyncio.StreamReader) -> Dict[str, str]:
        """Parse HTTP headers."""
        headers = {}
        while True:
            line = await reader.readline()
            if line == b'\r\n':
                break
            if b':' in line:
                key, value = line.decode().strip().split(':', 1)
                headers[key.lower()] = value.strip()
        return headers
    
    async def _process_message(self, message: Dict[str, Any]):
        """Process incoming message."""
        message_type = message.get('type')
        handler = self.message_handlers.get(message_type)
        if handler:
            await handler(message)
        else:
            logger.warning(f"No handler for message type: {message_type}")

# ============================================================================
# NODE DISCOVERY AND MANAGEMENT
# ============================================================================

class NodeDiscovery:
    """Manages node discovery and registration."""
    
    def __init__(self, config: DistributedProcessingConfig):
        self.config = config
        self.known_nodes: Dict[str, NodeInfo] = {}
        self.node_registry: Dict[str, datetime] = {}
        self.discovery_server: Optional[asyncio.Server] = None
        
    async def start_discovery(self) -> bool:
        """Start node discovery service."""
        try:
            self.discovery_server = await asyncio.start_server(
                self._handle_discovery,
                '0.0.0.0',
                self.config.discovery_port
            )
            logger.info(f"Discovery service started on port {self.config.discovery_port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start discovery service: {e}")
            return False
    
    async def stop_discovery(self) -> bool:
        """Stop node discovery service."""
        try:
            if self.discovery_server:
                self.discovery_server.close()
                await self.discovery_server.wait_closed()
            return True
        except Exception as e:
            logger.error(f"Failed to stop discovery service: {e}")
            return False
    
    async def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new node."""
        try:
            self.known_nodes[node_info.node_id] = node_info
            self.node_registry[node_info.node_id] = datetime.now()
            logger.info(f"Registered node: {node_info.node_name} ({node_info.node_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
            return False
    
    async def unregister_node(self, node_id: str) -> bool:
        """Unregister a node."""
        try:
            if node_id in self.known_nodes:
                del self.known_nodes[node_id]
            if node_id in self.node_registry:
                del self.node_registry[node_id]
            logger.info(f"Unregistered node: {node_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to unregister node: {e}")
            return False
    
    async def update_node_status(self, node_id: str, status: NodeStatus, **kwargs) -> bool:
        """Update node status and metrics."""
        try:
            if node_id in self.known_nodes:
                node = self.known_nodes[node_id]
                node.node_status = status
                node.last_heartbeat = datetime.now()
                
                for key, value in kwargs.items():
                    if hasattr(node, key):
                        setattr(node, key, value)
                
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to update node status: {e}")
            return False
    
    async def get_available_nodes(self) -> List[NodeInfo]:
        """Get list of available nodes."""
        current_time = datetime.now()
        available_nodes = []
        
        for node in self.known_nodes.values():
            # Check if node is online and not timed out
            if (node.node_status == NodeStatus.ONLINE and 
                (current_time - node.last_heartbeat).total_seconds() < self.config.node_timeout):
                available_nodes.append(node)
        
        return available_nodes
    
    async def _handle_discovery(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle node discovery requests."""
        try:
            data = await reader.read(1024)
            message = json.loads(data.decode())
            
            if message.get('type') == 'register':
                node_info = NodeInfo(**message['node_info'])
                await self.register_node(node_info)
                response = {'status': 'success', 'message': 'Node registered'}
            elif message.get('type') == 'heartbeat':
                node_id = message['node_id']
                status = NodeStatus(message['status'])
                await self.update_node_status(node_id, status, **message.get('metrics', {}))
                response = {'status': 'success', 'message': 'Heartbeat received'}
            else:
                response = {'status': 'error', 'message': 'Unknown message type'}
            
            writer.write(json.dumps(response).encode())
            await writer.drain()
            
        except Exception as e:
            logger.error(f"Error handling discovery request: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

# ============================================================================
# LOAD BALANCING
# ============================================================================

class LoadBalancer:
    """Intelligent load balancer for task distribution."""
    
    def __init__(self, config: DistributedProcessingConfig):
        self.config = config
        self.strategy = config.load_balancing_strategy
        self.node_weights: Dict[str, float] = {}
        self.node_loads: Dict[str, int] = {}
        self.last_assignment: Dict[str, datetime] = {}
        
    async def select_node(self, task: DistributedTask, available_nodes: List[NodeInfo]) -> Optional[str]:
        """Select the best node for a task based on load balancing strategy."""
        if not available_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return await self._round_robin_selection(available_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return await self._least_connections_selection(available_nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return await self._weighted_round_robin_selection(available_nodes)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return await self._consistent_hash_selection(task, available_nodes)
        elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
            return await self._adaptive_selection(task, available_nodes)
        else:
            return await self._round_robin_selection(available_nodes)
    
    async def _round_robin_selection(self, available_nodes: List[NodeInfo]) -> str:
        """Round-robin node selection."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        node = available_nodes[self._round_robin_index % len(available_nodes)]
        self._round_robin_index += 1
        return node.node_id
    
    async def _least_connections_selection(self, available_nodes: List[NodeInfo]) -> str:
        """Select node with least current load."""
        return min(available_nodes, key=lambda n: n.current_load).node_id
    
    async def _weighted_round_robin_selection(self, available_nodes: List[NodeInfo]) -> str:
        """Weighted round-robin selection based on node capacity."""
        total_weight = sum(node.node_weight for node in available_nodes)
        if total_weight == 0:
            return await self._round_robin_selection(available_nodes)
        
        # Simple weighted selection
        weights = [node.node_weight for node in available_nodes]
        selected = await self._weighted_random_selection(weights)
        return available_nodes[selected].node_id
    
    async def _consistent_hash_selection(self, task: DistributedTask, available_nodes: List[NodeInfo]) -> str:
        """Consistent hashing for task distribution."""
        if not available_nodes:
            return None
        
        # Simple consistent hashing based on task ID
        hash_value = int(hashlib.md5(task.task_id.encode()).hexdigest(), 16)
        node_index = hash_value % len(available_nodes)
        return available_nodes[node_index].node_id
    
    async def _adaptive_selection(self, task: DistributedTask, available_nodes: List[NodeInfo]) -> str:
        """Adaptive selection based on multiple factors."""
        best_node = None
        best_score = float('-inf')
        
        for node in available_nodes:
            # Calculate node score based on multiple factors
            load_score = 1.0 - (node.current_load / node.node_capacity)
            weight_score = node.node_weight
            health_score = 1.0 if node.node_status == NodeStatus.ONLINE else 0.0
            
            # Combine scores
            total_score = (load_score * 0.4 + 
                          weight_score * 0.3 + 
                          health_score * 0.3)
            
            if total_score > best_score:
                best_score = total_score
                best_node = node
        
        return best_node.node_id if best_node else None
    
    async def _weighted_random_selection(self, weights: List[float]) -> int:
        """Weighted random selection."""
        total_weight = sum(weights)
        if total_weight == 0:
            return 0
        
        random_value = asyncio.get_event_loop().time() % total_weight
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return i
        
        return len(weights) - 1

# ============================================================================
# TASK SCHEDULER
# ============================================================================

class TaskScheduler:
    """Manages task scheduling and execution."""
    
    def __init__(self, config: DistributedProcessingConfig):
        self.config = config
        self.pending_tasks: List[DistributedTask] = []
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.failed_tasks: Dict[str, DistributedTask] = {}
        self.task_dependencies: Dict[str, Set[str]] = {}
        
    async def submit_task(self, task: DistributedTask) -> bool:
        """Submit a new task for execution."""
        try:
            # Check dependencies
            if not await self._check_dependencies(task):
                self.pending_tasks.append(task)
                return True
            
            # Task is ready to run
            self.pending_tasks.append(task)
            return True
        except Exception as e:
            logger.error(f"Failed to submit task {task.task_id}: {e}")
            return False
    
    async def get_next_task(self) -> Optional[DistributedTask]:
        """Get next available task based on priority and dependencies."""
        if not self.pending_tasks:
            return None
        
        # Sort by priority and creation time
        self.pending_tasks.sort(key=lambda t: (t.priority.value, t.created_at))
        
        for task in self.pending_tasks:
            if await self._check_dependencies(task):
                self.pending_tasks.remove(task)
                return task
        
        return None
    
    async def start_task(self, task: DistributedTask) -> bool:
        """Mark task as started."""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            self.running_tasks[task.task_id] = task
            return True
        except Exception as e:
            logger.error(f"Failed to start task {task.task_id}: {e}")
            return False
    
    async def complete_task(self, task_id: str, result: Any) -> bool:
        """Mark task as completed."""
        try:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result
                
                del self.running_tasks[task_id]
                self.completed_tasks[task_id] = task
                
                # Check dependent tasks
                await self._check_dependent_tasks(task_id)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            return False
    
    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed."""
        try:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                task.error = error
                
                # Retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.started_at = None
                    self.pending_tasks.append(task)
                else:
                    self.failed_tasks[task_id] = task
                
                del self.running_tasks[task_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to fail task {task_id}: {e}")
            return False
    
    async def _check_dependencies(self, task: DistributedTask) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    async def _check_dependent_tasks(self, completed_task_id: str):
        """Check if any dependent tasks are now ready to run."""
        for task in self.pending_tasks[:]:  # Copy list to avoid modification during iteration
            if completed_task_id in task.dependencies:
                if await self._check_dependencies(task):
                    # Move task to front of queue
                    self.pending_tasks.remove(task)
                    self.pending_tasks.insert(0, task)

# ============================================================================
# FAULT TOLERANCE
# ============================================================================

class FaultToleranceManager:
    """Manages fault tolerance and recovery strategies."""
    
    def __init__(self, config: DistributedProcessingConfig):
        self.config = config
        self.strategy = config.fault_tolerance_strategy
        self.failure_counts: Dict[str, int] = {}
        self.circuit_breaker_states: Dict[str, str] = {}  # 'closed', 'open', 'half_open'
        self.circuit_breaker_timestamps: Dict[str, datetime] = {}
        
    async def handle_node_failure(self, node_id: str) -> bool:
        """Handle node failure and implement recovery strategy."""
        try:
            if self.strategy == FaultToleranceStrategy.CIRCUIT_BREAKER:
                return await self._handle_circuit_breaker(node_id)
            elif self.strategy == FaultToleranceStrategy.REPLICATION:
                return await self._handle_replication(node_id)
            elif self.strategy == FaultToleranceStrategy.CHECKPOINTING:
                return await self._handle_checkpointing(node_id)
            else:
                return await self._handle_retry(node_id)
        except Exception as e:
            logger.error(f"Failed to handle node failure: {e}")
            return False
    
    async def _handle_circuit_breaker(self, node_id: str) -> bool:
        """Handle failure using circuit breaker pattern."""
        current_time = datetime.now()
        
        # Increment failure count
        self.failure_counts[node_id] = self.failure_counts.get(node_id, 0) + 1
        
        if self.failure_counts[node_id] >= self.config.circuit_breaker_threshold:
            # Open circuit breaker
            self.circuit_breaker_states[node_id] = 'open'
            self.circuit_breaker_timestamps[node_id] = current_time
            logger.warning(f"Circuit breaker opened for node {node_id}")
            return False
        
        return True
    
    async def _handle_replication(self, node_id: str) -> bool:
        """Handle failure using task replication."""
        # This would replicate tasks to other nodes
        logger.info(f"Replicating tasks from failed node {node_id}")
        return True
    
    async def _handle_checkpointing(self, node_id: str) -> bool:
        """Handle failure using checkpointing."""
        # This would restore tasks from checkpoints
        logger.info(f"Restoring tasks from checkpoints for node {node_id}")
        return True
    
    async def _handle_retry(self, node_id: str) -> bool:
        """Handle failure using retry mechanism."""
        # This would retry failed tasks
        logger.info(f"Retrying tasks for node {node_id}")
        return True
    
    async def is_node_available(self, node_id: str) -> bool:
        """Check if node is available (circuit breaker not open)."""
        if node_id not in self.circuit_breaker_states:
            return True
        
        state = self.circuit_breaker_states[node_id]
        if state == 'closed':
            return True
        elif state == 'open':
            # Check if timeout has passed
            if node_id in self.circuit_breaker_timestamps:
                timeout_passed = (datetime.now() - self.circuit_breaker_timestamps[node_id]).total_seconds() > self.config.circuit_breaker_timeout
                if timeout_passed:
                    # Move to half-open state
                    self.circuit_breaker_states[node_id] = 'half_open'
                    return True
            return False
        elif state == 'half_open':
            return True
        
        return True
    
    async def record_success(self, node_id: str) -> bool:
        """Record successful operation for circuit breaker."""
        try:
            if node_id in self.circuit_breaker_states:
                # Reset circuit breaker
                self.circuit_breaker_states[node_id] = 'closed'
                self.failure_counts[node_id] = 0
                if node_id in self.circuit_breaker_timestamps:
                    del self.circuit_breaker_timestamps[node_id]
            return True
        except Exception as e:
            logger.error(f"Failed to record success for node {node_id}: {e}")
            return False

# ============================================================================
# MAIN DISTRIBUTED PROCESSING MODULE
# ============================================================================

class DistributedProcessingModule(BaseModule):
    """Comprehensive distributed processing module for Blaze AI system."""
    
    def __init__(self, config: DistributedProcessingConfig):
        super().__init__(config)
        self.config = config
        self.metrics = ClusterMetrics()
        
        # Initialize components
        self.network_protocol = HTTPProtocol(config)
        self.node_discovery = NodeDiscovery(config)
        self.load_balancer = LoadBalancer(config)
        self.task_scheduler = TaskScheduler(config)
        self.fault_tolerance = FaultToleranceManager(config)
        
        # Node and task management
        self.local_node: Optional[NodeInfo] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.scaling_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> bool:
        """Initialize the Distributed Processing module."""
        try:
            await super().initialize()
            
            # Initialize local node
            await self._initialize_local_node()
            
            # Start network services
            await self.network_protocol.start_listening()
            await self.node_discovery.start_discovery()
            
            # Start background tasks
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            if self.config.enable_auto_scaling:
                self.scaling_task = asyncio.create_task(self._auto_scaling_loop())
            
            self.status = ModuleStatus.ACTIVE
            logger.info("Distributed Processing module initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Distributed Processing module: {e}")
            self.status = ModuleStatus.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the Distributed Processing module."""
        try:
            # Stop background tasks
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()
            if self.scaling_task:
                self.scaling_task.cancel()
            
            # Stop network services
            await self.network_protocol.stop_listening()
            await self.node_discovery.stop_discovery()
            
            await super().shutdown()
            logger.info("Distributed Processing module shutdown successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during Distributed Processing module shutdown: {e}")
            return False
    
    async def submit_distributed_task(self, task_type: str, task_data: Dict[str, Any], 
                                    priority: TaskPriority = TaskPriority.NORMAL,
                                    timeout: float = None) -> Optional[str]:
        """Submit a task for distributed execution."""
        try:
            task = DistributedTask(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                task_type=task_type,
                task_data=task_data,
                priority=priority,
                status=TaskStatus.PENDING,
                source_node=self.config.node_id,
                timeout=timeout or self.config.task_timeout,
                max_retries=self.config.max_task_retries
            )
            
            success = await self.task_scheduler.submit_task(task)
            if success:
                logger.info(f"Submitted distributed task: {task.task_id}")
                return task.task_id
            else:
                logger.error(f"Failed to submit distributed task: {task.task_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to submit distributed task: {e}")
            return None
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get the status of a distributed task."""
        try:
            # Check running tasks
            if task_id in self.task_scheduler.running_tasks:
                return self.task_scheduler.running_tasks[task_id].status
            
            # Check completed tasks
            if task_id in self.task_scheduler.completed_tasks:
                return self.task_scheduler.completed_tasks[task_id].status
            
            # Check failed tasks
            if task_id in self.task_scheduler.failed_tasks:
                return self.task_scheduler.failed_tasks[task_id].status
            
            # Check pending tasks
            for task in self.task_scheduler.pending_tasks:
                if task.task_id == task_id:
                    return task.status
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return None
    
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get the result of a completed task."""
        try:
            if task_id in self.task_scheduler.completed_tasks:
                return self.task_scheduler.completed_tasks[task_id].result
            return None
        except Exception as e:
            logger.error(f"Failed to get task result: {e}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task."""
        try:
            # Check pending tasks
            for task in self.task_scheduler.pending_tasks[:]:
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    self.task_scheduler.pending_tasks.remove(task)
                    return True
            
            # Check running tasks
            if task_id in self.task_scheduler.running_tasks:
                task = self.task_scheduler.running_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                del self.task_scheduler.running_tasks[task_id]
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return False
    
    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get current cluster status and metrics."""
        try:
            available_nodes = await self.node_discovery.get_available_nodes()
            
            # Update metrics
            self.metrics.total_nodes = len(self.node_discovery.known_nodes)
            self.metrics.active_nodes = len(available_nodes)
            self.metrics.total_tasks = (len(self.task_scheduler.pending_tasks) + 
                                      len(self.task_scheduler.running_tasks) + 
                                      len(self.task_scheduler.completed_tasks) + 
                                      len(self.task_scheduler.failed_tasks))
            self.metrics.completed_tasks = len(self.task_scheduler.completed_tasks)
            self.metrics.failed_tasks = len(self.task_scheduler.failed_tasks)
            
            # Calculate cluster load
            total_capacity = sum(node.node_capacity for node in available_nodes)
            total_load = sum(node.current_load for node in available_nodes)
            self.metrics.cluster_load = total_load / total_capacity if total_capacity > 0 else 0.0
            
            self.metrics.last_updated = datetime.now()
            
            return {
                'metrics': self.metrics,
                'nodes': [node.__dict__ for node in available_nodes],
                'tasks': {
                    'pending': len(self.task_scheduler.pending_tasks),
                    'running': len(self.task_scheduler.running_tasks),
                    'completed': len(self.task_scheduler.completed_tasks),
                    'failed': len(self.task_scheduler.failed_tasks)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get cluster status: {e}")
            return {}
    
    async def _initialize_local_node(self):
        """Initialize local node information."""
        self.local_node = NodeInfo(
            node_id=self.config.node_id,
            node_name=self.config.node_name,
            node_address="127.0.0.1",
            node_port=self.config.communication_port,
            node_capacity=self.config.node_capacity,
            node_weight=self.config.node_weight,
            node_status=NodeStatus.ONLINE
        )
        
        # Register with discovery service
        await self.node_discovery.register_node(self.local_node)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to discovery service."""
        while self.status == ModuleStatus.ACTIVE:
            try:
                if self.local_node:
                    # Update local node metrics
                    self.local_node.current_load = len(self.task_scheduler.running_tasks)
                    self.local_node.last_heartbeat = datetime.now()
                    
                    # Send heartbeat
                    await self.network_protocol.send_message(
                        f"127.0.0.1:{self.config.discovery_port}",
                        {
                            'type': 'heartbeat',
                            'node_id': self.local_node.node_id,
                            'status': self.local_node.node_status.value,
                            'metrics': {
                                'current_load': self.local_node.current_load,
                                'cpu_usage': self.local_node.cpu_usage,
                                'memory_usage': self.local_node.memory_usage
                            }
                        }
                    )
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    async def _cleanup_loop(self):
        """Clean up expired tasks and nodes."""
        while self.status == ModuleStatus.ACTIVE:
            try:
                current_time = datetime.now()
                
                # Clean up expired tasks
                for task_id, task in list(self.task_scheduler.running_tasks.items()):
                    if task.started_at and (current_time - task.started_at).total_seconds() > task.timeout:
                        await self.task_scheduler.fail_task(task_id, "Task timeout")
                
                # Clean up stale nodes
                for node_id, last_seen in list(self.node_discovery.node_registry.items()):
                    if (current_time - last_seen).total_seconds() > self.config.node_timeout:
                        await self.node_discovery.unregister_node(node_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(30)
    
    async def _auto_scaling_loop(self):
        """Automatic scaling based on cluster load."""
        while self.status == ModuleStatus.ACTIVE:
            try:
                cluster_status = await self.get_cluster_status()
                cluster_load = cluster_status.get('metrics', {}).get('cluster_load', 0.0)
                
                if cluster_load > self.config.scaling_threshold:
                    # High load - consider scaling up
                    logger.info(f"High cluster load detected: {cluster_load:.2f}")
                    # This would trigger node creation or task redistribution
                
                elif cluster_load < (1.0 - self.config.scaling_threshold):
                    # Low load - consider scaling down
                    logger.info(f"Low cluster load detected: {cluster_load:.2f}")
                    # This would trigger node shutdown or task consolidation
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Auto-scaling loop error: {e}")
                await asyncio.sleep(60)
    
    async def get_metrics(self) -> ClusterMetrics:
        """Get cluster metrics."""
        return self.metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Check module health."""
        health = await super().health_check()
        health["total_nodes"] = self.metrics.total_nodes
        health["active_nodes"] = self.metrics.active_nodes
        health["total_tasks"] = self.metrics.total_tasks
        health["cluster_load"] = self.metrics.cluster_load
        return health

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_distributed_processing_module(config: Optional[DistributedProcessingConfig] = None) -> DistributedProcessingModule:
    """Create Distributed Processing module."""
    if config is None:
        config = DistributedProcessingConfig()
    return DistributedProcessingModule(config)

def create_distributed_processing_module_with_defaults(**kwargs) -> DistributedProcessingModule:
    """Create Distributed Processing module with default configuration."""
    config = DistributedProcessingConfig(**kwargs)
    return DistributedProcessingModule(config)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    "NodeStatus",
    "TaskStatus", 
    "TaskPriority",
    "LoadBalancingStrategy",
    "FaultToleranceStrategy",
    
    # Configuration and Data Classes
    "DistributedProcessingConfig",
    "NodeInfo",
    "DistributedTask",
    "ClusterMetrics",
    
    # Network and Discovery
    "NetworkProtocol",
    "HTTPProtocol",
    "NodeDiscovery",
    
    # Load Balancing and Scheduling
    "LoadBalancer",
    "TaskScheduler",
    
    # Fault Tolerance
    "FaultToleranceManager",
    
    # Main Module
    "DistributedProcessingModule",
    
    # Factory Functions
    "create_distributed_processing_module",
    "create_distributed_processing_module_with_defaults"
]

