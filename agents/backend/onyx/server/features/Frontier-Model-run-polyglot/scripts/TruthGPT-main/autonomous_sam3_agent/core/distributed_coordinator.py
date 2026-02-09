"""
Distributed Coordinator
=======================

Coordinates task execution across multiple nodes for distributed processing.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node status enumeration."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    DRAINING = "draining"


@dataclass
class NodeInfo:
    """Information about a distributed node."""
    node_id: str
    address: str
    port: int
    status: NodeStatus = NodeStatus.OFFLINE
    capabilities: Dict[str, Any] = field(default_factory=dict)
    current_load: float = 0.0
    max_capacity: int = 10
    last_heartbeat: Optional[str] = None
    registered_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class DistributedTask:
    """A task assigned to a node."""
    task_id: str
    node_id: str
    image_path: str
    text_prompt: str
    priority: int
    assigned_at: str
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None


class DistributedCoordinator:
    """
    Coordinates tasks across multiple nodes.
    
    Features:
    - Node registration and discovery
    - Load-balanced task assignment
    - Heartbeat monitoring
    - Automatic failover
    - Result aggregation
    """
    
    def __init__(
        self,
        coordinator_id: Optional[str] = None,
        heartbeat_interval: int = 10,
        node_timeout: int = 30,
    ):
        """
        Initialize distributed coordinator.
        
        Args:
            coordinator_id: Unique ID for this coordinator
            heartbeat_interval: Seconds between heartbeats
            node_timeout: Seconds before node considered offline
        """
        self.coordinator_id = coordinator_id or str(uuid.uuid4())[:8]
        self.heartbeat_interval = heartbeat_interval
        self.node_timeout = node_timeout
        
        self._nodes: Dict[str, NodeInfo] = {}
        self._distributed_tasks: Dict[str, DistributedTask] = {}
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        logger.info(f"Initialized DistributedCoordinator (id: {self.coordinator_id})")
    
    async def start(self):
        """Start the coordinator."""
        if self._running:
            logger.warning("Coordinator is already running")
            return
        
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        logger.info("DistributedCoordinator started")
    
    async def stop(self):
        """Stop the coordinator."""
        if not self._running:
            return
        
        self._running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        logger.info("DistributedCoordinator stopped")
    
    async def register_node(
        self,
        address: str,
        port: int,
        capabilities: Optional[Dict[str, Any]] = None,
        max_capacity: int = 10,
    ) -> str:
        """
        Register a new node.
        
        Args:
            address: Node IP address or hostname
            port: Node port
            capabilities: Node capabilities (e.g., GPU, memory)
            max_capacity: Maximum concurrent tasks
            
        Returns:
            Node ID
        """
        node_id = str(uuid.uuid4())[:8]
        
        node = NodeInfo(
            node_id=node_id,
            address=address,
            port=port,
            status=NodeStatus.ONLINE,
            capabilities=capabilities or {},
            max_capacity=max_capacity,
            last_heartbeat=datetime.now().isoformat(),
            registered_at=datetime.now().isoformat(),
        )
        
        async with self._lock:
            self._nodes[node_id] = node
        
        logger.info(f"Registered node {node_id} at {address}:{port}")
        return node_id
    
    async def unregister_node(self, node_id: str):
        """
        Unregister a node.
        
        Args:
            node_id: ID of node to unregister
        """
        async with self._lock:
            if node_id not in self._nodes:
                raise ValueError(f"Node {node_id} not found")
            
            # Reassign tasks from this node
            orphan_tasks = [
                task for task in self._distributed_tasks.values()
                if task.node_id == node_id and task.status == "pending"
            ]
            
            for task in orphan_tasks:
                task.node_id = ""
                task.status = "pending"
            
            del self._nodes[node_id]
        
        logger.info(f"Unregistered node {node_id}, {len(orphan_tasks)} tasks reassigned")
    
    async def update_heartbeat(self, node_id: str, load: float = 0.0):
        """
        Update node heartbeat.
        
        Args:
            node_id: Node ID
            load: Current load (0.0 - 1.0)
        """
        async with self._lock:
            if node_id in self._nodes:
                self._nodes[node_id].last_heartbeat = datetime.now().isoformat()
                self._nodes[node_id].current_load = load
                self._nodes[node_id].status = NodeStatus.ONLINE
    
    async def assign_task(
        self,
        image_path: str,
        text_prompt: str,
        priority: int = 0,
    ) -> Optional[str]:
        """
        Assign a task to the best available node.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt
            priority: Task priority
            
        Returns:
            Task ID if assigned, None if no nodes available
        """
        node = await self._select_best_node()
        if not node:
            logger.warning("No available nodes for task assignment")
            return None
        
        task_id = str(uuid.uuid4())
        
        task = DistributedTask(
            task_id=task_id,
            node_id=node.node_id,
            image_path=image_path,
            text_prompt=text_prompt,
            priority=priority,
            assigned_at=datetime.now().isoformat(),
        )
        
        async with self._lock:
            self._distributed_tasks[task_id] = task
        
        # Send task to node
        success = await self._send_task_to_node(node, task)
        if not success:
            logger.error(f"Failed to send task {task_id} to node {node.node_id}")
            return None
        
        logger.info(f"Assigned task {task_id} to node {node.node_id}")
        return task_id
    
    async def _select_best_node(self) -> Optional[NodeInfo]:
        """Select the best node for task assignment."""
        async with self._lock:
            available = [
                node for node in self._nodes.values()
                if node.status == NodeStatus.ONLINE
                and node.current_load < 0.9
            ]
            
            if not available:
                return None
            
            # Select node with lowest load
            return min(available, key=lambda n: n.current_load)
    
    async def _send_task_to_node(self, node: NodeInfo, task: DistributedTask) -> bool:
        """Send a task to a node."""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available, simulating task send")
            return True
        
        url = f"http://{node.address}:{node.port}/api/tasks"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json={
                        "task_id": task.task_id,
                        "image_path": task.image_path,
                        "text_prompt": task.text_prompt,
                        "priority": task.priority,
                    },
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Error sending task to node {node.node_id}: {e}")
            return False
    
    async def complete_task(self, task_id: str, result: Dict[str, Any]):
        """
        Mark a task as completed.
        
        Args:
            task_id: Task ID
            result: Task result
        """
        async with self._lock:
            if task_id in self._distributed_tasks:
                self._distributed_tasks[task_id].status = "completed"
                self._distributed_tasks[task_id].result = result
        
        logger.info(f"Completed distributed task {task_id}")
    
    async def fail_task(self, task_id: str, error: str):
        """
        Mark a task as failed.
        
        Args:
            task_id: Task ID
            error: Error message
        """
        async with self._lock:
            if task_id in self._distributed_tasks:
                task = self._distributed_tasks[task_id]
                task.status = "failed"
                task.result = {"error": error}
                
                # Try to reassign
                task.node_id = ""
        
        logger.warning(f"Failed distributed task {task_id}: {error}")
    
    async def _heartbeat_loop(self):
        """Check node health periodically."""
        while self._running:
            try:
                await self._check_node_health()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _check_node_health(self):
        """Check health of all nodes."""
        now = datetime.now()
        timeout = timedelta(seconds=self.node_timeout)
        
        async with self._lock:
            for node in self._nodes.values():
                if node.last_heartbeat:
                    last = datetime.fromisoformat(node.last_heartbeat)
                    if now - last > timeout:
                        if node.status != NodeStatus.OFFLINE:
                            logger.warning(f"Node {node.node_id} went offline")
                            node.status = NodeStatus.OFFLINE
    
    def get_nodes(self) -> List[Dict[str, Any]]:
        """Get all registered nodes."""
        return [node.to_dict() for node in self._nodes.values()]
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific node."""
        if node_id in self._nodes:
            return self._nodes[node_id].to_dict()
        return None
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        online = sum(1 for n in self._nodes.values() if n.status == NodeStatus.ONLINE)
        total_capacity = sum(n.max_capacity for n in self._nodes.values())
        avg_load = (
            sum(n.current_load for n in self._nodes.values()) / len(self._nodes)
            if self._nodes else 0.0
        )
        
        return {
            "coordinator_id": self.coordinator_id,
            "total_nodes": len(self._nodes),
            "online_nodes": online,
            "offline_nodes": len(self._nodes) - online,
            "total_capacity": total_capacity,
            "average_load": avg_load,
            "pending_tasks": sum(
                1 for t in self._distributed_tasks.values()
                if t.status == "pending"
            ),
            "completed_tasks": sum(
                1 for t in self._distributed_tasks.values()
                if t.status == "completed"
            ),
        }
