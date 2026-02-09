"""
Distributed Execution Module - Distributed benchmark execution.

Provides:
- Multi-node execution
- Task distribution
- Result aggregation
- Fault tolerance
"""

import logging
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Node status."""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class Node:
    """Execution node."""
    id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.IDLE
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "current_task": self.current_task,
            "last_heartbeat": self.last_heartbeat,
        }


@dataclass
class DistributedTask:
    """Distributed task."""
    id: str
    model_name: str
    benchmark_name: str
    node_id: Optional[str] = None
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "model_name": self.model_name,
            "benchmark_name": self.benchmark_name,
            "node_id": self.node_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }


class DistributedExecutor:
    """Distributed execution coordinator."""
    
    def __init__(self):
        """Initialize distributed executor."""
        self.nodes: Dict[str, Node] = {}
        self.tasks: Dict[str, DistributedTask] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    def register_node(
        self,
        node_id: str,
        host: str,
        port: int,
        capabilities: Optional[List[str]] = None,
    ) -> Node:
        """
        Register a new node.
        
        Args:
            node_id: Node identifier
            host: Node host
            port: Node port
            capabilities: Node capabilities
            
        Returns:
            Registered node
        """
        node = Node(
            id=node_id,
            host=host,
            port=port,
            capabilities=capabilities or [],
        )
        
        with self.lock:
            self.nodes[node_id] = node
        
        logger.info(f"Registered node: {node_id} ({host}:{port})")
        return node
    
    def unregister_node(self, node_id: str) -> None:
        """
        Unregister a node.
        
        Args:
            node_id: Node identifier
        """
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Unregistered node: {node_id}")
    
    def update_node_status(
        self,
        node_id: str,
        status: NodeStatus,
        current_task: Optional[str] = None,
    ) -> None:
        """
        Update node status.
        
        Args:
            node_id: Node identifier
            status: New status
            current_task: Current task ID
        """
        with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = status
                self.nodes[node_id].current_task = current_task
                self.nodes[node_id].last_heartbeat = datetime.now().isoformat()
    
    def get_available_nodes(self) -> List[Node]:
        """Get available nodes."""
        with self.lock:
            return [
                node for node in self.nodes.values()
                if node.status == NodeStatus.IDLE
            ]
    
    def create_task(
        self,
        model_name: str,
        benchmark_name: str,
    ) -> DistributedTask:
        """
        Create a distributed task.
        
        Args:
            model_name: Model name
            benchmark_name: Benchmark name
            
        Returns:
            Created task
        """
        task_id = f"{model_name}_{benchmark_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = DistributedTask(
            id=task_id,
            model_name=model_name,
            benchmark_name=benchmark_name,
        )
        
        with self.lock:
            self.tasks[task_id] = task
        
        logger.info(f"Created task: {task_id}")
        return task
    
    def assign_task(self, task_id: str, node_id: str) -> bool:
        """
        Assign task to node.
        
        Args:
            task_id: Task ID
            node_id: Node ID
            
        Returns:
            True if assigned successfully
        """
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            if node.status != NodeStatus.IDLE:
                return False
            
            task = self.tasks[task_id]
            task.node_id = node_id
            task.status = "assigned"
            node.status = NodeStatus.BUSY
            node.current_task = task_id
        
        logger.info(f"Assigned task {task_id} to node {node_id}")
        return True
    
    def complete_task(
        self,
        task_id: str,
        result: Dict[str, Any],
    ) -> DistributedTask:
        """
        Mark task as completed.
        
        Args:
            task_id: Task ID
            result: Task result
            
        Returns:
            Updated task
        """
        with self.lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            task.status = "completed"
            task.completed_at = datetime.now().isoformat()
            task.result = result
            
            if task.node_id:
                node = self.nodes.get(task.node_id)
                if node:
                    node.status = NodeStatus.IDLE
                    node.current_task = None
        
        logger.info(f"Completed task: {task_id}")
        return task
    
    def fail_task(
        self,
        task_id: str,
        error: str,
    ) -> DistributedTask:
        """
        Mark task as failed.
        
        Args:
            task_id: Task ID
            error: Error message
            
        Returns:
            Updated task
        """
        with self.lock:
            if task_id not in self.tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task = self.tasks[task_id]
            task.status = "failed"
            task.completed_at = datetime.now().isoformat()
            task.error = error
            
            if task.node_id:
                node = self.nodes.get(task.node_id)
                if node:
                    node.status = NodeStatus.IDLE
                    node.current_task = None
        
        logger.error(f"Failed task: {task_id} - {error}")
        return task
    
    def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """Get task by ID."""
        with self.lock:
            return self.tasks.get(task_id)
    
    def get_node_tasks(self, node_id: str) -> List[DistributedTask]:
        """Get tasks for a node."""
        with self.lock:
            return [
                task for task in self.tasks.values()
                if task.node_id == node_id
            ]
    
    def distribute_tasks(
        self,
        tasks: List[DistributedTask],
        strategy: str = "round_robin",
    ) -> Dict[str, str]:
        """
        Distribute tasks to available nodes.
        
        Args:
            tasks: List of tasks
            strategy: Distribution strategy (round_robin, least_busy, random)
            
        Returns:
            Dictionary mapping task_id to node_id
        """
        available_nodes = self.get_available_nodes()
        
        if not available_nodes:
            logger.warning("No available nodes for task distribution")
            return {}
        
        assignments = {}
        
        if strategy == "round_robin":
            for i, task in enumerate(tasks):
                node = available_nodes[i % len(available_nodes)]
                if self.assign_task(task.id, node.id):
                    assignments[task.id] = node.id
        
        return assignments
    
    def aggregate_results(
        self,
        task_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Aggregate results from multiple tasks.
        
        Args:
            task_ids: List of task IDs
            
        Returns:
            Aggregated results
        """
        results = []
        errors = []
        
        for task_id in task_ids:
            task = self.get_task(task_id)
            if task and task.result:
                results.append(task.result)
            elif task and task.error:
                errors.append(task.error)
        
        if not results:
            return {
                "success": False,
                "error": "No successful results",
                "errors": errors,
            }
        
        # Aggregate metrics
        aggregated = {
            "success": True,
            "total_tasks": len(task_ids),
            "successful_tasks": len(results),
            "failed_tasks": len(errors),
            "results": results,
            "aggregated_metrics": {
                "avg_accuracy": sum(r.get("accuracy", 0.0) for r in results) / len(results),
                "avg_throughput": sum(r.get("throughput", 0.0) for r in results) / len(results),
            },
        }
        
        if errors:
            aggregated["errors"] = errors
        
        return aggregated












