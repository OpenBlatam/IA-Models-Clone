"""
Edge Computing System for Facebook Posts
Ultra-low latency processing with distributed edge nodes
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict, deque
import aiohttp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import pickle
import hashlib

logger = logging.getLogger(__name__)


# Pure functions for edge computing

class EdgeNodeStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    OVERLOADED = "overloaded"


class ProcessingPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BATCH = "batch"


class EdgeTaskType(str, Enum):
    CONTENT_ANALYSIS = "content_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    KEYWORD_EXTRACTION = "keyword_extraction"
    READABILITY_CHECK = "readability_check"
    CACHE_UPDATE = "cache_update"
    REAL_TIME_PREDICTION = "real_time_prediction"
    IMAGE_PROCESSING = "image_processing"
    AUDIO_PROCESSING = "audio_processing"


@dataclass(frozen=True)
class EdgeNode:
    """Immutable edge node - pure data structure"""
    node_id: str
    location: str
    ip_address: str
    port: int
    status: EdgeNodeStatus
    capabilities: List[EdgeTaskType]
    cpu_usage: float
    memory_usage: float
    network_latency: float
    last_heartbeat: datetime
    processing_capacity: int
    current_load: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "node_id": node_id,
            "location": location,
            "ip_address": ip_address,
            "port": port,
            "status": status.value,
            "capabilities": [cap.value for cap in capabilities],
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "network_latency": network_latency,
            "last_heartbeat": last_heartbeat.isoformat(),
            "processing_capacity": processing_capacity,
            "current_load": current_load
        }


@dataclass(frozen=True)
class EdgeTask:
    """Immutable edge task - pure data structure"""
    task_id: str
    task_type: EdgeTaskType
    priority: ProcessingPriority
    data: Any
    target_location: Optional[str]
    timeout: float
    retry_count: int
    created_at: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "task_id": task_id,
            "task_type": task_type.value,
            "priority": priority.value,
            "data": data,
            "target_location": target_location,
            "timeout": timeout,
            "retry_count": retry_count,
            "created_at": created_at.isoformat(),
            "metadata": metadata
        }


@dataclass(frozen=True)
class EdgeTaskResult:
    """Immutable edge task result - pure data structure"""
    task_id: str
    success: bool
    result: Any
    processing_time: float
    node_id: str
    error_message: Optional[str]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary - pure function"""
        return {
            "task_id": task_id,
            "success": success,
            "result": result,
            "processing_time": processing_time,
            "node_id": node_id,
            "error_message": error_message,
            "timestamp": timestamp.isoformat()
        }


def calculate_node_score(node: EdgeNode, task_type: EdgeTaskType, user_location: Optional[str] = None) -> float:
    """Calculate node score for task assignment - pure function"""
    # Base score
    score = 1.0
    
    # Check if node supports the task type
    if task_type not in node.capabilities:
        return 0.0
    
    # Check if node is online
    if node.status != EdgeNodeStatus.ONLINE:
        return 0.0
    
    # CPU usage factor (lower is better)
    cpu_factor = max(0.1, 1.0 - node.cpu_usage)
    score *= cpu_factor
    
    # Memory usage factor (lower is better)
    memory_factor = max(0.1, 1.0 - node.memory_usage)
    score *= memory_factor
    
    # Load factor (lower is better)
    load_factor = max(0.1, 1.0 - (node.current_load / node.processing_capacity))
    score *= load_factor
    
    # Network latency factor (lower is better)
    latency_factor = max(0.1, 1.0 / (1.0 + node.network_latency / 100.0))
    score *= latency_factor
    
    # Location factor (closer is better)
    if user_location and user_location == node.location:
        location_factor = 1.5
    else:
        location_factor = 1.0
    score *= location_factor
    
    return max(0.0, min(10.0, score))


def select_optimal_node(
    nodes: List[EdgeNode],
    task_type: EdgeTaskType,
    user_location: Optional[str] = None
) -> Optional[EdgeNode]:
    """Select optimal node for task - pure function"""
    if not nodes:
        return None
    
    # Calculate scores for all nodes
    node_scores = [
        (node, calculate_node_score(node, task_type, user_location))
        for node in nodes
    ]
    
    # Filter out nodes with zero score
    valid_nodes = [(node, score) for node, score in node_scores if score > 0]
    
    if not valid_nodes:
        return None
    
    # Select node with highest score
    best_node, best_score = max(valid_nodes, key=lambda x: x[1])
    
    return best_node


def create_edge_task(
    task_type: EdgeTaskType,
    data: Any,
    priority: ProcessingPriority = ProcessingPriority.NORMAL,
    target_location: Optional[str] = None,
    timeout: float = 30.0
) -> EdgeTask:
    """Create edge task - pure function"""
    return EdgeTask(
        task_id=f"task_{uuid.uuid4().hex[:8]}",
        task_type=task_type,
        priority=priority,
        data=data,
        target_location=target_location,
        timeout=timeout,
        retry_count=0,
        created_at=datetime.utcnow(),
        metadata={}
    )


def create_edge_node(
    location: str,
    ip_address: str,
    port: int,
    capabilities: List[EdgeTaskType],
    processing_capacity: int = 100
) -> EdgeNode:
    """Create edge node - pure function"""
    return EdgeNode(
        node_id=f"node_{uuid.uuid4().hex[:8]}",
        location=location,
        ip_address=ip_address,
        port=port,
        status=EdgeNodeStatus.ONLINE,
        capabilities=capabilities,
        cpu_usage=0.0,
        memory_usage=0.0,
        network_latency=0.0,
        last_heartbeat=datetime.utcnow(),
        processing_capacity=processing_capacity,
        current_load=0
    )


# Edge Computing System Class

class EdgeComputingSystem:
    """Edge Computing System with distributed processing capabilities"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        
        # Edge nodes registry
        self.edge_nodes: Dict[str, EdgeNode] = {}
        self.nodes_by_location: Dict[str, List[EdgeNode]] = defaultdict(list)
        self.nodes_by_capability: Dict[EdgeTaskType, List[EdgeNode]] = defaultdict(list)
        
        # Task management
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self.active_tasks: Dict[str, EdgeTask] = {}
        self.completed_tasks: deque = deque(maxlen=10000)
        self.failed_tasks: deque = deque(maxlen=1000)
        
        # Processing threads
        self.worker_threads: List[threading.Thread] = []
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # HTTP client for node communication
        self.http_client: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_processing_time": 0.0,
            "nodes_online": 0,
            "total_processing_capacity": 0,
            "current_load": 0
        }
        
        # Task result callbacks
        self.task_callbacks: Dict[str, Callable] = {}
    
    async def start(self) -> None:
        """Start edge computing system"""
        if self.is_running:
            return
        
        try:
            # Initialize HTTP client
            self.http_client = aiohttp.ClientSession()
            
            # Start worker threads
            self.is_running = True
            self.shutdown_event.clear()
            
            for i in range(self.max_workers):
                worker = threading.Thread(
                    target=self._worker_loop,
                    name=f"EdgeWorker-{i}",
                    daemon=True
                )
                worker.start()
                self.worker_threads.append(worker)
            
            # Register default edge nodes
            await self._register_default_nodes()
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._stats_update_loop())
            
            logger.info("Edge computing system started")
            
        except Exception as e:
            logger.error(f"Error starting edge computing system: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop edge computing system"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Wait for worker threads to finish
        for worker in self.worker_threads:
            worker.join(timeout=5)
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.close()
        
        logger.info("Edge computing system stopped")
    
    async def _register_default_nodes(self) -> None:
        """Register default edge nodes"""
        try:
            # Register nodes in different locations
            locations = [
                ("us-east-1", "192.168.1.10", 8080),
                ("us-west-2", "192.168.2.10", 8080),
                ("eu-west-1", "192.168.3.10", 8080),
                ("ap-southeast-1", "192.168.4.10", 8080)
            ]
            
            for location, ip, port in locations:
                node = create_edge_node(
                    location=location,
                    ip_address=ip,
                    port=port,
                    capabilities=[
                        EdgeTaskType.CONTENT_ANALYSIS,
                        EdgeTaskType.SENTIMENT_ANALYSIS,
                        EdgeTaskType.KEYWORD_EXTRACTION,
                        EdgeTaskType.READABILITY_CHECK,
                        EdgeTaskType.CACHE_UPDATE,
                        EdgeTaskType.REAL_TIME_PREDICTION
                    ],
                    processing_capacity=100
                )
                
                await self.register_edge_node(node)
            
            logger.info("Default edge nodes registered")
            
        except Exception as e:
            logger.error(f"Error registering default nodes: {str(e)}")
    
    async def register_edge_node(self, node: EdgeNode) -> None:
        """Register edge node"""
        try:
            # Add to registries
            self.edge_nodes[node.node_id] = node
            self.nodes_by_location[node.location].append(node)
            
            for capability in node.capabilities:
                self.nodes_by_capability[capability].append(node)
            
            # Update statistics
            self.stats["nodes_online"] = len([n for n in self.edge_nodes.values() if n.status == EdgeNodeStatus.ONLINE])
            self.stats["total_processing_capacity"] = sum(n.processing_capacity for n in self.edge_nodes.values())
            
            logger.info(f"Registered edge node: {node.node_id} at {node.location}")
            
        except Exception as e:
            logger.error(f"Error registering edge node: {str(e)}")
    
    async def submit_task(
        self,
        task: EdgeTask,
        callback: Optional[Callable] = None
    ) -> str:
        """Submit task to edge computing system"""
        try:
            # Add callback if provided
            if callback:
                self.task_callbacks[task.task_id] = callback
            
            # Add to active tasks
            self.active_tasks[task.task_id] = task
            
            # Calculate priority score (lower number = higher priority)
            priority_scores = {
                ProcessingPriority.CRITICAL: 1,
                ProcessingPriority.HIGH: 2,
                ProcessingPriority.NORMAL: 3,
                ProcessingPriority.LOW: 4,
                ProcessingPriority.BATCH: 5
            }
            
            priority_score = priority_scores.get(task.priority, 3)
            
            # Add to task queue
            self.task_queue.put((priority_score, task.created_at, task))
            
            # Update statistics
            self.stats["total_tasks"] += 1
            
            logger.info(f"Submitted task: {task.task_id} ({task.task_type.value})")
            
            return task.task_id
            
        except Exception as e:
            logger.error(f"Error submitting task: {str(e)}")
            raise
    
    async def submit_batch_tasks(
        self,
        tasks: List[EdgeTask],
        callback: Optional[Callable] = None
    ) -> List[str]:
        """Submit batch of tasks"""
        try:
            task_ids = []
            
            for task in tasks:
                task_id = await self.submit_task(task, callback)
                task_ids.append(task_id)
            
            logger.info(f"Submitted batch of {len(tasks)} tasks")
            return task_ids
            
        except Exception as e:
            logger.error(f"Error submitting batch tasks: {str(e)}")
            raise
    
    def _worker_loop(self) -> None:
        """Worker thread loop"""
        while not self.shutdown_event.is_set():
            try:
                # Get task from queue (blocking with timeout)
                try:
                    priority, created_at, task = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process task
                asyncio.run(self._process_task(task))
                
            except Exception as e:
                logger.error(f"Error in worker loop: {str(e)}")
                time.sleep(1)
    
    async def _process_task(self, task: EdgeTask) -> None:
        """Process edge task"""
        try:
            start_time = time.time()
            
            # Find optimal node for task
            available_nodes = self.nodes_by_capability.get(task.task_type, [])
            optimal_node = select_optimal_node(
                available_nodes,
                task.task_type,
                task.target_location
            )
            
            if not optimal_node:
                # No suitable node available
                await self._handle_task_failure(
                    task,
                    "No suitable edge node available"
                )
                return
            
            # Execute task on selected node
            result = await self._execute_task_on_node(task, optimal_node)
            
            processing_time = time.time() - start_time
            
            # Create task result
            task_result = EdgeTaskResult(
                task_id=task.task_id,
                success=result["success"],
                result=result.get("data"),
                processing_time=processing_time,
                node_id=optimal_node.node_id,
                error_message=result.get("error"),
                timestamp=datetime.utcnow()
            )
            
            # Handle result
            if result["success"]:
                await self._handle_task_success(task, task_result)
            else:
                await self._handle_task_failure(task, result.get("error", "Unknown error"))
            
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {str(e)}")
            await self._handle_task_failure(task, str(e))
    
    async def _execute_task_on_node(self, task: EdgeTask, node: EdgeNode) -> Dict[str, Any]:
        """Execute task on specific edge node"""
        try:
            # Prepare request data
            request_data = {
                "task_id": task.task_id,
                "task_type": task.task_type.value,
                "data": task.data,
                "timeout": task.timeout
            }
            
            # Make HTTP request to edge node
            url = f"http://{node.ip_address}:{node.port}/api/v1/edge/execute"
            
            async with self.http_client.post(
                url,
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=task.timeout)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "data": result.get("result"),
                        "processing_time": result.get("processing_time", 0)
                    }
                else:
                    error_data = await response.json()
                    return {
                        "success": False,
                        "error": error_data.get("error", f"HTTP {response.status}")
                    }
        
        except asyncio.TimeoutError:
            return {
                "success": False,
                "error": "Task execution timeout"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Task execution failed: {str(e)}"
            }
    
    async def _handle_task_success(self, task: EdgeTask, result: EdgeTaskResult) -> None:
        """Handle successful task completion"""
        try:
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Add to completed tasks
            self.completed_tasks.append(result)
            
            # Update statistics
            self.stats["completed_tasks"] += 1
            
            # Update average processing time
            total_time = self.stats["average_processing_time"] * (self.stats["completed_tasks"] - 1)
            self.stats["average_processing_time"] = (total_time + result.processing_time) / self.stats["completed_tasks"]
            
            # Call callback if registered
            if task.task_id in self.task_callbacks:
                callback = self.task_callbacks[task.task_id]
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(result)
                    else:
                        callback(result)
                except Exception as e:
                    logger.error(f"Error in task callback: {str(e)}")
                finally:
                    del self.task_callbacks[task.task_id]
            
            logger.info(f"Task completed successfully: {task.task_id}")
            
        except Exception as e:
            logger.error(f"Error handling task success: {str(e)}")
    
    async def _handle_task_failure(self, task: EdgeTask, error_message: str) -> None:
        """Handle task failure"""
        try:
            # Remove from active tasks
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Create failure result
            failure_result = EdgeTaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                processing_time=0.0,
                node_id="",
                error_message=error_message,
                timestamp=datetime.utcnow()
            )
            
            # Add to failed tasks
            self.failed_tasks.append(failure_result)
            
            # Update statistics
            self.stats["failed_tasks"] += 1
            
            # Call callback if registered
            if task.task_id in self.task_callbacks:
                callback = self.task_callbacks[task.task_id]
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(failure_result)
                    else:
                        callback(failure_result)
                except Exception as e:
                    logger.error(f"Error in failure callback: {str(e)}")
                finally:
                    del self.task_callbacks[task.task_id]
            
            logger.warning(f"Task failed: {task.task_id} - {error_message}")
            
        except Exception as e:
            logger.error(f"Error handling task failure: {str(e)}")
    
    async def _heartbeat_loop(self) -> None:
        """Background heartbeat loop"""
        while self.is_running:
            try:
                # Check node health
                for node_id, node in list(self.edge_nodes.items()):
                    await self._check_node_health(node)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(10)
    
    async def _check_node_health(self, node: EdgeNode) -> None:
        """Check health of specific edge node"""
        try:
            # Make health check request
            url = f"http://{node.ip_address}:{node.port}/health"
            
            async with self.http_client.get(
                url,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                if response.status == 200:
                    health_data = await response.json()
                    
                    # Update node with health data
                    updated_node = EdgeNode(
                        node_id=node.node_id,
                        location=node.location,
                        ip_address=node.ip_address,
                        port=node.port,
                        status=EdgeNodeStatus.ONLINE,
                        capabilities=node.capabilities,
                        cpu_usage=health_data.get("cpu_usage", 0.0),
                        memory_usage=health_data.get("memory_usage", 0.0),
                        network_latency=health_data.get("network_latency", 0.0),
                        last_heartbeat=datetime.utcnow(),
                        processing_capacity=node.processing_capacity,
                        current_load=health_data.get("current_load", 0)
                    )
                    
                    # Update in registries
                    self.edge_nodes[node.node_id] = updated_node
                    
                    # Update location registry
                    for i, existing_node in enumerate(self.nodes_by_location[node.location]):
                        if existing_node.node_id == node.node_id:
                            self.nodes_by_location[node.location][i] = updated_node
                            break
                    
                    # Update capability registry
                    for capability in node.capabilities:
                        for i, existing_node in enumerate(self.nodes_by_capability[capability]):
                            if existing_node.node_id == node.node_id:
                                self.nodes_by_capability[capability][i] = updated_node
                                break
                
        except Exception as e:
            logger.debug(f"Health check failed for node {node.node_id}: {str(e)}")
            # Mark node as offline
            offline_node = EdgeNode(
                node_id=node.node_id,
                location=node.location,
                ip_address=node.ip_address,
                port=node.port,
                status=EdgeNodeStatus.OFFLINE,
                capabilities=node.capabilities,
                cpu_usage=node.cpu_usage,
                memory_usage=node.memory_usage,
                network_latency=node.network_latency,
                last_heartbeat=node.last_heartbeat,
                processing_capacity=node.processing_capacity,
                current_load=node.current_load
            )
            
            self.edge_nodes[node.node_id] = offline_node
    
    async def _stats_update_loop(self) -> None:
        """Background statistics update loop"""
        while self.is_running:
            try:
                # Update current load
                self.stats["current_load"] = sum(
                    node.current_load for node in self.edge_nodes.values()
                    if node.status == EdgeNodeStatus.ONLINE
                )
                
                # Update online nodes count
                self.stats["nodes_online"] = len([
                    node for node in self.edge_nodes.values()
                    if node.status == EdgeNodeStatus.ONLINE
                ])
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in stats update loop: {str(e)}")
                await asyncio.sleep(30)
    
    def get_edge_nodes(self) -> List[EdgeNode]:
        """Get all edge nodes"""
        return list(self.edge_nodes.values())
    
    def get_online_nodes(self) -> List[EdgeNode]:
        """Get online edge nodes"""
        return [
            node for node in self.edge_nodes.values()
            if node.status == EdgeNodeStatus.ONLINE
        ]
    
    def get_nodes_by_location(self, location: str) -> List[EdgeNode]:
        """Get nodes by location"""
        return self.nodes_by_location.get(location, [])
    
    def get_nodes_by_capability(self, capability: EdgeTaskType) -> List[EdgeNode]:
        """Get nodes by capability"""
        return self.nodes_by_capability.get(capability, [])
    
    def get_task_status(self, task_id: str) -> Optional[EdgeTaskResult]:
        """Get task status"""
        # Check completed tasks
        for result in self.completed_tasks:
            if result.task_id == task_id:
                return result
        
        # Check failed tasks
        for result in self.failed_tasks:
            if result.task_id == task_id:
                return result
        
        # Check active tasks
        if task_id in self.active_tasks:
            return EdgeTaskResult(
                task_id=task_id,
                success=False,
                result=None,
                processing_time=0.0,
                node_id="",
                error_message="Task is still processing",
                timestamp=datetime.utcnow()
            )
        
        return None
    
    def get_edge_statistics(self) -> Dict[str, Any]:
        """Get edge computing statistics"""
        return {
            "statistics": self.stats.copy(),
            "nodes": {
                "total": len(self.edge_nodes),
                "online": self.stats["nodes_online"],
                "by_location": {
                    location: len(nodes)
                    for location, nodes in self.nodes_by_location.items()
                },
                "by_capability": {
                    capability.value: len(nodes)
                    for capability, nodes in self.nodes_by_capability.items()
                }
            },
            "tasks": {
                "active": len(self.active_tasks),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "queue_size": self.task_queue.qsize()
            },
            "performance": {
                "average_processing_time": self.stats["average_processing_time"],
                "total_processing_capacity": self.stats["total_processing_capacity"],
                "current_load": self.stats["current_load"],
                "load_percentage": (
                    self.stats["current_load"] / max(1, self.stats["total_processing_capacity"]) * 100
                )
            }
        }


# Factory functions

def create_edge_computing_system(max_workers: int = 10) -> EdgeComputingSystem:
    """Create edge computing system - pure function"""
    return EdgeComputingSystem(max_workers)


async def get_edge_computing_system() -> EdgeComputingSystem:
    """Get edge computing system instance"""
    system = create_edge_computing_system()
    await system.start()
    return system

