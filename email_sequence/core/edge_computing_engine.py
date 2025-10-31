"""
Edge Computing Engine for Email Sequence System

This module provides edge computing capabilities including distributed processing,
edge AI inference, and real-time data processing at the network edge.
"""

import asyncio
import logging
import json
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from uuid import UUID
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import numpy as np

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .exceptions import EdgeComputingError
from .cache import cache_manager

logger = logging.getLogger(__name__)
settings = get_settings()


class EdgeNodeType(str, Enum):
    """Types of edge computing nodes"""
    AI_INFERENCE = "ai_inference"
    DATA_PROCESSING = "data_processing"
    REAL_TIME_ANALYTICS = "real_time_analytics"
    CONTENT_OPTIMIZATION = "content_optimization"
    PERSONALIZATION = "personalization"
    HYBRID = "hybrid"


class EdgeNodeStatus(str, Enum):
    """Edge node status"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class ProcessingPriority(str, Enum):
    """Processing priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EdgeNode:
    """Edge computing node information"""
    node_id: str
    node_type: EdgeNodeType
    location: str
    ip_address: str
    port: int
    capabilities: List[str]
    status: EdgeNodeStatus
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeTask:
    """Edge computing task"""
    task_id: str
    task_type: str
    priority: ProcessingPriority
    data: Dict[str, Any]
    target_node_type: EdgeNodeType
    timeout_seconds: int = 300
    retry_count: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class EdgeComputingEngine:
    """Edge computing engine for distributed processing"""
    
    def __init__(self):
        """Initialize edge computing engine"""
        self.nodes: Dict[str, EdgeNode] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.active_tasks: Dict[str, EdgeTask] = {}
        self.completed_tasks: Dict[str, EdgeTask] = {}
        
        # Performance metrics
        self.tasks_processed = 0
        self.tasks_failed = 0
        self.average_processing_time = 0.0
        self.total_processing_time = 0.0
        
        # Edge AI models
        self.edge_models: Dict[str, Any] = {}
        self.model_cache: Dict[str, bytes] = {}
        
        logger.info("Edge Computing Engine initialized")
    
    async def initialize(self) -> None:
        """Initialize the edge computing engine"""
        try:
            # Start background tasks
            asyncio.create_task(self._node_health_monitor())
            asyncio.create_task(self._task_processor())
            asyncio.create_task(self._load_balancer())
            asyncio.create_task(self._cleanup_completed_tasks())
            
            # Initialize edge AI models
            await self._initialize_edge_models()
            
            logger.info("Edge Computing Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing edge computing engine: {e}")
            raise EdgeComputingError(f"Failed to initialize edge computing engine: {e}")
    
    async def register_edge_node(
        self,
        node_id: str,
        node_type: EdgeNodeType,
        location: str,
        ip_address: str,
        port: int,
        capabilities: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> EdgeNode:
        """
        Register a new edge computing node.
        
        Args:
            node_id: Unique node identifier
            node_type: Type of edge node
            location: Geographic location
            ip_address: Node IP address
            port: Node port
            capabilities: List of node capabilities
            metadata: Additional node metadata
            
        Returns:
            EdgeNode object
        """
        try:
            # Create edge node
            edge_node = EdgeNode(
                node_id=node_id,
                node_type=node_type,
                location=location,
                ip_address=ip_address,
                port=port,
                capabilities=capabilities,
                status=EdgeNodeStatus.ONLINE,
                metadata=metadata or {}
            )
            
            # Register node
            self.nodes[node_id] = edge_node
            
            # Cache node information
            await cache_manager.set(f"edge_node:{node_id}", edge_node.__dict__, 3600)
            
            logger.info(f"Edge node registered: {node_id} ({node_type.value}) at {location}")
            return edge_node
            
        except Exception as e:
            logger.error(f"Error registering edge node: {e}")
            raise EdgeComputingError(f"Failed to register edge node: {e}")
    
    async def submit_edge_task(
        self,
        task_type: str,
        data: Dict[str, Any],
        target_node_type: EdgeNodeType,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        timeout_seconds: int = 300
    ) -> str:
        """
        Submit a task for edge processing.
        
        Args:
            task_type: Type of task to process
            data: Task data
            target_node_type: Target node type for processing
            priority: Task priority
            timeout_seconds: Task timeout in seconds
            
        Returns:
            Task ID
        """
        try:
            # Generate task ID
            task_id = f"task_{hashlib.sha256(f'{task_type}{datetime.utcnow().isoformat()}'.encode()).hexdigest()[:16]}"
            
            # Create edge task
            edge_task = EdgeTask(
                task_id=task_id,
                task_type=task_type,
                priority=priority,
                data=data,
                target_node_type=target_node_type,
                timeout_seconds=timeout_seconds
            )
            
            # Add to task queue
            await self.task_queue.put(edge_task)
            self.active_tasks[task_id] = edge_task
            
            logger.info(f"Edge task submitted: {task_id} ({task_type}) with priority {priority.value}")
            return task_id
            
        except Exception as e:
            logger.error(f"Error submitting edge task: {e}")
            raise EdgeComputingError(f"Failed to submit edge task: {e}")
    
    async def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """
        Get the result of an edge task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result
        """
        try:
            # Check active tasks
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": "processing",
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "progress": "Task is being processed"
                }
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                return {
                    "task_id": task_id,
                    "status": "completed",
                    "result": task.result,
                    "created_at": task.created_at.isoformat(),
                    "started_at": task.started_at.isoformat() if task.started_at else None,
                    "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                    "processing_time": (
                        (task.completed_at - task.started_at).total_seconds()
                        if task.started_at and task.completed_at else None
                    )
                }
            
            raise EdgeComputingError(f"Task not found: {task_id}")
            
        except Exception as e:
            logger.error(f"Error getting task result: {e}")
            raise EdgeComputingError(f"Failed to get task result: {e}")
    
    async def process_ai_inference(
        self,
        model_name: str,
        input_data: Dict[str, Any],
        priority: ProcessingPriority = ProcessingPriority.NORMAL
    ) -> str:
        """
        Submit AI inference task to edge nodes.
        
        Args:
            model_name: Name of the AI model
            input_data: Input data for inference
            priority: Task priority
            
        Returns:
            Task ID
        """
        try:
            task_data = {
                "model_name": model_name,
                "input_data": input_data,
                "inference_type": "ai_inference"
            }
            
            return await self.submit_edge_task(
                task_type="ai_inference",
                data=task_data,
                target_node_type=EdgeNodeType.AI_INFERENCE,
                priority=priority
            )
            
        except Exception as e:
            logger.error(f"Error submitting AI inference task: {e}")
            raise EdgeComputingError(f"Failed to submit AI inference task: {e}")
    
    async def process_content_optimization(
        self,
        content: str,
        optimization_type: str,
        target_audience: Dict[str, Any],
        priority: ProcessingPriority = ProcessingPriority.NORMAL
    ) -> str:
        """
        Submit content optimization task to edge nodes.
        
        Args:
            content: Content to optimize
            optimization_type: Type of optimization
            target_audience: Target audience data
            priority: Task priority
            
        Returns:
            Task ID
        """
        try:
            task_data = {
                "content": content,
                "optimization_type": optimization_type,
                "target_audience": target_audience,
                "optimization_goals": ["engagement", "conversion", "clarity"]
            }
            
            return await self.submit_edge_task(
                task_type="content_optimization",
                data=task_data,
                target_node_type=EdgeNodeType.CONTENT_OPTIMIZATION,
                priority=priority
            )
            
        except Exception as e:
            logger.error(f"Error submitting content optimization task: {e}")
            raise EdgeComputingError(f"Failed to submit content optimization task: {e}")
    
    async def process_real_time_analytics(
        self,
        analytics_data: Dict[str, Any],
        analysis_type: str,
        priority: ProcessingPriority = ProcessingPriority.HIGH
    ) -> str:
        """
        Submit real-time analytics task to edge nodes.
        
        Args:
            analytics_data: Analytics data to process
            analysis_type: Type of analysis
            priority: Task priority
            
        Returns:
            Task ID
        """
        try:
            task_data = {
                "analytics_data": analytics_data,
                "analysis_type": analysis_type,
                "real_time": True,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return await self.submit_edge_task(
                task_type="real_time_analytics",
                data=task_data,
                target_node_type=EdgeNodeType.REAL_TIME_ANALYTICS,
                priority=priority
            )
            
        except Exception as e:
            logger.error(f"Error submitting real-time analytics task: {e}")
            raise EdgeComputingError(f"Failed to submit real-time analytics task: {e}")
    
    async def get_edge_node_status(self, node_id: str) -> Dict[str, Any]:
        """
        Get edge node status and metrics.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node status information
        """
        try:
            if node_id not in self.nodes:
                raise EdgeComputingError(f"Edge node not found: {node_id}")
            
            node = self.nodes[node_id]
            
            return {
                "node_id": node_id,
                "node_type": node.node_type.value,
                "location": node.location,
                "status": node.status.value,
                "cpu_usage": node.cpu_usage,
                "memory_usage": node.memory_usage,
                "network_latency": node.network_latency,
                "capabilities": node.capabilities,
                "last_heartbeat": node.last_heartbeat.isoformat(),
                "uptime": (datetime.utcnow() - node.created_at).total_seconds(),
                "metadata": node.metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting edge node status: {e}")
            raise EdgeComputingError(f"Failed to get edge node status: {e}")
    
    async def get_edge_computing_stats(self) -> Dict[str, Any]:
        """
        Get edge computing engine statistics.
        
        Returns:
            Engine statistics
        """
        try:
            online_nodes = len([n for n in self.nodes.values() if n.status == EdgeNodeStatus.ONLINE])
            busy_nodes = len([n for n in self.nodes.values() if n.status == EdgeNodeStatus.BUSY])
            
            return {
                "total_nodes": len(self.nodes),
                "online_nodes": online_nodes,
                "busy_nodes": busy_nodes,
                "offline_nodes": len(self.nodes) - online_nodes,
                "active_tasks": len(self.active_tasks),
                "completed_tasks": len(self.completed_tasks),
                "tasks_processed": self.tasks_processed,
                "tasks_failed": self.tasks_failed,
                "average_processing_time": self.average_processing_time,
                "total_processing_time": self.total_processing_time,
                "success_rate": (
                    (self.tasks_processed / (self.tasks_processed + self.tasks_failed)) * 100
                    if (self.tasks_processed + self.tasks_failed) > 0 else 0
                ),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting edge computing stats: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _node_health_monitor(self) -> None:
        """Monitor edge node health and status"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = datetime.utcnow()
                
                for node_id, node in self.nodes.items():
                    # Check if node is responsive
                    if (current_time - node.last_heartbeat).total_seconds() > 60:
                        if node.status == EdgeNodeStatus.ONLINE:
                            node.status = EdgeNodeStatus.OFFLINE
                            logger.warning(f"Edge node went offline: {node_id}")
                    
                    # Simulate resource usage updates
                    node.cpu_usage = np.random.uniform(0.1, 0.9)
                    node.memory_usage = np.random.uniform(0.2, 0.8)
                    node.network_latency = np.random.uniform(1, 50)
                
            except Exception as e:
                logger.error(f"Error in node health monitor: {e}")
    
    async def _task_processor(self) -> None:
        """Process tasks from the queue"""
        while True:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                # Find suitable node
                suitable_node = await self._find_suitable_node(task)
                
                if suitable_node:
                    # Process task on node
                    await self._process_task_on_node(task, suitable_node)
                else:
                    # No suitable node available, retry later
                    logger.warning(f"No suitable node available for task: {task.task_id}")
                    await asyncio.sleep(5)
                    await self.task_queue.put(task)
                
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
    
    async def _find_suitable_node(self, task: EdgeTask) -> Optional[EdgeNode]:
        """Find a suitable node for task processing"""
        try:
            suitable_nodes = [
                node for node in self.nodes.values()
                if (node.node_type == task.target_node_type and
                    node.status == EdgeNodeStatus.ONLINE and
                    task.task_type in node.capabilities)
            ]
            
            if not suitable_nodes:
                return None
            
            # Select node with lowest resource usage
            best_node = min(suitable_nodes, key=lambda n: n.cpu_usage + n.memory_usage)
            return best_node
            
        except Exception as e:
            logger.error(f"Error finding suitable node: {e}")
            return None
    
    async def _process_task_on_node(self, task: EdgeTask, node: EdgeNode) -> None:
        """Process task on a specific edge node"""
        try:
            task.started_at = datetime.utcnow()
            node.status = EdgeNodeStatus.BUSY
            
            # Simulate task processing
            processing_time = np.random.uniform(1, 10)  # 1-10 seconds
            await asyncio.sleep(processing_time)
            
            # Process task based on type
            if task.task_type == "ai_inference":
                result = await self._process_ai_inference_task(task, node)
            elif task.task_type == "content_optimization":
                result = await self._process_content_optimization_task(task, node)
            elif task.task_type == "real_time_analytics":
                result = await self._process_analytics_task(task, node)
            else:
                result = await self._process_generic_task(task, node)
            
            # Complete task
            task.completed_at = datetime.utcnow()
            task.result = result
            node.status = EdgeNodeStatus.ONLINE
            
            # Move to completed tasks
            self.completed_tasks[task.task_id] = task
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Update metrics
            self.tasks_processed += 1
            actual_processing_time = (task.completed_at - task.started_at).total_seconds()
            self.total_processing_time += actual_processing_time
            self.average_processing_time = self.total_processing_time / self.tasks_processed
            
            logger.info(f"Task completed: {task.task_id} on node {node.node_id} in {actual_processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing task on node: {e}")
            task.error_message = str(e)
            self.tasks_failed += 1
            node.status = EdgeNodeStatus.ONLINE
    
    async def _process_ai_inference_task(self, task: EdgeTask, node: EdgeNode) -> Dict[str, Any]:
        """Process AI inference task"""
        try:
            model_name = task.data.get("model_name")
            input_data = task.data.get("input_data")
            
            # Simulate AI inference
            inference_result = {
                "model_name": model_name,
                "predictions": np.random.random(5).tolist(),
                "confidence_scores": np.random.random(5).tolist(),
                "processing_time": np.random.uniform(0.1, 2.0),
                "node_id": node.node_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return inference_result
            
        except Exception as e:
            logger.error(f"Error processing AI inference task: {e}")
            raise EdgeComputingError(f"Failed to process AI inference task: {e}")
    
    async def _process_content_optimization_task(self, task: EdgeTask, node: EdgeNode) -> Dict[str, Any]:
        """Process content optimization task"""
        try:
            content = task.data.get("content")
            optimization_type = task.data.get("optimization_type")
            
            # Simulate content optimization
            optimized_content = f"OPTIMIZED: {content}"
            
            optimization_result = {
                "original_content": content,
                "optimized_content": optimized_content,
                "optimization_type": optimization_type,
                "improvement_score": np.random.uniform(0.1, 0.5),
                "suggestions": [
                    "Improved readability",
                    "Enhanced engagement",
                    "Better call-to-action"
                ],
                "node_id": node.node_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error processing content optimization task: {e}")
            raise EdgeComputingError(f"Failed to process content optimization task: {e}")
    
    async def _process_analytics_task(self, task: EdgeTask, node: EdgeNode) -> Dict[str, Any]:
        """Process real-time analytics task"""
        try:
            analytics_data = task.data.get("analytics_data")
            analysis_type = task.data.get("analysis_type")
            
            # Simulate analytics processing
            analytics_result = {
                "analysis_type": analysis_type,
                "processed_data": analytics_data,
                "insights": [
                    "Engagement increased by 15%",
                    "Conversion rate improved by 8%",
                    "Peak activity at 2 PM"
                ],
                "metrics": {
                    "engagement_rate": np.random.uniform(0.1, 0.9),
                    "conversion_rate": np.random.uniform(0.05, 0.3),
                    "click_through_rate": np.random.uniform(0.02, 0.15)
                },
                "node_id": node.node_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return analytics_result
            
        except Exception as e:
            logger.error(f"Error processing analytics task: {e}")
            raise EdgeComputingError(f"Failed to process analytics task: {e}")
    
    async def _process_generic_task(self, task: EdgeTask, node: EdgeNode) -> Dict[str, Any]:
        """Process generic task"""
        try:
            generic_result = {
                "task_type": task.task_type,
                "processed_data": task.data,
                "status": "completed",
                "node_id": node.node_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return generic_result
            
        except Exception as e:
            logger.error(f"Error processing generic task: {e}")
            raise EdgeComputingError(f"Failed to process generic task: {e}")
    
    async def _load_balancer(self) -> None:
        """Load balancer for distributing tasks across nodes"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Analyze node load and redistribute if necessary
                for node in self.nodes.values():
                    if node.cpu_usage > 0.8 or node.memory_usage > 0.8:
                        logger.warning(f"High resource usage on node {node.node_id}: CPU={node.cpu_usage:.2f}, Memory={node.memory_usage:.2f}")
                
            except Exception as e:
                logger.error(f"Error in load balancer: {e}")
    
    async def _cleanup_completed_tasks(self) -> None:
        """Clean up old completed tasks"""
        while True:
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
                # Remove tasks older than 1 hour
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                
                tasks_to_remove = [
                    task_id for task_id, task in self.completed_tasks.items()
                    if task.completed_at and task.completed_at < cutoff_time
                ]
                
                for task_id in tasks_to_remove:
                    del self.completed_tasks[task_id]
                
                if tasks_to_remove:
                    logger.info(f"Cleaned up {len(tasks_to_remove)} old completed tasks")
                
            except Exception as e:
                logger.error(f"Error cleaning up completed tasks: {e}")
    
    async def _initialize_edge_models(self) -> None:
        """Initialize edge AI models"""
        try:
            # Load pre-trained models for edge inference
            edge_models = {
                "sentiment_analysis": "sentiment_model_v1.pkl",
                "content_optimization": "content_opt_model_v1.pkl",
                "engagement_prediction": "engagement_model_v1.pkl"
            }
            
            for model_name, model_file in edge_models.items():
                # Simulate model loading
                self.edge_models[model_name] = {
                    "file": model_file,
                    "loaded": True,
                    "version": "v1",
                    "size_mb": np.random.uniform(10, 100)
                }
            
            logger.info(f"Initialized {len(self.edge_models)} edge AI models")
            
        except Exception as e:
            logger.error(f"Error initializing edge models: {e}")


# Global edge computing engine instance
edge_computing_engine = EdgeComputingEngine()






























