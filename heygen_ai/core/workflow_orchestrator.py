#!/usr/bin/env python3
"""
Advanced Workflow Orchestrator for Enhanced HeyGen AI
Handles complex AI workflows, multi-step video generation pipelines, and intelligent task scheduling.
"""

import asyncio
import time
import json
import structlog
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from collections import defaultdict, deque
import hashlib
import secrets
import uuid
from pathlib import Path

logger = structlog.get_logger()

class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskStatus(Enum):
    """Individual task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"

class WorkflowType(Enum):
    """Types of workflows."""
    VIDEO_GENERATION = "video_generation"
    BATCH_PROCESSING = "batch_processing"
    MODEL_TRAINING = "model_training"
    DATA_PIPELINE = "data_pipeline"
    MULTI_MODAL_FUSION = "multi_modal_fusion"
    CUSTOM = "custom"

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    URGENT = 4

@dataclass
class WorkflowDefinition:
    """Workflow definition with tasks and dependencies."""
    workflow_id: str
    name: str
    description: str
    workflow_type: WorkflowType
    version: str
    tasks: List[Dict[str, Any]]
    dependencies: List[Tuple[str, str]]  # (from_task, to_task)
    parameters: Dict[str, Any]
    retry_policy: Dict[str, Any]
    timeout_seconds: int
    created_at: float
    updated_at: float

@dataclass
class WorkflowInstance:
    """Instance of a workflow execution."""
    instance_id: str
    workflow_id: str
    status: WorkflowStatus
    current_task: Optional[str]
    progress: float
    start_time: float
    end_time: Optional[float]
    parameters: Dict[str, Any]
    results: Dict[str, Any]
    error_message: Optional[str] = None
    retry_count: int = 0

@dataclass
class WorkflowTask:
    """Individual task within a workflow."""
    task_id: str
    name: str
    task_type: str
    parameters: Dict[str, Any]
    status: TaskStatus
    priority: TaskPriority
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    dependencies: List[str] = None
    estimated_duration: float = 0.0
    actual_duration: Optional[float] = None

@dataclass
class WorkflowMetrics:
    """Workflow performance metrics."""
    total_workflows: int
    completed_workflows: int
    failed_workflows: int
    average_execution_time: float
    success_rate: float
    active_workflows: int
    queued_workflows: int

class WorkflowOrchestrator:
    """Advanced workflow orchestration for HeyGen AI."""
    
    def __init__(
        self,
        max_concurrent_workflows: int = 50,
        max_workers_per_workflow: int = 8,
        enable_auto_scaling: bool = True,
        enable_intelligent_scheduling: bool = True,
        enable_workflow_optimization: bool = True,
        workflow_timeout_hours: int = 24
    ):
        self.max_concurrent_workflows = max_concurrent_workflows
        self.max_workers_per_workflow = max_workers_per_workflow
        self.enable_auto_scaling = enable_auto_scaling
        self.enable_intelligent_scheduling = enable_intelligent_scheduling
        self.enable_workflow_optimization = enable_workflow_optimization
        self.workflow_timeout_hours = workflow_timeout_hours
        
        # Workflow definitions and instances
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.workflow_instances: Dict[str, WorkflowInstance] = {}
        self.active_workflows: Dict[str, WorkflowInstance] = {}
        
        # Task management
        self.task_queue: deque = deque()
        self.running_tasks: Dict[str, WorkflowTask] = {}
        self.completed_tasks: Dict[str, WorkflowTask] = {}
        
        # Workflow execution graph
        self.execution_graphs: Dict[str, nx.DiGraph] = {}
        
        # Thread pool for task execution
        self.thread_pool = ThreadPoolExecutor(max_workers=16)
        
        # Background tasks
        self.workflow_execution_task: Optional[asyncio.Task] = None
        self.task_scheduling_task: Optional[asyncio.Task] = None
        self.optimization_task: Optional[asyncio.Task] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.metrics = WorkflowMetrics(
            total_workflows=0,
            completed_workflows=0,
            failed_workflows=0,
            average_execution_time=0.0,
            success_rate=0.0,
            active_workflows=0,
            queued_workflows=0
        )
        
        # Initialize default workflows
        self._initialize_default_workflows()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _initialize_default_workflows(self):
        """Initialize default workflow definitions."""
        # Video Generation Workflow
        video_workflow = WorkflowDefinition(
            workflow_id="video_generation_v1",
            name="Standard Video Generation",
            description="Complete video generation pipeline from script to final video",
            workflow_type=WorkflowType.VIDEO_GENERATION,
            version="1.0.0",
            tasks=[
                {
                    "task_id": "script_analysis",
                    "name": "Script Analysis",
                    "task_type": "nlp_analysis",
                    "parameters": {"model": "gpt-4", "max_tokens": 1000},
                    "priority": TaskPriority.HIGH.value,
                    "estimated_duration": 30.0
                },
                {
                    "task_id": "voice_synthesis",
                    "name": "Voice Synthesis",
                    "task_type": "tts_generation",
                    "parameters": {"voice_id": "natural_001", "quality": "high"},
                    "priority": TaskPriority.HIGH.value,
                    "estimated_duration": 120.0
                },
                {
                    "task_id": "avatar_generation",
                    "name": "Avatar Generation",
                    "task_type": "image_generation",
                    "parameters": {"model": "stable_diffusion_xl", "resolution": "1080p"},
                    "priority": TaskPriority.HIGH.value,
                    "estimated_duration": 180.0
                },
                {
                    "task_id": "lip_sync",
                    "name": "Lip Sync Generation",
                    "task_type": "lip_sync",
                    "parameters": {"model": "wav2lip", "quality": "high"},
                    "priority": TaskPriority.NORMAL.value,
                    "estimated_duration": 90.0
                },
                {
                    "task_id": "video_composition",
                    "name": "Video Composition",
                    "task_type": "video_rendering",
                    "parameters": {"format": "mp4", "codec": "h264"},
                    "priority": TaskPriority.NORMAL.value,
                    "estimated_duration": 60.0
                }
            ],
            dependencies=[
                ("script_analysis", "voice_synthesis"),
                ("script_analysis", "avatar_generation"),
                ("voice_synthesis", "lip_sync"),
                ("avatar_generation", "lip_sync"),
                ("lip_sync", "video_composition")
            ],
            parameters={"output_format": "mp4", "quality": "high"},
            retry_policy={"max_retries": 3, "backoff_factor": 2.0},
            timeout_seconds=3600,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.workflow_definitions["video_generation_v1"] = video_workflow
        
        # Batch Processing Workflow
        batch_workflow = WorkflowDefinition(
            workflow_id="batch_processing_v1",
            name="Batch Video Processing",
            description="Process multiple videos in parallel with resource optimization",
            workflow_type=WorkflowType.BATCH_PROCESSING,
            version="1.0.0",
            tasks=[
                {
                    "task_id": "batch_validation",
                    "name": "Batch Validation",
                    "task_type": "validation",
                    "parameters": {"max_batch_size": 100},
                    "priority": TaskPriority.HIGH.value,
                    "estimated_duration": 15.0
                },
                {
                    "task_id": "resource_allocation",
                    "name": "Resource Allocation",
                    "task_type": "resource_management",
                    "parameters": {"optimization": "auto"},
                    "priority": TaskPriority.HIGH.value,
                    "estimated_duration": 30.0
                },
                {
                    "task_id": "parallel_processing",
                    "name": "Parallel Processing",
                    "task_type": "parallel_execution",
                    "parameters": {"max_workers": 8},
                    "priority": TaskPriority.NORMAL.value,
                    "estimated_duration": 300.0
                },
                {
                    "task_id": "result_aggregation",
                    "name": "Result Aggregation",
                    "task_type": "aggregation",
                    "parameters": {"format": "json"},
                    "priority": TaskPriority.NORMAL.value,
                    "estimated_duration": 45.0
                }
            ],
            dependencies=[
                ("batch_validation", "resource_allocation"),
                ("resource_allocation", "parallel_processing"),
                ("parallel_processing", "result_aggregation")
            ],
            parameters={"batch_size": 50, "parallel_workers": 8},
            retry_policy={"max_retries": 2, "backoff_factor": 1.5},
            timeout_seconds=7200,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        self.workflow_definitions["batch_processing_v1"] = batch_workflow
    
    def _start_background_tasks(self):
        """Start background monitoring and processing tasks."""
        self.workflow_execution_task = asyncio.create_task(self._workflow_execution_loop())
        self.task_scheduling_task = asyncio.create_task(self._task_scheduling_loop())
        self.optimization_task = asyncio.create_task(self._workflow_optimization_loop())
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _workflow_execution_loop(self):
        """Main workflow execution loop."""
        while True:
            try:
                await self._execute_workflows()
                await asyncio.sleep(10)  # Execute every 10 seconds
                
            except Exception as e:
                logger.error(f"Workflow execution error: {e}")
                await asyncio.sleep(30)
    
    async def _task_scheduling_loop(self):
        """Task scheduling and optimization loop."""
        while True:
            try:
                if self.enable_intelligent_scheduling:
                    await self._optimize_task_scheduling()
                
                await asyncio.sleep(30)  # Optimize every 30 seconds
                
            except Exception as e:
                logger.error(f"Task scheduling error: {e}")
                await asyncio.sleep(60)
    
    async def _workflow_optimization_loop(self):
        """Workflow optimization and auto-scaling loop."""
        while True:
            try:
                if self.enable_workflow_optimization:
                    await self._optimize_workflows()
                
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
            except Exception as e:
                logger.error(f"Workflow optimization error: {e}")
                await asyncio.sleep(60)
    
    async def _monitoring_loop(self):
        """Workflow monitoring and metrics collection."""
        while True:
            try:
                await self._update_metrics()
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Cleanup completed and failed workflows."""
        while True:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(600)  # Cleanup every 10 minutes
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def create_workflow(
        self,
        name: str,
        description: str,
        workflow_type: WorkflowType,
        tasks: List[Dict[str, Any]],
        dependencies: List[Tuple[str, str]],
        parameters: Dict[str, Any] = None,
        retry_policy: Dict[str, Any] = None,
        timeout_seconds: int = 3600
    ) -> str:
        """Create a new workflow definition."""
        try:
            workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
            
            workflow = WorkflowDefinition(
                workflow_id=workflow_id,
                name=name,
                description=description,
                workflow_type=workflow_type,
                version="1.0.0",
                tasks=tasks,
                dependencies=dependencies,
                parameters=parameters or {},
                retry_policy=retry_policy or {"max_retries": 3, "backoff_factor": 2.0},
                timeout_seconds=timeout_seconds,
                created_at=time.time(),
                updated_at=time.time()
            )
            
            # Validate workflow
            if not self._validate_workflow(workflow):
                raise ValueError("Invalid workflow definition")
            
            # Build execution graph
            self.execution_graphs[workflow_id] = self._build_execution_graph(workflow)
            
            self.workflow_definitions[workflow_id] = workflow
            
            logger.info(f"Workflow created: {workflow_id} - {name}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise
    
    def _validate_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Validate workflow definition."""
        try:
            # Check if all tasks exist
            task_ids = {task["task_id"] for task in workflow.tasks}
            
            # Check dependencies
            for from_task, to_task in workflow.dependencies:
                if from_task not in task_ids or to_task not in task_ids:
                    logger.error(f"Invalid dependency: {from_task} -> {to_task}")
                    return False
            
            # Check for cycles
            graph = self._build_execution_graph(workflow)
            if not nx.is_directed_acyclic_graph(graph):
                logger.error("Workflow contains cycles")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Workflow validation error: {e}")
            return False
    
    def _build_execution_graph(self, workflow: WorkflowDefinition) -> nx.DiGraph:
        """Build execution graph from workflow definition."""
        graph = nx.DiGraph()
        
        # Add nodes (tasks)
        for task in workflow.tasks:
            graph.add_node(task["task_id"], **task)
        
        # Add edges (dependencies)
        for from_task, to_task in workflow.dependencies:
            graph.add_edge(from_task, to_task)
        
        return graph
    
    async def start_workflow(
        self,
        workflow_id: str,
        parameters: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """Start a workflow execution."""
        try:
            if workflow_id not in self.workflow_definitions:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            if len(self.active_workflows) >= self.max_concurrent_workflows:
                raise ValueError("Maximum concurrent workflows reached")
            
            workflow_def = self.workflow_definitions[workflow_id]
            instance_id = f"instance_{uuid.uuid4().hex[:8]}"
            
            instance = WorkflowInstance(
                instance_id=instance_id,
                workflow_id=workflow_id,
                status=WorkflowStatus.PENDING,
                current_task=None,
                progress=0.0,
                start_time=time.time(),
                end_time=None,
                parameters=parameters or {},
                results={}
            )
            
            self.workflow_instances[instance_id] = instance
            self.active_workflows[instance_id] = instance
            
            # Add to task queue
            self.task_queue.append((priority.value, time.time(), instance_id))
            
            self.metrics.total_workflows += 1
            self.metrics.active_workflows = len(self.active_workflows)
            self.metrics.queued_workflows = len(self.task_queue)
            
            logger.info(f"Workflow started: {instance_id} (workflow: {workflow_id})")
            return instance_id
            
        except Exception as e:
            logger.error(f"Failed to start workflow: {e}")
            raise
    
    async def _execute_workflows(self):
        """Execute pending workflows."""
        try:
            # Process task queue
            while self.task_queue and len(self.running_tasks) < self.max_workers_per_workflow:
                priority, timestamp, instance_id = self.task_queue.popleft()
                
                if instance_id not in self.active_workflows:
                    continue
                
                instance = self.active_workflows[instance_id]
                if instance.status != WorkflowStatus.PENDING:
                    continue
                
                # Start workflow execution
                await self._execute_workflow_instance(instance_id)
                
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
    
    async def _execute_workflow_instance(self, instance_id: str):
        """Execute a specific workflow instance."""
        try:
            instance = self.active_workflows[instance_id]
            workflow_def = self.workflow_definitions[instance.workflow_id]
            graph = self.execution_graphs[instance.workflow_id]
            
            # Update status
            instance.status = WorkflowStatus.RUNNING
            
            # Find ready tasks (no dependencies or all dependencies completed)
            ready_tasks = self._get_ready_tasks(instance_id, graph)
            
            if not ready_tasks:
                # Check if workflow is complete
                if self._is_workflow_complete(instance_id):
                    await self._complete_workflow(instance_id)
                    return
                else:
                    # Workflow is waiting for dependencies
                    return
            
            # Execute ready tasks
            for task_id in ready_tasks:
                await self._execute_task(instance_id, task_id)
                
        except Exception as e:
            logger.error(f"Workflow instance execution error: {e}")
            await self._fail_workflow(instance_id, str(e))
    
    def _get_ready_tasks(self, instance_id: str, graph: nx.DiGraph) -> List[str]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        try:
            instance = self.active_workflows[instance_id]
            ready_tasks = []
            
            for task in graph.nodes():
                if self._can_execute_task(instance_id, task):
                    ready_tasks.append(task)
            
            return ready_tasks
            
        except Exception as e:
            logger.error(f"Error getting ready tasks: {e}")
            return []
    
    def _can_execute_task(self, instance_id: str, task_id: str) -> bool:
        """Check if a task can be executed (dependencies satisfied)."""
        try:
            instance = self.active_workflows[instance_id]
            workflow_def = self.workflow_definitions[instance.workflow_id]
            graph = self.execution_graphs[instance.workflow_id]
            
            # Check if task is already completed or running
            if task_id in instance.results or task_id in self.running_tasks:
                return False
            
            # Check dependencies
            predecessors = list(graph.predecessors(task_id))
            for pred in predecessors:
                if pred not in instance.results:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking task execution: {e}")
            return False
    
    async def _execute_task(self, instance_id: str, task_id: str):
        """Execute a specific task."""
        try:
            instance = self.active_workflows[instance_id]
            workflow_def = self.workflow_definitions[instance.workflow_id]
            
            # Find task definition
            task_def = next((t for t in workflow_def.tasks if t["task_id"] == task_id), None)
            if not task_def:
                logger.error(f"Task definition not found: {task_id}")
                return
            
            # Create task instance
            task = WorkflowTask(
                task_id=task_id,
                name=task_def["name"],
                task_type=task_def["task_type"],
                parameters=task_def["parameters"],
                status=TaskStatus.RUNNING,
                priority=TaskPriority(task_def["priority"]),
                start_time=time.time(),
                dependencies=list(workflow_def.dependencies)
            )
            
            self.running_tasks[task_id] = task
            
            # Execute task (simulated)
            await self._simulate_task_execution(task, instance)
            
            # Update workflow progress
            await self._update_workflow_progress(instance_id)
            
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            await self._fail_task(instance_id, task_id, str(e))
    
    async def _simulate_task_execution(self, task: WorkflowTask, instance: WorkflowInstance):
        """Simulate task execution (in practice, this would execute real tasks)."""
        try:
            # Simulate execution time
            execution_time = task.estimated_duration or 60.0
            await asyncio.sleep(min(execution_time / 10, 5))  # Scale down for demo
            
            # Simulate success
            task.status = TaskStatus.COMPLETED
            task.end_time = time.time()
            task.actual_duration = task.end_time - task.start_time
            
            # Generate mock result
            task.result = {
                "status": "success",
                "output": f"Mock output for {task.name}",
                "metrics": {
                    "accuracy": 0.95,
                    "processing_time": task.actual_duration
                }
            }
            
            # Store result in workflow instance
            instance.results[task.task_id] = task.result
            
            # Remove from running tasks
            if task.task_id in self.running_tasks:
                del self.running_tasks[task.task_id]
            
            # Add to completed tasks
            self.completed_tasks[task.task_id] = task
            
            logger.info(f"Task completed: {task.task_id} in {task.actual_duration:.2f}s")
            
        except Exception as e:
            logger.error(f"Task simulation error: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
    
    async def _update_workflow_progress(self, instance_id: str):
        """Update workflow progress."""
        try:
            instance = self.active_workflows[instance_id]
            workflow_def = self.workflow_definitions[instance.workflow_id]
            
            total_tasks = len(workflow_def.tasks)
            completed_tasks = len(instance.results)
            
            instance.progress = (completed_tasks / total_tasks) * 100.0
            
        except Exception as e:
            logger.error(f"Progress update error: {e}")
    
    def _is_workflow_complete(self, instance_id: str) -> bool:
        """Check if workflow is complete."""
        try:
            instance = self.active_workflows[instance_id]
            workflow_def = self.workflow_definitions[instance.workflow_id]
            
            return len(instance.results) == len(workflow_def.tasks)
            
        except Exception as e:
            logger.error(f"Workflow completion check error: {e}")
            return False
    
    async def _complete_workflow(self, instance_id: str):
        """Mark workflow as completed."""
        try:
            instance = self.active_workflows[instance_id]
            instance.status = WorkflowStatus.COMPLETED
            instance.end_time = time.time()
            instance.progress = 100.0
            
            # Calculate execution time
            execution_time = instance.end_time - instance.start_time
            
            # Update metrics
            self.metrics.completed_workflows += 1
            self.metrics.active_workflows = len(self.active_workflows)
            
            # Update average execution time
            total_time = self.metrics.average_execution_time * (self.metrics.completed_workflows - 1)
            self.metrics.average_execution_time = (total_time + execution_time) / self.metrics.completed_workflows
            
            logger.info(f"Workflow completed: {instance_id} in {execution_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Workflow completion error: {e}")
    
    async def _fail_workflow(self, instance_id: str, error_message: str):
        """Mark workflow as failed."""
        try:
            instance = self.active_workflows[instance_id]
            instance.status = WorkflowStatus.FAILED
            instance.end_time = time.time()
            instance.error_message = error_message
            
            # Update metrics
            self.metrics.failed_workflows += 1
            self.metrics.active_workflows = len(self.active_workflows)
            
            logger.error(f"Workflow failed: {instance_id} - {error_message}")
            
        except Exception as e:
            logger.error(f"Workflow failure handling error: {e}")
    
    async def _fail_task(self, instance_id: str, task_id: str, error_message: str):
        """Mark task as failed."""
        try:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = TaskStatus.FAILED
                task.error_message = error_message
                task.end_time = time.time()
                
                del self.running_tasks[task_id]
                
                logger.error(f"Task failed: {task_id} - {error_message}")
            
        except Exception as e:
            logger.error(f"Task failure handling error: {e}")
    
    async def _optimize_task_scheduling(self):
        """Optimize task scheduling based on priorities and dependencies."""
        try:
            # Sort task queue by priority and timestamp
            self.task_queue = deque(
                sorted(self.task_queue, key=lambda x: (-x[0], x[1]))
            )
            
        except Exception as e:
            logger.error(f"Task scheduling optimization error: {e}")
    
    async def _optimize_workflows(self):
        """Optimize workflow execution and resource allocation."""
        try:
            if self.enable_auto_scaling:
                # Adjust thread pool size based on workload
                current_workers = len(self.running_tasks)
                if current_workers > self.max_workers_per_workflow * 0.8:
                    # Scale up
                    new_workers = min(current_workers + 2, self.max_workers_per_workflow)
                    if new_workers != self.thread_pool._max_workers:
                        self.thread_pool._max_workers = new_workers
                        logger.info(f"Scaled up to {new_workers} workers")
                
                elif current_workers < self.max_workers_per_workflow * 0.3:
                    # Scale down
                    new_workers = max(current_workers - 1, 4)
                    if new_workers != self.thread_pool._max_workers:
                        self.thread_pool._max_workers = new_workers
                        logger.info(f"Scaled down to {new_workers} workers")
            
        except Exception as e:
            logger.error(f"Workflow optimization error: {e}")
    
    async def _update_metrics(self):
        """Update workflow metrics."""
        try:
            # Update success rate
            total_completed = self.metrics.completed_workflows + self.metrics.failed_workflows
            if total_completed > 0:
                self.metrics.success_rate = self.metrics.completed_workflows / total_completed
            
            # Update active and queued counts
            self.metrics.active_workflows = len(self.active_workflows)
            self.metrics.queued_workflows = len(self.task_queue)
            
        except Exception as e:
            logger.error(f"Metrics update error: {e}")
    
    async def _perform_cleanup(self):
        """Cleanup old workflow instances and tasks."""
        try:
            current_time = time.time()
            cleanup_threshold = current_time - (self.workflow_timeout_hours * 3600)
            
            # Remove old workflow instances
            instances_to_remove = [
                instance_id for instance_id, instance in self.workflow_instances.items()
                if instance.end_time and instance.end_time < cleanup_threshold
            ]
            
            for instance_id in instances_to_remove:
                if instance_id in self.workflow_instances:
                    del self.workflow_instances[instance_id]
                if instance_id in self.active_workflows:
                    del self.active_workflows[instance_id]
            
            # Remove old completed tasks
            tasks_to_remove = [
                task_id for task_id, task in self.completed_tasks.items()
                if task.end_time and task.end_time < cleanup_threshold
            ]
            
            for task_id in tasks_to_remove:
                del self.completed_tasks[task_id]
            
            if instances_to_remove or tasks_to_remove:
                logger.info(f"Cleanup: removed {len(instances_to_remove)} instances, {len(tasks_to_remove)} tasks")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def get_workflow_definition(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition."""
        return self.workflow_definitions.get(workflow_id)
    
    def get_workflow_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """Get workflow instance."""
        return self.workflow_instances.get(instance_id)
    
    def get_active_workflows(self) -> List[WorkflowInstance]:
        """Get all active workflow instances."""
        return list(self.active_workflows.values())
    
    def get_workflow_metrics(self) -> WorkflowMetrics:
        """Get current workflow metrics."""
        return self.metrics
    
    async def cancel_workflow(self, instance_id: str) -> bool:
        """Cancel a running workflow."""
        try:
            if instance_id not in self.active_workflows:
                return False
            
            instance = self.active_workflows[instance_id]
            instance.status = WorkflowStatus.CANCELLED
            instance.end_time = time.time()
            
            # Cancel running tasks
            for task_id in list(self.running_tasks.keys()):
                if task_id in instance.results:
                    continue
                # Cancel task
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
            
            # Remove from active workflows
            del self.active_workflows[instance_id]
            
            self.metrics.active_workflows = len(self.active_workflows)
            
            logger.info(f"Workflow cancelled: {instance_id}")
            return True
            
        except Exception as e:
            logger.error(f"Workflow cancellation error: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the Workflow Orchestrator."""
        try:
            # Cancel background tasks
            if self.workflow_execution_task:
                self.workflow_execution_task.cancel()
            if self.task_scheduling_task:
                self.task_scheduling_task.cancel()
            if self.optimization_task:
                self.optimization_task.cancel()
            if self.monitoring_task:
                self.monitoring_task.cancel()
            if self.cleanup_task:
                self.cleanup_task.cancel()
            
            # Wait for tasks to complete
            tasks = [
                self.workflow_execution_task,
                self.task_scheduling_task,
                self.optimization_task,
                self.monitoring_task,
                self.cleanup_task
            ]
            
            for task in tasks:
                if task and not task.done():
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            logger.info("Workflow Orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Workflow Orchestrator shutdown error: {e}")

# Global Workflow Orchestrator instance
workflow_orchestrator: Optional[WorkflowOrchestrator] = None

def get_workflow_orchestrator() -> WorkflowOrchestrator:
    """Get global Workflow Orchestrator instance."""
    global workflow_orchestrator
    if workflow_orchestrator is None:
        workflow_orchestrator = WorkflowOrchestrator()
    return workflow_orchestrator

async def shutdown_workflow_orchestrator():
    """Shutdown global Workflow Orchestrator."""
    global workflow_orchestrator
    if workflow_orchestrator:
        await workflow_orchestrator.shutdown()
        workflow_orchestrator = None

