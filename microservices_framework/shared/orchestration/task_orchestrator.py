"""
Advanced Distributed Computing and Task Orchestration
Features: Distributed task execution, workflow orchestration, DAG processing, resource management
"""

import asyncio
import time
import json
import uuid
from typing import Dict, Any, Optional, List, Union, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import statistics
from abc import ABC, abstractmethod
import networkx as nx

# Distributed computing imports
try:
    import dask
    import dask.distributed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import celery
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

try:
    import redis
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskPriority(Enum):
    """Task priority"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class ResourceType(Enum):
    """Resource types"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"

class WorkflowStatus(Enum):
    """Workflow status"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Task definition"""
    task_id: str
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 300.0
    resources: Dict[ResourceType, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    worker_id: Optional[str] = None

@dataclass
class Worker:
    """Worker definition"""
    worker_id: str
    host: str
    port: int
    resources: Dict[ResourceType, float]
    available_resources: Dict[ResourceType, float] = field(default_factory=dict)
    status: str = "idle"
    current_tasks: Set[str] = field(default_factory=set)
    total_tasks_completed: int = 0
    last_heartbeat: float = field(default_factory=time.time)

@dataclass
class Workflow:
    """Workflow definition"""
    workflow_id: str
    name: str
    tasks: Dict[str, Task] = field(default_factory=dict)
    dag: nx.DiGraph = field(default_factory=nx.DiGraph)
    status: WorkflowStatus = WorkflowStatus.CREATED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class TaskExecutor(ABC):
    """Abstract task executor"""
    
    @abstractmethod
    async def execute_task(self, task: Task) -> Any:
        """Execute a task"""
        pass
    
    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        pass

class LocalTaskExecutor(TaskExecutor):
    """Local task executor"""
    
    def __init__(self):
        self.running_tasks: Dict[str, asyncio.Task] = {}
    
    async def execute_task(self, task: Task) -> Any:
        """Execute task locally"""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            # Create async task
            if asyncio.iscoroutinefunction(task.function):
                result = await task.function(*task.args, **task.kwargs)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, task.function, *task.args, **task.kwargs
                )
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            raise
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.running_tasks:
            self.running_tasks[task_id].cancel()
            del self.running_tasks[task_id]
            return True
        return False

class DaskTaskExecutor(TaskExecutor):
    """Dask distributed task executor"""
    
    def __init__(self, cluster_address: str = None):
        self.cluster_address = cluster_address
        self.client: Optional[dask.distributed.Client] = None
        self.connected = False
    
    async def connect(self):
        """Connect to Dask cluster"""
        try:
            if not DASK_AVAILABLE:
                raise ImportError("Dask not available")
            
            if self.cluster_address:
                self.client = dask.distributed.Client(self.cluster_address)
            else:
                self.client = dask.distributed.Client()
            
            self.connected = True
            logger.info("Connected to Dask cluster")
            
        except Exception as e:
            logger.error(f"Failed to connect to Dask cluster: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Dask cluster"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from Dask cluster")
    
    async def execute_task(self, task: Task) -> Any:
        """Execute task on Dask cluster"""
        try:
            if not self.connected or not self.client:
                raise RuntimeError("Not connected to Dask cluster")
            
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            # Submit task to Dask
            future = self.client.submit(
                task.function,
                *task.args,
                **task.kwargs,
                retries=task.max_retries
            )
            
            # Wait for result with timeout
            result = future.result(timeout=task.timeout)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            raise
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a Dask task"""
        # Dask doesn't provide direct task cancellation by ID
        # This would need to be implemented with task tracking
        return False

class RayTaskExecutor(TaskExecutor):
    """Ray distributed task executor"""
    
    def __init__(self, ray_address: str = None):
        self.ray_address = ray_address
        self.connected = False
    
    async def connect(self):
        """Connect to Ray cluster"""
        try:
            if not RAY_AVAILABLE:
                raise ImportError("Ray not available")
            
            if self.ray_address:
                ray.init(address=self.ray_address)
            else:
                ray.init()
            
            self.connected = True
            logger.info("Connected to Ray cluster")
            
        except Exception as e:
            logger.error(f"Failed to connect to Ray cluster: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Ray cluster"""
        if self.connected:
            ray.shutdown()
            self.connected = False
            logger.info("Disconnected from Ray cluster")
    
    async def execute_task(self, task: Task) -> Any:
        """Execute task on Ray cluster"""
        try:
            if not self.connected:
                raise RuntimeError("Not connected to Ray cluster")
            
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            # Create Ray remote function
            @ray.remote
            def ray_task(func, *args, **kwargs):
                return func(*args, **kwargs)
            
            # Submit task to Ray
            future = ray_task.remote(task.function, *task.args, **task.kwargs)
            
            # Wait for result with timeout
            result = ray.get(future, timeout=task.timeout)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            return result
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            raise
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a Ray task"""
        # Ray doesn't provide direct task cancellation by ID
        return False

class ResourceManager:
    """
    Resource management for distributed computing
    """
    
    def __init__(self):
        self.workers: Dict[str, Worker] = {}
        self.resource_allocations: Dict[str, Dict[ResourceType, float]] = {}
        self.resource_requests: deque = deque()
    
    def register_worker(self, worker: Worker):
        """Register a worker"""
        self.workers[worker.worker_id] = worker
        self.resource_allocations[worker.worker_id] = {}
        logger.info(f"Registered worker: {worker.worker_id}")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker"""
        if worker_id in self.workers:
            del self.workers[worker_id]
            del self.resource_allocations[worker_id]
            logger.info(f"Unregistered worker: {worker_id}")
    
    def update_worker_resources(self, worker_id: str, resources: Dict[ResourceType, float]):
        """Update worker resource availability"""
        if worker_id in self.workers:
            self.workers[worker_id].available_resources = resources.copy()
            self.workers[worker_id].last_heartbeat = time.time()
    
    def find_best_worker(self, required_resources: Dict[ResourceType, float]) -> Optional[str]:
        """Find the best worker for given resource requirements"""
        best_worker = None
        best_score = -1
        
        for worker_id, worker in self.workers.items():
            # Check if worker has sufficient resources
            if not self._has_sufficient_resources(worker, required_resources):
                continue
            
            # Calculate resource utilization score
            score = self._calculate_resource_score(worker, required_resources)
            
            if score > best_score:
                best_score = score
                best_worker = worker_id
        
        return best_worker
    
    def _has_sufficient_resources(self, worker: Worker, required: Dict[ResourceType, float]) -> bool:
        """Check if worker has sufficient resources"""
        for resource_type, amount in required.items():
            available = worker.available_resources.get(resource_type, 0)
            allocated = self.resource_allocations[worker.worker_id].get(resource_type, 0)
            
            if available - allocated < amount:
                return False
        
        return True
    
    def _calculate_resource_score(self, worker: Worker, required: Dict[ResourceType, float]) -> float:
        """Calculate resource utilization score"""
        total_score = 0.0
        
        for resource_type, amount in required.items():
            available = worker.available_resources.get(resource_type, 0)
            allocated = self.resource_allocations[worker.worker_id].get(resource_type, 0)
            utilization = (allocated + amount) / available if available > 0 else 0
            
            # Prefer workers with lower utilization
            score = 1.0 - utilization
            total_score += score
        
        return total_score / len(required) if required else 0.0
    
    def allocate_resources(self, worker_id: str, resources: Dict[ResourceType, float]) -> bool:
        """Allocate resources on a worker"""
        if worker_id not in self.workers:
            return False
        
        worker = self.workers[worker_id]
        
        # Check if resources are available
        if not self._has_sufficient_resources(worker, resources):
            return False
        
        # Allocate resources
        for resource_type, amount in resources.items():
            current_allocated = self.resource_allocations[worker_id].get(resource_type, 0)
            self.resource_allocations[worker_id][resource_type] = current_allocated + amount
        
        return True
    
    def deallocate_resources(self, worker_id: str, resources: Dict[ResourceType, float]):
        """Deallocate resources from a worker"""
        if worker_id not in self.resource_allocations:
            return
        
        for resource_type, amount in resources.items():
            current_allocated = self.resource_allocations[worker_id].get(resource_type, 0)
            self.resource_allocations[worker_id][resource_type] = max(0, current_allocated - amount)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource utilization statistics"""
        stats = {
            "total_workers": len(self.workers),
            "active_workers": len([w for w in self.workers.values() if w.status == "active"]),
            "total_resources": {},
            "allocated_resources": {},
            "available_resources": {}
        }
        
        # Calculate total resources
        for worker in self.workers.values():
            for resource_type, amount in worker.resources.items():
                stats["total_resources"][resource_type.value] = stats["total_resources"].get(resource_type.value, 0) + amount
        
        # Calculate allocated resources
        for worker_id, allocations in self.resource_allocations.items():
            for resource_type, amount in allocations.items():
                stats["allocated_resources"][resource_type.value] = stats["allocated_resources"].get(resource_type.value, 0) + amount
        
        # Calculate available resources
        for resource_type in ResourceType:
            total = stats["total_resources"].get(resource_type.value, 0)
            allocated = stats["allocated_resources"].get(resource_type.value, 0)
            stats["available_resources"][resource_type.value] = total - allocated
        
        return stats

class WorkflowEngine:
    """
    Workflow orchestration engine
    """
    
    def __init__(self, task_orchestrator):
        self.task_orchestrator = task_orchestrator
        self.workflows: Dict[str, Workflow] = {}
        self.running_workflows: Set[str] = set()
        self.workflow_tasks: Dict[str, asyncio.Task] = {}
    
    def create_workflow(self, workflow_id: str, name: str) -> Workflow:
        """Create a new workflow"""
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name
        )
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {workflow_id}")
        
        return workflow
    
    def add_task_to_workflow(self, workflow_id: str, task: Task):
        """Add task to workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        workflow.tasks[task.task_id] = task
        workflow.dag.add_node(task.task_id)
        
        # Add dependencies
        for dep_id in task.dependencies:
            if dep_id in workflow.tasks:
                workflow.dag.add_edge(dep_id, task.task_id)
        
        logger.info(f"Added task {task.task_id} to workflow {workflow_id}")
    
    def validate_workflow(self, workflow_id: str) -> bool:
        """Validate workflow DAG"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        
        # Check for cycles
        try:
            cycles = list(nx.simple_cycles(workflow.dag))
            if cycles:
                logger.error(f"Workflow {workflow_id} has cycles: {cycles}")
                return False
        except nx.NetworkXError:
            pass
        
        # Check for unreachable tasks
        reachable = set()
        for node in workflow.dag.nodes():
            if workflow.dag.in_degree(node) == 0:  # Root nodes
                reachable.update(nx.descendants(workflow.dag, node))
                reachable.add(node)
        
        unreachable = set(workflow.dag.nodes()) - reachable
        if unreachable:
            logger.warning(f"Workflow {workflow_id} has unreachable tasks: {unreachable}")
        
        return True
    
    async def execute_workflow(self, workflow_id: str) -> bool:
        """Execute a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if not self.validate_workflow(workflow_id):
            return False
        
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = time.time()
        self.running_workflows.add(workflow_id)
        
        # Create workflow execution task
        task = asyncio.create_task(self._execute_workflow_async(workflow_id))
        self.workflow_tasks[workflow_id] = task
        
        try:
            await task
            return True
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            workflow.status = WorkflowStatus.FAILED
            return False
        finally:
            self.running_workflows.discard(workflow_id)
            self.workflow_tasks.pop(workflow_id, None)
    
    async def _execute_workflow_async(self, workflow_id: str):
        """Execute workflow asynchronously"""
        workflow = self.workflows[workflow_id]
        
        # Get topological order of tasks
        try:
            task_order = list(nx.topological_sort(workflow.dag))
        except nx.NetworkXError:
            task_order = list(workflow.tasks.keys())
        
        # Execute tasks in order
        for task_id in task_order:
            if task_id not in workflow.tasks:
                continue
            
            task = workflow.tasks[task_id]
            
            # Wait for dependencies
            for dep_id in task.dependencies:
                if dep_id in workflow.tasks:
                    dep_task = workflow.tasks[dep_id]
                    while dep_task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        await asyncio.sleep(0.1)
                    
                    if dep_task.status == TaskStatus.FAILED:
                        logger.error(f"Dependency {dep_id} failed for task {task_id}")
                        workflow.status = WorkflowStatus.FAILED
                        return
            
            # Execute task
            try:
                await self.task_orchestrator.execute_task(task)
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                workflow.status = WorkflowStatus.FAILED
                return
        
        # All tasks completed successfully
        workflow.status = WorkflowStatus.COMPLETED
        workflow.completed_at = time.time()
        logger.info(f"Workflow {workflow_id} completed successfully")
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id not in self.running_workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.CANCELLED
        
        # Cancel workflow task
        if workflow_id in self.workflow_tasks:
            self.workflow_tasks[workflow_id].cancel()
        
        # Cancel running tasks
        for task in workflow.tasks.values():
            if task.status == TaskStatus.RUNNING:
                await self.task_orchestrator.cancel_task(task.task_id)
                task.status = TaskStatus.CANCELLED
        
        self.running_workflows.discard(workflow_id)
        logger.info(f"Cancelled workflow: {workflow_id}")
        
        return True
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow statistics"""
        return {
            "total_workflows": len(self.workflows),
            "running_workflows": len(self.running_workflows),
            "completed_workflows": len([w for w in self.workflows.values() if w.status == WorkflowStatus.COMPLETED]),
            "failed_workflows": len([w for w in self.workflows.values() if w.status == WorkflowStatus.FAILED]),
            "workflows": {
                workflow_id: {
                    "name": workflow.name,
                    "status": workflow.status.value,
                    "tasks": len(workflow.tasks),
                    "created_at": workflow.created_at,
                    "started_at": workflow.started_at,
                    "completed_at": workflow.completed_at
                }
                for workflow_id, workflow in self.workflows.items()
            }
        }

class TaskOrchestrator:
    """
    Main task orchestration manager
    """
    
    def __init__(self, executor_type: str = "local", **executor_kwargs):
        self.executor_type = executor_type
        self.executor: Optional[TaskExecutor] = None
        self.resource_manager = ResourceManager()
        self.workflow_engine = WorkflowEngine(self)
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.orchestration_active = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # Initialize executor
        self._initialize_executor(**executor_kwargs)
    
    def _initialize_executor(self, **kwargs):
        """Initialize task executor"""
        if self.executor_type == "local":
            self.executor = LocalTaskExecutor()
        elif self.executor_type == "dask":
            self.executor = DaskTaskExecutor(kwargs.get("cluster_address"))
        elif self.executor_type == "ray":
            self.executor = RayTaskExecutor(kwargs.get("ray_address"))
        else:
            raise ValueError(f"Unsupported executor type: {self.executor_type}")
    
    async def start_orchestration(self):
        """Start task orchestration"""
        if self.orchestration_active:
            return
        
        self.orchestration_active = True
        
        # Connect to distributed executor if needed
        if hasattr(self.executor, 'connect'):
            await self.executor.connect()
        
        # Start worker tasks
        for i in range(4):  # 4 worker tasks
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.append(worker)
        
        logger.info("Task orchestration started")
    
    async def stop_orchestration(self):
        """Stop task orchestration"""
        self.orchestration_active = False
        
        # Cancel worker tasks
        for worker in self.worker_tasks:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        
        # Disconnect from distributed executor if needed
        if hasattr(self.executor, 'disconnect'):
            await self.executor.disconnect()
        
        logger.info("Task orchestration stopped")
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task for execution"""
        # Calculate priority (higher number = higher priority)
        priority = task.priority.value
        
        # Add to queue
        await self.task_queue.put((priority, task.task_id, task))
        
        logger.info(f"Submitted task: {task.task_id}")
        return task.task_id
    
    async def execute_task(self, task: Task) -> Any:
        """Execute a task directly"""
        try:
            # Allocate resources if needed
            if task.resources:
                worker_id = self.resource_manager.find_best_worker(task.resources)
                if worker_id:
                    self.resource_manager.allocate_resources(worker_id, task.resources)
                    task.worker_id = worker_id
            
            # Execute task
            result = await self.executor.execute_task(task)
            
            # Deallocate resources
            if task.resources and task.worker_id:
                self.resource_manager.deallocate_resources(task.worker_id, task.resources)
            
            # Store completed task
            self.completed_tasks[task.task_id] = task
            
            return result
            
        except Exception as e:
            # Handle retries
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                
                # Wait before retry
                await asyncio.sleep(task.retry_delay * (2 ** task.retry_count))
                
                # Retry task
                return await self.execute_task(task)
            else:
                # Max retries exceeded
                task.status = TaskStatus.FAILED
                task.error = str(e)
                self.completed_tasks[task.task_id] = task
                raise
    
    async def _worker(self, worker_id: str):
        """Worker task for processing task queue"""
        while self.orchestration_active:
            try:
                # Get task from queue with timeout
                priority, task_id, task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Execute task
                self.running_tasks[task_id] = task
                await self.execute_task(task)
                
                # Remove from running tasks
                self.running_tasks.pop(task_id, None)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        # Cancel in executor
        if self.executor:
            await self.executor.cancel_task(task_id)
        
        # Update task status
        if task_id in self.running_tasks:
            self.running_tasks[task_id].status = TaskStatus.CANCELLED
            return True
        
        return False
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics"""
        return {
            "orchestration_active": self.orchestration_active,
            "executor_type": self.executor_type,
            "queue_size": self.task_queue.qsize(),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "resource_stats": self.resource_manager.get_resource_stats(),
            "workflow_stats": self.workflow_engine.get_workflow_stats()
        }

# Global task orchestrator
task_orchestrator = TaskOrchestrator()

# Utility functions
def create_task(
    name: str,
    function: Callable,
    args: tuple = (),
    kwargs: dict = None,
    priority: TaskPriority = TaskPriority.NORMAL,
    max_retries: int = 3,
    timeout: float = 300.0,
    resources: Dict[ResourceType, float] = None,
    dependencies: List[str] = None
) -> Task:
    """Create a new task"""
    return Task(
        task_id=str(uuid.uuid4()),
        name=name,
        function=function,
        args=args,
        kwargs=kwargs or {},
        priority=priority,
        max_retries=max_retries,
        timeout=timeout,
        resources=resources or {},
        dependencies=dependencies or []
    )

async def submit_task_async(task: Task) -> str:
    """Submit task asynchronously"""
    return await task_orchestrator.submit_task(task)

# Decorator for task execution
def distributed_task(
    priority: TaskPriority = TaskPriority.NORMAL,
    max_retries: int = 3,
    timeout: float = 300.0,
    resources: Dict[ResourceType, float] = None
):
    """Decorator for distributed task execution"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            task = create_task(
                name=func.__name__,
                function=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                max_retries=max_retries,
                timeout=timeout,
                resources=resources or {}
            )
            
            return await task_orchestrator.execute_task(task)
        
        def sync_wrapper(*args, **kwargs):
            task = create_task(
                name=func.__name__,
                function=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                max_retries=max_retries,
                timeout=timeout,
                resources=resources or {}
            )
            
            # For sync functions, run in thread pool
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(task_orchestrator.execute_task(task))
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator






























