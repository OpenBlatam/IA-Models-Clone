"""
Orchestrator System
===================

System for orchestrating complex operations and services.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OrchestrationStatus(Enum):
    """Orchestration status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


@dataclass
class OrchestrationTask:
    """Orchestration task definition."""
    id: str
    name: str
    service: str
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    timeout: Optional[float] = None
    priority: int = 0  # Higher = higher priority
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Orchestration execution result."""
    orchestration_id: str
    status: OrchestrationStatus
    task_results: Dict[str, Any] = field(default_factory=dict)
    task_errors: Dict[str, str] = field(default_factory=dict)
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class Orchestrator:
    """Service orchestrator."""
    
    def __init__(self):
        """Initialize orchestrator."""
        self.services: Dict[str, Any] = {}
        self.tasks: Dict[str, OrchestrationTask] = {}
        self.execution_order: List[str] = []
    
    def register_service(self, name: str, service: Any):
        """
        Register a service.
        
        Args:
            name: Service name
            service: Service instance
        """
        self.services[name] = service
        logger.debug(f"Registered service: {name}")
    
    def add_task(
        self,
        task_id: str,
        name: str,
        service: str,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        retry_count: int = 0,
        timeout: Optional[float] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add orchestration task.
        
        Args:
            task_id: Task ID
            name: Task name
            service: Service name
            operation: Operation name
            parameters: Optional parameters
            dependencies: Optional task dependencies
            retry_count: Retry count
            timeout: Optional timeout
            priority: Task priority
            metadata: Optional metadata
        """
        task = OrchestrationTask(
            id=task_id,
            name=name,
            service=service,
            operation=operation,
            parameters=parameters or {},
            dependencies=dependencies or [],
            retry_count=retry_count,
            timeout=timeout,
            priority=priority,
            metadata=metadata or {}
        )
        self.tasks[task_id] = task
        logger.debug(f"Added task {task_id} to orchestration")
    
    def _calculate_execution_order(self) -> List[str]:
        """Calculate execution order based on dependencies and priority."""
        # Topological sort with priority
        order = []
        visited = set()
        temp_visited = set()
        
        def visit(task_id: str):
            if task_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {task_id}")
            if task_id in visited:
                return
            
            temp_visited.add(task_id)
            
            if task_id in self.tasks:
                task = self.tasks[task_id]
                for dep in task.dependencies:
                    visit(dep)
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            order.append(task_id)
        
        # Sort by priority first
        sorted_tasks = sorted(
            self.tasks.items(),
            key=lambda x: (x[1].priority, x[0]),
            reverse=True
        )
        
        for task_id, _ in sorted_tasks:
            if task_id not in visited:
                visit(task_id)
        
        return order
    
    async def execute(self) -> OrchestrationResult:
        """
        Execute orchestration.
        
        Returns:
            Orchestration result
        """
        orchestration_id = f"orch_{datetime.now().timestamp()}"
        start = datetime.now()
        task_results = {}
        task_errors = {}
        
        self.execution_order = self._calculate_execution_order()
        
        try:
            for task_id in self.execution_order:
                task = self.tasks[task_id]
                
                # Check service exists
                if task.service not in self.services:
                    error = f"Service {task.service} not found"
                    task_errors[task_id] = error
                    raise ValueError(error)
                
                service = self.services[task.service]
                
                # Check operation exists
                if not hasattr(service, task.operation):
                    error = f"Operation {task.operation} not found in service {task.service}"
                    task_errors[task_id] = error
                    raise ValueError(error)
                
                operation = getattr(service, task.operation)
                
                # Execute operation
                try:
                    if task.timeout:
                        result = await asyncio.wait_for(
                            operation(**task.parameters),
                            timeout=task.timeout
                        )
                    else:
                        result = await operation(**task.parameters)
                    
                    task_results[task_id] = result
                    logger.info(f"Task {task_id} completed")
                    
                except Exception as e:
                    if task.retry_count > 0:
                        logger.warning(f"Task {task_id} failed, retrying: {e}")
                        for attempt in range(task.retry_count):
                            try:
                                result = await operation(**task.parameters)
                                task_results[task_id] = result
                                break
                            except Exception as retry_error:
                                if attempt == task.retry_count - 1:
                                    task_errors[task_id] = str(retry_error)
                                    raise
                    else:
                        task_errors[task_id] = str(e)
                        raise
            
            duration = (datetime.now() - start).total_seconds()
            status = OrchestrationStatus.PARTIAL if task_errors else OrchestrationStatus.COMPLETED
            
            return OrchestrationResult(
                orchestration_id=orchestration_id,
                status=status,
                task_results=task_results,
                task_errors=task_errors,
                duration=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            return OrchestrationResult(
                orchestration_id=orchestration_id,
                status=OrchestrationStatus.FAILED,
                task_results=task_results,
                task_errors=task_errors,
                duration=duration
            )




