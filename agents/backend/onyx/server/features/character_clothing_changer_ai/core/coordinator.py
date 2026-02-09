"""
Coordinator
===========

System for coordinating multiple components and operations.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CoordinationStatus(Enum):
    """Coordination status."""
    IDLE = "idle"
    COORDINATING = "coordinating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CoordinationTask:
    """Coordination task."""
    id: str
    name: str
    component: str
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationResult:
    """Coordination result."""
    coordination_id: str
    status: CoordinationStatus
    task_results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    duration: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class Coordinator:
    """Component coordinator."""
    
    def __init__(self):
        """Initialize coordinator."""
        self.components: Dict[str, Any] = {}
        self.tasks: Dict[str, CoordinationTask] = {}
        self.execution_order: List[str] = []
    
    def register_component(self, name: str, component: Any):
        """
        Register a component.
        
        Args:
            name: Component name
            component: Component instance
        """
        self.components[name] = component
        logger.debug(f"Registered component: {name}")
    
    def add_task(
        self,
        task_id: str,
        name: str,
        component: str,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        priority: int = 0,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add coordination task.
        
        Args:
            task_id: Task ID
            name: Task name
            component: Component name
            operation: Operation name
            parameters: Optional parameters
            dependencies: Optional task dependencies
            priority: Task priority
            timeout: Optional timeout
            metadata: Optional metadata
        """
        task = CoordinationTask(
            id=task_id,
            name=name,
            component=component,
            operation=operation,
            parameters=parameters or {},
            dependencies=dependencies or [],
            priority=priority,
            timeout=timeout,
            metadata=metadata or {}
        )
        self.tasks[task_id] = task
        logger.debug(f"Added task {task_id} to coordination")
    
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
    
    async def coordinate(self) -> CoordinationResult:
        """
        Coordinate tasks.
        
        Returns:
            Coordination result
        """
        coordination_id = f"coord_{datetime.now().timestamp()}"
        start = datetime.now()
        task_results = {}
        errors = {}
        
        self.execution_order = self._calculate_execution_order()
        
        try:
            for task_id in self.execution_order:
                task = self.tasks[task_id]
                
                # Check component exists
                if task.component not in self.components:
                    error = f"Component {task.component} not found"
                    errors[task_id] = error
                    continue
                
                component = self.components[task.component]
                
                # Check operation exists
                if not hasattr(component, task.operation):
                    error = f"Operation {task.operation} not found in component {task.component}"
                    errors[task_id] = error
                    continue
                
                operation = getattr(component, task.operation)
                
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
                    error = str(e)
                    errors[task_id] = error
                    logger.error(f"Task {task_id} failed: {error}")
            
            duration = (datetime.now() - start).total_seconds()
            status = CoordinationStatus.COMPLETED if not errors else CoordinationStatus.FAILED
            
            return CoordinationResult(
                coordination_id=coordination_id,
                status=status,
                task_results=task_results,
                errors=errors,
                duration=duration
            )
            
        except Exception as e:
            duration = (datetime.now() - start).total_seconds()
            return CoordinationResult(
                coordination_id=coordination_id,
                status=CoordinationStatus.FAILED,
                task_results=task_results,
                errors={**errors, "coordination": str(e)},
                duration=duration
            )

