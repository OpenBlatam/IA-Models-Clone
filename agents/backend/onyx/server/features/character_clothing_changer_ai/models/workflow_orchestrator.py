"""
Workflow Orchestrator for Flux2 Clothing Changer
==================================================

Workflow orchestration and task management.
"""

import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowTask:
    """Workflow task."""
    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class Workflow:
    """Workflow definition."""
    workflow_id: str
    name: str
    tasks: List[WorkflowTask]
    created_at: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING


class WorkflowOrchestrator:
    """Workflow orchestration system."""
    
    def __init__(self):
        """Initialize workflow orchestrator."""
        self.workflows: Dict[str, Workflow] = {}
        self.task_handlers: Dict[str, Callable] = {}
        self.execution_history: deque = deque(maxlen=1000)
    
    def register_handler(
        self,
        task_type: str,
        handler: Callable[[Dict[str, Any]], Any],
    ) -> None:
        """
        Register task handler.
        
        Args:
            task_type: Task type
            handler: Task handler function
        """
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")
    
    def create_workflow(
        self,
        name: str,
        tasks: List[Dict[str, Any]],
    ) -> Workflow:
        """
        Create workflow.
        
        Args:
            name: Workflow name
            tasks: List of task definitions
            
        Returns:
            Created workflow
        """
        workflow_id = str(uuid.uuid4())
        
        workflow_tasks = []
        for task_def in tasks:
            task = WorkflowTask(
                task_id=str(uuid.uuid4()),
                task_type=task_def["type"],
                dependencies=task_def.get("dependencies", []),
                data=task_def.get("data", {}),
            )
            workflow_tasks.append(task)
        
        workflow = Workflow(
            workflow_id=workflow_id,
            name=name,
            tasks=workflow_tasks,
        )
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {workflow_id}")
        return workflow
    
    def execute_workflow(
        self,
        workflow_id: str,
    ) -> Dict[str, Any]:
        """
        Execute workflow.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Execution result
        """
        if workflow_id not in self.workflows:
            return {"error": "Workflow not found"}
        
        workflow = self.workflows[workflow_id]
        workflow.status = TaskStatus.RUNNING
        
        # Execute tasks in dependency order
        completed_tasks = set()
        task_results = {}
        
        while len(completed_tasks) < len(workflow.tasks):
            # Find tasks ready to execute
            ready_tasks = [
                task for task in workflow.tasks
                if task.task_id not in completed_tasks
                and all(dep in completed_tasks for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                # Deadlock or error
                workflow.status = TaskStatus.FAILED
                return {"error": "Workflow deadlock detected"}
            
            # Execute ready tasks
            for task in ready_tasks:
                result = self._execute_task(task)
                task_results[task.task_id] = result
                completed_tasks.add(task.task_id)
                
                if task.status == TaskStatus.FAILED:
                    workflow.status = TaskStatus.FAILED
                    return {
                        "error": f"Task {task.task_id} failed: {task.error}",
                        "results": task_results,
                    }
        
        workflow.status = TaskStatus.COMPLETED
        self.execution_history.append({
            "workflow_id": workflow_id,
            "status": "completed",
            "timestamp": time.time(),
        })
        
        return {
            "status": "completed",
            "results": task_results,
        }
    
    def _execute_task(self, task: WorkflowTask) -> Any:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        if task.task_type not in self.task_handlers:
            task.status = TaskStatus.FAILED
            task.error = f"Handler not found for task type: {task.task_type}"
            return None
        
        try:
            handler = self.task_handlers[task.task_type]
            result = handler(task.data)
            
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            return result
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
            return None
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status."""
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "type": task.task_type,
                    "status": task.status.value,
                    "error": task.error,
                }
                for task in workflow.tasks
            ],
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "total_workflows": len(self.workflows),
            "completed_workflows": len([
                w for w in self.workflows.values()
                if w.status == TaskStatus.COMPLETED
            ]),
            "failed_workflows": len([
                w for w in self.workflows.values()
                if w.status == TaskStatus.FAILED
            ]),
            "execution_history_size": len(self.execution_history),
        }


