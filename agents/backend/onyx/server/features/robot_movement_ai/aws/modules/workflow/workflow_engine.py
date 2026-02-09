"""
Workflow Engine
===============

Workflow orchestration engine.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Workflow step definition."""
    name: str
    handler: Callable
    depends_on: List[str] = None
    retry_count: int = 0
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = []


@dataclass
class WorkflowExecution:
    """Workflow execution."""
    id: str
    workflow_name: str
    status: WorkflowStatus
    steps: List[WorkflowStep]
    current_step: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None


class WorkflowEngine:
    """Workflow orchestration engine."""
    
    def __init__(self):
        self._workflows: Dict[str, List[WorkflowStep]] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
    
    def register_workflow(self, name: str, steps: List[WorkflowStep]):
        """Register workflow."""
        self._workflows[name] = steps
        logger.info(f"Registered workflow: {name} with {len(steps)} steps")
    
    async def execute_workflow(
        self,
        workflow_name: str,
        initial_data: Any = None
    ) -> str:
        """Execute workflow."""
        if workflow_name not in self._workflows:
            raise ValueError(f"Workflow {workflow_name} not found")
        
        import uuid
        execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            id=execution_id,
            workflow_name=workflow_name,
            status=WorkflowStatus.RUNNING,
            steps=self._workflows[workflow_name].copy(),
            started_at=datetime.now()
        )
        
        self._executions[execution_id] = execution
        
        # Execute workflow asynchronously
        asyncio.create_task(self._execute_steps(execution, initial_data))
        
        return execution_id
    
    async def _execute_steps(self, execution: WorkflowExecution, data: Any):
        """Execute workflow steps."""
        try:
            completed_steps = set()
            
            while len(completed_steps) < len(execution.steps):
                # Find steps ready to execute
                ready_steps = [
                    step for step in execution.steps
                    if step.name not in completed_steps
                    and all(dep in completed_steps for dep in step.depends_on)
                ]
                
                if not ready_steps:
                    # Check for circular dependencies or missing steps
                    break
                
                # Execute ready steps in parallel
                tasks = []
                for step in ready_steps:
                    execution.current_step = step.name
                    task = self._execute_step(step, data)
                    tasks.append((step.name, task))
                
                results = await asyncio.gather(
                    *[task for _, task in tasks],
                    return_exceptions=True
                )
                
                for (step_name, _), result in zip(tasks, results):
                    if isinstance(result, Exception):
                        execution.status = WorkflowStatus.FAILED
                        execution.error = str(result)
                        logger.error(f"Step {step_name} failed: {result}")
                        return
                    
                    completed_steps.add(step_name)
                    data = result  # Pass result to next steps
            
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
            execution.result = data
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
            logger.error(f"Workflow execution failed: {e}")
    
    async def _execute_step(self, step: WorkflowStep, data: Any) -> Any:
        """Execute single step."""
        for attempt in range(step.retry_count + 1):
            try:
                if step.timeout:
                    result = await asyncio.wait_for(
                        step.handler(data) if asyncio.iscoroutinefunction(step.handler) else asyncio.to_thread(step.handler, data),
                        timeout=step.timeout
                    )
                else:
                    if asyncio.iscoroutinefunction(step.handler):
                        result = await step.handler(data)
                    else:
                        result = await asyncio.to_thread(step.handler, data)
                
                return result
            
            except Exception as e:
                if attempt < step.retry_count:
                    logger.warning(f"Step {step.name} failed, retrying: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution."""
        return self._executions.get(execution_id)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_executions": len(self._executions),
            "by_status": {
                status.value: sum(1 for e in self._executions.values() if e.status == status)
                for status in WorkflowStatus
            }
        }















