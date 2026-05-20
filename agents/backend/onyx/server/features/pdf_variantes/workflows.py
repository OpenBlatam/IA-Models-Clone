"""
Workflow Engine for PDF Variantes
==================================

Workflow automation and step execution engine.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import asyncio

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowTrigger(str, Enum):
    """Workflow trigger types."""
    MANUAL = "manual"
    FILE_UPLOAD = "file_upload"
    SCHEDULE = "schedule"
    WEBHOOK = "webhook"
    API = "api"


@dataclass
class WorkflowStep:
    """A step in a workflow."""
    step_id: str
    name: str
    action: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: int = 300
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "parameters": self.parameters,
            "depends_on": self.depends_on,
            "retry_count": self.retry_count,
            "timeout": self.timeout
        }


@dataclass
class WorkflowExecution:
    """Workflow execution instance."""
    execution_id: str
    workflow_name: str
    file_id: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    steps: List[WorkflowStep] = field(default_factory=list)
    current_step: Optional[str] = None
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_name": self.workflow_name,
            "file_id": self.file_id,
            "status": self.status.value,
            "steps": [s.to_dict() for s in self.steps],
            "current_step": self.current_step,
            "results": self.results,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat()
        }


class WorkflowEngine:
    """Workflow execution engine."""
    
    def __init__(self):
        self.workflows: Dict[str, List[WorkflowStep]] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.running_executions: Dict[str, bool] = {}
        logger.info("Initialized Workflow Engine")
    
    def register_workflow(
        self,
        name: str,
        steps: List[WorkflowStep]
    ):
        """Register a workflow."""
        self.workflows[name] = steps
        logger.info(f"Registered workflow: {name} with {len(steps)} steps")
    
    async def execute_workflow(
        self,
        workflow_name: str,
        file_id: Optional[str] = None,
        trigger: WorkflowTrigger = WorkflowTrigger.MANUAL
    ) -> str:
        """Execute a workflow."""
        
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_name}")
        
        execution_id = str(uuid4())
        steps = self.workflows[workflow_name]
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_name=workflow_name,
            file_id=file_id,
            status=WorkflowStatus.PENDING,
            steps=steps
        )
        
        self.executions[execution_id] = execution
        self.running_executions[execution_id] = True
        
        # Execute in background
        asyncio.create_task(self._run_workflow(execution))
        
        return execution_id
    
    async def _run_workflow(self, execution: WorkflowExecution):
        """Run workflow execution."""
        
        execution.status = WorkflowStatus.RUNNING
        execution.started_at = datetime.utcnow()
        
        try:
            for step in execution.steps:
                if not self.running_executions.get(execution.execution_id):
                    execution.status = WorkflowStatus.CANCELLED
                    break
                
                execution.current_step = step.step_id
                
                # Execute step
                result = await self._execute_step(step, execution.file_id)
                execution.results[step.step_id] = result
                
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.utcnow()
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()
            logger.error(f"Workflow execution failed: {e}")
        
        finally:
            self.running_executions.pop(execution.execution_id, None)
    
    async def _execute_step(
        self,
        step: WorkflowStep,
        file_id: Optional[str] = None
    ) -> Any:
        """Execute a single step."""
        
        try:
            # Build parameters
            params = {**step.parameters}
            if file_id:
                params['file_id'] = file_id
            
            # Execute with timeout
            result = await asyncio.wait_for(
                step.action(**params),
                timeout=step.timeout
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Step {step.name} timed out")
            raise
        except Exception as e:
            logger.error(f"Step {step.name} failed: {e}")
            raise
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        
        if execution_id in self.running_executions:
            self.running_executions[execution_id] = False
            
            if execution_id in self.executions:
                self.executions[execution_id].status = WorkflowStatus.CANCELLED
                self.executions[execution_id].completed_at = datetime.utcnow()
            
            logger.info(f"Cancelled execution: {execution_id}")
            return True
        
        return False
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution details."""
        return self.executions.get(execution_id)
    
    def get_executions_by_workflow(self, workflow_name: str) -> List[WorkflowExecution]:
        """Get all executions for a workflow."""
        return [
            exec for exec in self.executions.values()
            if exec.workflow_name == workflow_name
        ]
    
    async def pause_execution(self, execution_id: str) -> bool:
        """Pause a running execution."""
        
        if execution_id in self.executions:
            exec = self.executions[execution_id]
            if exec.status == WorkflowStatus.RUNNING:
                exec.status = WorkflowStatus.PAUSED
                self.running_executions[execution_id] = False
                logger.info(f"Paused execution: {execution_id}")
                return True
        
        return False
    
    async def resume_execution(self, execution_id: str) -> bool:
        """Resume a paused execution."""
        
        if execution_id in self.executions:
            exec = self.executions[execution_id]
            if exec.status == WorkflowStatus.PAUSED:
                exec.status = WorkflowStatus.RUNNING
                self.running_executions[execution_id] = True
                asyncio.create_task(self._run_workflow(exec))
                logger.info(f"Resumed execution: {execution_id}")
                return True
        
        return False
    
    def get_workflow_list(self) -> List[str]:
        """Get list of registered workflows."""
        return list(self.workflows.keys())
    
    def get_workflow_steps(self, workflow_name: str) -> List[WorkflowStep]:
        """Get steps for a workflow."""
        return self.workflows.get(workflow_name, [])
