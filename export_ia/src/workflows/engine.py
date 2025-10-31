"""
Workflow engine for Export IA automation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import uuid

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Workflow step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(Enum):
    """Workflow status."""
    DRAFT = "draft"
    ACTIVE = "active"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Definition of a workflow step."""
    id: str
    name: str
    description: str
    step_type: str
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    timeout: int = 300  # seconds
    condition: Optional[str] = None
    on_success: Optional[str] = None
    on_failure: Optional[str] = None


@dataclass
class WorkflowDefinition:
    """Definition of a workflow."""
    id: str
    name: str
    description: str
    version: str
    steps: List[WorkflowStep] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class WorkflowExecution:
    """Execution instance of a workflow."""
    id: str
    workflow_id: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    step_executions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    """Main workflow engine for automation."""
    
    def __init__(self):
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.step_handlers: Dict[str, Callable] = {}
        self.running_executions: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Register a workflow definition."""
        self.workflows[workflow.id] = workflow
        self.logger.info(f"Workflow registered: {workflow.name} (v{workflow.version})")
    
    def register_step_handler(self, step_type: str, handler: Callable) -> None:
        """Register a step handler."""
        self.step_handlers[step_type] = handler
        self.logger.info(f"Step handler registered: {step_type}")
    
    async def execute_workflow(
        self, 
        workflow_id: str, 
        variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        execution_id = str(uuid.uuid4())
        
        # Create execution instance
        execution = WorkflowExecution(
            id=execution_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING,
            started_at=datetime.now(),
            variables=variables or {}
        )
        
        self.executions[execution_id] = execution
        
        # Start execution
        task = asyncio.create_task(self._execute_workflow(execution))
        self.running_executions[execution_id] = task
        
        self.logger.info(f"Workflow execution started: {execution_id}")
        return execution_id
    
    async def _execute_workflow(self, execution: WorkflowExecution) -> None:
        """Execute a workflow instance."""
        try:
            workflow = self.workflows[execution.workflow_id]
            
            # Initialize step executions
            for step in workflow.steps:
                execution.step_executions[step.id] = {
                    "status": StepStatus.PENDING,
                    "started_at": None,
                    "completed_at": None,
                    "result": None,
                    "error": None
                }
            
            # Execute steps in dependency order
            await self._execute_steps(execution, workflow)
            
            # Check if all steps completed successfully
            all_completed = all(
                step_exec["status"] == StepStatus.COMPLETED
                for step_exec in execution.step_executions.values()
            )
            
            if all_completed:
                execution.status = WorkflowStatus.COMPLETED
            else:
                execution.status = WorkflowStatus.FAILED
            
            execution.completed_at = datetime.now()
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            self.logger.error(f"Workflow execution failed: {e}")
        
        finally:
            # Clean up running execution
            if execution.id in self.running_executions:
                del self.running_executions[execution.id]
    
    async def _execute_steps(self, execution: WorkflowExecution, workflow: WorkflowDefinition) -> None:
        """Execute workflow steps."""
        completed_steps = set()
        
        while len(completed_steps) < len(workflow.steps):
            # Find steps that can be executed (dependencies satisfied)
            ready_steps = []
            
            for step in workflow.steps:
                if step.id in completed_steps:
                    continue
                
                # Check if dependencies are satisfied
                dependencies_satisfied = all(
                    dep_id in completed_steps for dep_id in step.dependencies
                )
                
                if dependencies_satisfied:
                    ready_steps.append(step)
            
            if not ready_steps:
                # No ready steps, check for failures
                failed_steps = [
                    step_id for step_id, step_exec in execution.step_executions.items()
                    if step_exec["status"] == StepStatus.FAILED
                ]
                
                if failed_steps:
                    raise Exception(f"Workflow failed due to step failures: {failed_steps}")
                else:
                    raise Exception("No ready steps and no failures - workflow deadlock")
            
            # Execute ready steps in parallel
            tasks = []
            for step in ready_steps:
                task = asyncio.create_task(self._execute_step(execution, step))
                tasks.append(task)
            
            # Wait for all ready steps to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update completed steps
            for i, result in enumerate(results):
                step = ready_steps[i]
                if isinstance(result, Exception):
                    execution.step_executions[step.id]["status"] = StepStatus.FAILED
                    execution.step_executions[step.id]["error"] = str(result)
                else:
                    completed_steps.add(step.id)
    
    async def _execute_step(self, execution: WorkflowExecution, step: WorkflowStep) -> None:
        """Execute a single workflow step."""
        step_exec = execution.step_executions[step.id]
        step_exec["status"] = StepStatus.RUNNING
        step_exec["started_at"] = datetime.now()
        
        try:
            # Check if step handler exists
            if step.step_type not in self.step_handlers:
                raise ValueError(f"Step handler not found: {step.step_type}")
            
            handler = self.step_handlers[step.step_type]
            
            # Prepare step context
            context = {
                "step": step,
                "execution": execution,
                "variables": execution.variables,
                "step_results": {
                    step_id: step_exec["result"]
                    for step_id, step_exec in execution.step_executions.items()
                    if step_exec["result"] is not None
                }
            }
            
            # Execute step with timeout
            result = await asyncio.wait_for(
                handler(context),
                timeout=step.timeout
            )
            
            step_exec["status"] = StepStatus.COMPLETED
            step_exec["result"] = result
            step_exec["completed_at"] = datetime.now()
            
            self.logger.info(f"Step completed: {step.name}")
            
        except asyncio.TimeoutError:
            step_exec["status"] = StepStatus.FAILED
            step_exec["error"] = f"Step timeout after {step.timeout} seconds"
            self.logger.error(f"Step timeout: {step.name}")
            
        except Exception as e:
            step_exec["status"] = StepStatus.FAILED
            step_exec["error"] = str(e)
            self.logger.error(f"Step failed: {step.name} - {e}")
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution status."""
        if execution_id not in self.executions:
            return None
        
        execution = self.executions[execution_id]
        
        return {
            "id": execution.id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "started_at": execution.started_at.isoformat(),
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "error_message": execution.error_message,
            "step_executions": execution.step_executions,
            "variables": execution.variables
        }
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution."""
        if execution_id not in self.executions:
            return False
        
        execution = self.executions[execution_id]
        
        if execution.status != WorkflowStatus.RUNNING:
            return False
        
        # Cancel running task
        if execution_id in self.running_executions:
            self.running_executions[execution_id].cancel()
            del self.running_executions[execution_id]
        
        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.now()
        
        self.logger.info(f"Workflow execution cancelled: {execution_id}")
        return True
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """List all registered workflows."""
        return [
            {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "version": workflow.version,
                "steps_count": len(workflow.steps),
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat()
            }
            for workflow in self.workflows.values()
        ]
    
    def list_executions(self) -> List[Dict[str, Any]]:
        """List all executions."""
        return [
            {
                "id": execution.id,
                "workflow_id": execution.workflow_id,
                "status": execution.status.value,
                "started_at": execution.started_at.isoformat(),
                "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
                "error_message": execution.error_message
            }
            for execution in self.executions.values()
        ]


# Predefined workflow steps
class ExportWorkflowSteps:
    """Predefined workflow steps for export operations."""
    
    @staticmethod
    async def validate_content(context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate document content."""
        from src.core.validation import get_validation_manager
        
        content = context["variables"].get("content", {})
        config = context["variables"].get("config", {})
        
        validation_manager = get_validation_manager()
        results = validation_manager.validate_export_request(content, config)
        
        return {
            "validation_results": results,
            "has_errors": validation_manager.has_errors(results)
        }
    
    @staticmethod
    async def enhance_content(context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance document content."""
        from src.ai.enhancer import ContentEnhancer
        
        content = context["variables"].get("content", {})
        
        enhancer = ContentEnhancer()
        enhanced_content = await enhancer.enhance_document_content(content)
        
        return {"enhanced_content": enhanced_content}
    
    @staticmethod
    async def export_document(context: Dict[str, Any]) -> Dict[str, Any]:
        """Export document to specified format."""
        from src.core.engine import ExportIAEngine
        
        content = context["variables"].get("content", {})
        config = context["variables"].get("config", {})
        
        async with ExportIAEngine() as engine:
            task_id = await engine.export_document(content, config)
            
            # Wait for completion
            while True:
                status = await engine.get_task_status(task_id)
                if status["status"] == "completed":
                    return {"task_id": task_id, "file_path": status["file_path"]}
                elif status["status"] == "failed":
                    raise Exception(f"Export failed: {status.get('error', 'Unknown error')}")
                
                await asyncio.sleep(1)
    
    @staticmethod
    async def send_notification(context: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification about export completion."""
        # This would integrate with notification services
        return {"notification_sent": True, "timestamp": datetime.now().isoformat()}


# Global workflow engine instance
_workflow_engine: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """Get the global workflow engine instance."""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()
        
        # Register predefined step handlers
        _workflow_engine.register_step_handler("validate_content", ExportWorkflowSteps.validate_content)
        _workflow_engine.register_step_handler("enhance_content", ExportWorkflowSteps.enhance_content)
        _workflow_engine.register_step_handler("export_document", ExportWorkflowSteps.export_document)
        _workflow_engine.register_step_handler("send_notification", ExportWorkflowSteps.send_notification)
    
    return _workflow_engine




