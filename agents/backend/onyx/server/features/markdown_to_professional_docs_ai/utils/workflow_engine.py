"""Workflow and automation engine"""
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowStep:
    """Workflow step definition"""
    name: str
    action: str
    parameters: Dict[str, Any]
    condition: Optional[str] = None
    on_success: Optional[str] = None
    on_failure: Optional[str] = None


@dataclass
class WorkflowExecution:
    """Workflow execution record"""
    workflow_id: str
    status: WorkflowStatus
    current_step: int
    steps: List[WorkflowStep]
    results: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class WorkflowEngine:
    """Workflow automation engine"""
    
    def __init__(self):
        self.workflows: Dict[str, List[WorkflowStep]] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.actions: Dict[str, Callable] = {}
        self._register_default_actions()
    
    def _register_default_actions(self):
        """Register default workflow actions"""
        self.actions["convert"] = self._action_convert
        self.actions["validate"] = self._action_validate
        self.actions["compress"] = self._action_compress
        self.actions["sign"] = self._action_sign
        self.actions["notify"] = self._action_notify
        self.actions["export"] = self._action_export
    
    def create_workflow(
        self,
        workflow_name: str,
        steps: List[Dict[str, Any]]
    ) -> bool:
        """
        Create a new workflow
        
        Args:
            workflow_name: Workflow name
            steps: List of workflow steps
            
        Returns:
            True if successful
        """
        try:
            workflow_steps = []
            for step_data in steps:
                step = WorkflowStep(
                    name=step_data.get("name", ""),
                    action=step_data.get("action", ""),
                    parameters=step_data.get("parameters", {}),
                    condition=step_data.get("condition"),
                    on_success=step_data.get("on_success"),
                    on_failure=step_data.get("on_failure")
                )
                workflow_steps.append(step)
            
            self.workflows[workflow_name] = workflow_steps
            return True
        except Exception as e:
            logger.error(f"Error creating workflow: {e}")
            return False
    
    async def execute_workflow(
        self,
        workflow_name: str,
        initial_data: Dict[str, Any]
    ) -> str:
        """
        Execute a workflow
        
        Args:
            workflow_name: Workflow name
            initial_data: Initial data for workflow
            
        Returns:
            Execution ID
        """
        if workflow_name not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_name}")
        
        import uuid
        execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            workflow_id=execution_id,
            status=WorkflowStatus.RUNNING,
            current_step=0,
            steps=self.workflows[workflow_name],
            results={"initial": initial_data},
            started_at=datetime.now()
        )
        
        self.executions[execution_id] = execution
        
        # Execute workflow asynchronously
        import asyncio
        asyncio.create_task(self._run_workflow(execution_id))
        
        return execution_id
    
    async def _run_workflow(self, execution_id: str):
        """Run workflow execution"""
        execution = self.executions[execution_id]
        
        try:
            for i, step in enumerate(execution.steps):
                execution.current_step = i
                
                # Check condition
                if step.condition:
                    if not self._evaluate_condition(step.condition, execution.results):
                        continue
                
                # Execute action
                if step.action in self.actions:
                    action_func = self.actions[step.action]
                    result = await action_func(step.parameters, execution.results)
                    execution.results[step.name] = result
                    
                    # Handle success
                    if step.on_success:
                        next_step = self._find_step(step.on_success, execution.steps)
                        if next_step:
                            continue
                else:
                    logger.warning(f"Unknown action: {step.action}")
                    if step.on_failure:
                        next_step = self._find_step(step.on_failure, execution.steps)
                        if next_step:
                            continue
                    else:
                        raise ValueError(f"Unknown action: {step.action}")
            
            execution.status = WorkflowStatus.COMPLETED
            execution.completed_at = datetime.now()
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.completed_at = datetime.now()
    
    def _evaluate_condition(self, condition: str, results: Dict[str, Any]) -> bool:
        """Evaluate workflow condition"""
        # Simple condition evaluation
        # In production, would use a proper expression evaluator
        try:
            return eval(condition, {"results": results})
        except:
            return False
    
    def _find_step(self, step_name: str, steps: List[WorkflowStep]) -> Optional[WorkflowStep]:
        """Find step by name"""
        for step in steps:
            if step.name == step_name:
                return step
        return None
    
    async def _action_convert(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert action"""
        from services.converter_service import ConverterService
        from services.markdown_parser import MarkdownParser
        
        markdown_content = parameters.get("markdown_content") or context.get("markdown_content", "")
        output_format = parameters.get("output_format", "pdf")
        
        parser = MarkdownParser()
        parsed_content = parser.parse(markdown_content)
        
        converter = ConverterService()
        output_path = await converter.convert(
            parsed_content=parsed_content,
            output_format=output_format
        )
        
        return {
            "output_path": output_path,
            "format": output_format
        }
    
    async def _action_validate(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate action"""
        from utils.document_validator import get_document_validator
        
        document_path = parameters.get("document_path") or context.get("output_path", "")
        
        validator = get_document_validator()
        validation = validator.validate_document(document_path)
        
        return validation
    
    async def _action_compress(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compress action"""
        from utils.document_compressor import get_document_compressor
        
        document_path = parameters.get("document_path") or context.get("output_path", "")
        
        compressor = get_document_compressor()
        compressed_path = compressor.compress_document(document_path)
        
        return {
            "compressed_path": compressed_path
        }
    
    async def _action_sign(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sign action"""
        from utils.digital_signature import get_signature_manager
        
        document_path = parameters.get("document_path") or context.get("output_path", "")
        signer_name = parameters.get("signer_name", "System")
        
        signature_manager = get_signature_manager()
        signature = signature_manager.sign_document(document_path, signer_name)
        
        return signature
    
    async def _action_notify(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Notify action"""
        from utils.notifications import get_notification_manager
        
        recipient = parameters.get("recipient", "")
        message = parameters.get("message", "Workflow completed")
        
        notification_manager = get_notification_manager()
        notification_manager.send_notification(recipient, "workflow", message)
        
        return {
            "notified": True,
            "recipient": recipient
        }
    
    async def _action_export(
        self,
        parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Export action"""
        from utils.multi_export import get_multi_exporter
        
        document_path = parameters.get("document_path") or context.get("output_path", "")
        formats = parameters.get("formats", ["pdf"])
        
        # This would export to multiple formats
        return {
            "exported": True,
            "formats": formats
        }
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get workflow execution"""
        return self.executions.get(execution_id)
    
    def list_workflows(self) -> List[str]:
        """List all workflows"""
        return list(self.workflows.keys())


# Global workflow engine
_workflow_engine: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """Get global workflow engine"""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()
    return _workflow_engine

