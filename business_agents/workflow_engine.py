"""
Workflow Engine
===============

Advanced workflow creation and management system for business processes.
Supports complex multi-step workflows with conditional logic, parallel execution,
and integration with various business agents.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class StepType(Enum):
    TASK = "task"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENCE = "sequence"
    LOOP = "loop"
    API_CALL = "api_call"
    DOCUMENT_GENERATION = "document_generation"
    NOTIFICATION = "notification"

@dataclass
class WorkflowStep:
    id: str
    name: str
    step_type: StepType
    description: str
    agent_type: str
    parameters: Dict[str, Any]
    conditions: Optional[Dict[str, Any]] = None
    next_steps: List[str] = None
    parallel_steps: List[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300
    status: str = "pending"
    created_at: datetime = None
    completed_at: datetime = None
    error_message: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.next_steps is None:
            self.next_steps = []
        if self.parallel_steps is None:
            self.parallel_steps = []

@dataclass
class Workflow:
    id: str
    name: str
    description: str
    business_area: str
    steps: List[WorkflowStep]
    status: WorkflowStatus
    created_by: str
    created_at: datetime
    updated_at: datetime
    variables: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.variables is None:
            self.variables = {}
        if self.metadata is None:
            self.metadata = {}

class WorkflowEngine:
    """
    Advanced workflow engine for managing business processes.
    """
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.running_workflows: Dict[str, asyncio.Task] = {}
        self.agent_handlers: Dict[str, Callable] = {}
        self.step_handlers: Dict[StepType, Callable] = {}
        
    def register_agent_handler(self, agent_type: str, handler: Callable):
        """Register a handler for a specific agent type."""
        self.agent_handlers[agent_type] = handler
        
    def register_step_handler(self, step_type: StepType, handler: Callable):
        """Register a handler for a specific step type."""
        self.step_handlers[step_type] = handler
        
    async def create_workflow(
        self,
        name: str,
        description: str,
        business_area: str,
        steps: List[Dict[str, Any]],
        created_by: str,
        variables: Dict[str, Any] = None
    ) -> Workflow:
        """Create a new workflow."""
        
        workflow_id = str(uuid.uuid4())
        workflow_steps = []
        
        for step_data in steps:
            step = WorkflowStep(
                id=str(uuid.uuid4()),
                name=step_data["name"],
                step_type=StepType(step_data["step_type"]),
                description=step_data["description"],
                agent_type=step_data["agent_type"],
                parameters=step_data.get("parameters", {}),
                conditions=step_data.get("conditions"),
                next_steps=step_data.get("next_steps", []),
                parallel_steps=step_data.get("parallel_steps", []),
                max_retries=step_data.get("max_retries", 3),
                timeout=step_data.get("timeout", 300)
            )
            workflow_steps.append(step)
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            business_area=business_area,
            steps=workflow_steps,
            status=WorkflowStatus.DRAFT,
            created_by=created_by,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            variables=variables or {}
        )
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {workflow_id}")
        
        return workflow
        
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow."""
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflows[workflow_id]
        
        if workflow.status != WorkflowStatus.DRAFT:
            raise ValueError(f"Workflow {workflow_id} is not in draft status")
            
        workflow.status = WorkflowStatus.ACTIVE
        workflow.updated_at = datetime.now()
        
        # Start workflow execution
        task = asyncio.create_task(self._execute_workflow_steps(workflow))
        self.running_workflows[workflow_id] = task
        
        try:
            result = await task
            workflow.status = WorkflowStatus.COMPLETED
            workflow.updated_at = datetime.now()
            return result
        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.updated_at = datetime.now()
            logger.error(f"Workflow {workflow_id} failed: {str(e)}")
            raise
        finally:
            if workflow_id in self.running_workflows:
                del self.running_workflows[workflow_id]
                
    async def _execute_workflow_steps(self, workflow: Workflow) -> Dict[str, Any]:
        """Execute workflow steps."""
        
        results = {}
        completed_steps = set()
        
        # Find starting steps (steps with no dependencies)
        starting_steps = [step for step in workflow.steps if not any(
            step.id in other_step.next_steps for other_step in workflow.steps
        )]
        
        if not starting_steps:
            raise ValueError("No starting steps found in workflow")
            
        # Execute starting steps
        for step in starting_steps:
            result = await self._execute_step(step, workflow, results)
            results[step.id] = result
            completed_steps.add(step.id)
            
        # Execute remaining steps
        while len(completed_steps) < len(workflow.steps):
            next_steps = []
            
            for step in workflow.steps:
                if step.id in completed_steps:
                    continue
                    
                # Check if all dependencies are completed
                dependencies_met = all(
                    dep_id in completed_steps for dep_id in step.next_steps
                )
                
                if dependencies_met:
                    next_steps.append(step)
                    
            if not next_steps:
                raise ValueError("No more steps can be executed - possible circular dependency")
                
            # Execute next steps in parallel
            tasks = []
            for step in next_steps:
                task = asyncio.create_task(
                    self._execute_step(step, workflow, results)
                )
                tasks.append((step.id, task))
                
            for step_id, task in tasks:
                result = await task
                results[step_id] = result
                completed_steps.add(step_id)
                
        return results
        
    async def _execute_step(
        self,
        step: WorkflowStep,
        workflow: Workflow,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single workflow step."""
        
        step.status = "running"
        step.created_at = datetime.now()
        
        try:
            # Check conditions
            if step.conditions:
                if not self._evaluate_conditions(step.conditions, workflow.variables, previous_results):
                    step.status = "skipped"
                    return {"status": "skipped", "reason": "conditions_not_met"}
                    
            # Execute step based on type
            if step.step_type in self.step_handlers:
                handler = self.step_handlers[step.step_type]
                result = await handler(step, workflow, previous_results)
            else:
                # Default execution
                result = await self._default_step_execution(step, workflow, previous_results)
                
            step.status = "completed"
            step.completed_at = datetime.now()
            
            return result
            
        except Exception as e:
            step.status = "failed"
            step.error_message = str(e)
            step.retry_count += 1
            
            if step.retry_count < step.max_retries:
                logger.warning(f"Step {step.id} failed, retrying ({step.retry_count}/{step.max_retries})")
                await asyncio.sleep(2 ** step.retry_count)  # Exponential backoff
                return await self._execute_step(step, workflow, previous_results)
            else:
                logger.error(f"Step {step.id} failed after {step.max_retries} retries")
                raise
                
    def _evaluate_conditions(
        self,
        conditions: Dict[str, Any],
        variables: Dict[str, Any],
        previous_results: Dict[str, Any]
    ) -> bool:
        """Evaluate step conditions."""
        
        # Simple condition evaluation
        # Can be extended with more complex logic
        for key, expected_value in conditions.items():
            if key in variables:
                if variables[key] != expected_value:
                    return False
            elif key in previous_results:
                if previous_results[key] != expected_value:
                    return False
            else:
                return False
                
        return True
        
    async def _default_step_execution(
        self,
        step: WorkflowStep,
        workflow: Workflow,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Default step execution logic."""
        
        if step.agent_type in self.agent_handlers:
            handler = self.agent_handlers[step.agent_type]
            return await handler(step, workflow, previous_results)
        else:
            # Simulate step execution
            await asyncio.sleep(1)
            return {
                "status": "completed",
                "message": f"Step {step.name} executed successfully",
                "timestamp": datetime.now().isoformat()
            }
            
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID."""
        return self.workflows.get(workflow_id)
        
    def list_workflows(
        self,
        business_area: str = None,
        status: WorkflowStatus = None,
        created_by: str = None
    ) -> List[Workflow]:
        """List workflows with optional filters."""
        
        workflows = list(self.workflows.values())
        
        if business_area:
            workflows = [w for w in workflows if w.business_area == business_area]
            
        if status:
            workflows = [w for w in workflows if w.status == status]
            
        if created_by:
            workflows = [w for w in workflows if w.created_by == created_by]
            
        return workflows
        
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        
        if workflow_id in self.running_workflows:
            task = self.running_workflows[workflow_id]
            task.cancel()
            del self.running_workflows[workflow_id]
            
        if workflow_id in self.workflows:
            workflow = self.workflows[workflow_id]
            workflow.status = WorkflowStatus.PAUSED
            workflow.updated_at = datetime.now()
            return True
            
        return False
        
    async def resume_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Resume a paused workflow."""
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflows[workflow_id]
        
        if workflow.status != WorkflowStatus.PAUSED:
            raise ValueError(f"Workflow {workflow_id} is not paused")
            
        return await self.execute_workflow(workflow_id)
        
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        
        if workflow_id in self.running_workflows:
            task = self.running_workflows[workflow_id]
            task.cancel()
            del self.running_workflows[workflow_id]
            
        if workflow_id in self.workflows:
            del self.workflows[workflow_id]
            return True
            
        return False
        
    def export_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Export workflow as JSON."""
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
            
        workflow = self.workflows[workflow_id]
        
        # Convert to serializable format
        export_data = {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "business_area": workflow.business_area,
            "status": workflow.status.value,
            "created_by": workflow.created_by,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat(),
            "variables": workflow.variables,
            "metadata": workflow.metadata,
            "steps": []
        }
        
        for step in workflow.steps:
            step_data = {
                "id": step.id,
                "name": step.name,
                "step_type": step.step_type.value,
                "description": step.description,
                "agent_type": step.agent_type,
                "parameters": step.parameters,
                "conditions": step.conditions,
                "next_steps": step.next_steps,
                "parallel_steps": step.parallel_steps,
                "max_retries": step.max_retries,
                "timeout": step.timeout
            }
            export_data["steps"].append(step_data)
            
        return export_data
        
    def import_workflow(self, workflow_data: Dict[str, Any]) -> Workflow:
        """Import workflow from JSON."""
        
        # Convert steps
        steps = []
        for step_data in workflow_data["steps"]:
            step = WorkflowStep(
                id=step_data["id"],
                name=step_data["name"],
                step_type=StepType(step_data["step_type"]),
                description=step_data["description"],
                agent_type=step_data["agent_type"],
                parameters=step_data["parameters"],
                conditions=step_data.get("conditions"),
                next_steps=step_data.get("next_steps", []),
                parallel_steps=step_data.get("parallel_steps", []),
                max_retries=step_data.get("max_retries", 3),
                timeout=step_data.get("timeout", 300)
            )
            steps.append(step)
            
        # Create workflow
        workflow = Workflow(
            id=workflow_data["id"],
            name=workflow_data["name"],
            description=workflow_data["description"],
            business_area=workflow_data["business_area"],
            steps=steps,
            status=WorkflowStatus(workflow_data["status"]),
            created_by=workflow_data["created_by"],
            created_at=datetime.fromisoformat(workflow_data["created_at"]),
            updated_at=datetime.fromisoformat(workflow_data["updated_at"]),
            variables=workflow_data.get("variables", {}),
            metadata=workflow_data.get("metadata", {})
        )
        
        self.workflows[workflow.id] = workflow
        return workflow





























