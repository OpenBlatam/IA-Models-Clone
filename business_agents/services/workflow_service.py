"""
Workflow Service
================

Service layer for workflow operations.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from ..business_agents import BusinessAgentManager, BusinessArea
from ..workflow_engine import Workflow, WorkflowStatus, StepType

logger = logging.getLogger(__name__)

class WorkflowService:
    """Service for workflow operations."""
    
    def __init__(self, agent_manager: BusinessAgentManager):
        self.agent_manager = agent_manager
    
    async def list_workflows(
        self,
        business_area: Optional[BusinessArea] = None,
        created_by: Optional[str] = None,
        status: Optional[WorkflowStatus] = None
    ) -> List[Dict[str, Any]]:
        """List all workflows with optional filters."""
        
        workflows = self.agent_manager.list_workflows(business_area=business_area, created_by=created_by)
        
        # Apply status filter if provided
        if status:
            workflows = [w for w in workflows if w.status == status]
        
        return [
            {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "business_area": workflow.business_area,
                "status": workflow.status.value,
                "created_by": workflow.created_by,
                "created_at": workflow.created_at.isoformat(),
                "updated_at": workflow.updated_at.isoformat(),
                "steps_count": len(workflow.steps),
                "variables": workflow.variables
            }
            for workflow in workflows
        ]
    
    async def get_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Get specific workflow details."""
        
        workflow = self.agent_manager.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        return {
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
            "steps": [
                {
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
                    "timeout": step.timeout,
                    "status": step.status,
                    "created_at": step.created_at.isoformat() if step.created_at else None,
                    "completed_at": step.completed_at.isoformat() if step.completed_at else None,
                    "error_message": step.error_message
                }
                for step in workflow.steps
            ]
        }
    
    async def create_workflow(
        self,
        name: str,
        description: str,
        business_area: BusinessArea,
        steps: List[Dict[str, Any]],
        created_by: str,
        variables: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a new workflow."""
        
        try:
            workflow = await self.agent_manager.create_business_workflow(
                name=name,
                description=description,
                business_area=business_area,
                steps=steps,
                created_by=created_by,
                variables=variables
            )
            
            return {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "business_area": workflow.business_area,
                "status": workflow.status.value,
                "created_by": workflow.created_by,
                "created_at": workflow.created_at.isoformat(),
                "steps_count": len(workflow.steps)
            }
            
        except Exception as e:
            logger.error(f"Workflow creation failed: {str(e)}")
            raise Exception("Failed to create workflow")
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Execute a workflow."""
        
        try:
            result = await self.agent_manager.execute_business_workflow(workflow_id)
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "execution_results": result,
                "executed_at": datetime.now().isoformat()
            }
            
        except ValueError as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in workflow execution: {str(e)}")
            raise Exception("Failed to execute workflow")
    
    async def pause_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Pause a running workflow."""
        
        try:
            success = await self.agent_manager.workflow_engine.pause_workflow(workflow_id)
            
            if not success:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            return {
                "workflow_id": workflow_id,
                "status": "paused",
                "paused_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Workflow pause failed: {str(e)}")
            raise Exception("Failed to pause workflow")
    
    async def resume_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Resume a paused workflow."""
        
        try:
            result = await self.agent_manager.workflow_engine.resume_workflow(workflow_id)
            
            return {
                "workflow_id": workflow_id,
                "status": "resumed",
                "execution_results": result,
                "resumed_at": datetime.now().isoformat()
            }
            
        except ValueError as e:
            logger.error(f"Workflow resume failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in workflow resume: {str(e)}")
            raise Exception("Failed to resume workflow")
    
    async def delete_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Delete a workflow."""
        
        try:
            success = self.agent_manager.workflow_engine.delete_workflow(workflow_id)
            
            if not success:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            return {
                "workflow_id": workflow_id,
                "status": "deleted",
                "deleted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Workflow deletion failed: {str(e)}")
            raise Exception("Failed to delete workflow")
    
    async def export_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Export workflow as JSON."""
        
        try:
            export_data = self.agent_manager.workflow_engine.export_workflow(workflow_id)
            return export_data
            
        except ValueError as e:
            logger.error(f"Workflow export failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in workflow export: {str(e)}")
            raise Exception("Failed to export workflow")
    
    async def import_workflow(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Import workflow from JSON."""
        
        try:
            workflow = self.agent_manager.workflow_engine.import_workflow(workflow_data)
            
            return {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "business_area": workflow.business_area,
                "status": workflow.status.value,
                "imported_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Workflow import failed: {str(e)}")
            raise Exception("Failed to import workflow")
    
    async def get_workflow_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get predefined workflow templates for each business area."""
        
        return self.agent_manager.get_workflow_templates()