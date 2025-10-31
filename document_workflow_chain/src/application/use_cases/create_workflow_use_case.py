"""
Create Workflow Use Case
=======================

Use case for creating a new workflow with business logic and validation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

from ...domain.entities.workflow_chain import WorkflowChain
from ...domain.value_objects.workflow_id import WorkflowId
from ...domain.value_objects.workflow_status import WorkflowStatus
from ...domain.repositories.workflow_repository import WorkflowRepository
from ...domain.services.workflow_domain_service import WorkflowDomainService
from ...shared.events.event_bus import EventBus
from ...shared.exceptions.application_exceptions import ApplicationException


logger = logging.getLogger(__name__)


@dataclass
class CreateWorkflowRequest:
    """Request for creating a workflow"""
    name: str
    description: str = ""
    settings: Optional[Dict[str, Any]] = None


@dataclass
class CreateWorkflowResponse:
    """Response for creating a workflow"""
    workflow_id: str
    name: str
    description: str
    status: str
    created_at: str
    success: bool
    message: str = ""


class CreateWorkflowUseCase:
    """
    Use case for creating a new workflow
    
    This use case handles the business logic for creating a workflow,
    including validation, domain service calls, and event publishing.
    """
    
    def __init__(
        self,
        workflow_repository: WorkflowRepository,
        workflow_domain_service: WorkflowDomainService,
        event_bus: EventBus
    ):
        self._workflow_repository = workflow_repository
        self._workflow_domain_service = workflow_domain_service
        self._event_bus = event_bus
    
    async def execute(self, request: CreateWorkflowRequest) -> CreateWorkflowResponse:
        """
        Execute the create workflow use case
        
        Args:
            request: The create workflow request
            
        Returns:
            CreateWorkflowResponse: The response with workflow details
            
        Raises:
            ApplicationException: If the operation fails
        """
        try:
            logger.info(f"Creating workflow with name: {request.name}")
            
            # Validate request
            self._validate_request(request)
            
            # Check if workflow name already exists
            await self._workflow_domain_service.validate_workflow_name_uniqueness(request.name)
            
            # Generate workflow ID
            workflow_id = WorkflowId.generate()
            
            # Create workflow entity
            workflow = WorkflowChain(
                workflow_id=workflow_id,
                name=request.name,
                description=request.description,
                status=WorkflowStatus.DRAFT,
                settings=request.settings or {}
            )
            
            # Save workflow
            await self._workflow_repository.save(workflow)
            
            # Publish domain events
            await self._publish_domain_events(workflow)
            
            logger.info(f"Successfully created workflow: {workflow_id}")
            
            return CreateWorkflowResponse(
                workflow_id=str(workflow.id),
                name=workflow.name,
                description=workflow.description,
                status=workflow.status.value,
                created_at=workflow.created_at.isoformat(),
                success=True,
                message="Workflow created successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise ApplicationException(f"Failed to create workflow: {str(e)}") from e
    
    def _validate_request(self, request: CreateWorkflowRequest) -> None:
        """Validate the create workflow request"""
        if not request.name or not request.name.strip():
            raise ApplicationException("Workflow name is required")
        
        if len(request.name) > 255:
            raise ApplicationException("Workflow name cannot exceed 255 characters")
        
        if len(request.description) > 1000:
            raise ApplicationException("Workflow description cannot exceed 1000 characters")
        
        if request.settings:
            self._validate_settings(request.settings)
    
    def _validate_settings(self, settings: Dict[str, Any]) -> None:
        """Validate workflow settings"""
        if "max_nodes" in settings:
            max_nodes = settings["max_nodes"]
            if not isinstance(max_nodes, int) or max_nodes <= 0:
                raise ApplicationException("max_nodes must be a positive integer")
            
            if max_nodes > 10000:
                raise ApplicationException("max_nodes cannot exceed 10000")
        
        if "timeout" in settings:
            timeout = settings["timeout"]
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ApplicationException("timeout must be a positive number")
            
            if timeout > 3600:  # 1 hour
                raise ApplicationException("timeout cannot exceed 3600 seconds")
    
    async def _publish_domain_events(self, workflow: WorkflowChain) -> None:
        """Publish domain events for the workflow"""
        try:
            for event in workflow.domain_events:
                await self._event_bus.publish(event)
            
            # Clear domain events after publishing
            workflow.clear_domain_events()
            
        except Exception as e:
            logger.error(f"Failed to publish domain events for workflow {workflow.id}: {e}")
            # Don't raise exception here as the workflow was already saved




