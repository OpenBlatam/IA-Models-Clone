"""
Add Node Use Case
================

Use case for adding a node to a workflow with business logic and validation.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging

from ...domain.entities.workflow_chain import WorkflowChain
from ...domain.entities.workflow_node import WorkflowNode
from ...domain.value_objects.workflow_id import WorkflowId
from ...domain.value_objects.node_id import NodeId
from ...domain.value_objects.priority import Priority
from ...domain.repositories.workflow_repository import WorkflowRepository
from ...domain.services.workflow_domain_service import WorkflowDomainService
from ...shared.events.event_bus import EventBus
from ...shared.exceptions.application_exceptions import ApplicationException


logger = logging.getLogger(__name__)


@dataclass
class AddNodeRequest:
    """Request for adding a node to a workflow"""
    workflow_id: str
    title: str
    content: str
    prompt: str
    parent_id: Optional[str] = None
    priority: int = Priority.NORMAL
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AddNodeResponse:
    """Response for adding a node to a workflow"""
    node_id: str
    workflow_id: str
    title: str
    content: str
    prompt: str
    parent_id: Optional[str]
    priority: int
    tags: List[str]
    created_at: str
    success: bool
    message: str = ""


class AddNodeUseCase:
    """
    Use case for adding a node to a workflow
    
    This use case handles the business logic for adding a node to a workflow,
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
    
    async def execute(self, request: AddNodeRequest) -> AddNodeResponse:
        """
        Execute the add node use case
        
        Args:
            request: The add node request
            
        Returns:
            AddNodeResponse: The response with node details
            
        Raises:
            ApplicationException: If the operation fails
        """
        try:
            logger.info(f"Adding node to workflow: {request.workflow_id}")
            
            # Validate request
            self._validate_request(request)
            
            # Get workflow
            workflow_id = WorkflowId.from_string(request.workflow_id)
            workflow = await self._workflow_repository.get_by_id(workflow_id)
            
            if not workflow:
                raise ApplicationException(f"Workflow {request.workflow_id} not found")
            
            # Validate workflow is editable
            if not workflow.status.is_editable():
                raise ApplicationException(f"Workflow {request.workflow_id} is not editable")
            
            # Validate parent node if specified
            parent_id = None
            if request.parent_id:
                parent_id = NodeId.from_string(request.parent_id)
                parent_node = workflow.get_node(parent_id)
                if not parent_node:
                    raise ApplicationException(f"Parent node {request.parent_id} not found")
            
            # Generate node ID
            node_id = NodeId.generate()
            
            # Create node entity
            node = WorkflowNode(
                node_id=node_id,
                title=request.title,
                content=request.content,
                prompt=request.prompt,
                parent_id=parent_id,
                priority=Priority(request.priority),
                tags=request.tags or [],
                metadata=request.metadata or {}
            )
            
            # Add node to workflow
            workflow.add_node(node)
            
            # Save workflow
            await self._workflow_repository.save(workflow)
            
            # Publish domain events
            await self._publish_domain_events(workflow)
            
            logger.info(f"Successfully added node {node_id} to workflow {workflow_id}")
            
            return AddNodeResponse(
                node_id=str(node.id),
                workflow_id=str(workflow.id),
                title=node.title,
                content=node.content,
                prompt=node.prompt,
                parent_id=str(node.parent_id) if node.parent_id else None,
                priority=node.priority.value,
                tags=node.tags,
                created_at=node.created_at.isoformat(),
                success=True,
                message="Node added successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to add node to workflow: {e}")
            raise ApplicationException(f"Failed to add node: {str(e)}") from e
    
    def _validate_request(self, request: AddNodeRequest) -> None:
        """Validate the add node request"""
        if not request.workflow_id or not request.workflow_id.strip():
            raise ApplicationException("Workflow ID is required")
        
        if not request.title or not request.title.strip():
            raise ApplicationException("Node title is required")
        
        if len(request.title) > 255:
            raise ApplicationException("Node title cannot exceed 255 characters")
        
        if not request.content or not request.content.strip():
            raise ApplicationException("Node content is required")
        
        if len(request.content) > 100000:
            raise ApplicationException("Node content cannot exceed 100,000 characters")
        
        if not request.prompt or not request.prompt.strip():
            raise ApplicationException("Node prompt is required")
        
        if len(request.prompt) > 2000:
            raise ApplicationException("Node prompt cannot exceed 2,000 characters")
        
        # Validate priority
        try:
            Priority(request.priority)
        except ValueError:
            raise ApplicationException(f"Invalid priority: {request.priority}")
        
        # Validate tags
        if request.tags:
            if len(request.tags) > 20:
                raise ApplicationException("Node cannot have more than 20 tags")
            
            for tag in request.tags:
                if not tag or not tag.strip():
                    raise ApplicationException("Tag cannot be empty")
                
                if len(tag) > 50:
                    raise ApplicationException("Tag cannot exceed 50 characters")
    
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




