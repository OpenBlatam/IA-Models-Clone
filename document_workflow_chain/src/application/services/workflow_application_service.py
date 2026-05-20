"""
Workflow Application Service
============================

Application service for workflow orchestration and business logic.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from ...domain.entities.workflow_chain import WorkflowChain
from ...domain.entities.workflow_node import WorkflowNode
from ...domain.value_objects.workflow_id import WorkflowId
from ...domain.value_objects.node_id import NodeId
from ...domain.value_objects.workflow_status import WorkflowStatus
from ...domain.value_objects.priority import Priority
from ...domain.repositories.workflow_repository import IWorkflowRepository
from ...domain.services.workflow_domain_service import WorkflowDomainService, WorkflowStatistics
from ...shared.events.event_bus import EventBus
from ...shared.exceptions.application_exceptions import (
    ValidationException,
    BusinessRuleException,
    ResourceNotFoundException,
    ConcurrencyException
)
from ..dto.workflow_dto import WorkflowCreateDTO, WorkflowUpdateDTO, WorkflowResponseDTO
from ..dto.node_dto import NodeCreateDTO, NodeUpdateDTO, NodeResponseDTO


logger = logging.getLogger(__name__)


class WorkflowApplicationService:
    """
    Workflow application service
    
    Orchestrates workflow operations and coordinates between
    domain services, repositories, and external services.
    """
    
    def __init__(
        self,
        workflow_repository: IWorkflowRepository,
        domain_service: WorkflowDomainService,
        event_bus: EventBus
    ):
        self.workflow_repository = workflow_repository
        self.domain_service = domain_service
        self.event_bus = event_bus
    
    async def create_workflow(
        self,
        workflow_data: WorkflowCreateDTO,
        user_id: Optional[str] = None
    ) -> WorkflowResponseDTO:
        """Create a new workflow chain"""
        try:
            # Validate workflow creation
            self.domain_service.validate_workflow_creation(
                name=workflow_data.name,
                description=workflow_data.description,
                settings=workflow_data.settings
            )
            
            # Create workflow entity
            workflow = WorkflowChain(
                id=WorkflowId(),
                name=workflow_data.name,
                description=workflow_data.description,
                status=WorkflowStatus.CREATED,
                settings=workflow_data.settings or {},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1
            )
            
            # Add initial nodes if provided
            for node_data in workflow_data.nodes:
                node = self._create_node_from_dto(workflow.id, node_data)
                workflow.add_node(node)
            
            # Save workflow
            saved_workflow = await self.workflow_repository.save(workflow)
            
            # Publish domain event
            await self.event_bus.publish(
                "workflow.created",
                {
                    "workflow_id": str(saved_workflow.id),
                    "name": saved_workflow.name,
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Workflow created: {saved_workflow.id}")
            return self._workflow_to_dto(saved_workflow)
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise
    
    async def get_workflow(
        self,
        workflow_id: WorkflowId,
        user_id: Optional[str] = None
    ) -> WorkflowResponseDTO:
        """Get workflow by ID"""
        try:
            workflow = await self.workflow_repository.get_by_id(workflow_id)
            if not workflow:
                raise ResourceNotFoundException(
                    resource_type="Workflow",
                    resource_id=str(workflow_id)
                )
            
            # Check permissions (simplified)
            if not self._can_access_workflow(workflow, user_id):
                raise BusinessRuleException("Access denied to workflow")
            
            return self._workflow_to_dto(workflow)
            
        except Exception as e:
            logger.error(f"Failed to get workflow {workflow_id}: {e}")
            raise
    
    async def update_workflow(
        self,
        workflow_id: WorkflowId,
        updates: WorkflowUpdateDTO,
        user_id: Optional[str] = None,
        expected_version: Optional[int] = None
    ) -> WorkflowResponseDTO:
        """Update workflow"""
        try:
            # Get existing workflow
            workflow = await self.workflow_repository.get_by_id(workflow_id)
            if not workflow:
                raise ResourceNotFoundException(
                    resource_type="Workflow",
                    resource_id=str(workflow_id)
                )
            
            # Check permissions
            if not self._can_modify_workflow(workflow, user_id):
                raise BusinessRuleException("Modification denied for workflow")
            
            # Check concurrency
            if expected_version and workflow.version != expected_version:
                raise ConcurrencyException(
                    resource_type="Workflow",
                    resource_id=str(workflow_id),
                    expected_version=expected_version,
                    actual_version=workflow.version
                )
            
            # Validate updates
            update_dict = updates.dict(exclude_unset=True)
            self.domain_service.validate_workflow_update(workflow, update_dict)
            
            # Apply updates
            if "name" in update_dict:
                workflow.name = update_dict["name"]
            if "description" in update_dict:
                workflow.description = update_dict["description"]
            if "status" in update_dict:
                workflow.status = WorkflowStatus(update_dict["status"])
            if "settings" in update_dict:
                workflow.settings = update_dict["settings"]
            
            workflow.updated_at = datetime.utcnow()
            workflow.version += 1
            
            # Save updated workflow
            saved_workflow = await self.workflow_repository.save(workflow)
            
            # Publish domain event
            await self.event_bus.publish(
                "workflow.updated",
                {
                    "workflow_id": str(saved_workflow.id),
                    "updated_fields": list(update_dict.keys()),
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Workflow updated: {saved_workflow.id}")
            return self._workflow_to_dto(saved_workflow)
            
        except Exception as e:
            logger.error(f"Failed to update workflow {workflow_id}: {e}")
            raise
    
    async def delete_workflow(
        self,
        workflow_id: WorkflowId,
        user_id: Optional[str] = None
    ) -> bool:
        """Delete workflow"""
        try:
            # Get existing workflow
            workflow = await self.workflow_repository.get_by_id(workflow_id)
            if not workflow:
                raise ResourceNotFoundException(
                    resource_type="Workflow",
                    resource_id=str(workflow_id)
                )
            
            # Check permissions
            if not self._can_delete_workflow(workflow, user_id):
                raise BusinessRuleException("Deletion denied for workflow")
            
            # Check if workflow can be deleted
            if not self.domain_service.can_delete_workflow(workflow):
                raise BusinessRuleException("Workflow cannot be deleted in current status")
            
            # Soft delete by changing status
            workflow.status = WorkflowStatus.DELETED
            workflow.updated_at = datetime.utcnow()
            workflow.version += 1
            
            await self.workflow_repository.save(workflow)
            
            # Publish domain event
            await self.event_bus.publish(
                "workflow.deleted",
                {
                    "workflow_id": str(workflow_id),
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Workflow deleted: {workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            raise
    
    async def add_node_to_workflow(
        self,
        workflow_id: WorkflowId,
        node_data: NodeCreateDTO,
        user_id: Optional[str] = None
    ) -> NodeResponseDTO:
        """Add node to workflow"""
        try:
            # Get workflow
            workflow = await self.workflow_repository.get_by_id(workflow_id)
            if not workflow:
                raise ResourceNotFoundException(
                    resource_type="Workflow",
                    resource_id=str(workflow_id)
                )
            
            # Check permissions
            if not self._can_modify_workflow(workflow, user_id):
                raise BusinessRuleException("Modification denied for workflow")
            
            # Check if node can be added
            if not self.domain_service.can_add_node_to_workflow(workflow):
                raise BusinessRuleException("Cannot add node to workflow in current status")
            
            # Validate node creation
            parent_id = NodeId(node_data.parent_id) if node_data.parent_id else None
            self.domain_service.validate_node_creation(
                workflow=workflow,
                title=node_data.title,
                content=node_data.content,
                prompt=node_data.prompt,
                parent_id=parent_id
            )
            
            # Create node
            node = self._create_node_from_dto(workflow_id, node_data)
            workflow.add_node(node)
            
            # Save workflow with new node
            saved_workflow = await self.workflow_repository.save(workflow)
            
            # Find the added node
            added_node = next(n for n in saved_workflow.nodes if n.id == node.id)
            
            # Publish domain event
            await self.event_bus.publish(
                "node.added",
                {
                    "workflow_id": str(workflow_id),
                    "node_id": str(node.id),
                    "title": node.title,
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Node added to workflow {workflow_id}: {node.id}")
            return self._node_to_dto(added_node)
            
        except Exception as e:
            logger.error(f"Failed to add node to workflow {workflow_id}: {e}")
            raise
    
    async def update_node(
        self,
        workflow_id: WorkflowId,
        node_id: NodeId,
        updates: NodeUpdateDTO,
        user_id: Optional[str] = None,
        expected_version: Optional[int] = None
    ) -> NodeResponseDTO:
        """Update node in workflow"""
        try:
            # Get workflow
            workflow = await self.workflow_repository.get_by_id(workflow_id)
            if not workflow:
                raise ResourceNotFoundException(
                    resource_type="Workflow",
                    resource_id=str(workflow_id)
                )
            
            # Find node
            node = next((n for n in workflow.nodes if n.id == node_id), None)
            if not node:
                raise ResourceNotFoundException(
                    resource_type="Node",
                    resource_id=str(node_id)
                )
            
            # Check permissions
            if not self._can_modify_workflow(workflow, user_id):
                raise BusinessRuleException("Modification denied for workflow")
            
            # Check concurrency
            if expected_version and node.version != expected_version:
                raise ConcurrencyException(
                    resource_type="Node",
                    resource_id=str(node_id),
                    expected_version=expected_version,
                    actual_version=node.version
                )
            
            # Validate updates
            update_dict = updates.dict(exclude_unset=True)
            self.domain_service.validate_node_update(workflow, node, update_dict)
            
            # Apply updates
            if "title" in update_dict:
                node.title = update_dict["title"]
            if "content" in update_dict:
                node.content = update_dict["content"]
            if "prompt" in update_dict:
                node.prompt = update_dict["prompt"]
            if "parent_id" in update_dict:
                node.parent_id = NodeId(update_dict["parent_id"]) if update_dict["parent_id"] else None
            if "priority" in update_dict:
                node.priority = Priority(update_dict["priority"])
            if "status" in update_dict:
                node.status = WorkflowStatus(update_dict["status"])
            if "tags" in update_dict:
                node.tags = update_dict["tags"]
            if "metadata" in update_dict:
                node.metadata = update_dict["metadata"]
            
            node.updated_at = datetime.utcnow()
            node.version += 1
            
            # Save workflow with updated node
            saved_workflow = await self.workflow_repository.save(workflow)
            
            # Find the updated node
            updated_node = next(n for n in saved_workflow.nodes if n.id == node_id)
            
            # Publish domain event
            await self.event_bus.publish(
                "node.updated",
                {
                    "workflow_id": str(workflow_id),
                    "node_id": str(node_id),
                    "updated_fields": list(update_dict.keys()),
                    "user_id": user_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Node updated in workflow {workflow_id}: {node_id}")
            return self._node_to_dto(updated_node)
            
        except Exception as e:
            logger.error(f"Failed to update node {node_id} in workflow {workflow_id}: {e}")
            raise
    
    async def get_workflow_statistics(
        self,
        workflow_id: WorkflowId,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get workflow statistics"""
        try:
            # Get workflow
            workflow = await self.workflow_repository.get_by_id(workflow_id)
            if not workflow:
                raise ResourceNotFoundException(
                    resource_type="Workflow",
                    resource_id=str(workflow_id)
                )
            
            # Check permissions
            if not self._can_access_workflow(workflow, user_id):
                raise BusinessRuleException("Access denied to workflow")
            
            # Calculate statistics
            stats = self.domain_service.calculate_workflow_statistics(workflow)
            health_score = self.domain_service.get_workflow_health_score(workflow)
            complexity_score = self.domain_service.get_workflow_complexity_score(workflow)
            
            return {
                "workflow_id": str(workflow_id),
                "statistics": {
                    "total_nodes": stats.total_nodes,
                    "active_nodes": stats.active_nodes,
                    "completed_nodes": stats.completed_nodes,
                    "error_nodes": stats.error_nodes,
                    "average_quality_score": stats.average_quality_score,
                    "total_word_count": stats.total_word_count,
                    "estimated_reading_time": stats.estimated_reading_time,
                    "most_used_tags": stats.most_used_tags
                },
                "health_score": health_score,
                "complexity_score": complexity_score,
                "calculated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get workflow statistics {workflow_id}: {e}")
            raise
    
    async def list_workflows(
        self,
        user_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[WorkflowResponseDTO]:
        """List workflows with filtering"""
        try:
            workflows = await self.workflow_repository.get_all(
                status=status,
                limit=limit,
                offset=offset
            )
            
            # Filter by user permissions (simplified)
            accessible_workflows = [
                w for w in workflows
                if self._can_access_workflow(w, user_id)
            ]
            
            return [self._workflow_to_dto(w) for w in accessible_workflows]
            
        except Exception as e:
            logger.error(f"Failed to list workflows: {e}")
            raise
    
    def _create_node_from_dto(
        self,
        workflow_id: WorkflowId,
        node_data: NodeCreateDTO
    ) -> WorkflowNode:
        """Create node entity from DTO"""
        return WorkflowNode(
            id=NodeId(),
            workflow_id=workflow_id,
            title=node_data.title,
            content=node_data.content,
            prompt=node_data.prompt,
            parent_id=NodeId(node_data.parent_id) if node_data.parent_id else None,
            priority=node_data.priority,
            status=WorkflowStatus.CREATED,
            tags=node_data.tags or [],
            metadata=node_data.metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1
        )
    
    def _workflow_to_dto(self, workflow: WorkflowChain) -> WorkflowResponseDTO:
        """Convert workflow entity to DTO"""
        return WorkflowResponseDTO(
            id=str(workflow.id),
            name=workflow.name,
            description=workflow.description,
            status=workflow.status.value,
            settings=workflow.settings,
            created_at=workflow.created_at.isoformat() if workflow.created_at else None,
            updated_at=workflow.updated_at.isoformat() if workflow.updated_at else None,
            version=workflow.version,
            nodes=[self._node_to_dto(n) for n in workflow.nodes]
        )
    
    def _node_to_dto(self, node: WorkflowNode) -> NodeResponseDTO:
        """Convert node entity to DTO"""
        return NodeResponseDTO(
            id=str(node.id),
            workflow_id=str(node.workflow_id),
            title=node.title,
            content=node.content,
            prompt=node.prompt,
            parent_id=str(node.parent_id) if node.parent_id else None,
            priority=node.priority.value,
            status=node.status.value,
            tags=node.tags,
            metadata=node.metadata,
            created_at=node.created_at.isoformat() if node.created_at else None,
            updated_at=node.updated_at.isoformat() if node.updated_at else None,
            version=node.version
        )
    
    def _can_access_workflow(self, workflow: WorkflowChain, user_id: Optional[str]) -> bool:
        """Check if user can access workflow (simplified)"""
        # In a real implementation, this would check user permissions
        return True
    
    def _can_modify_workflow(self, workflow: WorkflowChain, user_id: Optional[str]) -> bool:
        """Check if user can modify workflow (simplified)"""
        # In a real implementation, this would check user permissions
        return True
    
    def _can_delete_workflow(self, workflow: WorkflowChain, user_id: Optional[str]) -> bool:
        """Check if user can delete workflow (simplified)"""
        # In a real implementation, this would check user permissions
        return True




