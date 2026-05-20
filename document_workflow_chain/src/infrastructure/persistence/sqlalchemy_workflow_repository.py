"""
SQLAlchemy Workflow Repository Implementation
===========================================

Concrete implementation of WorkflowRepository using SQLAlchemy ORM.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from ...domain.entities.workflow_chain import WorkflowChain
from ...domain.entities.workflow_node import WorkflowNode
from ...domain.value_objects.workflow_id import WorkflowId
from ...domain.value_objects.workflow_status import WorkflowStatus
from ...domain.repositories.workflow_repository import WorkflowRepository
from .models import WorkflowModel, NodeModel


logger = logging.getLogger(__name__)


class SQLAlchemyWorkflowRepository(WorkflowRepository):
    """
    SQLAlchemy implementation of WorkflowRepository
    
    This implementation uses SQLAlchemy ORM for database operations
    with async support and proper error handling.
    """
    
    def __init__(self, session: AsyncSession):
        self._session = session
    
    async def save(self, workflow: WorkflowChain) -> None:
        """Save a workflow to the database"""
        try:
            # Check if workflow exists
            existing_workflow = await self._get_workflow_model(workflow.id)
            
            if existing_workflow:
                # Update existing workflow
                await self._update_workflow_model(existing_workflow, workflow)
            else:
                # Create new workflow
                await self._create_workflow_model(workflow)
            
            await self._session.commit()
            logger.debug(f"Saved workflow {workflow.id}")
            
        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to save workflow {workflow.id}: {e}")
            raise
    
    async def get_by_id(self, workflow_id: WorkflowId) -> Optional[WorkflowChain]:
        """Get a workflow by ID"""
        try:
            workflow_model = await self._get_workflow_model(workflow_id)
            if workflow_model:
                return await self._model_to_entity(workflow_model)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get workflow {workflow_id}: {e}")
            raise
    
    async def get_by_name(self, name: str) -> Optional[WorkflowChain]:
        """Get a workflow by name"""
        try:
            stmt = select(WorkflowModel).where(WorkflowModel.name == name)
            result = await self._session.execute(stmt)
            workflow_model = result.scalar_one_or_none()
            
            if workflow_model:
                return await self._model_to_entity(workflow_model)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get workflow by name {name}: {e}")
            raise
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[WorkflowChain]:
        """Get all workflows with pagination"""
        try:
            stmt = (
                select(WorkflowModel)
                .options(selectinload(WorkflowModel.nodes))
                .offset(offset)
                .limit(limit)
                .order_by(WorkflowModel.created_at.desc())
            )
            result = await self._session.execute(stmt)
            workflow_models = result.scalars().all()
            
            workflows = []
            for model in workflow_models:
                workflow = await self._model_to_entity(model)
                workflows.append(workflow)
            
            return workflows
            
        except Exception as e:
            logger.error(f"Failed to get all workflows: {e}")
            raise
    
    async def get_by_status(self, status: WorkflowStatus, limit: int = 100, offset: int = 0) -> List[WorkflowChain]:
        """Get workflows by status"""
        try:
            stmt = (
                select(WorkflowModel)
                .options(selectinload(WorkflowModel.nodes))
                .where(WorkflowModel.status == status.value)
                .offset(offset)
                .limit(limit)
                .order_by(WorkflowModel.created_at.desc())
            )
            result = await self._session.execute(stmt)
            workflow_models = result.scalars().all()
            
            workflows = []
            for model in workflow_models:
                workflow = await self._model_to_entity(model)
                workflows.append(workflow)
            
            return workflows
            
        except Exception as e:
            logger.error(f"Failed to get workflows by status {status}: {e}")
            raise
    
    async def get_by_created_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[WorkflowChain]:
        """Get workflows created within a date range"""
        try:
            stmt = (
                select(WorkflowModel)
                .options(selectinload(WorkflowModel.nodes))
                .where(
                    and_(
                        WorkflowModel.created_at >= start_date,
                        WorkflowModel.created_at <= end_date
                    )
                )
                .offset(offset)
                .limit(limit)
                .order_by(WorkflowModel.created_at.desc())
            )
            result = await self._session.execute(stmt)
            workflow_models = result.scalars().all()
            
            workflows = []
            for model in workflow_models:
                workflow = await self._model_to_entity(model)
                workflows.append(workflow)
            
            return workflows
            
        except Exception as e:
            logger.error(f"Failed to get workflows by created date range: {e}")
            raise
    
    async def get_by_updated_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        limit: int = 100, 
        offset: int = 0
    ) -> List[WorkflowChain]:
        """Get workflows updated within a date range"""
        try:
            stmt = (
                select(WorkflowModel)
                .options(selectinload(WorkflowModel.nodes))
                .where(
                    and_(
                        WorkflowModel.updated_at >= start_date,
                        WorkflowModel.updated_at <= end_date
                    )
                )
                .offset(offset)
                .limit(limit)
                .order_by(WorkflowModel.updated_at.desc())
            )
            result = await self._session.execute(stmt)
            workflow_models = result.scalars().all()
            
            workflows = []
            for model in workflow_models:
                workflow = await self._model_to_entity(model)
                workflows.append(workflow)
            
            return workflows
            
        except Exception as e:
            logger.error(f"Failed to get workflows by updated date range: {e}")
            raise
    
    async def search_by_name(self, name_pattern: str, limit: int = 100, offset: int = 0) -> List[WorkflowChain]:
        """Search workflows by name pattern"""
        try:
            stmt = (
                select(WorkflowModel)
                .options(selectinload(WorkflowModel.nodes))
                .where(WorkflowModel.name.ilike(f"%{name_pattern}%"))
                .offset(offset)
                .limit(limit)
                .order_by(WorkflowModel.name)
            )
            result = await self._session.execute(stmt)
            workflow_models = result.scalars().all()
            
            workflows = []
            for model in workflow_models:
                workflow = await self._model_to_entity(model)
                workflows.append(workflow)
            
            return workflows
            
        except Exception as e:
            logger.error(f"Failed to search workflows by name pattern {name_pattern}: {e}")
            raise
    
    async def count(self) -> int:
        """Get total count of workflows"""
        try:
            stmt = select(func.count(WorkflowModel.id))
            result = await self._session.execute(stmt)
            return result.scalar()
            
        except Exception as e:
            logger.error(f"Failed to count workflows: {e}")
            raise
    
    async def count_by_status(self, status: WorkflowStatus) -> int:
        """Get count of workflows by status"""
        try:
            stmt = select(func.count(WorkflowModel.id)).where(WorkflowModel.status == status.value)
            result = await self._session.execute(stmt)
            return result.scalar()
            
        except Exception as e:
            logger.error(f"Failed to count workflows by status {status}: {e}")
            raise
    
    async def delete(self, workflow_id: WorkflowId) -> bool:
        """Delete a workflow"""
        try:
            workflow_model = await self._get_workflow_model(workflow_id)
            if workflow_model:
                await self._session.delete(workflow_model)
                await self._session.commit()
                logger.debug(f"Deleted workflow {workflow_id}")
                return True
            return False
            
        except Exception as e:
            await self._session.rollback()
            logger.error(f"Failed to delete workflow {workflow_id}: {e}")
            raise
    
    async def exists(self, workflow_id: WorkflowId) -> bool:
        """Check if a workflow exists"""
        try:
            workflow_model = await self._get_workflow_model(workflow_id)
            return workflow_model is not None
            
        except Exception as e:
            logger.error(f"Failed to check if workflow {workflow_id} exists: {e}")
            raise
    
    async def exists_by_name(self, name: str) -> bool:
        """Check if a workflow with the given name exists"""
        try:
            stmt = select(func.count(WorkflowModel.id)).where(WorkflowModel.name == name)
            result = await self._session.execute(stmt)
            count = result.scalar()
            return count > 0
            
        except Exception as e:
            logger.error(f"Failed to check if workflow name {name} exists: {e}")
            raise
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics"""
        try:
            # Total workflows
            total_stmt = select(func.count(WorkflowModel.id))
            total_result = await self._session.execute(total_stmt)
            total_workflows = total_result.scalar()
            
            # Workflows by status
            status_stmt = (
                select(WorkflowModel.status, func.count(WorkflowModel.id))
                .group_by(WorkflowModel.status)
            )
            status_result = await self._session.execute(status_stmt)
            status_counts = dict(status_result.fetchall())
            
            # Average nodes per workflow
            avg_nodes_stmt = select(func.avg(func.count(NodeModel.id))).select_from(
                WorkflowModel.join(NodeModel)
            ).group_by(WorkflowModel.id)
            avg_nodes_result = await self._session.execute(avg_nodes_stmt)
            avg_nodes = avg_nodes_result.scalar() or 0
            
            return {
                "total_workflows": total_workflows,
                "workflows_by_status": status_counts,
                "average_nodes_per_workflow": float(avg_nodes),
                "repository_type": "SQLAlchemy"
            }
            
        except Exception as e:
            logger.error(f"Failed to get repository statistics: {e}")
            raise
    
    async def _get_workflow_model(self, workflow_id: WorkflowId) -> Optional[WorkflowModel]:
        """Get workflow model by ID"""
        stmt = (
            select(WorkflowModel)
            .options(selectinload(WorkflowModel.nodes))
            .where(WorkflowModel.id == str(workflow_id))
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _create_workflow_model(self, workflow: WorkflowChain) -> None:
        """Create a new workflow model"""
        workflow_model = WorkflowModel(
            id=str(workflow.id),
            name=workflow.name,
            description=workflow.description,
            status=workflow.status.value,
            settings=workflow.settings,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            version=workflow.version
        )
        
        self._session.add(workflow_model)
        
        # Create node models
        for node in workflow.nodes.values():
            node_model = NodeModel(
                id=str(node.id),
                workflow_id=str(workflow.id),
                title=node.title,
                content=node.content,
                prompt=node.prompt,
                parent_id=str(node.parent_id) if node.parent_id else None,
                priority=node.priority.value,
                tags=node.tags,
                metadata=node.metadata,
                created_at=node.created_at,
                updated_at=node.updated_at,
                version=node.version
            )
            self._session.add(node_model)
    
    async def _update_workflow_model(self, workflow_model: WorkflowModel, workflow: WorkflowChain) -> None:
        """Update an existing workflow model"""
        workflow_model.name = workflow.name
        workflow_model.description = workflow.description
        workflow_model.status = workflow.status.value
        workflow_model.settings = workflow.settings
        workflow_model.updated_at = workflow.updated_at
        workflow_model.version = workflow.version
        
        # Update nodes
        await self._update_nodes(workflow_model, workflow)
    
    async def _update_nodes(self, workflow_model: WorkflowModel, workflow: WorkflowChain) -> None:
        """Update workflow nodes"""
        # Get existing nodes
        existing_nodes_stmt = select(NodeModel).where(NodeModel.workflow_id == str(workflow.id))
        existing_nodes_result = await self._session.execute(existing_nodes_stmt)
        existing_nodes = {node.id: node for node in existing_nodes_result.scalars()}
        
        # Update or create nodes
        for node in workflow.nodes.values():
            node_id = str(node.id)
            if node_id in existing_nodes:
                # Update existing node
                node_model = existing_nodes[node_id]
                node_model.title = node.title
                node_model.content = node.content
                node_model.prompt = node.prompt
                node_model.parent_id = str(node.parent_id) if node.parent_id else None
                node_model.priority = node.priority.value
                node_model.tags = node.tags
                node_model.metadata = node.metadata
                node_model.updated_at = node.updated_at
                node_model.version = node.version
            else:
                # Create new node
                node_model = NodeModel(
                    id=node_id,
                    workflow_id=str(workflow.id),
                    title=node.title,
                    content=node.content,
                    prompt=node.prompt,
                    parent_id=str(node.parent_id) if node.parent_id else None,
                    priority=node.priority.value,
                    tags=node.tags,
                    metadata=node.metadata,
                    created_at=node.created_at,
                    updated_at=node.updated_at,
                    version=node.version
                )
                self._session.add(node_model)
        
        # Remove deleted nodes
        current_node_ids = {str(node.id) for node in workflow.nodes.values()}
        for node_id, node_model in existing_nodes.items():
            if node_id not in current_node_ids:
                await self._session.delete(node_model)
    
    async def _model_to_entity(self, workflow_model: WorkflowModel) -> WorkflowChain:
        """Convert workflow model to domain entity"""
        from ...domain.value_objects.workflow_id import WorkflowId
        from ...domain.value_objects.workflow_status import WorkflowStatus
        from ...domain.value_objects.node_id import NodeId
        from ...domain.value_objects.priority import Priority
        
        # Create workflow entity
        workflow = WorkflowChain(
            workflow_id=WorkflowId.from_string(workflow_model.id),
            name=workflow_model.name,
            description=workflow_model.description,
            status=WorkflowStatus(workflow_model.status),
            settings=workflow_model.settings
        )
        
        # Set private attributes
        workflow._created_at = workflow_model.created_at
        workflow._updated_at = workflow_model.updated_at
        workflow._version = workflow_model.version
        
        # Create node entities
        for node_model in workflow_model.nodes:
            node = WorkflowNode(
                node_id=NodeId.from_string(node_model.id),
                title=node_model.title,
                content=node_model.content,
                prompt=node_model.prompt,
                parent_id=NodeId.from_string(node_model.parent_id) if node_model.parent_id else None,
                priority=Priority(node_model.priority),
                tags=node_model.tags,
                metadata=node_model.metadata
            )
            
            # Set private attributes
            node._created_at = node_model.created_at
            node._updated_at = node_model.updated_at
            node._version = node_model.version
            
            # Add to workflow
            workflow._nodes[node.id] = node
        
        return workflow




