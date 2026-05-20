"""
Workflow API
============

Simple and clear workflow API for the Document Workflow Chain system.
"""

from __future__ import annotations
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from ..core.database import get_database
from ..models.workflow import WorkflowChain, WorkflowNode

# Create router
router = APIRouter()


# Request/Response models
class WorkflowCreate(BaseModel):
    """Workflow creation request"""
    name: str
    description: Optional[str] = None
    config: Optional[dict] = None


class WorkflowUpdate(BaseModel):
    """Workflow update request"""
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    config: Optional[dict] = None


class WorkflowResponse(BaseModel):
    """Workflow response"""
    id: int
    name: str
    description: Optional[str]
    status: str
    priority: str
    config: Optional[dict]
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True


class NodeCreate(BaseModel):
    """Node creation request"""
    name: str
    description: Optional[str] = None
    node_type: str
    config: Optional[dict] = None
    input_data: Optional[dict] = None


class NodeResponse(BaseModel):
    """Node response"""
    id: int
    name: str
    description: Optional[str]
    node_type: str
    status: str
    config: Optional[dict]
    input_data: Optional[dict]
    output_data: Optional[dict]
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True


# Workflow endpoints
@router.get("/", response_model=List[WorkflowResponse])
async def get_workflows(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_database)
):
    """Get all workflows - simple and clear"""
    try:
        from sqlalchemy import select
        
        result = await db.execute(
            select(WorkflowChain).offset(skip).limit(limit)
        )
        workflows = result.scalars().all()
        
        return [
            WorkflowResponse(
                id=w.id,
                name=w.name,
                description=w.description,
                status=w.status,
                priority=w.priority,
                config=w.config,
                created_at=w.created_at.isoformat(),
                updated_at=w.updated_at.isoformat()
            )
            for w in workflows
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflows: {str(e)}"
        )


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: int,
    db: AsyncSession = Depends(get_database)
):
    """Get workflow by ID - simple and clear"""
    try:
        from sqlalchemy import select
        
        result = await db.execute(
            select(WorkflowChain).where(WorkflowChain.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow not found"
            )
        
        return WorkflowResponse(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            status=workflow.status,
            priority=workflow.priority,
            config=workflow.config,
            created_at=workflow.created_at.isoformat(),
            updated_at=workflow.updated_at.isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow: {str(e)}"
        )


@router.post("/", response_model=WorkflowResponse)
async def create_workflow(
    workflow: WorkflowCreate,
    db: AsyncSession = Depends(get_database)
):
    """Create new workflow - simple and clear"""
    try:
        new_workflow = WorkflowChain(
            name=workflow.name,
            description=workflow.description,
            config=workflow.config
        )
        
        db.add(new_workflow)
        await db.commit()
        await db.refresh(new_workflow)
        
        return WorkflowResponse(
            id=new_workflow.id,
            name=new_workflow.name,
            description=new_workflow.description,
            status=new_workflow.status,
            priority=new_workflow.priority,
            config=new_workflow.config,
            created_at=new_workflow.created_at.isoformat(),
            updated_at=new_workflow.updated_at.isoformat()
        )
    
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: int,
    workflow: WorkflowUpdate,
    db: AsyncSession = Depends(get_database)
):
    """Update workflow - simple and clear"""
    try:
        from sqlalchemy import select
        
        result = await db.execute(
            select(WorkflowChain).where(WorkflowChain.id == workflow_id)
        )
        existing_workflow = result.scalar_one_or_none()
        
        if not existing_workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow not found"
            )
        
        # Update fields
        if workflow.name is not None:
            existing_workflow.name = workflow.name
        if workflow.description is not None:
            existing_workflow.description = workflow.description
        if workflow.status is not None:
            existing_workflow.status = workflow.status
        if workflow.config is not None:
            existing_workflow.config = workflow.config
        
        await db.commit()
        await db.refresh(existing_workflow)
        
        return WorkflowResponse(
            id=existing_workflow.id,
            name=existing_workflow.name,
            description=existing_workflow.description,
            status=existing_workflow.status,
            priority=existing_workflow.priority,
            config=existing_workflow.config,
            created_at=existing_workflow.created_at.isoformat(),
            updated_at=existing_workflow.updated_at.isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update workflow: {str(e)}"
        )


@router.delete("/{workflow_id}")
async def delete_workflow(
    workflow_id: int,
    db: AsyncSession = Depends(get_database)
):
    """Delete workflow - simple and clear"""
    try:
        from sqlalchemy import select
        
        result = await db.execute(
            select(WorkflowChain).where(WorkflowChain.id == workflow_id)
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow not found"
            )
        
        await db.delete(workflow)
        await db.commit()
        
        return {"message": "Workflow deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete workflow: {str(e)}"
        )


# Node endpoints
@router.get("/{workflow_id}/nodes", response_model=List[NodeResponse])
async def get_workflow_nodes(
    workflow_id: int,
    db: AsyncSession = Depends(get_database)
):
    """Get workflow nodes - simple and clear"""
    try:
        from sqlalchemy import select
        
        result = await db.execute(
            select(WorkflowNode).where(WorkflowNode.workflow_id == workflow_id)
        )
        nodes = result.scalars().all()
        
        return [
            NodeResponse(
                id=n.id,
                name=n.name,
                description=n.description,
                node_type=n.node_type,
                status=n.status,
                config=n.config,
                input_data=n.input_data,
                output_data=n.output_data,
                created_at=n.created_at.isoformat(),
                updated_at=n.updated_at.isoformat()
            )
            for n in nodes
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow nodes: {str(e)}"
        )


@router.post("/{workflow_id}/nodes", response_model=NodeResponse)
async def create_workflow_node(
    workflow_id: int,
    node: NodeCreate,
    db: AsyncSession = Depends(get_database)
):
    """Create workflow node - simple and clear"""
    try:
        new_node = WorkflowNode(
            name=node.name,
            description=node.description,
            node_type=node.node_type,
            config=node.config,
            input_data=node.input_data,
            workflow_id=workflow_id
        )
        
        db.add(new_node)
        await db.commit()
        await db.refresh(new_node)
        
        return NodeResponse(
            id=new_node.id,
            name=new_node.name,
            description=new_node.description,
            node_type=new_node.node_type,
            status=new_node.status,
            config=new_node.config,
            input_data=new_node.input_data,
            output_data=new_node.output_data,
            created_at=new_node.created_at.isoformat(),
            updated_at=new_node.updated_at.isoformat()
        )
    
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow node: {str(e)}"
        )


