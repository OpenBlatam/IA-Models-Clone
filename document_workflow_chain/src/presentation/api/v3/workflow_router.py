"""
Workflow API Router v3
======================

Advanced REST API endpoints for workflow management.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ....application.services.workflow_application_service import WorkflowApplicationService
from ....application.dto.workflow_dto import WorkflowCreateDTO, WorkflowUpdateDTO
from ....application.dto.workflow_response_dto import (
    WorkflowResponseDTO,
    WorkflowListResponseDTO,
    WorkflowStatisticsResponseDTO,
    WorkflowHealthResponseDTO
)
from ....application.dto.node_dto import NodeCreateDTO, NodeUpdateDTO
from ....application.dto.node_response_dto import NodeResponseDTO
from ....domain.value_objects.workflow_id import WorkflowId
from ....domain.value_objects.node_id import NodeId
from ....domain.value_objects.workflow_status import WorkflowStatus
from ....shared.container import Container
from ....shared.exceptions.application_exceptions import (
    ValidationException,
    ResourceNotFoundException,
    BusinessRuleException,
    ConcurrencyException
)
from ....shared.utils.decorators import rate_limit, cache, log_execution
from ....shared.utils.helpers import PaginationHelpers


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3/workflows", tags=["Workflows v3"])


# Dependency injection
def get_workflow_service() -> WorkflowApplicationService:
    """Get workflow application service"""
    container = Container()
    return container.get_workflow_application_service()


# Request/Response models
class WorkflowCreateRequest(BaseModel):
    """Workflow creation request"""
    name: str = Field(..., min_length=3, max_length=255, description="Workflow name")
    description: Optional[str] = Field(None, max_length=1000, description="Workflow description")
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Workflow settings")
    nodes: List[NodeCreateDTO] = Field(default_factory=list, description="Initial nodes")


class WorkflowUpdateRequest(BaseModel):
    """Workflow update request"""
    name: Optional[str] = Field(None, min_length=3, max_length=255, description="Workflow name")
    description: Optional[str] = Field(None, max_length=1000, description="Workflow description")
    status: Optional[str] = Field(None, description="Workflow status")
    settings: Optional[Dict[str, Any]] = Field(None, description="Workflow settings")


class NodeCreateRequest(BaseModel):
    """Node creation request"""
    title: str = Field(..., min_length=3, max_length=255, description="Node title")
    content: str = Field(..., min_length=1, max_length=100000, description="Node content")
    prompt: str = Field(..., min_length=1, max_length=10000, description="Node prompt")
    parent_id: Optional[str] = Field(None, description="Parent node ID")
    priority: int = Field(3, ge=1, le=5, description="Node priority (1-5)")
    tags: List[str] = Field(default_factory=list, description="Node tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Node metadata")


class NodeUpdateRequest(BaseModel):
    """Node update request"""
    title: Optional[str] = Field(None, min_length=3, max_length=255, description="Node title")
    content: Optional[str] = Field(None, min_length=1, max_length=100000, description="Node content")
    prompt: Optional[str] = Field(None, min_length=1, max_length=10000, description="Node prompt")
    parent_id: Optional[str] = Field(None, description="Parent node ID")
    priority: Optional[int] = Field(None, ge=1, le=5, description="Node priority (1-5)")
    status: Optional[str] = Field(None, description="Node status")
    tags: Optional[List[str]] = Field(None, description="Node tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Node metadata")


class BulkOperationRequest(BaseModel):
    """Bulk operation request"""
    operation: str = Field(..., description="Operation type")
    workflow_ids: List[str] = Field(..., description="List of workflow IDs")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Operation parameters")


# Error handlers
def handle_validation_error(e: ValidationException) -> JSONResponse:
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Validation Error",
            "message": e.message,
            "details": e.details
        }
    )


def handle_not_found_error(e: ResourceNotFoundException) -> JSONResponse:
    """Handle not found errors"""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Resource Not Found",
            "message": e.message,
            "details": e.details
        }
    )


def handle_business_rule_error(e: BusinessRuleException) -> JSONResponse:
    """Handle business rule errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Business Rule Violation",
            "message": e.message,
            "details": e.details
        }
    )


def handle_concurrency_error(e: ConcurrencyException) -> JSONResponse:
    """Handle concurrency errors"""
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "error": "Concurrency Conflict",
            "message": e.message,
            "details": e.details
        }
    )


# Workflow endpoints
@router.post(
    "/",
    response_model=WorkflowResponseDTO,
    status_code=status.HTTP_201_CREATED,
    summary="Create Workflow",
    description="Create a new workflow chain with optional initial nodes"
)
@rate_limit(max_calls=10, time_window=60)
@log_execution()
async def create_workflow(
    request: WorkflowCreateRequest,
    workflow_service: WorkflowApplicationService = Depends(get_workflow_service),
    user_id: Optional[str] = Query(None, description="User ID")
) -> WorkflowResponseDTO:
    """Create a new workflow"""
    try:
        workflow_data = WorkflowCreateDTO(
            name=request.name,
            description=request.description,
            settings=request.settings,
            nodes=request.nodes
        )
        
        return await workflow_service.create_workflow(workflow_data, user_id)
        
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/{workflow_id}",
    response_model=WorkflowResponseDTO,
    summary="Get Workflow",
    description="Get workflow by ID with all nodes"
)
@cache(ttl_seconds=300)
@log_execution()
async def get_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    workflow_service: WorkflowApplicationService = Depends(get_workflow_service),
    user_id: Optional[str] = Query(None, description="User ID")
) -> WorkflowResponseDTO:
    """Get workflow by ID"""
    try:
        workflow_id_obj = WorkflowId(workflow_id)
        return await workflow_service.get_workflow(workflow_id_obj, user_id)
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid workflow ID format"
        )
    except ResourceNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Failed to get workflow {workflow_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put(
    "/{workflow_id}",
    response_model=WorkflowResponseDTO,
    summary="Update Workflow",
    description="Update workflow with optimistic locking"
)
@rate_limit(max_calls=20, time_window=60)
@log_execution()
async def update_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    request: WorkflowUpdateRequest = Body(...),
    workflow_service: WorkflowApplicationService = Depends(get_workflow_service),
    user_id: Optional[str] = Query(None, description="User ID"),
    expected_version: Optional[int] = Query(None, description="Expected version for optimistic locking")
) -> WorkflowResponseDTO:
    """Update workflow"""
    try:
        workflow_id_obj = WorkflowId(workflow_id)
        
        updates = WorkflowUpdateDTO(
            name=request.name,
            description=request.description,
            status=request.status,
            settings=request.settings
        )
        
        return await workflow_service.update_workflow(
            workflow_id_obj, updates, user_id, expected_version
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid workflow ID format"
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except ResourceNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except ConcurrencyException as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Failed to update workflow {workflow_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.delete(
    "/{workflow_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete Workflow",
    description="Soft delete workflow"
)
@rate_limit(max_calls=5, time_window=60)
@log_execution()
async def delete_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    workflow_service: WorkflowApplicationService = Depends(get_workflow_service),
    user_id: Optional[str] = Query(None, description="User ID")
):
    """Delete workflow"""
    try:
        workflow_id_obj = WorkflowId(workflow_id)
        await workflow_service.delete_workflow(workflow_id_obj, user_id)
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid workflow ID format"
        )
    except ResourceNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except BusinessRuleException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Failed to delete workflow {workflow_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/",
    response_model=WorkflowListResponseDTO,
    summary="List Workflows",
    description="List workflows with filtering and pagination"
)
@cache(ttl_seconds=60)
@log_execution()
async def list_workflows(
    workflow_service: WorkflowApplicationService = Depends(get_workflow_service),
    user_id: Optional[str] = Query(None, description="User ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Page size"),
    search: Optional[str] = Query(None, description="Search query")
) -> WorkflowListResponseDTO:
    """List workflows with filtering and pagination"""
    try:
        # Convert status string to enum if provided
        status_enum = None
        if status:
            try:
                status_enum = WorkflowStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status}"
                )
        
        # Calculate offset
        offset = PaginationHelpers.calculate_offset(page, page_size)
        
        # Get workflows
        workflows = await workflow_service.list_workflows(
            user_id=user_id,
            status=status_enum,
            limit=page_size,
            offset=offset
        )
        
        # Calculate pagination info
        total_count = len(workflows)  # This would be the actual total from the service
        total_pages = (total_count + page_size - 1) // page_size
        
        return WorkflowListResponseDTO(
            workflows=workflows,
            total_count=total_count,
            page=page,
            page_size=page_size,
            has_next=page < total_pages,
            has_previous=page > 1
        )
        
    except Exception as e:
        logger.error(f"Failed to list workflows: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get(
    "/{workflow_id}/statistics",
    response_model=WorkflowStatisticsResponseDTO,
    summary="Get Workflow Statistics",
    description="Get detailed statistics for a workflow"
)
@cache(ttl_seconds=300)
@log_execution()
async def get_workflow_statistics(
    workflow_id: str = Path(..., description="Workflow ID"),
    workflow_service: WorkflowApplicationService = Depends(get_workflow_service),
    user_id: Optional[str] = Query(None, description="User ID")
) -> WorkflowStatisticsResponseDTO:
    """Get workflow statistics"""
    try:
        workflow_id_obj = WorkflowId(workflow_id)
        stats_data = await workflow_service.get_workflow_statistics(workflow_id_obj, user_id)
        
        return WorkflowStatisticsResponseDTO(**stats_data)
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid workflow ID format"
        )
    except ResourceNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Failed to get workflow statistics {workflow_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Node endpoints
@router.post(
    "/{workflow_id}/nodes",
    response_model=NodeResponseDTO,
    status_code=status.HTTP_201_CREATED,
    summary="Add Node to Workflow",
    description="Add a new node to an existing workflow"
)
@rate_limit(max_calls=30, time_window=60)
@log_execution()
async def add_node_to_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    request: NodeCreateRequest = Body(...),
    workflow_service: WorkflowApplicationService = Depends(get_workflow_service),
    user_id: Optional[str] = Query(None, description="User ID")
) -> NodeResponseDTO:
    """Add node to workflow"""
    try:
        workflow_id_obj = WorkflowId(workflow_id)
        
        node_data = NodeCreateDTO(
            title=request.title,
            content=request.content,
            prompt=request.prompt,
            parent_id=request.parent_id,
            priority=request.priority,
            tags=request.tags,
            metadata=request.metadata
        )
        
        return await workflow_service.add_node_to_workflow(workflow_id_obj, node_data, user_id)
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid workflow ID format"
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except ResourceNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except BusinessRuleException as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Failed to add node to workflow {workflow_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put(
    "/{workflow_id}/nodes/{node_id}",
    response_model=NodeResponseDTO,
    summary="Update Node",
    description="Update node in workflow with optimistic locking"
)
@rate_limit(max_calls=50, time_window=60)
@log_execution()
async def update_node(
    workflow_id: str = Path(..., description="Workflow ID"),
    node_id: str = Path(..., description="Node ID"),
    request: NodeUpdateRequest = Body(...),
    workflow_service: WorkflowApplicationService = Depends(get_workflow_service),
    user_id: Optional[str] = Query(None, description="User ID"),
    expected_version: Optional[int] = Query(None, description="Expected version for optimistic locking")
) -> NodeResponseDTO:
    """Update node in workflow"""
    try:
        workflow_id_obj = WorkflowId(workflow_id)
        node_id_obj = NodeId(node_id)
        
        updates = NodeUpdateDTO(
            title=request.title,
            content=request.content,
            prompt=request.prompt,
            parent_id=request.parent_id,
            priority=request.priority,
            status=request.status,
            tags=request.tags,
            metadata=request.metadata
        )
        
        return await workflow_service.update_node(
            workflow_id_obj, node_id_obj, updates, user_id, expected_version
        )
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ID format"
        )
    except ValidationException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )
    except ResourceNotFoundException as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except ConcurrencyException as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message
        )
    except Exception as e:
        logger.error(f"Failed to update node {node_id} in workflow {workflow_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Bulk operations
@router.post(
    "/bulk",
    summary="Bulk Operations",
    description="Perform bulk operations on multiple workflows"
)
@rate_limit(max_calls=5, time_window=60)
@log_execution()
async def bulk_operations(
    request: BulkOperationRequest = Body(...),
    workflow_service: WorkflowApplicationService = Depends(get_workflow_service),
    user_id: Optional[str] = Query(None, description="User ID")
):
    """Perform bulk operations"""
    try:
        results = []
        
        for workflow_id_str in request.workflow_ids:
            try:
                workflow_id = WorkflowId(workflow_id_str)
                
                if request.operation == "delete":
                    await workflow_service.delete_workflow(workflow_id, user_id)
                    results.append({"workflow_id": workflow_id_str, "status": "success"})
                elif request.operation == "activate":
                    updates = WorkflowUpdateDTO(status="active")
                    await workflow_service.update_workflow(workflow_id, updates, user_id)
                    results.append({"workflow_id": workflow_id_str, "status": "success"})
                elif request.operation == "pause":
                    updates = WorkflowUpdateDTO(status="paused")
                    await workflow_service.update_workflow(workflow_id, updates, user_id)
                    results.append({"workflow_id": workflow_id_str, "status": "success"})
                else:
                    results.append({
                        "workflow_id": workflow_id_str,
                        "status": "error",
                        "error": f"Unknown operation: {request.operation}"
                    })
                    
            except Exception as e:
                results.append({
                    "workflow_id": workflow_id_str,
                    "status": "error",
                    "error": str(e)
                })
        
        return {
            "operation": request.operation,
            "total_processed": len(request.workflow_ids),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to perform bulk operation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Health check endpoint
@router.get(
    "/health",
    summary="Health Check",
    description="Check the health of the workflow service"
)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "workflow-api-v3",
        "timestamp": "2024-01-01T00:00:00Z"
    }




