"""
Workflow Controller
==================

FastAPI controller for workflow operations with advanced features:
- Async/await throughout
- Dependency injection
- Request/response validation
- Error handling
- Rate limiting
- Caching
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ...application.use_cases.create_workflow_use_case import (
    CreateWorkflowUseCase, CreateWorkflowRequest, CreateWorkflowResponse
)
from ...application.use_cases.add_node_use_case import (
    AddNodeUseCase, AddNodeRequest, AddNodeResponse
)
from ...application.use_cases.get_workflow_use_case import (
    GetWorkflowUseCase, GetWorkflowRequest, GetWorkflowResponse
)
from ...application.use_cases.list_workflows_use_case import (
    ListWorkflowsUseCase, ListWorkflowsRequest, ListWorkflowsResponse
)
from ...shared.container import DependencyInjectionContainer
from ...shared.exceptions.application_exceptions import ApplicationException
from ...shared.middleware.rate_limiter import RateLimiter
from ...shared.middleware.cache import CacheMiddleware
from ...shared.middleware.auth import AuthMiddleware


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v3/workflows", tags=["workflows"])


# Request/Response Models
class CreateWorkflowApiRequest(BaseModel):
    """API request model for creating workflow"""
    name: str = Field(..., min_length=1, max_length=255, description="Workflow name")
    description: str = Field("", max_length=1000, description="Workflow description")
    settings: Optional[Dict[str, Any]] = Field(None, description="Workflow settings")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()


class CreateWorkflowApiResponse(BaseModel):
    """API response model for creating workflow"""
    workflow_id: str
    name: str
    description: str
    status: str
    created_at: str
    success: bool
    message: str


class AddNodeApiRequest(BaseModel):
    """API request model for adding node"""
    workflow_id: str = Field(..., description="Workflow ID")
    title: str = Field(..., min_length=1, max_length=255, description="Node title")
    content: str = Field(..., min_length=1, description="Node content")
    prompt: str = Field(..., min_length=1, description="Node prompt")
    parent_id: Optional[str] = Field(None, description="Parent node ID")
    priority: int = Field(2, ge=1, le=5, description="Node priority (1-5)")
    tags: Optional[List[str]] = Field(None, description="Node tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Node metadata")


class AddNodeApiResponse(BaseModel):
    """API response model for adding node"""
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
    message: str


class WorkflowApiResponse(BaseModel):
    """API response model for workflow"""
    workflow_id: str
    name: str
    description: str
    status: str
    created_at: str
    updated_at: str
    version: int
    statistics: Dict[str, Any]


class WorkflowListApiResponse(BaseModel):
    """API response model for workflow list"""
    workflows: List[WorkflowApiResponse]
    total: int
    limit: int
    offset: int
    has_more: bool


# Dependency functions
async def get_container() -> DependencyInjectionContainer:
    """Get dependency injection container"""
    from ...shared.container import get_container
    return get_container()


async def get_create_workflow_use_case(
    container: DependencyInjectionContainer = Depends(get_container)
) -> CreateWorkflowUseCase:
    """Get create workflow use case"""
    return await container.resolve(CreateWorkflowUseCase)


async def get_add_node_use_case(
    container: DependencyInjectionContainer = Depends(get_container)
) -> AddNodeUseCase:
    """Get add node use case"""
    return await container.resolve(AddNodeUseCase)


async def get_workflow_use_case(
    container: DependencyInjectionContainer = Depends(get_container)
) -> GetWorkflowUseCase:
    """Get workflow use case"""
    return await container.resolve(GetWorkflowUseCase)


async def get_list_workflows_use_case(
    container: DependencyInjectionContainer = Depends(get_container)
) -> ListWorkflowsUseCase:
    """Get list workflows use case"""
    return await container.resolve(ListWorkflowsUseCase)


# API Endpoints
@router.post("/", response_model=CreateWorkflowApiResponse)
async def create_workflow(
    request: CreateWorkflowApiRequest,
    background_tasks: BackgroundTasks,
    use_case: CreateWorkflowUseCase = Depends(get_create_workflow_use_case),
    rate_limiter: RateLimiter = Depends(),
    cache: CacheMiddleware = Depends(),
    auth: AuthMiddleware = Depends()
):
    """
    Create a new workflow
    
    Creates a new workflow with the specified name, description, and settings.
    The workflow will be created in DRAFT status and can be activated later.
    """
    try:
        # Rate limiting
        await rate_limiter.check_limit("create_workflow")
        
        # Authentication
        user_id = await auth.get_current_user_id()
        
        # Convert API request to use case request
        use_case_request = CreateWorkflowRequest(
            name=request.name,
            description=request.description,
            settings=request.settings
        )
        
        # Execute use case
        use_case_response = await use_case.execute(use_case_request)
        
        # Cache the result
        await cache.set(f"workflow:{use_case_response.workflow_id}", use_case_response, ttl=300)
        
        # Background task: Send notification
        background_tasks.add_task(
            send_workflow_created_notification,
            use_case_response.workflow_id,
            user_id
        )
        
        logger.info(f"Created workflow {use_case_response.workflow_id} for user {user_id}")
        
        return CreateWorkflowApiResponse(
            workflow_id=use_case_response.workflow_id,
            name=use_case_response.name,
            description=use_case_response.description,
            status=use_case_response.status,
            created_at=use_case_response.created_at,
            success=use_case_response.success,
            message=use_case_response.message
        )
        
    except ApplicationException as e:
        logger.warning(f"Application error creating workflow: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error creating workflow: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/", response_model=WorkflowListApiResponse)
async def list_workflows(
    limit: int = Query(20, ge=1, le=100, description="Number of workflows to return"),
    offset: int = Query(0, ge=0, description="Number of workflows to skip"),
    status: Optional[str] = Query(None, description="Filter by workflow status"),
    search: Optional[str] = Query(None, description="Search workflows by name"),
    use_case: ListWorkflowsUseCase = Depends(get_list_workflows_use_case),
    cache: CacheMiddleware = Depends(),
    auth: AuthMiddleware = Depends()
):
    """
    List workflows with pagination and filtering
    
    Returns a paginated list of workflows with optional filtering by status
    and search by name.
    """
    try:
        # Authentication
        user_id = await auth.get_current_user_id()
        
        # Check cache
        cache_key = f"workflows:list:{limit}:{offset}:{status}:{search}:{user_id}"
        cached_result = await cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Convert API parameters to use case request
        use_case_request = ListWorkflowsRequest(
            limit=limit,
            offset=offset,
            status=status,
            search=search,
            user_id=user_id
        )
        
        # Execute use case
        use_case_response = await use_case.execute(use_case_request)
        
        # Convert to API response
        api_response = WorkflowListApiResponse(
            workflows=[
                WorkflowApiResponse(
                    workflow_id=w.workflow_id,
                    name=w.name,
                    description=w.description,
                    status=w.status,
                    created_at=w.created_at,
                    updated_at=w.updated_at,
                    version=w.version,
                    statistics=w.statistics
                )
                for w in use_case_response.workflows
            ],
            total=use_case_response.total,
            limit=use_case_response.limit,
            offset=use_case_response.offset,
            has_more=use_case_response.has_more
        )
        
        # Cache the result
        await cache.set(cache_key, api_response, ttl=60)
        
        return api_response
        
    except ApplicationException as e:
        logger.warning(f"Application error listing workflows: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error listing workflows: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{workflow_id}", response_model=WorkflowApiResponse)
async def get_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    use_case: GetWorkflowUseCase = Depends(get_workflow_use_case),
    cache: CacheMiddleware = Depends(),
    auth: AuthMiddleware = Depends()
):
    """
    Get workflow by ID
    
    Returns detailed information about a specific workflow including
    its nodes and statistics.
    """
    try:
        # Authentication
        user_id = await auth.get_current_user_id()
        
        # Check cache
        cache_key = f"workflow:{workflow_id}:{user_id}"
        cached_result = await cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Convert API parameters to use case request
        use_case_request = GetWorkflowRequest(
            workflow_id=workflow_id,
            user_id=user_id
        )
        
        # Execute use case
        use_case_response = await use_case.execute(use_case_request)
        
        if not use_case_response.workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Convert to API response
        api_response = WorkflowApiResponse(
            workflow_id=use_case_response.workflow.workflow_id,
            name=use_case_response.workflow.name,
            description=use_case_response.workflow.description,
            status=use_case_response.workflow.status,
            created_at=use_case_response.workflow.created_at,
            updated_at=use_case_response.workflow.updated_at,
            version=use_case_response.workflow.version,
            statistics=use_case_response.workflow.statistics
        )
        
        # Cache the result
        await cache.set(cache_key, api_response, ttl=300)
        
        return api_response
        
    except HTTPException:
        raise
    except ApplicationException as e:
        logger.warning(f"Application error getting workflow: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error getting workflow: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{workflow_id}/nodes", response_model=AddNodeApiResponse)
async def add_node_to_workflow(
    workflow_id: str = Path(..., description="Workflow ID"),
    request: AddNodeApiRequest = ...,
    background_tasks: BackgroundTasks = ...,
    use_case: AddNodeUseCase = Depends(get_add_node_use_case),
    rate_limiter: RateLimiter = Depends(),
    cache: CacheMiddleware = Depends(),
    auth: AuthMiddleware = Depends()
):
    """
    Add node to workflow
    
    Adds a new node to the specified workflow with the given content,
    prompt, and metadata.
    """
    try:
        # Rate limiting
        await rate_limiter.check_limit("add_node")
        
        # Authentication
        user_id = await auth.get_current_user_id()
        
        # Validate workflow_id matches path parameter
        if request.workflow_id != workflow_id:
            raise HTTPException(status_code=400, detail="Workflow ID mismatch")
        
        # Convert API request to use case request
        use_case_request = AddNodeRequest(
            workflow_id=request.workflow_id,
            title=request.title,
            content=request.content,
            prompt=request.prompt,
            parent_id=request.parent_id,
            priority=request.priority,
            tags=request.tags,
            metadata=request.metadata
        )
        
        # Execute use case
        use_case_response = await use_case.execute(use_case_request)
        
        # Invalidate cache for this workflow
        await cache.delete(f"workflow:{workflow_id}:{user_id}")
        await cache.delete(f"workflows:list:*")  # Invalidate list cache
        
        # Background task: Send notification
        background_tasks.add_task(
            send_node_added_notification,
            use_case_response.node_id,
            workflow_id,
            user_id
        )
        
        logger.info(f"Added node {use_case_response.node_id} to workflow {workflow_id} for user {user_id}")
        
        return AddNodeApiResponse(
            node_id=use_case_response.node_id,
            workflow_id=use_case_response.workflow_id,
            title=use_case_response.title,
            content=use_case_response.content,
            prompt=use_case_response.prompt,
            parent_id=use_case_response.parent_id,
            priority=use_case_response.priority,
            tags=use_case_response.tags,
            created_at=use_case_response.created_at,
            success=use_case_response.success,
            message=use_case_response.message
        )
        
    except HTTPException:
        raise
    except ApplicationException as e:
        logger.warning(f"Application error adding node: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error adding node: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Background task functions
async def send_workflow_created_notification(workflow_id: str, user_id: str) -> None:
    """Send notification when workflow is created"""
    try:
        # Implementation would send notification via email, push, etc.
        logger.info(f"Sending workflow created notification for {workflow_id} to user {user_id}")
    except Exception as e:
        logger.error(f"Failed to send workflow created notification: {e}")


async def send_node_added_notification(node_id: str, workflow_id: str, user_id: str) -> None:
    """Send notification when node is added"""
    try:
        # Implementation would send notification via email, push, etc.
        logger.info(f"Sending node added notification for {node_id} in workflow {workflow_id} to user {user_id}")
    except Exception as e:
        logger.error(f"Failed to send node added notification: {e}")


# Error handlers
@router.exception_handler(ApplicationException)
async def application_exception_handler(request, exc: ApplicationException):
    """Handle application exceptions"""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Application Error",
            "message": str(exc),
            "error_code": getattr(exc, 'error_code', 'APPLICATION_ERROR')
        }
    )


@router.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception in workflow controller: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )




