"""
Database API Endpoints
======================

REST API endpoints for database operations, analytics, and data management.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from ..services.database_service import DatabaseService
from ..middleware.auth_middleware import get_current_user, require_permission, User

# Create API router
router = APIRouter(prefix="/database", tags=["Database"])

# Pydantic models
class UserCreateRequest(BaseModel):
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    password_hash: str = Field(..., description="Password hash")
    first_name: str = Field(..., description="First name")
    last_name: str = Field(..., description="Last name")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class UserUpdateRequest(BaseModel):
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    is_active: Optional[bool] = Field(None, description="Active status")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class BusinessAgentCreateRequest(BaseModel):
    name: str = Field(..., description="Agent name")
    business_area: str = Field(..., description="Business area")
    description: Optional[str] = Field(None, description="Agent description")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class WorkflowCreateRequest(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    business_area: str = Field(..., description="Business area")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow steps")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Workflow configuration")
    created_by: str = Field(..., description="User ID who created the workflow")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class DocumentCreateRequest(BaseModel):
    title: str = Field(..., description="Document title")
    document_type: str = Field(..., description="Document type")
    business_area: str = Field(..., description="Business area")
    content: Optional[str] = Field(None, description="Document content")
    format: str = Field("markdown", description="Document format")
    file_path: Optional[str] = Field(None, description="File path")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    created_by: str = Field(..., description="User ID who created the document")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class MetricRecordRequest(BaseModel):
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    metric_type: str = Field(..., description="Metric type")
    tags: Optional[Dict[str, str]] = Field(None, description="Metric tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: Optional[int] = Field(50, description="Maximum number of results")

# Global database service instance
db_service = None

def get_db_service() -> DatabaseService:
    """Get global database service instance."""
    global db_service
    if db_service is None:
        db_service = DatabaseService({"cache_enabled": True, "cache_ttl": 3600})
    return db_service

# API Endpoints

@router.get("/health", response_model=Dict[str, Any])
async def database_health_check():
    """Database health check."""
    
    db_service = get_db_service()
    
    try:
        health_status = await db_service.health_check()
        return health_status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database health check failed: {str(e)}")

@router.get("/cache/stats", response_model=Dict[str, Any])
async def get_cache_statistics(
    current_user: User = Depends(require_permission("database:view"))
):
    """Get cache statistics."""
    
    db_service = get_db_service()
    
    try:
        cache_stats = await db_service.get_cache_stats()
        return cache_stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache statistics: {str(e)}")

@router.post("/cache/clear", response_model=Dict[str, str])
async def clear_cache(
    pattern: Optional[str] = Query(None, description="Cache pattern to clear"),
    current_user: User = Depends(require_permission("database:manage"))
):
    """Clear cache."""
    
    db_service = get_db_service()
    
    try:
        await db_service.clear_cache(pattern)
        message = f"Cache cleared successfully" + (f" for pattern: {pattern}" if pattern else "")
        return {"message": message}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

# User endpoints
@router.post("/users", response_model=Dict[str, Any])
async def create_user(
    request: UserCreateRequest,
    current_user: User = Depends(require_permission("users:create"))
):
    """Create a new user."""
    
    db_service = get_db_service()
    
    try:
        user_data = request.dict()
        user = await db_service.create_user(user_data)
        return user.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create user: {str(e)}")

@router.get("/users/{user_id}", response_model=Dict[str, Any])
async def get_user(
    user_id: str,
    current_user: User = Depends(require_permission("users:view"))
):
    """Get user by ID."""
    
    db_service = get_db_service()
    
    try:
        user = await db_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user: {str(e)}")

@router.put("/users/{user_id}", response_model=Dict[str, Any])
async def update_user(
    user_id: str,
    request: UserUpdateRequest,
    current_user: User = Depends(require_permission("users:update"))
):
    """Update user."""
    
    db_service = get_db_service()
    
    try:
        update_data = {k: v for k, v in request.dict().items() if v is not None}
        user = await db_service.update_user(user_id, update_data)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update user: {str(e)}")

# Business Agent endpoints
@router.post("/agents", response_model=Dict[str, Any])
async def create_business_agent(
    request: BusinessAgentCreateRequest,
    current_user: User = Depends(require_permission("agents:create"))
):
    """Create a new business agent."""
    
    db_service = get_db_service()
    
    try:
        agent_data = request.dict()
        agent = await db_service.create_business_agent(agent_data)
        return agent.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create business agent: {str(e)}")

@router.get("/agents", response_model=List[Dict[str, Any]])
async def get_business_agents(
    business_area: Optional[str] = Query(None, description="Filter by business area"),
    current_user: User = Depends(require_permission("agents:view"))
):
    """Get business agents."""
    
    db_service = get_db_service()
    
    try:
        agents = await db_service.get_business_agents(business_area)
        return [agent.to_dict() for agent in agents]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get business agents: {str(e)}")

@router.get("/agents/{agent_id}", response_model=Dict[str, Any])
async def get_business_agent(
    agent_id: str,
    current_user: User = Depends(require_permission("agents:view"))
):
    """Get business agent by ID."""
    
    db_service = get_db_service()
    
    try:
        agent = await db_service.get_business_agent_by_id(agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Business agent not found")
        
        return agent.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get business agent: {str(e)}")

# Workflow endpoints
@router.post("/workflows", response_model=Dict[str, Any])
async def create_workflow(
    request: WorkflowCreateRequest,
    current_user: User = Depends(require_permission("workflows:create"))
):
    """Create a new workflow."""
    
    db_service = get_db_service()
    
    try:
        workflow_data = request.dict()
        workflow = await db_service.create_workflow(workflow_data)
        return workflow.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")

@router.get("/workflows", response_model=List[Dict[str, Any]])
async def get_workflows(
    business_area: Optional[str] = Query(None, description="Filter by business area"),
    status: Optional[str] = Query(None, description="Filter by status"),
    current_user: User = Depends(require_permission("workflows:view"))
):
    """Get workflows."""
    
    db_service = get_db_service()
    
    try:
        workflows = await db_service.get_workflows(business_area, status)
        return [workflow.to_dict() for workflow in workflows]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflows: {str(e)}")

@router.get("/workflows/{workflow_id}", response_model=Dict[str, Any])
async def get_workflow(
    workflow_id: str,
    current_user: User = Depends(require_permission("workflows:view"))
):
    """Get workflow by ID."""
    
    db_service = get_db_service()
    
    try:
        workflow = await db_service.get_workflow_by_id(workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return workflow.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow: {str(e)}")

@router.put("/workflows/{workflow_id}", response_model=Dict[str, Any])
async def update_workflow(
    workflow_id: str,
    request: Dict[str, Any],
    current_user: User = Depends(require_permission("workflows:update"))
):
    """Update workflow."""
    
    db_service = get_db_service()
    
    try:
        workflow = await db_service.update_workflow(workflow_id, request)
        
        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return workflow.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update workflow: {str(e)}")

# Document endpoints
@router.post("/documents", response_model=Dict[str, Any])
async def create_document(
    request: DocumentCreateRequest,
    current_user: User = Depends(require_permission("documents:create"))
):
    """Create a new document."""
    
    db_service = get_db_service()
    
    try:
        document_data = request.dict()
        document = await db_service.create_document(document_data)
        return document.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create document: {str(e)}")

@router.get("/documents", response_model=List[Dict[str, Any]])
async def get_documents(
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    business_area: Optional[str] = Query(None, description="Filter by business area"),
    current_user: User = Depends(require_permission("documents:view"))
):
    """Get documents."""
    
    db_service = get_db_service()
    
    try:
        documents = await db_service.get_documents(document_type, business_area)
        return [document.to_dict() for document in documents]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

# Metric endpoints
@router.post("/metrics", response_model=Dict[str, str])
async def record_metric(
    request: MetricRecordRequest,
    current_user: User = Depends(require_permission("metrics:create"))
):
    """Record a metric."""
    
    db_service = get_db_service()
    
    try:
        metric_data = request.dict()
        metric_data["timestamp"] = datetime.utcnow()
        metric = await db_service.record_metric(metric_data)
        
        return {"message": "Metric recorded successfully", "metric_id": metric.id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record metric: {str(e)}")

@router.get("/metrics", response_model=List[Dict[str, Any]])
async def get_metrics(
    name: Optional[str] = Query(None, description="Filter by metric name"),
    start_time: Optional[datetime] = Query(None, description="Start time filter"),
    end_time: Optional[datetime] = Query(None, description="End time filter"),
    current_user: User = Depends(require_permission("metrics:view"))
):
    """Get metrics."""
    
    db_service = get_db_service()
    
    try:
        metrics = await db_service.get_metrics(name, start_time, end_time)
        return [metric.to_dict() for metric in metrics]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

# Analytics endpoints
@router.get("/analytics/workflows", response_model=Dict[str, Any])
async def get_workflow_analytics(
    days: int = Query(30, description="Number of days to analyze"),
    current_user: User = Depends(require_permission("analytics:view"))
):
    """Get workflow analytics."""
    
    db_service = get_db_service()
    
    try:
        analytics = await db_service.get_workflow_analytics(days)
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get workflow analytics: {str(e)}")

@router.get("/analytics/agents", response_model=Dict[str, Any])
async def get_agent_analytics(
    days: int = Query(30, description="Number of days to analyze"),
    current_user: User = Depends(require_permission("analytics:view"))
):
    """Get agent analytics."""
    
    db_service = get_db_service()
    
    try:
        analytics = await db_service.get_agent_analytics(days)
        return analytics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent analytics: {str(e)}")

# Search endpoints
@router.post("/search/workflows", response_model=List[Dict[str, Any]])
async def search_workflows(
    request: SearchRequest,
    business_area: Optional[str] = Query(None, description="Filter by business area"),
    current_user: User = Depends(require_permission("workflows:view"))
):
    """Search workflows."""
    
    db_service = get_db_service()
    
    try:
        workflows = await db_service.search_workflows(request.query, business_area)
        # Limit results
        limited_workflows = workflows[:request.limit] if request.limit else workflows
        return [workflow.to_dict() for workflow in limited_workflows]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search workflows: {str(e)}")

@router.post("/search/documents", response_model=List[Dict[str, Any]])
async def search_documents(
    request: SearchRequest,
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    current_user: User = Depends(require_permission("documents:view"))
):
    """Search documents."""
    
    db_service = get_db_service()
    
    try:
        documents = await db_service.search_documents(request.query, document_type)
        # Limit results
        limited_documents = documents[:request.limit] if request.limit else documents
        return [document.to_dict() for document in limited_documents]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search documents: {str(e)}")

# Maintenance endpoints
@router.post("/maintenance/cleanup", response_model=Dict[str, str])
async def cleanup_old_data(
    days: int = Query(90, description="Number of days to keep data"),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(require_permission("database:manage"))
):
    """Clean up old data."""
    
    db_service = get_db_service()
    
    try:
        if background_tasks:
            background_tasks.add_task(db_service.cleanup_old_data, days)
            return {"message": f"Cleanup task scheduled for data older than {days} days"}
        else:
            await db_service.cleanup_old_data(days)
            return {"message": f"Cleaned up data older than {days} days"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup old data: {str(e)}")

@router.get("/stats", response_model=Dict[str, Any])
async def get_database_statistics(
    current_user: User = Depends(require_permission("database:view"))
):
    """Get database statistics."""
    
    db_service = get_db_service()
    
    try:
        # Get workflow analytics
        workflow_analytics = await db_service.get_workflow_analytics(30)
        
        # Get agent analytics
        agent_analytics = await db_service.get_agent_analytics(30)
        
        # Get cache stats
        cache_stats = await db_service.get_cache_stats()
        
        return {
            "workflows": workflow_analytics,
            "agents": agent_analytics,
            "cache": cache_stats,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database statistics: {str(e)}")




























