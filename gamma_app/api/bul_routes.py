"""
BUL Integration API Routes for Gamma App
API endpoints for integrating BUL (Business Universal Language) system
with Gamma App for advanced document generation capabilities.
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, status
from ..services.bul_integration_service import (
    BULIntegrationService,
    BULDocumentRequest,
    BULDocumentResponse,
    BULTask,
    BULDocument,
    BusinessArea,
    DocumentType,
    TaskStatus
)
from .error_handlers import handle_route_errors
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bul", tags=["BUL Integration"])
def get_bul_service() -> BULIntegrationService:
    """Get BUL integration service instance"""
    return BULIntegrationService()
@router.get("/")
async def bul_root() -> Dict[str, Any]:
    """BUL integration root endpoint"""
    return {
        "message": "BUL Integration Service for Gamma App",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoints": [
            "/bul/documents/generate",
            "/bul/tasks/{task_id}/status",
            "/bul/documents",
            "/bul/documents/{document_id}",
            "/bul/business-areas",
            "/bul/document-types",
            "/bul/health",
            "/bul/statistics"
        ]
    }
@router.get("/health")
@handle_route_errors
async def bul_health(
    bul_service: BULIntegrationService = Depends(get_bul_service)
) -> Dict[str, Any]:
    """Get BUL system health status"""
    health_status = await bul_service.get_system_health()
    return {
        "bul_system": health_status,
        "integration_service": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
@router.get("/business-areas")
@handle_route_errors
async def get_business_areas(
    bul_service: BULIntegrationService = Depends(get_bul_service)
) -> Dict[str, Any]:
    """Get available business areas from BUL system"""
    business_areas = await bul_service.get_business_areas()
    return {
        "business_areas": business_areas,
        "total": len(business_areas),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
@router.get("/document-types")
@handle_route_errors
async def get_document_types(
    bul_service: BULIntegrationService = Depends(get_bul_service)
) -> Dict[str, Any]:
    """Get available document types from BUL system"""
    document_types = await bul_service.get_document_types()
    return {
        "document_types": document_types,
        "total": len(document_types),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
@router.post("/documents/generate", response_model=BULDocumentResponse)
@handle_route_errors
async def generate_document(
    request: BULDocumentRequest,
    background_tasks: BackgroundTasks,
    bul_service: BULIntegrationService = Depends(get_bul_service)
) -> BULDocumentResponse:
    """Generate a document using BUL system"""
    if request.business_area and request.business_area not in BusinessArea:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid business area. Available: {[area.value for area in BusinessArea]}"
        )
    if request.document_type and request.document_type not in DocumentType:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid document type. Available: {[doc_type.value for doc_type in DocumentType]}"
        )
    if not (1 <= request.priority <= 5):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Priority must be between 1 and 5"
        )
    response = await bul_service.generate_document(request)
    logger.info(
        "Document generation initiated",
        extra={"task_id": response.task_id}
    )
    return response
@router.get("/tasks/{task_id}/status")
@handle_route_errors
async def get_task_status(
    task_id: str = Path(..., description="Task ID"),
    bul_service: BULIntegrationService = Depends(get_bul_service)
) -> Dict[str, Any]:
    """Get the status of a BUL task"""
    task = await bul_service.get_task_status(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    return {
        "task_id": task.task_id,
        "status": task.status.value,
        "progress": task.progress,
        "created_at": task.created_at.isoformat(),
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "business_area": task.request.business_area.value if task.request.business_area else None,
        "document_type": task.request.document_type.value if task.request.document_type else None,
        "query": task.request.query,
        "result": task.result,
        "error": task.error
    }
@router.get("/tasks")
@handle_route_errors
async def list_tasks(
    status: Optional[TaskStatus] = Query(None, description="Filter by task status"),
    business_area: Optional[BusinessArea] = Query(None, description="Filter by business area"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of tasks to return"),
    offset: int = Query(0, ge=0, description="Number of tasks to skip"),
    bul_service: BULIntegrationService = Depends(get_bul_service)
) -> Dict[str, Any]:
    """List BUL tasks with optional filtering"""
    all_tasks = list(bul_service.tasks.values())
    if status:
        all_tasks = [t for t in all_tasks if t.status == status]
    if business_area:
        all_tasks = [t for t in all_tasks if t.request.business_area == business_area]
    all_tasks.sort(key=lambda x: x.created_at, reverse=True)
    paginated_tasks = all_tasks[offset:offset + limit]
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "status": task.status.value,
                "progress": task.progress,
                "created_at": task.created_at.isoformat(),
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "business_area": task.request.business_area.value if task.request.business_area else None,
                "document_type": task.request.document_type.value if task.request.document_type else None,
                "query": task.request.query,
                "priority": task.request.priority
            }
            for task in paginated_tasks
        ],
        "total": len(all_tasks),
        "limit": limit,
        "offset": offset,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
@router.delete("/tasks/{task_id}")
@handle_route_errors
async def delete_task(
    task_id: str = Path(..., description="Task ID"),
    bul_service: BULIntegrationService = Depends(get_bul_service)
) -> Dict[str, str]:
    """Delete a BUL task"""
    success = await bul_service.delete_task(task_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found or could not be deleted"
        )
    return {"message": "Task deleted successfully", "task_id": task_id}
@router.get("/documents")
@handle_route_errors
async def list_documents(
    business_area: Optional[BusinessArea] = Query(None, description="Filter by business area"),
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    bul_service: BULIntegrationService = Depends(get_bul_service)
) -> Dict[str, Any]:
    """List generated documents with optional filtering"""
    documents = await bul_service.list_documents(
        business_area=business_area,
        document_type=document_type,
        limit=limit,
        offset=offset
    )
    return {
        "documents": [
            {
                "document_id": doc.document_id,
                "title": doc.title,
                "format": doc.format,
                "word_count": doc.word_count,
                "business_area": doc.business_area.value,
                "document_type": doc.document_type.value,
                "query": doc.query,
                "generated_at": doc.generated_at.isoformat(),
                "metadata": doc.metadata
            }
            for doc in documents
        ],
        "total": len(documents),
        "limit": limit,
        "offset": offset,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
@router.get("/documents/{document_id}")
@handle_route_errors
async def get_document(
    document_id: str = Path(..., description="Document ID"),
    bul_service: BULIntegrationService = Depends(get_bul_service)
) -> Dict[str, Any]:
    """Get a specific generated document"""
    document = await bul_service.get_document(document_id)
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return {
        "document_id": document.document_id,
        "title": document.title,
        "content": document.content,
        "format": document.format,
        "word_count": document.word_count,
        "business_area": document.business_area.value,
        "document_type": document.document_type.value,
        "query": document.query,
        "generated_at": document.generated_at.isoformat(),
        "metadata": document.metadata
    }
@router.get("/documents/search")
@handle_route_errors
async def search_documents(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=50, description="Maximum number of results"),
    bul_service: BULIntegrationService = Depends(get_bul_service)
) -> Dict[str, Any]:
    """Search documents by content or title"""
    documents = await bul_service.search_documents(query=q, limit=limit)
    return {
        "query": q,
        "documents": [
            {
                "document_id": doc.document_id,
                "title": doc.title,
                "format": doc.format,
                "word_count": doc.word_count,
                "business_area": doc.business_area.value,
                "document_type": doc.document_type.value,
                "query": doc.query,
                "generated_at": doc.generated_at.isoformat(),
                "metadata": doc.metadata
            }
            for doc in documents
        ],
        "total": len(documents),
        "limit": limit,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
@router.get("/statistics")
@handle_route_errors
async def get_statistics(
    bul_service: BULIntegrationService = Depends(get_bul_service)
) -> Dict[str, Any]:
    """Get BUL integration statistics"""
    stats = await bul_service.get_statistics()
    return stats
@router.post("/cleanup")
@handle_route_errors
async def cleanup_old_tasks(
    max_age_hours: int = Query(24, ge=1, le=168, description="Maximum age of tasks to keep (in hours)"),
    bul_service: BULIntegrationService = Depends(get_bul_service)
) -> Dict[str, Any]:
    """Clean up old completed tasks"""
    cleaned_count = await bul_service.cleanup_old_tasks(max_age_hours)
    return {
        "message": f"Cleaned up {cleaned_count} old tasks",
        "max_age_hours": max_age_hours,
        "cleaned_count": cleaned_count,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
