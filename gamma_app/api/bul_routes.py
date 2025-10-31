"""
BUL Integration API Routes for Gamma App
========================================

API endpoints for integrating BUL (Business Universal Language) system
with Gamma App for advanced document generation capabilities.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

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

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/bul", tags=["BUL Integration"])

# Dependency to get BUL service
def get_bul_service() -> BULIntegrationService:
    """Get BUL integration service instance."""
    # In a real implementation, this would come from dependency injection
    return BULIntegrationService()

@router.get("/")
async def bul_root():
    """BUL integration root endpoint."""
    return {
        "message": "BUL Integration Service for Gamma App",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
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
async def bul_health(bul_service: BULIntegrationService = Depends(get_bul_service)):
    """Get BUL system health status."""
    try:
        health_status = await bul_service.get_system_health()
        return {
            "bul_system": health_status,
            "integration_service": "healthy",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error checking BUL health: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@router.get("/business-areas")
async def get_business_areas(bul_service: BULIntegrationService = Depends(get_bul_service)):
    """Get available business areas from BUL system."""
    try:
        business_areas = await bul_service.get_business_areas()
        return {
            "business_areas": business_areas,
            "total": len(business_areas),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting business areas: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get business areas: {e}")

@router.get("/document-types")
async def get_document_types(bul_service: BULIntegrationService = Depends(get_bul_service)):
    """Get available document types from BUL system."""
    try:
        document_types = await bul_service.get_document_types()
        return {
            "document_types": document_types,
            "total": len(document_types),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting document types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document types: {e}")

@router.post("/documents/generate", response_model=BULDocumentResponse)
async def generate_document(
    request: BULDocumentRequest,
    background_tasks: BackgroundTasks,
    bul_service: BULIntegrationService = Depends(get_bul_service)
):
    """Generate a document using BUL system."""
    try:
        # Validate business area
        if request.business_area and request.business_area not in BusinessArea:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid business area. Available: {[area.value for area in BusinessArea]}"
            )
        
        # Validate document type
        if request.document_type and request.document_type not in DocumentType:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid document type. Available: {[doc_type.value for doc_type in DocumentType]}"
            )
        
        # Validate priority
        if not (1 <= request.priority <= 5):
            raise HTTPException(
                status_code=400,
                detail="Priority must be between 1 and 5"
            )
        
        # Generate document
        response = await bul_service.generate_document(request)
        
        logger.info(f"Document generation initiated: {response.task_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate document: {e}")

@router.get("/tasks/{task_id}/status")
async def get_task_status(
    task_id: str,
    bul_service: BULIntegrationService = Depends(get_bul_service)
):
    """Get the status of a BUL task."""
    try:
        task = await bul_service.get_task_status(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {e}")

@router.get("/tasks")
async def list_tasks(
    status: Optional[TaskStatus] = Query(None, description="Filter by task status"),
    business_area: Optional[BusinessArea] = Query(None, description="Filter by business area"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of tasks to return"),
    offset: int = Query(0, ge=0, description="Number of tasks to skip"),
    bul_service: BULIntegrationService = Depends(get_bul_service)
):
    """List BUL tasks with optional filtering."""
    try:
        # Get all tasks (in a real implementation, this would be filtered from database)
        all_tasks = list(bul_service.tasks.values())
        
        # Apply filters
        if status:
            all_tasks = [t for t in all_tasks if t.status == status]
        
        if business_area:
            all_tasks = [t for t in all_tasks if t.request.business_area == business_area]
        
        # Sort by creation date (newest first)
        all_tasks.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
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
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {e}")

@router.delete("/tasks/{task_id}")
async def delete_task(
    task_id: str,
    bul_service: BULIntegrationService = Depends(get_bul_service)
):
    """Delete a BUL task."""
    try:
        success = await bul_service.delete_task(task_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or could not be deleted")
        
        return {"message": "Task deleted successfully", "task_id": task_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete task: {e}")

@router.get("/documents")
async def list_documents(
    business_area: Optional[BusinessArea] = Query(None, description="Filter by business area"),
    document_type: Optional[DocumentType] = Query(None, description="Filter by document type"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    bul_service: BULIntegrationService = Depends(get_bul_service)
):
    """List generated documents with optional filtering."""
    try:
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
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {e}")

@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    bul_service: BULIntegrationService = Depends(get_bul_service)
):
    """Get a specific generated document."""
    try:
        document = await bul_service.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {e}")

@router.get("/documents/search")
async def search_documents(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=50, description="Maximum number of results"),
    bul_service: BULIntegrationService = Depends(get_bul_service)
):
    """Search documents by content or title."""
    try:
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
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to search documents: {e}")

@router.get("/statistics")
async def get_statistics(bul_service: BULIntegrationService = Depends(get_bul_service)):
    """Get BUL integration statistics."""
    try:
        stats = await bul_service.get_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {e}")

@router.post("/cleanup")
async def cleanup_old_tasks(
    max_age_hours: int = Query(24, ge=1, le=168, description="Maximum age of tasks to keep (in hours)"),
    bul_service: BULIntegrationService = Depends(get_bul_service)
):
    """Clean up old completed tasks."""
    try:
        cleaned_count = await bul_service.cleanup_old_tasks(max_age_hours)
        
        return {
            "message": f"Cleaned up {cleaned_count} old tasks",
            "max_age_hours": max_age_hours,
            "cleaned_count": cleaned_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up tasks: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup tasks: {e}")





















