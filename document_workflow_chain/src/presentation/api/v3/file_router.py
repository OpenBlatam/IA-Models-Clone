"""
File Router
===========

FastAPI router for file operations and storage management.
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...shared.services.file_service import (
    FileType,
    StorageBackend,
    CompressionType,
    upload_file,
    download_file,
    delete_file,
    list_files,
    get_file_metadata,
    update_file_metadata,
    generate_thumbnail,
    get_storage_stats
)
from ...shared.middleware.auth import get_current_user_optional
from ...shared.middleware.rate_limiter import rate_limit
from ...shared.middleware.metrics_middleware import record_file_metrics
from ...shared.utils.decorators import log_execution, measure_performance


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/files", tags=["File Operations"])


# Request/Response models
class FileUploadResponse(BaseModel):
    """File upload response"""
    filename: str
    original_filename: str
    file_type: str
    mime_type: str
    size: int
    checksum: str
    created_at: str
    tags: List[str]
    metadata: Dict[str, Any]


class FileListResponse(BaseModel):
    """File list response"""
    files: List[FileUploadResponse]
    total: int
    limit: int
    offset: int


class FileMetadataUpdate(BaseModel):
    """File metadata update request"""
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None


class StorageStatsResponse(BaseModel):
    """Storage statistics response"""
    total_files: int
    total_size: int
    total_size_mb: float
    backend: str
    timestamp: str


# File operations endpoints
@router.post("/upload", response_model=FileUploadResponse)
@rate_limit(requests=20, window=60)  # 20 requests per minute
@record_file_metrics
@log_execution
@measure_performance
async def upload_file_endpoint(
    file: UploadFile = File(...),
    tags: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> FileUploadResponse:
    """Upload file to storage"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Parse tags and metadata
        file_tags = tags.split(",") if tags else []
        file_metadata = {}
        if metadata:
            import json
            try:
                file_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                file_metadata = {"description": metadata}
        
        # Add user metadata
        if current_user:
            file_metadata["uploaded_by"] = current_user.get("id")
            file_metadata["uploaded_by_email"] = current_user.get("email")
        
        # Upload file
        file_metadata_obj = await upload_file(
            file_content=file_content,
            filename=file.filename,
            metadata=file_metadata,
            tags=file_tags
        )
        
        # Return response
        return FileUploadResponse(
            filename=file_metadata_obj.filename,
            original_filename=file_metadata_obj.original_filename,
            file_type=file_metadata_obj.file_type.value,
            mime_type=file_metadata_obj.mime_type,
            size=file_metadata_obj.size,
            checksum=file_metadata_obj.checksum,
            created_at=file_metadata_obj.created_at.isoformat(),
            tags=file_metadata_obj.tags,
            metadata=file_metadata_obj.metadata
        )
    
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@router.get("/download/{filename}")
@rate_limit(requests=50, window=60)  # 50 requests per minute
@record_file_metrics
@log_execution
@measure_performance
async def download_file_endpoint(
    filename: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Download file from storage"""
    try:
        # Download file
        file_content = await download_file(filename)
        
        # Get file metadata
        file_metadata = await get_file_metadata(filename)
        if not file_metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Return file as streaming response
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type=file_metadata.mime_type,
            headers={
                "Content-Disposition": f"attachment; filename={file_metadata.original_filename}",
                "Content-Length": str(len(file_content))
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download failed: {e}")
        raise HTTPException(status_code=500, detail=f"File download failed: {str(e)}")


@router.delete("/{filename}")
@rate_limit(requests=10, window=60)  # 10 requests per minute
@record_file_metrics
@log_execution
@measure_performance
async def delete_file_endpoint(
    filename: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, str]:
    """Delete file from storage"""
    try:
        # Check if user has permission to delete
        if current_user:
            file_metadata = await get_file_metadata(filename)
            if file_metadata and file_metadata.metadata.get("uploaded_by") != current_user.get("id"):
                raise HTTPException(status_code=403, detail="Permission denied")
        
        # Delete file
        success = await delete_file(filename)
        
        if success:
            return {"message": f"File {filename} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete file")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"File deletion failed: {str(e)}")


@router.get("/list", response_model=FileListResponse)
@rate_limit(requests=30, window=60)  # 30 requests per minute
@record_file_metrics
@log_execution
@measure_performance
async def list_files_endpoint(
    prefix: Optional[str] = Query(None, description="Filter by filename prefix"),
    tags: Optional[str] = Query(None, description="Filter by tags (comma-separated)"),
    file_type: Optional[FileType] = Query(None, description="Filter by file type"),
    limit: int = Query(100, description="Maximum number of files to return", ge=1, le=1000),
    offset: int = Query(0, description="Number of files to skip", ge=0),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> FileListResponse:
    """List files with filtering"""
    try:
        # Parse tags
        file_tags = tags.split(",") if tags else None
        
        # List files
        files = await list_files(
            prefix=prefix,
            tags=file_tags,
            file_type=file_type,
            limit=limit,
            offset=offset
        )
        
        # Convert to response format
        file_responses = []
        for file_metadata in files:
            file_responses.append(FileUploadResponse(
                filename=file_metadata.filename,
                original_filename=file_metadata.original_filename,
                file_type=file_metadata.file_type.value,
                mime_type=file_metadata.mime_type,
                size=file_metadata.size,
                checksum=file_metadata.checksum,
                created_at=file_metadata.created_at.isoformat(),
                tags=file_metadata.tags,
                metadata=file_metadata.metadata
            ))
        
        return FileListResponse(
            files=file_responses,
            total=len(file_responses),
            limit=limit,
            offset=offset
        )
    
    except Exception as e:
        logger.error(f"File listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"File listing failed: {str(e)}")


@router.get("/{filename}/metadata")
@rate_limit(requests=50, window=60)  # 50 requests per minute
@record_file_metrics
@log_execution
@measure_performance
async def get_file_metadata_endpoint(
    filename: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> FileUploadResponse:
    """Get file metadata"""
    try:
        # Get file metadata
        file_metadata = await get_file_metadata(filename)
        if not file_metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Return response
        return FileUploadResponse(
            filename=file_metadata.filename,
            original_filename=file_metadata.original_filename,
            file_type=file_metadata.file_type.value,
            mime_type=file_metadata.mime_type,
            size=file_metadata.size,
            checksum=file_metadata.checksum,
            created_at=file_metadata.created_at.isoformat(),
            tags=file_metadata.tags,
            metadata=file_metadata.metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get file metadata: {str(e)}")


@router.put("/{filename}/metadata")
@rate_limit(requests=20, window=60)  # 20 requests per minute
@record_file_metrics
@log_execution
@measure_performance
async def update_file_metadata_endpoint(
    filename: str,
    metadata_update: FileMetadataUpdate,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> Dict[str, str]:
    """Update file metadata"""
    try:
        # Check if user has permission to update
        if current_user:
            file_metadata = await get_file_metadata(filename)
            if file_metadata and file_metadata.metadata.get("uploaded_by") != current_user.get("id"):
                raise HTTPException(status_code=403, detail="Permission denied")
        
        # Update metadata
        success = await update_file_metadata(
            filename=filename,
            metadata=metadata_update.metadata or {},
            tags=metadata_update.tags
        )
        
        if success:
            return {"message": f"File metadata updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update file metadata")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update file metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update file metadata: {str(e)}")


@router.get("/{filename}/thumbnail")
@rate_limit(requests=100, window=60)  # 100 requests per minute
@record_file_metrics
@log_execution
@measure_performance
async def generate_thumbnail_endpoint(
    filename: str,
    size: int = Query(200, description="Thumbnail size", ge=50, le=500),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """Generate thumbnail for image files"""
    try:
        # Get file metadata
        file_metadata = await get_file_metadata(filename)
        if not file_metadata:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if file is an image
        if file_metadata.file_type != FileType.IMAGE:
            raise HTTPException(status_code=400, detail="File is not an image")
        
        # Generate thumbnail
        thumbnail_content = await generate_thumbnail(filename, (size, size))
        
        # Return thumbnail as streaming response
        return StreamingResponse(
            io.BytesIO(thumbnail_content),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"inline; filename=thumbnail_{filename}",
                "Content-Length": str(len(thumbnail_content))
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Thumbnail generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Thumbnail generation failed: {str(e)}")


@router.get("/stats", response_model=StorageStatsResponse)
@rate_limit(requests=10, window=60)  # 10 requests per minute
@record_file_metrics
@log_execution
@measure_performance
async def get_storage_stats_endpoint(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
) -> StorageStatsResponse:
    """Get storage statistics"""
    try:
        # Get storage stats
        stats = await get_storage_stats()
        
        return StorageStatsResponse(
            total_files=stats.get("total_files", 0),
            total_size=stats.get("total_size", 0),
            total_size_mb=stats.get("total_size_mb", 0.0),
            backend=stats.get("backend", "unknown"),
            timestamp=stats.get("timestamp", "2024-01-01T00:00:00Z")
        )
    
    except Exception as e:
        logger.error(f"Failed to get storage stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get storage stats: {str(e)}")


# Health check endpoint
@router.get("/health")
@log_execution
async def file_service_health_check() -> Dict[str, Any]:
    """File service health check"""
    try:
        # Check if file service is running
        stats = await get_storage_stats()
        
        return {
            "status": "healthy",
            "backend": stats.get("backend", "unknown"),
            "total_files": stats.get("total_files", 0),
            "total_size_mb": stats.get("total_size_mb", 0.0),
            "timestamp": "2024-01-01T00:00:00Z"  # In real implementation, use actual timestamp
        }
    
    except Exception as e:
        logger.error(f"File service health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"  # In real implementation, use actual timestamp
        }


