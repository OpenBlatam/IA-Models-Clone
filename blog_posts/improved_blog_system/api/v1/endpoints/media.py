"""
Advanced Media Management API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, UploadFile, File
from pydantic import BaseModel, Field
from datetime import datetime

from ....services.advanced_media_service import AdvancedMediaService, MediaType, MediaStatus
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class MediaUploadRequest(BaseModel):
    """Request model for media upload."""
    filename: str = Field(..., description="Original filename")
    collection_id: Optional[str] = Field(default=None, description="Collection ID")
    tags: Optional[List[str]] = Field(default=None, description="Media tags")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class MediaListRequest(BaseModel):
    """Request model for listing media."""
    media_type: Optional[str] = Field(default=None, description="Filter by media type")
    status: Optional[str] = Field(default=None, description="Filter by status")
    user_id: Optional[str] = Field(default=None, description="Filter by user")
    collection_id: Optional[str] = Field(default=None, description="Filter by collection")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Page size")
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order")


async def get_media_service(session: DatabaseSessionDep) -> AdvancedMediaService:
    """Get media service instance."""
    return AdvancedMediaService(session)


@router.post("/upload", response_model=Dict[str, Any])
async def upload_media(
    file: UploadFile = File(...),
    filename: Optional[str] = Query(default=None, description="Custom filename"),
    collection_id: Optional[str] = Query(default=None, description="Collection ID"),
    tags: Optional[str] = Query(default=None, description="Comma-separated tags"),
    metadata: Optional[str] = Query(default=None, description="JSON metadata"),
    media_service: AdvancedMediaService = Depends(get_media_service),
    current_user: CurrentUserDep = Depends()
):
    """Upload media file with advanced processing."""
    try:
        # Read file data
        file_data = await file.read()
        
        # Use custom filename or original
        upload_filename = filename or file.filename
        
        # Parse tags
        tag_list = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
        
        # Parse metadata
        metadata_dict = {}
        if metadata:
            try:
                import json
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise ValidationError("Invalid JSON metadata")
        
        result = await media_service.upload_media(
            file_data=file_data,
            filename=upload_filename,
            user_id=str(current_user.id) if current_user else None,
            collection_id=collection_id,
            tags=tag_list,
            metadata=metadata_dict
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Media uploaded successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload media"
        )


@router.get("/{file_id}", response_model=Dict[str, Any])
async def get_media(
    file_id: str,
    include_metadata: bool = Query(default=True, description="Include metadata"),
    media_service: AdvancedMediaService = Depends(get_media_service),
    current_user: CurrentUserDep = Depends()
):
    """Get media file by ID."""
    try:
        result = await media_service.get_media(
            file_id=file_id,
            include_metadata=include_metadata
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Media retrieved successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get media"
        )


@router.post("/list", response_model=Dict[str, Any])
async def list_media(
    request: MediaListRequest = Depends(),
    media_service: AdvancedMediaService = Depends(get_media_service),
    current_user: CurrentUserDep = Depends()
):
    """List media files with filtering and pagination."""
    try:
        # Convert media type to enum if provided
        media_type_enum = None
        if request.media_type:
            try:
                media_type_enum = MediaType(request.media_type.lower())
            except ValueError:
                raise ValidationError(f"Invalid media type: {request.media_type}")
        
        # Convert status to enum if provided
        status_enum = None
        if request.status:
            try:
                status_enum = MediaStatus(request.status.lower())
            except ValueError:
                raise ValidationError(f"Invalid status: {request.status}")
        
        result = await media_service.list_media(
            media_type=media_type_enum,
            status=status_enum,
            user_id=request.user_id,
            collection_id=request.collection_id,
            page=request.page,
            page_size=request.page_size,
            sort_by=request.sort_by,
            sort_order=request.sort_order
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Media list retrieved successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list media"
        )


@router.get("/", response_model=Dict[str, Any])
async def list_media_get(
    media_type: Optional[str] = Query(default=None, description="Filter by media type"),
    status: Optional[str] = Query(default=None, description="Filter by status"),
    user_id: Optional[str] = Query(default=None, description="Filter by user"),
    collection_id: Optional[str] = Query(default=None, description="Filter by collection"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Page size"),
    sort_by: str = Query(default="created_at", description="Sort field"),
    sort_order: str = Query(default="desc", description="Sort order"),
    media_service: AdvancedMediaService = Depends(get_media_service),
    current_user: CurrentUserDep = Depends()
):
    """List media files via GET request."""
    try:
        # Convert media type to enum if provided
        media_type_enum = None
        if media_type:
            try:
                media_type_enum = MediaType(media_type.lower())
            except ValueError:
                raise ValidationError(f"Invalid media type: {media_type}")
        
        # Convert status to enum if provided
        status_enum = None
        if status:
            try:
                status_enum = MediaStatus(status.lower())
            except ValueError:
                raise ValidationError(f"Invalid status: {status}")
        
        result = await media_service.list_media(
            media_type=media_type_enum,
            status=status_enum,
            user_id=user_id,
            collection_id=collection_id,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Media list retrieved successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list media"
        )


@router.delete("/{file_id}", response_model=Dict[str, Any])
async def delete_media(
    file_id: str,
    media_service: AdvancedMediaService = Depends(get_media_service),
    current_user: CurrentUserDep = Depends()
):
    """Delete media file (soft delete)."""
    try:
        result = await media_service.delete_media(file_id=file_id)
        
        return {
            "success": True,
            "data": result,
            "message": "Media deleted successfully"
        }
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete media"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_media_stats(
    media_service: AdvancedMediaService = Depends(get_media_service),
    current_user: CurrentUserDep = Depends()
):
    """Get media statistics."""
    try:
        result = await media_service.get_media_stats()
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Media statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get media statistics"
        )


@router.get("/types", response_model=Dict[str, Any])
async def get_media_types():
    """Get available media types and their descriptions."""
    media_types = {
        "image": {
            "name": "Image",
            "description": "Image files (JPEG, PNG, GIF, WebP, SVG)",
            "supported_formats": ["JPEG", "PNG", "GIF", "WebP", "SVG"],
            "processing_features": ["Thumbnail generation", "Multiple sizes", "OCR", "Face detection", "Object detection", "Image captioning"],
            "max_size": "100MB"
        },
        "video": {
            "name": "Video",
            "description": "Video files (MP4, AVI, MOV, WMV, WebM)",
            "supported_formats": ["MP4", "AVI", "MOV", "WMV", "WebM"],
            "processing_features": ["Thumbnail generation", "Audio extraction", "Transcription", "Video summarization"],
            "max_size": "100MB"
        },
        "audio": {
            "name": "Audio",
            "description": "Audio files (MP3, WAV, OGG, M4A, FLAC)",
            "supported_formats": ["MP3", "WAV", "OGG", "M4A", "FLAC"],
            "processing_features": ["Transcription", "Audio feature extraction", "Audio summarization"],
            "max_size": "100MB"
        },
        "document": {
            "name": "Document",
            "description": "Document files (PDF, DOC, DOCX)",
            "supported_formats": ["PDF", "DOC", "DOCX"],
            "processing_features": ["Text extraction", "Metadata extraction"],
            "max_size": "100MB"
        },
        "archive": {
            "name": "Archive",
            "description": "Archive files (ZIP, RAR, 7Z)",
            "supported_formats": ["ZIP", "RAR", "7Z"],
            "processing_features": ["Archive extraction", "File listing"],
            "max_size": "100MB"
        },
        "other": {
            "name": "Other",
            "description": "Other file types",
            "supported_formats": ["Any"],
            "processing_features": ["Basic metadata extraction"],
            "max_size": "100MB"
        }
    }
    
    return {
        "success": True,
        "data": {
            "media_types": media_types,
            "total_types": len(media_types)
        },
        "message": "Media types retrieved successfully"
    }


@router.get("/statuses", response_model=Dict[str, Any])
async def get_media_statuses():
    """Get available media statuses and their descriptions."""
    media_statuses = {
        "uploading": {
            "name": "Uploading",
            "description": "Media file is being uploaded",
            "visibility": "Private",
            "processing": "In progress"
        },
        "processing": {
            "name": "Processing",
            "description": "Media file is being processed",
            "visibility": "Private",
            "processing": "In progress"
        },
        "ready": {
            "name": "Ready",
            "description": "Media file is ready for use",
            "visibility": "Public",
            "processing": "Completed"
        },
        "error": {
            "name": "Error",
            "description": "Error occurred during processing",
            "visibility": "Private",
            "processing": "Failed"
        },
        "deleted": {
            "name": "Deleted",
            "description": "Media file has been deleted",
            "visibility": "Hidden",
            "processing": "Completed"
        }
    }
    
    return {
        "success": True,
        "data": {
            "media_statuses": media_statuses,
            "total_statuses": len(media_statuses)
        },
        "message": "Media statuses retrieved successfully"
    }


@router.get("/processing/types", response_model=Dict[str, Any])
async def get_processing_types():
    """Get available processing types and their descriptions."""
    processing_types = {
        "thumbnail": {
            "name": "Thumbnail Generation",
            "description": "Generate thumbnail images",
            "applies_to": ["image", "video"],
            "output": "Image file"
        },
        "resize": {
            "name": "Image Resizing",
            "description": "Resize images to different dimensions",
            "applies_to": ["image"],
            "output": "Image files"
        },
        "compress": {
            "name": "Compression",
            "description": "Compress media files for smaller size",
            "applies_to": ["image", "video", "audio"],
            "output": "Compressed file"
        },
        "watermark": {
            "name": "Watermarking",
            "description": "Add watermarks to images",
            "applies_to": ["image"],
            "output": "Watermarked image"
        },
        "filter": {
            "name": "Filter Application",
            "description": "Apply filters to images",
            "applies_to": ["image"],
            "output": "Filtered image"
        },
        "ocr": {
            "name": "OCR (Optical Character Recognition)",
            "description": "Extract text from images",
            "applies_to": ["image"],
            "output": "Text file"
        },
        "transcription": {
            "name": "Transcription",
            "description": "Convert audio to text",
            "applies_to": ["audio", "video"],
            "output": "Text file"
        },
        "face_detection": {
            "name": "Face Detection",
            "description": "Detect faces in images",
            "applies_to": ["image"],
            "output": "Face data"
        },
        "object_detection": {
            "name": "Object Detection",
            "description": "Detect objects in images",
            "applies_to": ["image"],
            "output": "Object data"
        },
        "speech_to_text": {
            "name": "Speech to Text",
            "description": "Convert speech to text",
            "applies_to": ["audio"],
            "output": "Text file"
        },
        "text_to_speech": {
            "name": "Text to Speech",
            "description": "Convert text to speech",
            "applies_to": ["text"],
            "output": "Audio file"
        },
        "translation": {
            "name": "Translation",
            "description": "Translate text content",
            "applies_to": ["text"],
            "output": "Translated text"
        },
        "enhancement": {
            "name": "Enhancement",
            "description": "Enhance media quality",
            "applies_to": ["image", "video", "audio"],
            "output": "Enhanced media"
        }
    }
    
    return {
        "success": True,
        "data": {
            "processing_types": processing_types,
            "total_types": len(processing_types)
        },
        "message": "Processing types retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_media_health(
    media_service: AdvancedMediaService = Depends(get_media_service),
    current_user: CurrentUserDep = Depends()
):
    """Get media management system health status."""
    try:
        # Get media stats
        stats = await media_service.get_media_stats()
        
        # Calculate health metrics
        total_media = stats["data"].get("total_media", 0)
        media_by_status = stats["data"].get("media_by_status", {})
        ready_media = media_by_status.get("ready", 0)
        error_media = media_by_status.get("error", 0)
        
        # Calculate health score
        health_score = 100
        
        # Check media distribution
        if total_media > 0:
            ready_ratio = ready_media / total_media
            error_ratio = error_media / total_media
            
            if ready_ratio < 0.8:
                health_score -= 20
            if error_ratio > 0.1:
                health_score -= 30
        
        # Check AI models
        ai_models_loaded = stats["data"].get("ai_models_loaded", 0)
        if ai_models_loaded < 3:
            health_score -= 15
        
        # Check cache
        cache_size = stats["data"].get("cache_size", 0)
        if cache_size > 1000:
            health_score -= 10
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_media": total_media,
                "ready_media": ready_media,
                "error_media": error_media,
                "ai_models_loaded": ai_models_loaded,
                "cache_size": cache_size,
                "media_by_type": stats["data"].get("media_by_type", {}),
                "average_file_size": stats["data"].get("average_file_size", 0),
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "Media health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get media health status"
        )
























