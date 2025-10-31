"""
Content Management API Endpoints

This module provides API endpoints for content management functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from ...core.config import get_config, SystemConfig
from ...core.exceptions import ValidationError, StorageError

logger = logging.getLogger(__name__)

router = APIRouter()


class ContentRequest(BaseModel):
    """Request model for content operations"""
    content: str = Field(..., description="Content to process", min_length=1)
    content_type: str = Field(default="text", description="Type of content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Content metadata")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")


class ContentResponse(BaseModel):
    """Response model for content operations"""
    content_id: str
    content_hash: str
    content_type: str
    status: str
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    processing_time: float


class ContentSearchRequest(BaseModel):
    """Request model for content search"""
    query: str = Field(..., description="Search query", min_length=1)
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    limit: int = Field(default=10, description="Maximum number of results", ge=1, le=100)
    offset: int = Field(default=0, description="Number of results to skip", ge=0)
    options: Dict[str, Any] = Field(default_factory=dict, description="Search options")


class ContentSearchResponse(BaseModel):
    """Response model for content search"""
    search_id: str
    query: str
    total_results: int
    results: List[Dict[str, Any]]
    processing_time: float


class ContentVersionRequest(BaseModel):
    """Request model for content versioning"""
    content_id: str = Field(..., description="Content ID to version")
    version_notes: str = Field(default="", description="Notes for this version")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Version metadata")


class ContentVersionResponse(BaseModel):
    """Response model for content versioning"""
    version_id: str
    content_id: str
    version_number: int
    version_notes: str
    created_at: str
    metadata: Dict[str, Any]


@router.post("/create", response_model=ContentResponse)
async def create_content(
    request: ContentRequest,
    background_tasks: BackgroundTasks,
    config: SystemConfig = Depends(get_config)
):
    """
    Create new content entry
    
    This endpoint creates a new content entry in the system with
    automatic analysis and metadata extraction.
    """
    try:
        if not config.features.get("content_lifecycle", False):
            raise HTTPException(
                status_code=403,
                detail="Content lifecycle feature is not enabled"
            )
        
        # Import here to avoid circular imports
        from ...engines.content_lifecycle_engine import ContentLifecycleEngine
        
        engine = ContentLifecycleEngine(config)
        await engine.initialize()
        
        # Process content
        import time
        start_time = time.time()
        
        results = await engine.create_content(
            request.content,
            content_type=request.content_type,
            metadata=request.metadata,
            **request.options
        )
        
        processing_time = time.time() - start_time
        
        # Generate content ID and hash
        import hashlib
        content_hash = hashlib.md5(request.content.encode()).hexdigest()
        content_id = f"content_{content_hash}_{int(time.time())}"
        
        response = ContentResponse(
            content_id=content_id,
            content_hash=content_hash,
            content_type=request.content_type,
            status="created",
            results=results,
            metadata={
                "engine_version": "1.0.0",
                "timestamp": time.time(),
                "options": request.options
            },
            processing_time=processing_time
        )
        
        # Clean up
        await engine.shutdown()
        
        return response
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StorageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Content creation failed: {e}")
        raise HTTPException(status_code=500, detail="Content creation failed")


@router.get("/{content_id}")
async def get_content(
    content_id: str,
    config: SystemConfig = Depends(get_config)
):
    """
    Get content by ID
    
    Retrieves content and its associated metadata by ID.
    """
    try:
        if not config.features.get("content_lifecycle", False):
            raise HTTPException(
                status_code=403,
                detail="Content lifecycle feature is not enabled"
            )
        
        # This would typically query a database
        # For now, return a placeholder response
        return {
            "content_id": content_id,
            "status": "not_found",
            "message": "Content retrieval not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Failed to get content: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve content")


@router.put("/{content_id}")
async def update_content(
    content_id: str,
    request: ContentRequest,
    config: SystemConfig = Depends(get_config)
):
    """
    Update existing content
    
    Updates content and creates a new version in the version history.
    """
    try:
        if not config.features.get("content_lifecycle", False):
            raise HTTPException(
                status_code=403,
                detail="Content lifecycle feature is not enabled"
            )
        
        # Import here to avoid circular imports
        from ...engines.content_lifecycle_engine import ContentLifecycleEngine
        
        engine = ContentLifecycleEngine(config)
        await engine.initialize()
        
        # Update content
        import time
        start_time = time.time()
        
        results = await engine.update_content(
            content_id,
            request.content,
            metadata=request.metadata,
            **request.options
        )
        
        processing_time = time.time() - start_time
        
        # Generate new content hash
        import hashlib
        content_hash = hashlib.md5(request.content.encode()).hexdigest()
        
        response = ContentResponse(
            content_id=content_id,
            content_hash=content_hash,
            content_type=request.content_type,
            status="updated",
            results=results,
            metadata={
                "engine_version": "1.0.0",
                "timestamp": time.time(),
                "options": request.options
            },
            processing_time=processing_time
        )
        
        # Clean up
        await engine.shutdown()
        
        return response
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StorageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Content update failed: {e}")
        raise HTTPException(status_code=500, detail="Content update failed")


@router.delete("/{content_id}")
async def delete_content(
    content_id: str,
    config: SystemConfig = Depends(get_config)
):
    """
    Delete content
    
    Soft deletes content and marks it as archived.
    """
    try:
        if not config.features.get("content_lifecycle", False):
            raise HTTPException(
                status_code=403,
                detail="Content lifecycle feature is not enabled"
            )
        
        # Import here to avoid circular imports
        from ...engines.content_lifecycle_engine import ContentLifecycleEngine
        
        engine = ContentLifecycleEngine(config)
        await engine.initialize()
        
        # Delete content
        results = await engine.delete_content(content_id)
        
        # Clean up
        await engine.shutdown()
        
        return {
            "content_id": content_id,
            "status": "deleted",
            "results": results
        }
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StorageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Content deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Content deletion failed")


@router.post("/search", response_model=ContentSearchResponse)
async def search_content(
    request: ContentSearchRequest,
    config: SystemConfig = Depends(get_config)
):
    """
    Search content
    
    Performs full-text search across all content in the system.
    """
    try:
        if not config.features.get("content_lifecycle", False):
            raise HTTPException(
                status_code=403,
                detail="Content lifecycle feature is not enabled"
            )
        
        # Import here to avoid circular imports
        from ...engines.content_lifecycle_engine import ContentLifecycleEngine
        
        engine = ContentLifecycleEngine(config)
        await engine.initialize()
        
        # Perform search
        import time
        start_time = time.time()
        
        results = await engine.search_content(
            request.query,
            filters=request.filters,
            limit=request.limit,
            offset=request.offset,
            **request.options
        )
        
        processing_time = time.time() - start_time
        
        # Generate search ID
        search_id = f"search_{int(time.time())}"
        
        response = ContentSearchResponse(
            search_id=search_id,
            query=request.query,
            total_results=len(results),
            results=results,
            processing_time=processing_time
        )
        
        # Clean up
        await engine.shutdown()
        
        return response
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StorageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Content search failed: {e}")
        raise HTTPException(status_code=500, detail="Content search failed")


@router.post("/{content_id}/versions", response_model=ContentVersionResponse)
async def create_content_version(
    content_id: str,
    request: ContentVersionRequest,
    config: SystemConfig = Depends(get_config)
):
    """
    Create a new version of content
    
    Creates a new version in the content's version history.
    """
    try:
        if not config.features.get("content_lifecycle", False):
            raise HTTPException(
                status_code=403,
                detail="Content lifecycle feature is not enabled"
            )
        
        # Import here to avoid circular imports
        from ...engines.content_lifecycle_engine import ContentLifecycleEngine
        
        engine = ContentLifecycleEngine(config)
        await engine.initialize()
        
        # Create version
        results = await engine.create_version(
            content_id,
            version_notes=request.version_notes,
            metadata=request.metadata
        )
        
        # Clean up
        await engine.shutdown()
        
        return ContentVersionResponse(
            version_id=results.get("version_id", f"version_{int(time.time())}"),
            content_id=content_id,
            version_number=results.get("version_number", 1),
            version_notes=request.version_notes,
            created_at=results.get("created_at", time.time()),
            metadata=request.metadata
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except StorageError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Content versioning failed: {e}")
        raise HTTPException(status_code=500, detail="Content versioning failed")


@router.get("/{content_id}/versions")
async def get_content_versions(
    content_id: str,
    config: SystemConfig = Depends(get_config)
):
    """
    Get content version history
    
    Retrieves all versions of a content item.
    """
    try:
        if not config.features.get("content_lifecycle", False):
            raise HTTPException(
                status_code=403,
                detail="Content lifecycle feature is not enabled"
            )
        
        # This would typically query a database
        # For now, return a placeholder response
        return {
            "content_id": content_id,
            "versions": [],
            "message": "Content version history retrieval not yet implemented"
        }
        
    except Exception as e:
        logger.error(f"Failed to get content versions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve content versions")





















