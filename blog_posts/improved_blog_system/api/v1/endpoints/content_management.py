"""
Advanced Content Management API endpoints
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ....services.advanced_content_management_service import AdvancedContentManagementService, ContentType, ContentStatus
from ....api.dependencies import CurrentUserDep, DatabaseSessionDep
from ....core.exceptions import DatabaseError, ValidationError

router = APIRouter()


class CreateContentRequest(BaseModel):
    """Request model for creating content."""
    title: str = Field(..., description="Content title")
    content: str = Field(..., description="Content body")
    content_type: str = Field(default="blog_post", description="Content type")
    template_name: Optional[str] = Field(default=None, description="Template name")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class UpdateContentRequest(BaseModel):
    """Request model for updating content."""
    title: Optional[str] = Field(default=None, description="Content title")
    content: Optional[str] = Field(default=None, description="Content body")
    status: Optional[str] = Field(default=None, description="Content status")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ContentListRequest(BaseModel):
    """Request model for listing content."""
    content_type: Optional[str] = Field(default=None, description="Filter by content type")
    status: Optional[str] = Field(default=None, description="Filter by status")
    author_id: Optional[str] = Field(default=None, description="Filter by author")
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Page size")
    sort_by: str = Field(default="created_at", description="Sort field")
    sort_order: str = Field(default="desc", description="Sort order")


async def get_content_service(session: DatabaseSessionDep) -> AdvancedContentManagementService:
    """Get content management service instance."""
    return AdvancedContentManagementService(session)


@router.post("/create", response_model=Dict[str, Any])
async def create_content(
    request: CreateContentRequest = Depends(),
    content_service: AdvancedContentManagementService = Depends(get_content_service),
    current_user: CurrentUserDep = Depends()
):
    """Create new content with advanced processing."""
    try:
        # Convert content type to enum
        try:
            content_type = ContentType(request.content_type.lower())
        except ValueError:
            raise ValidationError(f"Invalid content type: {request.content_type}")
        
        result = await content_service.create_content(
            title=request.title,
            content=request.content,
            content_type=content_type,
            author_id=str(current_user.id) if current_user else None,
            template_name=request.template_name,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Content created successfully"
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
            detail="Failed to create content"
        )


@router.put("/{content_id}", response_model=Dict[str, Any])
async def update_content(
    content_id: str,
    request: UpdateContentRequest = Depends(),
    content_service: AdvancedContentManagementService = Depends(get_content_service),
    current_user: CurrentUserDep = Depends()
):
    """Update existing content."""
    try:
        # Convert status to enum if provided
        status_enum = None
        if request.status:
            try:
                status_enum = ContentStatus(request.status.lower())
            except ValueError:
                raise ValidationError(f"Invalid status: {request.status}")
        
        result = await content_service.update_content(
            content_id=content_id,
            title=request.title,
            content=request.content,
            status=status_enum,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "data": result,
            "message": "Content updated successfully"
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
            detail="Failed to update content"
        )


@router.get("/{content_id}", response_model=Dict[str, Any])
async def get_content(
    content_id: str,
    include_metadata: bool = Query(default=True, description="Include metadata"),
    content_service: AdvancedContentManagementService = Depends(get_content_service),
    current_user: CurrentUserDep = Depends()
):
    """Get content by ID."""
    try:
        result = await content_service.get_content(
            content_id=content_id,
            include_metadata=include_metadata
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Content retrieved successfully"
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
            detail="Failed to get content"
        )


@router.get("/slug/{slug}", response_model=Dict[str, Any])
async def get_content_by_slug(
    slug: str,
    include_metadata: bool = Query(default=True, description="Include metadata"),
    content_service: AdvancedContentManagementService = Depends(get_content_service),
    current_user: CurrentUserDep = Depends()
):
    """Get content by slug."""
    try:
        result = await content_service.get_content_by_slug(
            slug=slug,
            include_metadata=include_metadata
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Content retrieved successfully"
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
            detail="Failed to get content by slug"
        )


@router.post("/list", response_model=Dict[str, Any])
async def list_content(
    request: ContentListRequest = Depends(),
    content_service: AdvancedContentManagementService = Depends(get_content_service),
    current_user: CurrentUserDep = Depends()
):
    """List content with filtering and pagination."""
    try:
        # Convert content type to enum if provided
        content_type_enum = None
        if request.content_type:
            try:
                content_type_enum = ContentType(request.content_type.lower())
            except ValueError:
                raise ValidationError(f"Invalid content type: {request.content_type}")
        
        # Convert status to enum if provided
        status_enum = None
        if request.status:
            try:
                status_enum = ContentStatus(request.status.lower())
            except ValueError:
                raise ValidationError(f"Invalid status: {request.status}")
        
        result = await content_service.list_content(
            content_type=content_type_enum,
            status=status_enum,
            author_id=request.author_id,
            page=request.page,
            page_size=request.page_size,
            sort_by=request.sort_by,
            sort_order=request.sort_order
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Content list retrieved successfully"
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
            detail="Failed to list content"
        )


@router.get("/", response_model=Dict[str, Any])
async def list_content_get(
    content_type: Optional[str] = Query(default=None, description="Filter by content type"),
    status: Optional[str] = Query(default=None, description="Filter by status"),
    author_id: Optional[str] = Query(default=None, description="Filter by author"),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Page size"),
    sort_by: str = Query(default="created_at", description="Sort field"),
    sort_order: str = Query(default="desc", description="Sort order"),
    content_service: AdvancedContentManagementService = Depends(get_content_service),
    current_user: CurrentUserDep = Depends()
):
    """List content via GET request."""
    try:
        # Convert content type to enum if provided
        content_type_enum = None
        if content_type:
            try:
                content_type_enum = ContentType(content_type.lower())
            except ValueError:
                raise ValidationError(f"Invalid content type: {content_type}")
        
        # Convert status to enum if provided
        status_enum = None
        if status:
            try:
                status_enum = ContentStatus(status.lower())
            except ValueError:
                raise ValidationError(f"Invalid status: {status}")
        
        result = await content_service.list_content(
            content_type=content_type_enum,
            status=status_enum,
            author_id=author_id,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Content list retrieved successfully"
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
            detail="Failed to list content"
        )


@router.delete("/{content_id}", response_model=Dict[str, Any])
async def delete_content(
    content_id: str,
    content_service: AdvancedContentManagementService = Depends(get_content_service),
    current_user: CurrentUserDep = Depends()
):
    """Delete content (soft delete)."""
    try:
        result = await content_service.delete_content(content_id=content_id)
        
        return {
            "success": True,
            "data": result,
            "message": "Content deleted successfully"
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
            detail="Failed to delete content"
        )


@router.get("/templates", response_model=Dict[str, Any])
async def get_content_templates(
    content_service: AdvancedContentManagementService = Depends(get_content_service),
    current_user: CurrentUserDep = Depends()
):
    """Get available content templates."""
    try:
        result = await content_service.get_content_templates()
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Content templates retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get content templates"
        )


@router.get("/analytics", response_model=Dict[str, Any])
async def get_content_analytics(
    content_id: Optional[str] = Query(default=None, description="Specific content ID"),
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    content_service: AdvancedContentManagementService = Depends(get_content_service),
    current_user: CurrentUserDep = Depends()
):
    """Get content analytics."""
    try:
        result = await content_service.get_content_analytics(
            content_id=content_id,
            days=days
        )
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Content analytics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get content analytics"
        )


@router.get("/stats", response_model=Dict[str, Any])
async def get_content_stats(
    content_service: AdvancedContentManagementService = Depends(get_content_service),
    current_user: CurrentUserDep = Depends()
):
    """Get content statistics."""
    try:
        result = await content_service.get_content_stats()
        
        return {
            "success": True,
            "data": result["data"],
            "message": "Content statistics retrieved successfully"
        }
        
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get content statistics"
        )


@router.get("/types", response_model=Dict[str, Any])
async def get_content_types():
    """Get available content types and their descriptions."""
    content_types = {
        "blog_post": {
            "name": "Blog Post",
            "description": "Standard blog post content",
            "features": ["SEO optimization", "Social sharing", "Comments"],
            "use_cases": ["Personal blogs", "Company blogs", "News articles"]
        },
        "article": {
            "name": "Article",
            "description": "Long-form article content",
            "features": ["In-depth analysis", "Research citations", "Professional formatting"],
            "use_cases": ["Magazine articles", "Research papers", "Industry analysis"]
        },
        "tutorial": {
            "name": "Tutorial",
            "description": "Step-by-step tutorial content",
            "features": ["Structured steps", "Difficulty levels", "Progress tracking"],
            "use_cases": ["How-to guides", "Educational content", "Technical tutorials"]
        },
        "news": {
            "name": "News Article",
            "description": "News and current events content",
            "features": ["Breaking news", "Source attribution", "Timeline tracking"],
            "use_cases": ["News websites", "Press releases", "Industry updates"]
        },
        "review": {
            "name": "Review",
            "description": "Product or service review content",
            "features": ["Rating system", "Pros and cons", "Comparison tables"],
            "use_cases": ["Product reviews", "Service reviews", "Comparison articles"]
        },
        "interview": {
            "name": "Interview",
            "description": "Interview and Q&A content",
            "features": ["Question-answer format", "Speaker attribution", "Timestamps"],
            "use_cases": ["Podcast transcripts", "Expert interviews", "Q&A sessions"]
        },
        "case_study": {
            "name": "Case Study",
            "description": "Detailed case study content",
            "features": ["Problem-solution format", "Data visualization", "Outcome tracking"],
            "use_cases": ["Business case studies", "Success stories", "Problem analysis"]
        },
        "whitepaper": {
            "name": "Whitepaper",
            "description": "Technical whitepaper content",
            "features": ["Technical depth", "Research methodology", "Executive summary"],
            "use_cases": ["Technical documentation", "Research papers", "Industry reports"]
        }
    }
    
    return {
        "success": True,
        "data": {
            "content_types": content_types,
            "total_types": len(content_types)
        },
        "message": "Content types retrieved successfully"
    }


@router.get("/statuses", response_model=Dict[str, Any])
async def get_content_statuses():
    """Get available content statuses and their descriptions."""
    content_statuses = {
        "draft": {
            "name": "Draft",
            "description": "Content is being worked on and not yet published",
            "visibility": "Private",
            "workflow": "In progress"
        },
        "published": {
            "name": "Published",
            "description": "Content is live and visible to the public",
            "visibility": "Public",
            "workflow": "Completed"
        },
        "archived": {
            "name": "Archived",
            "description": "Content is no longer active but preserved",
            "visibility": "Hidden",
            "workflow": "Completed"
        },
        "scheduled": {
            "name": "Scheduled",
            "description": "Content is scheduled for future publication",
            "visibility": "Private",
            "workflow": "Scheduled"
        },
        "review": {
            "name": "Under Review",
            "description": "Content is being reviewed before publication",
            "visibility": "Private",
            "workflow": "In review"
        },
        "rejected": {
            "name": "Rejected",
            "description": "Content was rejected and needs revision",
            "visibility": "Private",
            "workflow": "Needs revision"
        }
    }
    
    return {
        "success": True,
        "data": {
            "content_statuses": content_statuses,
            "total_statuses": len(content_statuses)
        },
        "message": "Content statuses retrieved successfully"
    }


@router.get("/health", response_model=Dict[str, Any])
async def get_content_health(
    content_service: AdvancedContentManagementService = Depends(get_content_service),
    current_user: CurrentUserDep = Depends()
):
    """Get content management system health status."""
    try:
        # Get content stats
        stats = await content_service.get_content_stats()
        
        # Calculate health metrics
        total_content = stats["data"].get("total_content", 0)
        content_by_status = stats["data"].get("content_by_status", {})
        published_content = content_by_status.get("published", 0)
        draft_content = content_by_status.get("draft", 0)
        
        # Calculate health score
        health_score = 100
        
        # Check content distribution
        if total_content > 0:
            published_ratio = published_content / total_content
            if published_ratio < 0.3:
                health_score -= 20
            elif published_ratio > 0.8:
                health_score -= 10
        
        # Check average metrics
        avg_seo_score = stats["data"].get("average_seo_score", 0)
        avg_readability_score = stats["data"].get("average_readability_score", 0)
        
        if avg_seo_score < 50:
            health_score -= 15
        if avg_readability_score < 40:
            health_score -= 15
        
        health_status = "excellent" if health_score >= 90 else "good" if health_score >= 70 else "fair" if health_score >= 50 else "poor"
        
        return {
            "success": True,
            "data": {
                "health_status": health_status,
                "health_score": health_score,
                "total_content": total_content,
                "published_content": published_content,
                "draft_content": draft_content,
                "average_seo_score": avg_seo_score,
                "average_readability_score": avg_readability_score,
                "templates_available": stats["data"].get("templates_available", 0),
                "cache_size": stats["data"].get("cache_size", 0),
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "Content health status retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get content health status"
        )