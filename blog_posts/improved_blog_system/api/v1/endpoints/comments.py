"""
Comments API endpoints
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path

from ....models.schemas import CommentCreate, CommentResponse, PaginationParams, PaginatedResponse
from ....api.dependencies import CommentServiceDep, CurrentUserDep
from ....core.exceptions import NotFoundError, ValidationError
from ....utils.pagination import create_paginated_response

router = APIRouter()


@router.post("/", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
async def create_comment(
    post_id: int = Query(..., description="Blog post ID"),
    comment_data: CommentCreate = ...,
    comment_service: CommentServiceDep = Depends(),
    current_user: CurrentUserDep = Depends()
):
    """Create a new comment."""
    try:
        comment = await comment_service.create_comment(
            comment_data, post_id, current_user["user_id"]
        )
        return comment
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create comment"
        )


@router.get("/", response_model=PaginatedResponse[CommentResponse])
async def list_comments(
    post_id: int = Query(..., description="Blog post ID"),
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    approved_only: bool = Query(True, description="Show only approved comments"),
    comment_service: CommentServiceDep = Depends()
):
    """List comments for a blog post."""
    try:
        from ....models.schemas import PaginationParams
        pagination = PaginationParams(page=page, size=size)
        
        comments, total = await comment_service.list_comments(
            post_id, pagination, approved_only
        )
        
        return create_paginated_response(comments, total, pagination)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list comments"
        )


@router.get("/{comment_id}", response_model=CommentResponse)
async def get_comment(
    comment_id: int = Path(..., description="Comment ID"),
    comment_service: CommentServiceDep = Depends()
):
    """Get comment by ID."""
    try:
        comment = await comment_service.get_comment(comment_id)
        return comment
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get comment"
        )


@router.put("/{comment_id}/approve", response_model=CommentResponse)
async def approve_comment(
    comment_id: int = Path(..., description="Comment ID"),
    comment_service: CommentServiceDep = Depends(),
    current_user: CurrentUserDep = Depends()
):
    """Approve a comment (admin/moderator only)."""
    try:
        comment = await comment_service.approve_comment(comment_id)
        return comment
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to approve comment"
        )


@router.put("/{comment_id}/reject", response_model=CommentResponse)
async def reject_comment(
    comment_id: int = Path(..., description="Comment ID"),
    comment_service: CommentServiceDep = Depends(),
    current_user: CurrentUserDep = Depends()
):
    """Reject a comment (admin/moderator only)."""
    try:
        comment = await comment_service.reject_comment(comment_id)
        return comment
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reject comment"
        )


@router.delete("/{comment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_comment(
    comment_id: int = Path(..., description="Comment ID"),
    comment_service: CommentServiceDep = Depends(),
    current_user: CurrentUserDep = Depends()
):
    """Delete a comment."""
    try:
        await comment_service.delete_comment(comment_id, current_user["user_id"])
        return None
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete comment"
        )






























