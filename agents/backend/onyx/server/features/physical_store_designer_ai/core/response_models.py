"""
Response models for consistent API responses
"""

from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime


class SuccessResponse(BaseModel):
    """Standard success response"""
    success: bool = True
    data: Any
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Standard error response"""
    success: bool = False
    error: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class PaginatedResponse(BaseModel):
    """Paginated response"""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


def create_success_response(data: Any, message: Optional[str] = None) -> SuccessResponse:
    """Create a success response"""
    return SuccessResponse(data=data, message=message)


def create_error_response(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
    """Create an error response"""
    error = {
        "code": error_code,
        "message": message
    }
    if details:
        error["details"] = details
    return ErrorResponse(error=error)


def create_paginated_response(
    items: List[Any],
    total: int,
    page: int,
    page_size: int
) -> PaginatedResponse:
    """Create a paginated response"""
    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )


class MetadataResponse(BaseModel):
    """Response with metadata"""
    data: Any
    metadata: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


def create_metadata_response(
    data: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> MetadataResponse:
    """Create a response with metadata"""
    return MetadataResponse(
        data=data,
        metadata=metadata or {}
    )


class ListResponse(BaseModel):
    """Simple list response"""
    items: List[Any]
    count: int
    timestamp: datetime = Field(default_factory=datetime.now)


def create_list_response(items: List[Any]) -> ListResponse:
    """Create a simple list response"""
    return ListResponse(items=items, count=len(items))

