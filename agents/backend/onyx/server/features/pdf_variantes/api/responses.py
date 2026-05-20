"""
PDF Variantes API - Standard Response Models
Consistent response formats for all endpoints
"""

from typing import Generic, TypeVar, Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

T = TypeVar('T')


class SuccessResponse(BaseModel, Generic[T]):
    """Standard success response format"""
    success: bool = True
    data: T
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response format"""
    success: bool = False
    data: None = None
    error: Dict[str, Any]
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None


class PaginationMeta(BaseModel):
    """Pagination metadata"""
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    limit: int = Field(..., description="Items per page")
    offset: int = Field(..., description="Items offset")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response with metadata"""
    success: bool = True
    data: List[T]
    pagination: PaginationMeta
    message: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None


def create_success_response(
    data: T,
    message: Optional[str] = None,
    request_id: Optional[str] = None
) -> SuccessResponse[T]:
    """Create a standard success response"""
    return SuccessResponse(
        success=True,
        data=data,
        message=message,
        request_id=request_id
    )


def create_error_response(
    message: str,
    status_code: int = 400,
    error_type: str = "Error",
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ErrorResponse:
    """Create a standard error response"""
    error = {
        "message": message,
        "status_code": status_code,
        "type": error_type
    }
    if details:
        error["details"] = details
    
    return ErrorResponse(
        success=False,
        error=error,
        request_id=request_id
    )


def create_paginated_response(
    items: List[T],
    total: int,
    page: int = 1,
    limit: int = 20,
    offset: int = 0,
    message: Optional[str] = None,
    request_id: Optional[str] = None
) -> PaginatedResponse[T]:
    """Create a paginated response with metadata"""
    total_pages = (total + limit - 1) // limit if limit > 0 else 0
    
    pagination = PaginationMeta(
        total=total,
        page=page,
        limit=limit,
        offset=offset,
        total_pages=total_pages,
        has_next=(offset + limit) < total,
        has_previous=offset > 0
    )
    
    return PaginatedResponse(
        success=True,
        data=items,
        pagination=pagination,
        message=message,
        request_id=request_id
    )


class FilterParams(BaseModel):
    """Common filter parameters"""
    search: Optional[str] = Field(None, description="Search query")
    sort_by: Optional[str] = Field(None, description="Field to sort by")
    sort_order: Optional[str] = Field("asc", description="Sort order (asc/desc)")
    start_date: Optional[datetime] = Field(None, description="Start date filter")
    end_date: Optional[datetime] = Field(None, description="End date filter")
    

class PaginationParams(BaseModel):
    """Common pagination parameters"""
    page: int = Field(1, ge=1, description="Page number")
    limit: int = Field(20, ge=1, le=100, description="Items per page")
    offset: int = Field(0, ge=0, description="Items offset")


class StatsResponse(BaseModel):
    """Statistics response format"""
    success: bool = True
    data: Dict[str, Any]
    period: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())






