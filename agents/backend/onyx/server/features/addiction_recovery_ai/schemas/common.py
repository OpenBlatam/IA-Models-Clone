"""
Common Pydantic schemas for shared response types
"""

from typing import Optional, Generic, TypeVar, List
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar('T')


class SuccessResponse(BaseModel):
    """Standard success response"""
    status: str = Field(default="success", description="Response status")
    message: Optional[str] = Field(default=None, description="Success message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class ErrorResponse(BaseModel):
    """Standard error response"""
    status: str = Field(default="error", description="Response status")
    detail: str = Field(..., description="Error detail message")
    error_code: Optional[str] = Field(default=None, description="Error code")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper"""
    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(default=1, ge=1, description="Current page number")
    page_size: int = Field(default=10, ge=1, le=100, description="Items per page")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_previous: bool = Field(..., description="Whether there is a previous page")

