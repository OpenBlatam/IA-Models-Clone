"""
Common Schemas
==============

Common Pydantic models used across the Business Agents System.
"""

from typing import Dict, Any, List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field
from datetime import datetime

T = TypeVar('T')

class ErrorResponse(BaseModel):
    """Standard error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

class SuccessResponse(BaseModel, Generic[T]):
    """Standard success response schema."""
    
    success: bool = Field(True, description="Success status")
    data: T = Field(..., description="Response data")
    message: Optional[str] = Field(None, description="Success message")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class PaginationParams(BaseModel):
    """Pagination parameters."""
    
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(10, ge=1, le=100, description="Page size")
    sort_by: Optional[str] = Field(None, description="Sort field")
    sort_order: str = Field("asc", regex="^(asc|desc)$", description="Sort order")

class PaginationResponse(BaseModel, Generic[T]):
    """Paginated response schema."""
    
    items: List[T] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")

class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="System version")
    components: Dict[str, str] = Field(..., description="Component health status")
    metrics: Dict[str, Any] = Field(..., description="System metrics")

class SystemInfoResponse(BaseModel):
    """System information response schema."""
    
    system: Dict[str, Any] = Field(..., description="System information")
    capabilities: Dict[str, bool] = Field(..., description="System capabilities")
    business_areas: List[Dict[str, Any]] = Field(..., description="Available business areas")
    workflow_templates: Dict[str, int] = Field(..., description="Workflow template counts")
    configuration: Dict[str, Any] = Field(..., description="System configuration")

class MetricsResponse(BaseModel):
    """System metrics response schema."""
    
    agents: Dict[str, Any] = Field(..., description="Agent metrics")
    workflows: Dict[str, Any] = Field(..., description="Workflow metrics")
    business_areas: Dict[str, Any] = Field(..., description="Business area metrics")
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
