"""Schemas for comparison endpoints."""

from pydantic import BaseModel, Field
from typing import List, Optional
from api.schemas.visualization import SurgeryType


class ComparisonRequest(BaseModel):
    """Request for before/after comparison."""
    
    visualization_id: str = Field(..., description="ID of the visualization to compare")
    include_original: bool = Field(True, description="Include original image in comparison")
    layout: str = Field("side_by_side", description="Layout: side_by_side, overlay, or grid")


class ComparisonResponse(BaseModel):
    """Response for comparison."""
    
    comparison_id: str = Field(..., description="ID of the comparison")
    image_url: str = Field(..., description="URL to the comparison image")
    original_url: Optional[str] = Field(None, description="URL to original image")
    visualization_url: str = Field(..., description="URL to visualization image")
    layout: str = Field(..., description="Layout used")
    created_at: str = Field(..., description="ISO timestamp of creation")


class BatchVisualizationRequest(BaseModel):
    """Request for batch visualization processing."""
    
    requests: List[dict] = Field(..., description="List of visualization requests")
    max_concurrent: int = Field(3, ge=1, le=10, description="Maximum concurrent processing")


class BatchVisualizationResponse(BaseModel):
    """Response for batch processing."""
    
    total: int = Field(..., description="Total requests")
    processed: int = Field(..., description="Successfully processed")
    failed: int = Field(..., description="Failed requests")
    results: List[dict] = Field(..., description="List of results")
    processing_time: float = Field(..., description="Total processing time in seconds")

