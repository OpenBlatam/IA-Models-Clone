"""Pydantic schemas for visualization requests and responses."""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class SurgeryType(str, Enum):
    """Available surgery types."""
    RHINOPLASTY = "rhinoplasty"
    FACELIFT = "facelift"
    BLEPHAROPLASTY = "blepharoplasty"
    LIPOSUCTION = "liposuction"
    BREAST_AUGMENTATION = "breast_augmentation"
    CHIN_AUGMENTATION = "chin_augmentation"


class VisualizationRequest(BaseModel):
    """Request model for visualization creation."""
    
    surgery_type: SurgeryType = Field(..., description="Type of surgery to visualize")
    intensity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Intensity of the surgery effect (0.0 to 1.0)"
    )
    image_data: Optional[bytes] = Field(None, description="Image data as bytes")
    image_url: Optional[str] = Field(None, description="URL of the image to process")
    target_areas: Optional[List[str]] = Field(
        None,
        description="Specific areas to modify (e.g., ['nose', 'chin'])"
    )
    additional_notes: Optional[str] = Field(
        None,
        description="Additional notes or preferences for the visualization"
    )


class VisualizationResponse(BaseModel):
    """Response model for visualization creation."""
    
    visualization_id: str = Field(..., description="Unique ID of the visualization")
    image_url: str = Field(..., description="URL to access the visualization image")
    preview_url: Optional[str] = Field(None, description="URL to a preview thumbnail")
    surgery_type: SurgeryType = Field(..., description="Type of surgery visualized")
    intensity: float = Field(..., description="Intensity used for the visualization")
    created_at: str = Field(..., description="ISO timestamp of creation")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

