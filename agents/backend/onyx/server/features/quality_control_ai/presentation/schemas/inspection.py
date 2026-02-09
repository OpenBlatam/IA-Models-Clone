"""
Inspection Schemas

Pydantic schemas for inspection requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any
from datetime import datetime

from .defect import DefectSchema
from .anomaly import AnomalySchema


class InspectionRequestSchema(BaseModel):
    """Schema for inspection request."""
    
    image_data: str = Field(..., description="Base64 encoded image or file path")
    image_format: str = Field(default="base64", description="Image format: 'base64', 'file_path'")
    config_overrides: Optional[dict] = Field(None, description="Optional configuration overrides")
    include_visualization: bool = Field(default=False, description="Include visualization in response")
    timeout_seconds: Optional[float] = Field(None, description="Optional timeout for inspection")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_data": "base64_encoded_string_or_file_path",
                "image_format": "base64",
                "include_visualization": False,
            }
        }


class InspectionResponseSchema(BaseModel):
    """Schema for inspection response."""
    
    inspection_id: str = Field(..., description="Unique inspection identifier")
    quality_score: float = Field(..., ge=0.0, le=100.0, description="Quality score (0-100)")
    quality_status: str = Field(..., description="Quality status")
    defects: List[DefectSchema] = Field(default_factory=list, description="List of detected defects")
    anomalies: List[AnomalySchema] = Field(default_factory=list, description="List of detected anomalies")
    is_acceptable: bool = Field(..., description="Whether quality is acceptable")
    recommendation: str = Field(..., description="Recommendation based on quality")
    inference_time_ms: Optional[float] = Field(None, description="Inference time in milliseconds")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization image")
    metadata: Optional[dict] = Field(None, description="Additional metadata")
    created_at: Optional[datetime] = Field(None, description="Inspection creation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "inspection_id": "123e4567-e89b-12d3-a456-426614174000",
                "quality_score": 85.5,
                "quality_status": "good",
                "defects": [],
                "anomalies": [],
                "is_acceptable": True,
                "recommendation": "Product meets good quality standards. Approve.",
            }
        }


class BatchInspectionRequestSchema(BaseModel):
    """Schema for batch inspection request."""
    
    images: List[InspectionRequestSchema] = Field(..., description="List of images to inspect")
    batch_size: Optional[int] = Field(None, gt=0, description="Optional batch size for processing")
    parallel: bool = Field(default=True, description="Whether to process in parallel")
    max_workers: Optional[int] = Field(None, gt=0, description="Maximum number of worker threads")
    
    class Config:
        json_schema_extra = {
            "example": {
                "images": [
                    {
                        "image_data": "base64_encoded_string",
                        "image_format": "base64",
                    }
                ],
                "parallel": True,
                "max_workers": 4,
            }
        }


class BatchInspectionResponseSchema(BaseModel):
    """Schema for batch inspection response."""
    
    inspections: List[InspectionResponseSchema] = Field(..., description="List of inspection results")
    total_processed: int = Field(..., ge=0, description="Total number of images processed")
    total_succeeded: int = Field(..., ge=0, description="Number of successful inspections")
    total_failed: int = Field(..., ge=0, description="Number of failed inspections")
    average_quality_score: float = Field(..., ge=0.0, le=100.0, description="Average quality score")
    total_processing_time_ms: Optional[float] = Field(None, description="Total processing time in milliseconds")
    created_at: Optional[datetime] = Field(None, description="Batch processing timestamp")



