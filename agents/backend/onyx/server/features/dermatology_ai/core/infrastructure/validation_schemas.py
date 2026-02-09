"""
Validation Schemas
Pydantic schemas for request/response validation
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AnalysisRequest(BaseModel):
    """Request schema for image analysis"""
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "metadata": {
                    "device": "mobile",
                    "location": "indoor"
                }
            }
        }


class AnalysisResponse(BaseModel):
    """Response schema for analysis"""
    success: bool
    analysis_id: str
    status: str
    metrics: Optional[Dict[str, float]] = None
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "metrics": {
                    "overall_score": 75.5,
                    "hydration_score": 68.0
                },
                "conditions": [
                    {
                        "name": "acne",
                        "confidence": 0.85,
                        "severity": "moderate"
                    }
                ]
            }
        }


class RecommendationResponse(BaseModel):
    """Response schema for recommendations"""
    success: bool
    count: int
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "count": 3,
                "recommendations": [
                    {
                        "product_id": "hydrating-serum-001",
                        "product_name": "Hydrating Serum",
                        "category": "serum",
                        "priority": 1,
                        "reason": "Low hydration score detected",
                        "confidence": 0.85
                    }
                ]
            }
        }


class HistoryResponse(BaseModel):
    """Response schema for analysis history"""
    success: bool
    count: int
    analyses: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "count": 10,
                "analyses": [
                    {
                        "id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "completed",
                        "created_at": "2024-01-01T12:00:00Z"
                    }
                ]
            }
        }


class PaginationParams(BaseModel):
    """Pagination parameters"""
    limit: int = Field(default=10, ge=1, le=100, description="Number of items per page")
    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    
    @validator('limit')
    def validate_limit(cls, v):
        if v < 1 or v > 100:
            raise ValueError('limit must be between 1 and 100')
        return v
    
    @validator('offset')
    def validate_offset(cls, v):
        if v < 0:
            raise ValueError('offset must be non-negative')
        return v


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: Dict[str, Any]
    request_id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "error": {
                    "type": "ValidationError",
                    "message": "Invalid input",
                    "status_code": 400,
                    "timestamp": "2024-01-01T12:00:00Z"
                },
                "request_id": "req-123"
            }
        }















