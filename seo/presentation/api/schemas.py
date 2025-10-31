from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, HttpUrl, Field, validator
from datetime import datetime
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
API Schemas
Pydantic schemas for request/response validation
"""



class AnalyzeURLSchema(BaseModel):
    """Schema for single URL analysis request"""
    
    url: str = Field(..., description="URL to analyze", example="https://example.com")
    force_refresh: bool = Field(False, description="Force refresh analysis", example=False)
    include_recommendations: bool = Field(True, description="Include recommendations", example=True)
    include_warnings: bool = Field(True, description="Include warnings", example=True)
    include_errors: bool = Field(True, description="Include errors", example=True)
    
    @validator('url')
    def validate_url(cls, v) -> bool:
        """Validate URL format"""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class AnalyzeBatchSchema(BaseModel):
    """Schema for batch URL analysis request"""
    
    urls: List[str] = Field(..., description="List of URLs to analyze", min_items=1, max_items=100)
    batch_id: Optional[str] = Field(None, description="Optional batch ID", example="batch_123")
    max_concurrent: int = Field(5, description="Maximum concurrent analyses", ge=1, le=20, example=5)
    force_refresh: bool = Field(False, description="Force refresh all analyses", example=False)
    
    @validator('urls')
    def validate_urls(cls, v) -> bool:
        """Validate all URLs"""
        for url in v:
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f'URL must start with http:// or https://: {url}')
        return v


class AnalyzeURLResponseSchema(BaseModel):
    """Schema for URL analysis response"""
    
    id: str = Field(..., description="Analysis ID", example="analysis_123")
    url: str = Field(..., description="Analyzed URL", example="https://example.com")
    title: str = Field(..., description="Page title", example="Example Page")
    meta_description: str = Field(..., description="Meta description", example="Example description")
    meta_keywords: str = Field(..., description="Meta keywords", example="example, keywords")
    seo_score: float = Field(..., description="Overall SEO score", ge=0, le=100, example=85.5)
    grade: str = Field(..., description="SEO grade", example="B")
    recommendations: List[str] = Field(..., description="SEO recommendations", example=["Improve title length"])
    warnings: List[str] = Field(..., description="SEO warnings", example=["Missing meta description"])
    errors: List[str] = Field(..., description="SEO errors", example=[])
    stats: Dict[str, Any] = Field(..., description="Analysis statistics")
    cached: bool = Field(..., description="Whether result was cached", example=False)
    analysis_time: float = Field(..., description="Analysis duration in seconds", example=1.234)
    created_at: datetime = Field(..., description="Analysis creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Analysis update timestamp")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "id": "analysis_123",
                "url": "https://example.com",
                "title": "Example Page - SEO Optimized",
                "meta_description": "This is an example page with good SEO practices",
                "meta_keywords": "example, seo, optimization",
                "seo_score": 85.5,
                "grade": "B",
                "recommendations": [
                    "Add more internal links",
                    "Optimize image alt tags"
                ],
                "warnings": [
                    "Meta description could be longer"
                ],
                "errors": [],
                "stats": {
                    "word_count": 1500,
                    "character_count": 8500,
                    "link_count": 25,
                    "image_count": 10
                },
                "cached": False,
                "analysis_time": 1.234,
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:01Z"
            }
        }


class AnalyzeBatchResponse(BaseModel):
    """Schema for batch analysis response"""
    
    batch_id: str = Field(..., description="Batch ID", example="batch_123")
    total_urls: int = Field(..., description="Total URLs in batch", example=10)
    successful_analyses: int = Field(..., description="Successful analyses", example=8)
    failed_analyses: int = Field(..., description="Failed analyses", example=2)
    analyses: List[AnalyzeURLResponseSchema] = Field(..., description="Analysis results")
    batch_time: float = Field(..., description="Total batch processing time", example=5.678)
    created_at: datetime = Field(..., description="Batch creation timestamp")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_123",
                "total_urls": 10,
                "successful_analyses": 8,
                "failed_analyses": 2,
                "analyses": [],
                "batch_time": 5.678,
                "created_at": "2024-01-01T12:00:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response"""
    
    status: str = Field(..., description="Service status", example="healthy")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Service version", example="1.0.0")
    uptime: float = Field(..., description="Service uptime in seconds", example=3600.0)
    checks: Dict[str, str] = Field(..., description="Health checks status")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "1.0.0",
                "uptime": 3600.0,
                "checks": {
                    "database": "healthy",
                    "cache": "healthy",
                    "analyzer": "healthy"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error response"""
    
    error: str = Field(..., description="Error message", example="Analysis failed")
    error_code: Optional[str] = Field(None, description="Error code", example="ANALYSIS_ERROR")
    details: Optional[str] = Field(None, description="Error details", example="Failed to parse HTML")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID", example="req_123")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "error": "Analysis failed",
                "error_code": "ANALYSIS_ERROR",
                "details": "Failed to parse HTML content",
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_123"
            }
        }


class StatsResponse(BaseModel):
    """Schema for statistics response"""
    
    repository: Dict[str, Any] = Field(..., description="Repository statistics")
    analyzer: Dict[str, Any] = Field(..., description="Analyzer statistics")
    scoring: Dict[str, Any] = Field(..., description="Scoring service statistics")
    timestamp: float = Field(..., description="Statistics timestamp")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "repository": {
                    "total_analyses": 1000,
                    "cache_hits": 750,
                    "cache_miss_rate": 0.25
                },
                "analyzer": {
                    "total_requests": 1000,
                    "success_rate": 0.95,
                    "average_response_time": 1.2
                },
                "scoring": {
                    "total_scores_calculated": 1000,
                    "average_score": 75.5
                },
                "timestamp": 1704110400.0
            }
        }


class CacheClearResponse(BaseModel):
    """Schema for cache clear response"""
    
    message: str = Field(..., description="Success message", example="Cache cleared successfully")
    timestamp: datetime = Field(..., description="Clear timestamp")
    items_cleared: Optional[int] = Field(None, description="Number of items cleared", example=100)
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "message": "Cache cleared successfully",
                "timestamp": "2024-01-01T12:00:00Z",
                "items_cleared": 100
            }
        }


class DeleteResponse(BaseModel):
    """Schema for delete response"""
    
    message: str = Field(..., description="Success message", example="Analysis deleted successfully")
    timestamp: datetime = Field(..., description="Delete timestamp")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "message": "Analysis deleted successfully",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        } 