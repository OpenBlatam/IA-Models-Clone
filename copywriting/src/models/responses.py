from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
TIMEOUT_SECONDS = 60

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from .entities import CopywritingVariant, PerformanceMetrics
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Response Models
==============

Pydantic models for API response validation.
"""



class CopywritingResponse(BaseModel):
    """Copywriting generation response model"""
    
    request_id: str = Field(..., description="Unique request identifier")
    content: str = Field(..., description="Generated content")
    variants: List[CopywritingVariant] = Field(default=[], description="Content variants")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Model used for generation")
    
    # Optional fields
    error: Optional[str] = Field(default=None, description="Error message if any")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    cache_hit: bool = Field(default=False, description="Whether response was served from cache")
    optimization_applied: bool = Field(default=False, description="Whether optimization was applied")
    
    # Performance metrics
    tokens_generated: Optional[int] = Field(default=None, description="Number of tokens generated")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "request_id": "req_1234567890",
                "content": "🚀 Transform your business with our cutting-edge digital marketing solutions!",
                "variants": [
                    {
                        "id": "variant_1",
                        "content": "🚀 Transform your business with our cutting-edge digital marketing solutions!",
                        "score": 0.85,
                        "metadata": {
                            "relevance_score": 0.9,
                            "engagement_score": 0.8,
                            "conversion_score": 0.85
                        }
                    }
                ],
                "processing_time": 0.45,
                "model_used": "gpt2-medium",
                "cache_hit": False,
                "optimization_applied": True,
                "tokens_generated": 15,
                "confidence_score": 0.85
            }
        }


class BatchResponse(BaseModel):
    """Batch copywriting response model"""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    results: List[CopywritingResponse] = Field(..., description="List of generation results")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    batch_size: int = Field(..., description="Number of requests in batch")
    success_count: int = Field(..., description="Number of successful generations")
    error_count: int = Field(..., description="Number of failed generations")
    
    # Batch metadata
    batch_options: Optional[Dict[str, Any]] = Field(default=None, description="Batch processing options")
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Batch start time")
    completed_at: datetime = Field(default_factory=datetime.utcnow, description="Batch completion time")
    
    # Performance metrics
    avg_processing_time: float = Field(..., description="Average processing time per request")
    throughput: float = Field(..., description="Requests processed per second")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_1234567890",
                "results": [
                    {
                        "request_id": "req_1",
                        "content": "Generated content 1",
                        "variants": [],
                        "processing_time": 0.3,
                        "model_used": "gpt2-medium"
                    }
                ],
                "total_processing_time": 1.2,
                "batch_size": 3,
                "success_count": 3,
                "error_count": 0,
                "avg_processing_time": 0.4,
                "throughput": 2.5
            }
        }


class OptimizationResponse(BaseModel):
    """Text optimization response model"""
    
    request_id: str = Field(..., description="Unique request identifier")
    original_text: str = Field(..., description="Original text")
    optimized_text: str = Field(..., description="Optimized text")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    # Optimization details
    optimization_type: str = Field(..., description="Type of optimization applied")
    improvements: List[str] = Field(default=[], description="List of improvements made")
    metrics: Dict[str, float] = Field(default={}, description="Optimization metrics")
    
    # Optional fields
    suggestions: Optional[List[str]] = Field(default=None, description="Additional suggestions")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "request_id": "opt_1234567890",
                "original_text": "Check out our amazing product!",
                "optimized_text": "🚀 Discover our revolutionary product that transforms your workflow!",
                "processing_time": 0.25,
                "optimization_type": "engagement",
                "improvements": [
                    "Added emoji for visual appeal",
                    "Enhanced emotional impact",
                    "Improved call-to-action"
                ],
                "metrics": {
                    "engagement_score": 0.85,
                    "readability_score": 0.92,
                    "conversion_potential": 0.78
                }
            }
        }


class AnalysisResponse(BaseModel):
    """Content analysis response model"""
    
    request_id: str = Field(..., description="Unique request identifier")
    text: str = Field(..., description="Analyzed text")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    # Analysis results
    sentiment: Optional[Dict[str, Any]] = Field(default=None, description="Sentiment analysis results")
    readability: Optional[Dict[str, Any]] = Field(default=None, description="Readability analysis results")
    tone: Optional[Dict[str, Any]] = Field(default=None, description="Tone analysis results")
    engagement: Optional[Dict[str, Any]] = Field(default=None, description="Engagement analysis results")
    seo: Optional[Dict[str, Any]] = Field(default=None, description="SEO analysis results")
    grammar: Optional[Dict[str, Any]] = Field(default=None, description="Grammar analysis results")
    
    # Overall assessment
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall content score")
    suggestions: List[str] = Field(default=[], description="Improvement suggestions")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "request_id": "analysis_1234567890",
                "text": "Our product is the best solution for your needs!",
                "processing_time": 0.15,
                "sentiment": {
                    "score": 0.8,
                    "label": "positive",
                    "confidence": 0.85
                },
                "readability": {
                    "score": 0.75,
                    "grade_level": "8th grade",
                    "complexity": "moderate"
                },
                "overall_score": 0.78,
                "suggestions": [
                    "Consider adding more specific benefits",
                    "Include social proof or testimonials",
                    "Make the call-to-action more compelling"
                ]
            }
        }


class SystemMetrics(BaseModel):
    """System metrics response model"""
    
    # Engine status
    engine_status: Dict[str, Any] = Field(..., description="Engine status information")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    memory_usage: Dict[str, Any] = Field(..., description="Memory usage information")
    cache_stats: Dict[str, Any] = Field(..., description="Cache statistics")
    batch_stats: Dict[str, Any] = Field(..., description="Batch processing statistics")
    
    # System information
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="System version")
    environment: str = Field(..., description="Environment (development/staging/production)")
    
    # Health information
    health_status: str = Field(..., description="Overall health status")
    last_health_check: datetime = Field(..., description="Last health check timestamp")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "engine_status": {
                    "initialized": True,
                    "active_requests": 5,
                    "total_requests": 1250
                },
                "performance_metrics": {
                    "avg_processing_time": 0.45,
                    "cache_hit_rate": 0.75,
                    "throughput": 2.2
                },
                "memory_usage": {
                    "total": 8589934592,
                    "used": 2147483648,
                    "percent": 25.0
                },
                "uptime": 3600.0,
                "version": "3.0.0",
                "environment": "production",
                "health_status": "healthy",
                "last_health_check": "2024-01-15T10:30:00Z"
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "error": "Invalid request parameters",
                "error_code": "VALIDATION_ERROR",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_1234567890",
                "details": {
                    "field": "prompt",
                    "message": "Field is required"
                }
            }
        } 