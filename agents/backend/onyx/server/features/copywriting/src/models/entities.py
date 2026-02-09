from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Entity Models
============

Core entity models for the copywriting system.
"""



class ModelType(str, Enum):
    """Model type enumeration"""
    GPT2 = "gpt2"
    GPT2_MEDIUM = "gpt2-medium"
    GPT2_LARGE = "gpt2-large"
    DISTILGPT2 = "distilgpt2"
    CUSTOM = "custom"


class PlatformType(str, Enum):
    """Platform type enumeration"""
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    EMAIL = "email"
    BLOG = "blog"
    WEBSITE = "website"
    AD = "ad"
    GENERAL = "general"


class ContentType(str, Enum):
    """Content type enumeration"""
    POST = "post"
    CAPTION = "caption"
    AD = "ad"
    EMAIL = "email"
    ARTICLE = "article"
    HEADLINE = "headline"
    DESCRIPTION = "description"
    BIO = "bio"
    GENERAL = "general"


class ToneType(str, Enum):
    """Tone type enumeration"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    CONVERSATIONAL = "conversational"
    PERSUASIVE = "persuasive"
    EDUCATIONAL = "educational"
    HUMOROUS = "humorous"


class CopywritingVariant(BaseModel):
    """Copywriting variant entity"""
    
    id: str = Field(..., description="Unique variant identifier")
    content: str = Field(..., description="Generated content")
    score: float = Field(..., ge=0.0, le=1.0, description="Quality score")
    
    # Variant metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Variant metadata")
    generation_method: str = Field(default="torch", description="Generation method used")
    variant_index: int = Field(default=0, description="Variant index in sequence")
    
    # Quality metrics
    relevance_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Relevance score")
    engagement_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Engagement score")
    conversion_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Conversion score")
    readability_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Readability score")
    
    # Generation details
    tokens_used: Optional[int] = Field(default=None, description="Number of tokens used")
    generation_time: Optional[float] = Field(default=None, description="Generation time in seconds")
    model_used: Optional[str] = Field(default=None, description="Model used for generation")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "id": "variant_1",
                "content": "🚀 Transform your business with our cutting-edge digital marketing solutions!",
                "score": 0.85,
                "metadata": {
                    "relevance_score": 0.9,
                    "engagement_score": 0.8,
                    "conversion_score": 0.85,
                    "readability_score": 0.92
                },
                "generation_method": "torch",
                "variant_index": 0,
                "tokens_used": 15,
                "generation_time": 0.3,
                "model_used": "gpt2-medium"
            }
        }


class PerformanceMetrics(BaseModel):
    """Performance metrics entity"""
    
    # Request metrics
    total_requests: int = Field(default=0, description="Total requests processed")
    successful_requests: int = Field(default=0, description="Successful requests")
    failed_requests: int = Field(default=0, description="Failed requests")
    active_requests: int = Field(default=0, description="Currently active requests")
    
    # Timing metrics
    avg_processing_time: float = Field(default=0.0, description="Average processing time")
    min_processing_time: float = Field(default=0.0, description="Minimum processing time")
    max_processing_time: float = Field(default=0.0, description="Maximum processing time")
    total_processing_time: float = Field(default=0.0, description="Total processing time")
    
    # Cache metrics
    cache_hits: int = Field(default=0, description="Cache hits")
    cache_misses: int = Field(default=0, description="Cache misses")
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Cache hit rate")
    
    # Batch metrics
    total_batches: int = Field(default=0, description="Total batches processed")
    avg_batch_size: float = Field(default=0.0, description="Average batch size")
    avg_batch_processing_time: float = Field(default=0.0, description="Average batch processing time")
    
    # Error metrics
    total_errors: int = Field(default=0, description="Total errors")
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Error rate")
    last_error: Optional[str] = Field(default=None, description="Last error message")
    
    # System metrics
    memory_usage_mb: float = Field(default=0.0, description="Memory usage in MB")
    cpu_usage_percent: float = Field(default=0.0, description="CPU usage percentage")
    gpu_usage_percent: Optional[float] = Field(default=None, description="GPU usage percentage")
    
    # Timestamp
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last metrics update")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "total_requests": 1250,
                "successful_requests": 1200,
                "failed_requests": 50,
                "active_requests": 5,
                "avg_processing_time": 0.45,
                "min_processing_time": 0.1,
                "max_processing_time": 2.5,
                "cache_hits": 900,
                "cache_misses": 350,
                "cache_hit_rate": 0.72,
                "total_batches": 25,
                "avg_batch_size": 8.5,
                "total_errors": 50,
                "error_rate": 0.04,
                "memory_usage_mb": 2048.0,
                "cpu_usage_percent": 25.5
            }
        }


class RequestLog(BaseModel):
    """Request log entity"""
    
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Request timestamp")
    
    # Request details
    prompt: str = Field(..., description="Original prompt")
    platform: str = Field(..., description="Target platform")
    content_type: str = Field(..., description="Content type")
    tone: str = Field(..., description="Tone requested")
    
    # Processing details
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Model used")
    cache_hit: bool = Field(default=False, description="Whether cache was hit")
    
    # Response details
    success: bool = Field(..., description="Whether request was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    content_length: int = Field(default=0, description="Generated content length")
    variants_count: int = Field(default=0, description="Number of variants generated")
    
    # Performance details
    tokens_used: Optional[int] = Field(default=None, description="Tokens used")
    memory_usage_mb: Optional[float] = Field(default=None, description="Memory usage during processing")
    
    # Metadata
    user_agent: Optional[str] = Field(default=None, description="User agent")
    ip_address: Optional[str] = Field(default=None, description="IP address")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "request_id": "req_1234567890",
                "timestamp": "2024-01-15T10:30:00Z",
                "prompt": "Create engaging content about digital marketing",
                "platform": "instagram",
                "content_type": "post",
                "tone": "professional",
                "processing_time": 0.45,
                "model_used": "gpt2-medium",
                "cache_hit": False,
                "success": True,
                "content_length": 120,
                "variants_count": 3,
                "tokens_used": 45,
                "memory_usage_mb": 512.0
            }
        }


class SystemHealth(BaseModel):
    """System health entity"""
    
    # Overall status
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    
    # Component health
    engine_healthy: bool = Field(..., description="Engine health status")
    cache_healthy: bool = Field(..., description="Cache health status")
    database_healthy: bool = Field(..., description="Database health status")
    gpu_healthy: Optional[bool] = Field(default=None, description="GPU health status")
    
    # Resource usage
    memory_usage_percent: float = Field(..., ge=0.0, le=100.0, description="Memory usage percentage")
    cpu_usage_percent: float = Field(..., ge=0.0, le=100.0, description="CPU usage percentage")
    disk_usage_percent: float = Field(..., ge=0.0, le=100.0, description="Disk usage percentage")
    gpu_usage_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="GPU usage percentage")
    
    # Performance indicators
    response_time_avg: float = Field(..., description="Average response time")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="Error rate")
    throughput: float = Field(..., description="Requests per second")
    
    # Issues and warnings
    issues: List[str] = Field(default=[], description="List of health issues")
    warnings: List[str] = Field(default=[], description="List of health warnings")
    
    # Uptime
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    last_restart: Optional[datetime] = Field(default=None, description="Last restart timestamp")
    
    @dataclass
class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "engine_healthy": True,
                "cache_healthy": True,
                "database_healthy": True,
                "gpu_healthy": True,
                "memory_usage_percent": 25.5,
                "cpu_usage_percent": 15.2,
                "disk_usage_percent": 45.8,
                "gpu_usage_percent": 30.0,
                "response_time_avg": 0.45,
                "error_rate": 0.02,
                "throughput": 2.2,
                "uptime_seconds": 3600.0
            }
        } 