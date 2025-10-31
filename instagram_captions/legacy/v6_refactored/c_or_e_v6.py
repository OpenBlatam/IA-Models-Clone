from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import secrets
import hashlib
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator
    from pydantic_settings import BaseSettings
    from pydantic import BaseSettings
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v6.0 - Consolidated Core Module

Refactored architecture combining configuration, schemas, and utilities 
for maximum simplicity while maintaining all functionality.
"""


try:
except ImportError:


# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

class UltraConfig(BaseSettings):
    """Consolidated configuration for maximum simplicity."""
    
    # API Information
    API_VERSION: str = "6.0.0"
    API_NAME: str = "Instagram Captions API v6.0 - Refactored Architecture"
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8080, env="PORT")
    
    # Security Configuration
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32), env="SECRET_KEY")
    VALID_API_KEYS: List[str] = Field(
        default=["ultra-key-123", "speed-key-456", "refactor-key-789"],
        env="VALID_API_KEYS"
    )
    
    # Performance Configuration
    MAX_BATCH_SIZE: int = Field(default=100, env="MAX_BATCH_SIZE")
    AI_PARALLEL_WORKERS: int = Field(default=20, env="AI_PARALLEL_WORKERS")
    CACHE_MAX_SIZE: int = Field(default=50000, env="CACHE_MAX_SIZE")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")
    RATE_LIMIT_REQUESTS: int = Field(default=10000, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=3600, env="RATE_LIMIT_WINDOW")
    
    # Quality Configuration
    AI_QUALITY_THRESHOLD: float = Field(default=85.0, env="AI_QUALITY_THRESHOLD")
    STYLE_BONUS: int = Field(default=15, env="STYLE_BONUS")
    AUDIENCE_BONUS: int = Field(default=15, env="AUDIENCE_BONUS")
    PRIORITY_BONUS: int = Field(default=20, env="PRIORITY_BONUS")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = '{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}'
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global configuration instance
config = UltraConfig()


# =============================================================================
# SCHEMAS SECTION
# =============================================================================

class CaptionRequest(BaseModel):
    """Simplified caption request with essential fields only."""
    
    content_description: str = Field(
        ..., 
        min_length=5, 
        max_length=1000,
        description="Content description for caption generation"
    )
    style: str = Field(
        default="casual",
        pattern="^(casual|professional|playful|inspirational|educational|promotional)$",
        description="Caption style"
    )
    audience: str = Field(
        default="general",
        pattern="^(general|business|millennials|gen_z|creators|lifestyle)$",
        description="Target audience"
    )
    hashtag_count: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Number of hashtags to generate"
    )
    priority: str = Field(
        default="normal",
        pattern="^(low|normal|high|urgent)$",
        description="Processing priority"
    )
    client_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Client identifier for tracking"
    )
    
    @field_validator('content_description')
    @classmethod
    def validate_content_description(cls, v: str) -> str:
        """Validate and sanitize content description."""
        if not v.strip():
            raise ValueError("Content description cannot be empty")
        
        # Remove dangerous patterns
        dangerous_patterns = [
            '<script>', '</script>', '<iframe>', '</iframe>',
            'javascript:', 'onload=', 'onerror=', 'onclick='
        ]
        
        v_lower = v.lower()
        for pattern in dangerous_patterns:
            if pattern in v_lower:
                raise ValueError(f"Content contains potentially dangerous pattern: {pattern}")
        
        return v.strip()
    
    @field_validator('client_id')
    @classmethod
    def validate_client_id(cls, v: str) -> str:
        """Validate client ID format."""
        if not v.strip():
            raise ValueError("Client ID cannot be empty")
        
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_')
        if not all(c in allowed_chars for c in v):
            raise ValueError("Client ID contains invalid characters")
        
        return v.strip()


class BatchRequest(BaseModel):
    """Simplified batch processing request."""
    
    requests: List[CaptionRequest] = Field(
        ...,
        max_length=config.MAX_BATCH_SIZE,
        description=f"List of caption requests (max {config.MAX_BATCH_SIZE})"
    )
    batch_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique batch identifier"
    )


class CaptionResponse(BaseModel):
    """Simplified caption response with essential data."""
    
    request_id: str = Field(..., description="Unique request identifier")
    status: str = Field(..., description="Response status")
    caption: str = Field(..., description="Generated caption")
    hashtags: List[str] = Field(..., description="Generated hashtags")
    quality_score: float = Field(..., ge=0, le=100, description="Quality score out of 100")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Response timestamp")
    cache_hit: bool = Field(..., description="Whether response came from cache")
    api_version: str = Field(..., description="API version")


class BatchResponse(BaseModel):
    """Simplified batch processing response."""
    
    batch_id: str = Field(..., description="Batch identifier")
    status: str = Field(..., description="Batch processing status")
    results: List[CaptionResponse] = Field(..., description="Individual caption results")
    total_processed: int = Field(..., ge=0, description="Total captions processed")
    total_time_ms: float = Field(..., ge=0, description="Total processing time")
    avg_quality_score: float = Field(..., ge=0, le=100, description="Average quality score")
    timestamp: datetime = Field(..., description="Batch completion timestamp")
    api_version: str = Field(..., description="API version")


class HealthResponse(BaseModel):
    """Simplified health check response."""
    
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")
    performance_grade: str = Field(..., description="Performance grade (A+, A, B, C)")
    metrics: Dict[str, Any] = Field(..., description="Performance metrics")


class ErrorResponse(BaseModel):
    """Standardized error response."""
    
    error: Dict[str, Any] = Field(..., description="Error details")
    
    @classmethod
    def create(cls, message: str, status_code: int, request_id: str = "unknown") -> "ErrorResponse":
        """Create standardized error response."""
        return cls(
            error={
                "message": message,
                "status_code": status_code,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "api_version": config.API_VERSION
            }
        )


# =============================================================================
# UTILITIES SECTION
# =============================================================================

class Utils:
    """Consolidated utility functions for common operations."""
    
    @staticmethod
    async def generate_request_id(prefix: str = "req") -> str:
        """Generate unique request ID for tracking."""
        timestamp = int(time.time() * 1000000)
        return f"{prefix}-{timestamp % 1000000:06d}"
    
    @staticmethod
    def generate_batch_id(client_id: str = "batch") -> str:
        """Generate unique batch ID."""
        timestamp = int(time.time())
        return f"ultra-{client_id}-{timestamp}"
    
    @staticmethod
    def create_cache_key(data: Dict[str, Any], prefix: str = "v6") -> str:
        """Create optimized cache key from request data."""
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        hash_obj = hashlib.md5(json_str.encode())
        return f"{prefix}:{hash_obj.hexdigest()[:16]}"
    
    @staticmethod
    def get_current_timestamp() -> datetime:
        """Get current UTC timestamp."""
        return datetime.now(timezone.utc)
    
    @staticmethod
    def calculate_processing_time(start_time: float) -> float:
        """Calculate processing time in milliseconds."""
        return round((time.time() - start_time) * 1000, 3)
    
    @staticmethod
    def sanitize_content(content: str) -> str:
        """Sanitize content for safe processing."""
        if not content:
            return ""
        
        dangerous_patterns = [
            '<script>', '</script>', '<iframe>', '</iframe>',
            'javascript:', 'onload=', 'onerror=', 'onclick='
        ]
        
        sanitized = content
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern, '')
        
        return sanitized.strip()
    
    @staticmethod
    def calculate_quality_bonus(style: str, audience: str, priority: str) -> float:
        """Calculate quality bonus based on request parameters."""
        bonus = 0.0
        
        if style in ["professional", "inspirational"]:
            bonus += config.STYLE_BONUS
        if audience != "general":
            bonus += config.AUDIENCE_BONUS
        if priority in ["high", "urgent"]:
            bonus += config.PRIORITY_BONUS
        
        return bonus
    
    @staticmethod
    def format_response_time(time_ms: float) -> str:
        """Format response time for display."""
        if time_ms < 1:
            return f"{time_ms:.3f}ms"
        elif time_ms < 1000:
            return f"{time_ms:.1f}ms"
        else:
            return f"{time_ms/1000:.2f}s"


class ResponseBuilder:
    """Builder for creating standardized API responses."""
    
    @staticmethod
    def success(data: Any, message: str = "Success", request_id: str = None, 
                processing_time_ms: float = None) -> Dict[str, Any]:
        """Build standardized success response."""
        response = {
            "status": "success",
            "message": message,
            "data": data,
            "timestamp": Utils.get_current_timestamp().isoformat(),
            "api_version": config.API_VERSION
        }
        
        if request_id:
            response["request_id"] = request_id
        if processing_time_ms is not None:
            response["processing_time_ms"] = processing_time_ms
        
        return response
    
    @staticmethod
    def error(message: str, status_code: int = 500, request_id: str = None,
              details: Any = None) -> Dict[str, Any]:
        """Build standardized error response."""
        response = {
            "status": "error",
            "error": {
                "message": message,
                "status_code": status_code,
                "timestamp": Utils.get_current_timestamp().isoformat(),
                "api_version": config.API_VERSION
            }
        }
        
        if request_id:
            response["error"]["request_id"] = request_id
        if details:
            response["error"]["details"] = details
        
        return response


class CacheKeyGenerator:
    """Specialized cache key generation for different types of requests."""
    
    @staticmethod
    def caption_key(request_data: Dict[str, Any]) -> str:
        """Generate cache key for single caption requests."""
        relevant_fields = {
            "content_description": request_data.get("content_description", ""),
            "style": request_data.get("style", "casual"),
            "audience": request_data.get("audience", "general"),
            "hashtag_count": request_data.get("hashtag_count", 10)
        }
        return Utils.create_cache_key(relevant_fields, "caption")
    
    @staticmethod
    def batch_key(batch_data: Dict[str, Any]) -> str:
        """Generate cache key for batch requests."""
        request_hashes = []
        for req in batch_data.get("requests", []):
            req_key = CacheKeyGenerator.caption_key(req)
            request_hashes.append(req_key)
        
        batch_signature = {
            "request_hashes": sorted(request_hashes),
            "batch_size": len(request_hashes)
        }
        
        return Utils.create_cache_key(batch_signature, "batch")
    
    @staticmethod
    def health_key() -> str:
        """Generate cache key for health checks (time-based)."""
        minute_timestamp = int(time.time()) // 60
        return f"health:{minute_timestamp}"


# =============================================================================
# METRICS SECTION
# =============================================================================

class SimpleMetrics:
    """Simplified metrics collection for essential monitoring."""
    
    def __init__(self) -> Any:
        # Request metrics
        self.requests_total = 0
        self.requests_success = 0
        self.requests_error = 0
        
        # Performance metrics
        self.processing_times = []
        self.quality_scores = []
        self.start_time = time.time()
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
    
    def record_request(self, success: bool, response_time: float, quality_score: float = None):
        """Record request completion."""
        self.requests_total += 1
        
        if success:
            self.requests_success += 1
        else:
            self.requests_error += 1
        
        # Store processing time (keep last 1000)
        self.processing_times.append(response_time)
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
        
        # Store quality score (keep last 1000)
        if quality_score is not None:
            self.quality_scores.append(quality_score)
            if len(self.quality_scores) > 1000:
                self.quality_scores = self.quality_scores[-1000:]
    
    def record_cache_hit(self) -> Any:
        """Record cache hit."""
        self.cache_hits += 1
    
    def record_cache_miss(self) -> Any:
        """Record cache miss."""
        self.cache_misses += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        uptime = time.time() - self.start_time
        
        # Calculate averages
        avg_response_time = (
            sum(self.processing_times) / len(self.processing_times) 
            if self.processing_times else 0
        )
        
        avg_quality = (
            sum(self.quality_scores) / len(self.quality_scores)
            if self.quality_scores else 0
        )
        
        cache_total = self.cache_hits + self.cache_misses
        
        return {
            "requests": {
                "total": self.requests_total,
                "success": self.requests_success,
                "error": self.requests_error,
                "success_rate": round((self.requests_success / max(1, self.requests_total)) * 100, 2),
                "rps": round(self.requests_total / max(1, uptime), 2)
            },
            "performance": {
                "avg_response_time_ms": round(avg_response_time * 1000, 2),
                "avg_quality_score": round(avg_quality, 2),
                "uptime_seconds": round(uptime, 2)
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": round((self.cache_hits / max(1, cache_total)) * 100, 2)
            }
        }
    
    def get_performance_grade(self) -> str:
        """Calculate performance grade based on key metrics."""
        stats = self.get_stats()
        
        avg_response = stats["performance"]["avg_response_time_ms"]
        success_rate = stats["requests"]["success_rate"]
        cache_hit_rate = stats["cache"]["hit_rate"]
        avg_quality = stats["performance"]["avg_quality_score"]
        
        score = 0
        
        # Response time scoring (40 points)
        if avg_response < 50:
            score += 40
        elif avg_response < 100:
            score += 30
        elif avg_response < 200:
            score += 20
        elif avg_response < 500:
            score += 10
        
        # Success rate scoring (30 points)
        if success_rate >= 99.5:
            score += 30
        elif success_rate >= 95:
            score += 20
        elif success_rate >= 90:
            score += 10
        elif success_rate >= 80:
            score += 5
        
        # Cache hit rate scoring (15 points)
        if cache_hit_rate >= 90:
            score += 15
        elif cache_hit_rate >= 80:
            score += 12
        elif cache_hit_rate >= 70:
            score += 8
        elif cache_hit_rate >= 50:
            score += 5
        
        # Quality scoring (15 points)
        if avg_quality >= 90:
            score += 15
        elif avg_quality >= 85:
            score += 12
        elif avg_quality >= 80:
            score += 8
        elif avg_quality >= 70:
            score += 5
        
        # Convert score to grade
        if score >= 90:
            return "A+ ULTRA-FAST"
        elif score >= 80:
            return "A FAST"
        elif score >= 70:
            return "B GOOD"
        elif score >= 60:
            return "C ACCEPTABLE"
        else:
            return "D NEEDS_OPTIMIZATION"


# Global instances
metrics = SimpleMetrics()


# Export all components
__all__ = [
    # Configuration
    'config',
    
    # Schemas
    'CaptionRequest',
    'BatchRequest',
    'CaptionResponse',
    'BatchResponse',
    'HealthResponse',
    'ErrorResponse',
    
    # Utilities
    'Utils',
    'ResponseBuilder',
    'CacheKeyGenerator',
    
    # Metrics
    'metrics',
    'SimpleMetrics'
] 