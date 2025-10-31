from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import json
import hashlib
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator, root_validator
import numpy as np
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Optimized Data Models for Copywriting System
============================================

Advanced data models with validation, serialization, and performance optimizations.
"""



class ToneType(str, Enum):
    """Copywriting tone types"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    INSPIRATIONAL = "inspirational"
    HUMOROUS = "humorous"
    URGENT = "urgent"
    FRIENDLY = "friendly"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"


class PlatformType(str, Enum):
    """Target platform types"""
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    EMAIL = "email"
    WEBSITE = "website"
    ADS = "ads"
    BLOG = "blog"


class LanguageType(str, Enum):
    """Supported languages"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"
    ITALIAN = "it"


class OptimizationLevel(str, Enum):
    """Optimization levels"""
    BASIC = "basic"
    ADVANCED = "advanced"
    ULTRA = "ultra"
    ENTERPRISE = "enterprise"


class CopywritingRequest(BaseModel):
    """Optimized copywriting request model"""
    
    # Core fields
    product_description: str = Field(..., min_length=10, max_length=1000)
    target_platform: PlatformType = Field(default=PlatformType.INSTAGRAM)
    tone: ToneType = Field(default=ToneType.PROFESSIONAL)
    language: LanguageType = Field(default=LanguageType.ENGLISH)
    
    # Advanced options
    optimization_level: OptimizationLevel = Field(default=OptimizationLevel.ADVANCED)
    max_variants: int = Field(default=5, ge=1, le=20)
    target_audience: Optional[str] = Field(default=None, max_length=200)
    keywords: Optional[List[str]] = Field(default=None, max_items=20)
    call_to_action: Optional[str] = Field(default=None, max_length=100)
    
    # Performance settings
    use_cache: bool = Field(default=True)
    enable_optimization: bool = Field(default=True)
    batch_size: int = Field(default=1, ge=1, le=100)
    
    # Metadata
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    priority: int = Field(default=5, ge=1, le=10)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('product_description')
    def validate_description(cls, v) -> bool:
        """Validate product description"""
        if len(v.strip()) < 10:
            raise ValueError('Product description must be at least 10 characters')
        return v.strip()
    
    @validator('keywords')
    def validate_keywords(cls, v) -> bool:
        """Validate keywords"""
        if v is not None:
            return [kw.strip().lower() for kw in v if kw.strip()]
        return v
    
    @root_validator
    async def validate_request(cls, values) -> bool:
        """Root validation for the request"""
        # Ensure at least one optimization feature is enabled
        if not values.get('use_cache') and not values.get('enable_optimization'):
            raise ValueError('At least one optimization feature must be enabled')
        
        return values
    
    def get_cache_key(self) -> str:
        """Generate cache key for this request"""
        key_data = {
            'product_description': self.product_description,
            'target_platform': self.target_platform,
            'tone': self.tone,
            'language': self.language,
            'target_audience': self.target_audience,
            'keywords': sorted(self.keywords) if self.keywords else None,
            'call_to_action': self.call_to_action
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with optimized serialization"""
        return {
            'product_description': self.product_description,
            'target_platform': self.target_platform.value,
            'tone': self.tone.value,
            'language': self.language.value,
            'optimization_level': self.optimization_level.value,
            'max_variants': self.max_variants,
            'target_audience': self.target_audience,
            'keywords': self.keywords,
            'call_to_action': self.call_to_action,
            'use_cache': self.use_cache,
            'enable_optimization': self.enable_optimization,
            'batch_size': self.batch_size,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'priority': self.priority,
            'created_at': self.created_at.isoformat()
        }
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CopywritingVariant(BaseModel):
    """Optimized copywriting variant model"""
    
    # Core content
    id: str = Field(..., description="Unique variant identifier")
    content: str = Field(..., min_length=10, max_length=2000)
    title: Optional[str] = Field(default=None, max_length=200)
    
    # Metrics and scores
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    engagement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    conversion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Performance metrics
    processing_time: float = Field(default=0.0, ge=0.0)
    token_count: int = Field(default=0, ge=0)
    model_used: Optional[str] = Field(default=None)
    
    # Metadata
    variant_type: str = Field(default="standard")
    optimization_applied: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('overall_score', pre=True, always=True)
    def calculate_overall_score(cls, v, values) -> Any:
        """Calculate overall score from individual scores"""
        if v == 0.0:  # Only calculate if not explicitly set
            relevance = values.get('relevance_score', 0.0)
            engagement = values.get('engagement_score', 0.0)
            conversion = values.get('conversion_score', 0.0)
            return (relevance * 0.4 + engagement * 0.3 + conversion * 0.3)
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with optimized serialization"""
        return {
            'id': self.id,
            'content': self.content,
            'title': self.title,
            'relevance_score': round(self.relevance_score, 4),
            'engagement_score': round(self.engagement_score, 4),
            'conversion_score': round(self.conversion_score, 4),
            'overall_score': round(self.overall_score, 4),
            'processing_time': round(self.processing_time, 4),
            'token_count': self.token_count,
            'model_used': self.model_used,
            'variant_type': self.variant_type,
            'optimization_applied': self.optimization_applied,
            'created_at': self.created_at.isoformat()
        }
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "forbid"


class CopywritingResponse(BaseModel):
    """Optimized copywriting response model"""
    
    # Core response data
    request_id: str = Field(..., description="Unique request identifier")
    variants: List[CopywritingVariant] = Field(..., min_items=1, max_items=20)
    
    # Performance metrics
    total_processing_time: float = Field(default=0.0, ge=0.0)
    cache_hit: bool = Field(default=False)
    optimization_applied: List[str] = Field(default_factory=list)
    
    # Quality metrics
    average_score: float = Field(default=0.0, ge=0.0, le=1.0)
    best_variant_id: Optional[str] = Field(default=None)
    
    # Metadata
    model_version: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)
    
    @validator('average_score', pre=True, always=True)
    def calculate_average_score(cls, v, values) -> Any:
        """Calculate average score from variants"""
        if v == 0.0 and 'variants' in values:
            variants = values['variants']
            if variants:
                scores = [v.overall_score for v in variants]
                return np.mean(scores)
        return v
    
    @validator('best_variant_id', pre=True, always=True)
    def set_best_variant(cls, v, values) -> Any:
        """Set best variant ID based on scores"""
        if v is None and 'variants' in values:
            variants = values['variants']
            if variants:
                best_variant = max(variants, key=lambda x: x.overall_score)
                return best_variant.id
        return v
    
    def get_best_variant(self) -> Optional[CopywritingVariant]:
        """Get the best performing variant"""
        if self.best_variant_id:
            for variant in self.variants:
                if variant.id == self.best_variant_id:
                    return variant
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with optimized serialization"""
        return {
            'request_id': self.request_id,
            'variants': [v.to_dict() for v in self.variants],
            'total_processing_time': round(self.total_processing_time, 4),
            'cache_hit': self.cache_hit,
            'optimization_applied': self.optimization_applied,
            'average_score': round(self.average_score, 4),
            'best_variant_id': self.best_variant_id,
            'model_version': self.model_version,
            'created_at': self.created_at.isoformat()
        }
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        extra = "forbid"


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring"""
    request_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_processing_time: float = 0.0
    total_processing_time: float = 0.0
    error_count: int = 0
    optimization_count: int = 0
    
    def update(self, processing_time: float, cache_hit: bool = False, 
               optimization_applied: bool = False, error: bool = False):
        """Update metrics"""
        self.request_count += 1
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.request_count
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        if optimization_applied:
            self.optimization_count += 1
            
        if error:
            self.error_count += 1
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'request_count': self.request_count,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': round(self.get_cache_hit_rate(), 4),
            'average_processing_time': round(self.average_processing_time, 4),
            'total_processing_time': round(self.total_processing_time, 4),
            'error_count': self.error_count,
            'optimization_count': self.optimization_count
        }


# Type aliases for better code readability
RequestDict = Dict[str, Any]
ResponseDict = Dict[str, Any]
VariantDict = Dict[str, Any]
MetricsDict = Dict[str, Any] 