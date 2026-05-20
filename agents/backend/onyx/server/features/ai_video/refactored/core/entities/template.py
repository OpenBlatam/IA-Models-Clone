from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID
from pydantic import Field, field_validator
from .base import AggregateRoot
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Template Entity
==============

Template entity representing video templates with configuration and metadata.
"""





class TemplateCategory(str, Enum):
    """Template categories."""
    BUSINESS = "business"
    EDUCATION = "education"
    MARKETING = "marketing"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    SOCIAL = "social"
    PRODUCT = "product"
    TUTORIAL = "tutorial"


class TemplateStatus(str, Enum):
    """Template status."""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class Template(AggregateRoot):
    """
    Template entity for video generation.
    
    Templates define the structure, styling, and configuration
    for video generation with AI avatars and image synchronization.
    """
    
    # Basic information
    name: str = Field(..., min_length=1, max_length=100, description="Template name")
    description: str = Field(..., min_length=10, max_length=500, description="Template description")
    category: TemplateCategory = Field(..., description="Template category")
    status: TemplateStatus = Field(default=TemplateStatus.DRAFT, description="Template status")
    
    # Media assets
    thumbnail_url: str = Field(..., description="Template thumbnail URL")
    preview_video_url: Optional[str] = Field(None, description="Preview video URL")
    template_file_url: Optional[str] = Field(None, description="Template file URL")
    
    # Configuration
    duration_range: Dict[str, int] = Field(
        ..., 
        description="Duration range in seconds",
        examples=[{"min": 15, "max": 180}]
    )
    supported_ratios: List[str] = Field(
        ...,
        min_items=1,
        description="Supported aspect ratios",
        examples=[["16:9", "9:16", "1:1"]]
    )
    features: List[str] = Field(
        default=[],
        description="Template features",
        examples=[["avatar_support", "text_overlay", "background_music"]]
    )
    
    # Metadata
    tags: List[str] = Field(default=[], description="Template tags")
    is_premium: bool = Field(default=False, description="Premium template flag")
    creator_id: UUID = Field(..., description="Template creator ID")
    
    # Usage statistics
    usage_count: int = Field(default=0, description="Number of times used")
    rating: float = Field(default=0.0, ge=0.0, le=5.0, description="Average rating")
    rating_count: int = Field(default=0, description="Number of ratings")
    
    # Configuration schema
    config_schema: Dict = Field(
        default={},
        description="Template configuration schema"
    )
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate template name."""
        if not v.strip():
            raise ValueError("Template name cannot be empty")
        return v.strip()
    
    @field_validator("duration_range")
    @classmethod
    def validate_duration_range(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Validate duration range."""
        if "min" not in v or "max" not in v:
            raise ValueError("Duration range must include min and max")
        if v["min"] <= 0 or v["max"] <= 0:
            raise ValueError("Duration values must be positive")
        if v["min"] >= v["max"]:
            raise ValueError("Min duration must be less than max duration")
        return v
    
    @field_validator("supported_ratios")
    @classmethod
    def validate_ratios(cls, v: List[str]) -> List[str]:
        """Validate aspect ratios."""
        valid_ratios = ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"]
        for ratio in v:
            if ratio not in valid_ratios:
                raise ValueError(f"Invalid aspect ratio: {ratio}")
        return v
    
    def _validate_entity(self) -> None:
        """Validate template business rules."""
        # Additional business rule validations
        if self.is_premium and not self.preview_video_url:
            raise ValueError("Premium templates must have a preview video")
        
        if self.status == TemplateStatus.ACTIVE and not self.template_file_url:
            raise ValueError("Active templates must have a template file")
    
    def activate(self) -> None:
        """Activate the template."""
        if not self.template_file_url:
            raise ValueError("Cannot activate template without template file")
        
        self.status = TemplateStatus.ACTIVE
        self.mark_as_dirty()
    
    def deactivate(self) -> None:
        """Deactivate the template."""
        self.status = TemplateStatus.INACTIVE
        self.mark_as_dirty()
    
    def archive(self) -> None:
        """Archive the template."""
        self.status = TemplateStatus.ARCHIVED
        self.mark_as_dirty()
    
    def increment_usage(self) -> None:
        """Increment usage count."""
        self.usage_count += 1
        self.mark_as_dirty()
    
    def add_rating(self, rating: float) -> None:
        """Add a rating to the template."""
        if not 0.0 <= rating <= 5.0:
            raise ValueError("Rating must be between 0.0 and 5.0")
        
        # Calculate new average rating
        total_rating = self.rating * self.rating_count + rating
        self.rating_count += 1
        self.rating = total_rating / self.rating_count
        
        self.mark_as_dirty()
    
    def supports_feature(self, feature: str) -> bool:
        """Check if template supports a specific feature."""
        return feature in self.features
    
    def supports_ratio(self, ratio: str) -> bool:
        """Check if template supports a specific aspect ratio."""
        return ratio in self.supported_ratios
    
    def is_available_for_user(self, user_id: UUID, is_premium_user: bool = False) -> bool:
        """Check if template is available for a specific user."""
        if self.status != TemplateStatus.ACTIVE:
            return False
        
        if self.is_premium and not is_premium_user:
            return False
        
        return True
    
    def get_estimated_duration(self, content_length: int) -> int:
        """Estimate video duration based on content length."""
        # Simple estimation: 150 words per minute
        estimated_seconds = (content_length / 150) * 60
        
        # Clamp to template duration range
        min_duration = self.duration_range["min"]
        max_duration = self.duration_range["max"]
        
        return max(min_duration, min(max_duration, int(estimated_seconds)))
    
    def to_summary_dict(self) -> Dict:
        """Convert to summary dictionary for listings."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "thumbnail_url": self.thumbnail_url,
            "is_premium": self.is_premium,
            "rating": self.rating,
            "usage_count": self.usage_count,
            "features": self.features,
            "supported_ratios": self.supported_ratios,
        } 