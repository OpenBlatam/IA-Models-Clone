from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Template and Avatar AI Schemas
=============================

Pydantic schemas for template selection, AI avatars, and image synchronization.
"""




class TemplateCategory(str, Enum):
    """Template categories."""
    BUSINESS = "business"
    EDUCATION = "education"
    MARKETING = "marketing"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    SOCIAL = "social"


class AvatarGender(str, Enum):
    """Avatar gender options."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class AvatarStyle(str, Enum):
    """Avatar style options."""
    REALISTIC = "realistic"
    CARTOON = "cartoon"
    ANIME = "anime"
    BUSINESS = "business"
    CASUAL = "casual"


class ImageSyncMode(str, Enum):
    """Image synchronization modes."""
    AUTO = "auto"           # Sync automático con el script
    MANUAL = "manual"       # Sincronización manual
    BEAT_SYNC = "beat_sync" # Sincronizar con beats del audio
    WORD_SYNC = "word_sync" # Sincronizar con palabras específicas


class ScriptTone(str, Enum):
    """Script tone options."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    ENERGETIC = "energetic"
    CALM = "calm"
    AUTHORITATIVE = "authoritative"


class TemplateInfo(BaseModel):
    """Template information schema."""
    
    template_id: str = Field(..., description="Unique template identifier")
    name: str = Field(..., description="Template display name")
    description: str = Field(..., description="Template description")
    category: TemplateCategory = Field(..., description="Template category")
    thumbnail_url: str = Field(..., description="Template thumbnail image")
    preview_video_url: Optional[str] = Field(None, description="Template preview video")
    duration_range: Dict[str, int] = Field(
        ..., 
        description="Duration range in seconds",
        examples=[{"min": 15, "max": 180}]
    )
    supported_ratios: List[str] = Field(
        ...,
        description="Supported aspect ratios",
        examples=[["16:9", "9:16", "1:1"]]
    )
    features: List[str] = Field(
        ...,
        description="Template features",
        examples=[["avatar_support", "text_overlay", "background_music"]]
    )
    tags: List[str] = Field(
        default=[],
        description="Template tags for filtering"
    )
    is_premium: bool = Field(default=False, description="Whether template is premium")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AvatarConfig(BaseModel):
    """AI Avatar configuration schema."""
    
    avatar_id: Optional[str] = Field(None, description="Predefined avatar ID")
    gender: AvatarGender = Field(default=AvatarGender.NEUTRAL, description="Avatar gender")
    style: AvatarStyle = Field(default=AvatarStyle.REALISTIC, description="Avatar style")
    age_range: str = Field(
        default="25-35",
        description="Avatar age range",
        examples=["18-25", "25-35", "35-45", "45-60"]
    )
    ethnicity: Optional[str] = Field(
        None,
        description="Avatar ethnicity preference",
        examples=["caucasian", "asian", "hispanic", "african", "mixed"]
    )
    outfit: Optional[str] = Field(
        None,
        description="Avatar outfit style",
        examples=["business", "casual", "formal", "creative"]
    )
    voice_settings: Dict[str, Any] = Field(
        default={
            "language": "es",
            "accent": "neutral",
            "speed": 1.0,
            "pitch": 1.0
        },
        description="Voice synthesis settings"
    )
    custom_appearance: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom appearance settings"
    )


class ImageSyncConfig(BaseModel):
    """Image synchronization configuration."""
    
    sync_mode: ImageSyncMode = Field(
        default=ImageSyncMode.AUTO,
        description="Image synchronization mode"
    )
    images: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of image URLs or paths"
    )
    sync_points: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Manual sync points",
        examples=[[
            {"timestamp": 5.0, "image_index": 0, "transition": "fade"},
            {"timestamp": 10.0, "image_index": 1, "transition": "slide"}
        ]]
    )
    transition_duration: float = Field(
        default=0.5,
        ge=0.1,
        le=3.0,
        description="Transition duration between images in seconds"
    )
    default_image_duration: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="Default duration for each image in seconds"
    )
    effects: Optional[Dict[str, Any]] = Field(
        None,
        description="Image effects and filters"
    )
    
    @field_validator("images")
    @classmethod
    def validate_images(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one image is required")
        return v


class ScriptConfig(BaseModel):
    """Script generation configuration."""
    
    content: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Script content or prompt for generation"
    )
    tone: ScriptTone = Field(
        default=ScriptTone.PROFESSIONAL,
        description="Script tone"
    )
    language: str = Field(
        default="es",
        description="Script language code"
    )
    target_duration: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Target script duration in seconds"
    )
    include_pauses: bool = Field(
        default=True,
        description="Include natural pauses in script"
    )
    speaking_rate: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speaking rate multiplier"
    )
    custom_instructions: Optional[str] = Field(
        None,
        description="Custom instructions for script generation"
    )
    keywords: Optional[List[str]] = Field(
        None,
        description="Keywords to emphasize in script"
    )


class TemplateVideoRequest(BaseModel):
    """Complete template-based video request."""
    
    template_id: str = Field(..., description="Selected template ID")
    user_id: str = Field(..., description="User identifier")
    
    # Avatar configuration
    avatar_config: AvatarConfig = Field(..., description="AI Avatar settings")
    
    # Image synchronization
    image_sync: ImageSyncConfig = Field(..., description="Image sync configuration")
    
    # Script configuration
    script_config: ScriptConfig = Field(..., description="Script settings")
    
    # Video settings
    output_format: str = Field(default="mp4", description="Output video format")
    quality: str = Field(default="high", description="Video quality")
    aspect_ratio: str = Field(default="16:9", description="Video aspect ratio")
    
    # Additional settings
    background_music: Optional[str] = Field(
        None,
        description="Background music URL or ID"
    )
    watermark: Optional[str] = Field(
        None,
        description="Watermark text or image URL"
    )
    custom_branding: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom branding settings"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata"
    )
    
    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, v: str) -> str:
        valid_ratios = ["16:9", "9:16", "1:1", "4:3", "21:9"]
        if v not in valid_ratios:
            raise ValueError(f"Aspect ratio must be one of: {valid_ratios}")
        return v


class TemplateVideoResponse(BaseModel):
    """Template video generation response."""
    
    request_id: str = Field(..., description="Request identifier")
    template_id: str = Field(..., description="Used template ID")
    status: str = Field(..., description="Processing status")
    
    # Generated content URLs
    avatar_video_url: Optional[str] = Field(None, description="Generated avatar video")
    final_video_url: Optional[str] = Field(None, description="Final composed video")
    thumbnail_url: Optional[str] = Field(None, description="Video thumbnail")
    
    # Processing details
    processing_stages: Dict[str, str] = Field(
        default={
            "script_generation": "pending",
            "avatar_creation": "pending", 
            "image_sync": "pending",
            "video_composition": "pending",
            "final_render": "pending"
        },
        description="Processing stage statuses"
    )
    
    # Timing information
    estimated_completion: Optional[datetime] = Field(
        None,
        description="Estimated completion time"
    )
    processing_time: Optional[float] = Field(
        None,
        description="Actual processing time in seconds"
    )
    
    # Generated assets
    generated_script: Optional[str] = Field(
        None,
        description="Final generated script"
    )
    avatar_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Generated avatar information"
    )
    sync_timeline: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Image synchronization timeline"
    )
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Error message if failed")
    warnings: Optional[List[str]] = Field(None, description="Processing warnings")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TemplateListResponse(BaseModel):
    """Template list response schema."""
    
    templates: List[TemplateInfo] = Field(..., description="Available templates")
    total_count: int = Field(..., description="Total number of templates")
    categories: List[str] = Field(..., description="Available categories")
    filters: Dict[str, List[str]] = Field(
        ...,
        description="Available filter options"
    )


class AvatarPreviewRequest(BaseModel):
    """Avatar preview generation request."""
    
    avatar_config: AvatarConfig = Field(..., description="Avatar configuration")
    sample_text: str = Field(
        default="Hola, soy tu avatar de IA. ¿Cómo puedo ayudarte hoy?",
        description="Sample text for preview"
    )
    preview_duration: int = Field(
        default=10,
        ge=5,
        le=30,
        description="Preview duration in seconds"
    )


class AvatarPreviewResponse(BaseModel):
    """Avatar preview response."""
    
    preview_id: str = Field(..., description="Preview identifier")
    avatar_video_url: str = Field(..., description="Preview video URL")
    avatar_info: Dict[str, Any] = Field(..., description="Avatar details")
    expires_at: datetime = Field(..., description="Preview expiration time") 