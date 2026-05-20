"""
Data models for Faceless Video AI system
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4


class VideoStatus(str, Enum):
    """Status of video generation"""
    PENDING = "pending"
    PROCESSING = "processing"
    GENERATING_IMAGES = "generating_images"
    GENERATING_AUDIO = "generating_audio"
    ADDING_SUBTITLES = "adding_subtitles"
    COMPOSITING = "compositing"
    COMPLETED = "completed"
    FAILED = "failed"


class SubtitleStyle(str, Enum):
    """Subtitle styling options"""
    SIMPLE = "simple"
    MODERN = "modern"
    BOLD = "bold"
    ELEGANT = "elegant"
    MINIMAL = "minimal"


class VideoStyle(str, Enum):
    """Video style options"""
    REALISTIC = "realistic"
    ANIMATED = "animated"
    ABSTRACT = "abstract"
    MINIMALIST = "minimalist"
    DYNAMIC = "dynamic"


class AudioVoice(str, Enum):
    """Available TTS voices"""
    MALE_1 = "male_1"
    MALE_2 = "male_2"
    FEMALE_1 = "female_1"
    FEMALE_2 = "female_2"
    NEUTRAL = "neutral"


class SubtitleConfig(BaseModel):
    """Configuration for subtitle generation"""
    enabled: bool = True
    style: SubtitleStyle = SubtitleStyle.MODERN
    font_size: int = Field(default=48, ge=24, le=120)
    font_color: str = Field(default="#FFFFFF", pattern="^#[0-9A-Fa-f]{6}$")
    background_color: Optional[str] = Field(default=None, pattern="^#[0-9A-Fa-f]{6}$")
    position: str = Field(default="bottom", pattern="^(top|center|bottom)$")
    animation: bool = True
    max_chars_per_line: int = Field(default=42, ge=20, le=80)
    fade_in: bool = True
    fade_out: bool = True

    class Config:
        json_schema_extra = {
            "example": {
                "enabled": True,
                "style": "modern",
                "font_size": 48,
                "font_color": "#FFFFFF",
                "background_color": "#00000080",
                "position": "bottom",
                "animation": True,
                "max_chars_per_line": 42,
                "fade_in": True,
                "fade_out": True,
            }
        }


class AudioConfig(BaseModel):
    """Configuration for audio generation"""
    voice: AudioVoice = AudioVoice.NEUTRAL
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch: float = Field(default=1.0, ge=0.5, le=2.0)
    volume: float = Field(default=1.0, ge=0.0, le=1.0)
    background_music: bool = False
    music_volume: float = Field(default=0.3, ge=0.0, le=1.0)
    music_style: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "voice": "neutral",
                "speed": 1.0,
                "pitch": 1.0,
                "volume": 1.0,
                "background_music": False,
                "music_volume": 0.3,
                "music_style": None,
            }
        }


class VideoConfig(BaseModel):
    """Configuration for video generation"""
    resolution: str = Field(default="1920x1080", pattern="^\\d+x\\d+$")
    fps: int = Field(default=30, ge=24, le=60)
    duration: Optional[float] = None  # Auto-calculated if None
    style: VideoStyle = VideoStyle.REALISTIC
    transition_duration: float = Field(default=0.5, ge=0.1, le=2.0)
    image_duration: float = Field(default=3.0, ge=1.0, le=10.0)
    background_color: str = Field(default="#000000", pattern="^#[0-9A-Fa-f]{6}$")

    class Config:
        json_schema_extra = {
            "example": {
                "resolution": "1920x1080",
                "fps": 30,
                "duration": None,
                "style": "realistic",
                "transition_duration": 0.5,
                "image_duration": 3.0,
                "background_color": "#000000",
            }
        }


class VideoScript(BaseModel):
    """Script model for video generation"""
    text: str = Field(..., min_length=10, description="Script text content")
    segments: Optional[List[Dict[str, Any]]] = None  # Auto-generated if None
    language: str = Field(default="es", pattern="^[a-z]{2}$")
    metadata: Optional[Dict[str, Any]] = None

    @validator("text")
    def validate_text(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Script text must be at least 10 characters")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Bienvenidos a este increíble video sobre inteligencia artificial. Hoy exploraremos las últimas tendencias en IA.",
                "language": "es",
                "metadata": None,
            }
        }


class GenerationProgress(BaseModel):
    """Progress tracking for video generation"""
    status: VideoStatus
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    estimated_time_remaining: Optional[float] = None
    message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "processing",
                "progress": 45.5,
                "current_step": "Generating images",
                "total_steps": 5,
                "completed_steps": 2,
                "estimated_time_remaining": 120.0,
                "message": "Processing segment 3 of 10",
            }
        }


class VideoGenerationRequest(BaseModel):
    """Request model for video generation"""
    script: VideoScript
    video_config: VideoConfig = Field(default_factory=VideoConfig)
    audio_config: AudioConfig = Field(default_factory=AudioConfig)
    subtitle_config: SubtitleConfig = Field(default_factory=SubtitleConfig)
    output_format: str = Field(default="mp4", pattern="^(mp4|webm|mov)$")
    output_quality: str = Field(default="high", pattern="^(low|medium|high|ultra)$")

    class Config:
        json_schema_extra = {
            "example": {
                "script": {
                    "text": "Bienvenidos a este increíble video sobre inteligencia artificial.",
                    "language": "es",
                },
                "video_config": {
                    "resolution": "1920x1080",
                    "fps": 30,
                    "style": "realistic",
                },
                "audio_config": {
                    "voice": "neutral",
                    "speed": 1.0,
                },
                "subtitle_config": {
                    "enabled": True,
                    "style": "modern",
                },
                "output_format": "mp4",
                "output_quality": "high",
            }
        }


class VideoGenerationResponse(BaseModel):
    """Response model for video generation"""
    video_id: UUID = Field(default_factory=uuid4)
    status: VideoStatus = VideoStatus.PENDING
    progress: GenerationProgress = Field(default_factory=lambda: GenerationProgress(
        status=VideoStatus.PENDING,
        progress=0.0,
        current_step="Initializing",
        total_steps=5,
        completed_steps=0,
    ))
    video_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: Optional[float] = None
    file_size: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "video_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "progress": {
                    "status": "completed",
                    "progress": 100.0,
                    "current_step": "Completed",
                    "total_steps": 5,
                    "completed_steps": 5,
                },
                "video_url": "https://example.com/videos/video.mp4",
                "thumbnail_url": "https://example.com/thumbnails/thumb.jpg",
                "duration": 120.5,
                "file_size": 15728640,
                "created_at": "2024-01-01T12:00:00Z",
                "error": None,
            }
        }

