"""
Pydantic Models for OpusClip Improved
====================================

Advanced data models with validation and type safety.
"""

from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum
from datetime import datetime
from uuid import UUID, uuid4
import base64

from pydantic import BaseModel, Field, validator, ConfigDict, computed_field
from pydantic_settings import BaseSettings


class VideoFormat(str, Enum):
    """Supported video formats"""
    MP4 = "mp4"
    AVI = "avi"
    MOV = "mov"
    MKV = "mkv"
    WEBM = "webm"
    FLV = "flv"


class AudioFormat(str, Enum):
    """Supported audio formats"""
    MP3 = "mp3"
    AAC = "aac"
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"


class ProcessingStatus(str, Enum):
    """Processing status options"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClipType(str, Enum):
    """Types of clips that can be generated"""
    HIGHLIGHT = "highlight"
    VIRAL = "viral"
    EDUCATIONAL = "educational"
    PROMOTIONAL = "promotional"
    TEASER = "teaser"
    RECAP = "recap"
    MONTAGE = "montage"


class PlatformType(str, Enum):
    """Target platforms for export"""
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"
    FACEBOOK = "facebook"
    TWITCH = "twitch"
    CUSTOM = "custom"


class QualityLevel(str, Enum):
    """Video quality levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"
    CUSTOM = "custom"


class AIProvider(str, Enum):
    """AI service providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    LOCAL = "local"
    HUGGINGFACE = "huggingface"


class VideoAnalysisRequest(BaseModel):
    """Request model for video analysis"""
    model_config = ConfigDict(extra="forbid")
    
    video_url: Optional[str] = Field(None, description="URL of the video to analyze")
    video_file: Optional[str] = Field(None, description="Base64 encoded video file")
    video_path: Optional[str] = Field(None, description="Local path to video file")
    
    # Analysis options
    extract_audio: bool = Field(True, description="Extract and analyze audio")
    detect_faces: bool = Field(True, description="Detect faces in video")
    detect_objects: bool = Field(True, description="Detect objects in video")
    analyze_sentiment: bool = Field(True, description="Analyze sentiment")
    extract_transcript: bool = Field(True, description="Extract transcript")
    detect_scenes: bool = Field(True, description="Detect scene changes")
    
    # Advanced options
    custom_prompts: Optional[List[str]] = Field(None, description="Custom analysis prompts")
    ai_provider: AIProvider = Field(AIProvider.OPENAI, description="AI provider to use")
    language: str = Field("en", description="Language for analysis")
    
    # Processing options
    max_duration: Optional[int] = Field(None, description="Maximum duration to analyze (seconds)")
    sample_rate: int = Field(1, description="Sample every N seconds for analysis")
    
    @validator('video_url', 'video_file', 'video_path')
    def validate_video_source(cls, v, values):
        """Ensure at least one video source is provided"""
        if not any([values.get('video_url'), values.get('video_file'), values.get('video_path')]):
            raise ValueError("At least one video source must be provided")
        return v


class VideoAnalysisResponse(BaseModel):
    """Response model for video analysis"""
    model_config = ConfigDict(extra="forbid")
    
    analysis_id: UUID = Field(default_factory=uuid4, description="Unique analysis ID")
    status: ProcessingStatus = Field(description="Analysis status")
    
    # Video metadata
    duration: float = Field(description="Video duration in seconds")
    fps: float = Field(description="Frames per second")
    resolution: str = Field(description="Video resolution")
    format: VideoFormat = Field(description="Video format")
    file_size: int = Field(description="File size in bytes")
    
    # Analysis results
    transcript: Optional[str] = Field(None, description="Extracted transcript")
    sentiment_scores: Optional[Dict[str, float]] = Field(None, description="Sentiment analysis scores")
    key_moments: Optional[List[Dict[str, Any]]] = Field(None, description="Key moments detected")
    scene_changes: Optional[List[float]] = Field(None, description="Scene change timestamps")
    face_detections: Optional[List[Dict[str, Any]]] = Field(None, description="Face detection results")
    object_detections: Optional[List[Dict[str, Any]]] = Field(None, description="Object detection results")
    
    # AI insights
    content_summary: Optional[str] = Field(None, description="AI-generated content summary")
    topics: Optional[List[str]] = Field(None, description="Identified topics")
    emotions: Optional[List[str]] = Field(None, description="Detected emotions")
    viral_potential: Optional[float] = Field(None, description="Viral potential score (0-1)")
    
    # Processing info
    processing_time: float = Field(description="Processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ClipGenerationRequest(BaseModel):
    """Request model for clip generation"""
    model_config = ConfigDict(extra="forbid")
    
    analysis_id: UUID = Field(description="Video analysis ID")
    clip_type: ClipType = Field(description="Type of clip to generate")
    
    # Clip specifications
    target_duration: int = Field(30, ge=5, le=300, description="Target duration in seconds")
    max_clips: int = Field(5, ge=1, le=20, description="Maximum number of clips to generate")
    
    # Content preferences
    include_intro: bool = Field(True, description="Include intro in clips")
    include_outro: bool = Field(True, description="Include outro in clips")
    add_captions: bool = Field(True, description="Add captions to clips")
    add_watermark: bool = Field(False, description="Add watermark to clips")
    
    # AI customization
    custom_prompt: Optional[str] = Field(None, description="Custom generation prompt")
    ai_provider: AIProvider = Field(AIProvider.OPENAI, description="AI provider to use")
    style_preference: Optional[str] = Field(None, description="Style preference for clips")
    
    # Platform optimization
    target_platforms: List[PlatformType] = Field([PlatformType.YOUTUBE], description="Target platforms")
    aspect_ratio: Optional[str] = Field(None, description="Aspect ratio (e.g., '16:9', '9:16')")
    
    # Quality settings
    quality: QualityLevel = Field(QualityLevel.HIGH, description="Output quality")
    custom_quality: Optional[Dict[str, Any]] = Field(None, description="Custom quality settings")


class ClipGenerationResponse(BaseModel):
    """Response model for clip generation"""
    model_config = ConfigDict(extra="forbid")
    
    generation_id: UUID = Field(default_factory=uuid4, description="Unique generation ID")
    analysis_id: UUID = Field(description="Source analysis ID")
    status: ProcessingStatus = Field(description="Generation status")
    
    # Generated clips
    clips: List[Dict[str, Any]] = Field(description="Generated clips")
    
    # Processing info
    processing_time: float = Field(description="Processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    @computed_field
    @property
    def clip_count(self) -> int:
        """Number of generated clips"""
        return len(self.clips)


class ClipExportRequest(BaseModel):
    """Request model for clip export"""
    model_config = ConfigDict(extra="forbid")
    
    generation_id: UUID = Field(description="Clip generation ID")
    clip_ids: List[UUID] = Field(description="Specific clip IDs to export")
    
    # Export options
    format: VideoFormat = Field(VideoFormat.MP4, description="Export format")
    quality: QualityLevel = Field(QualityLevel.HIGH, description="Export quality")
    resolution: Optional[str] = Field(None, description="Custom resolution")
    bitrate: Optional[int] = Field(None, description="Custom bitrate")
    
    # Platform optimization
    target_platform: PlatformType = Field(description="Target platform")
    optimize_for_platform: bool = Field(True, description="Optimize for target platform")
    
    # Additional options
    add_metadata: bool = Field(True, description="Add metadata to exported files")
    create_thumbnail: bool = Field(True, description="Create thumbnail for clips")
    generate_subtitles: bool = Field(False, description="Generate subtitle files")


class ClipExportResponse(BaseModel):
    """Response model for clip export"""
    model_config = ConfigDict(extra="forbid")
    
    export_id: UUID = Field(default_factory=uuid4, description="Unique export ID")
    generation_id: UUID = Field(description="Source generation ID")
    status: ProcessingStatus = Field(description="Export status")
    
    # Export results
    exported_files: List[Dict[str, Any]] = Field(description="Exported file information")
    download_urls: List[str] = Field(description="Download URLs for exported files")
    
    # Processing info
    processing_time: float = Field(description="Processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = Field(None, description="Error message if failed")


class BatchProcessingRequest(BaseModel):
    """Request model for batch processing"""
    model_config = ConfigDict(extra="forbid")
    
    videos: List[VideoAnalysisRequest] = Field(description="List of videos to process")
    clip_types: List[ClipType] = Field(description="Types of clips to generate")
    
    # Batch options
    parallel_processing: bool = Field(True, description="Process videos in parallel")
    max_concurrent: int = Field(3, ge=1, le=10, description="Maximum concurrent processing")
    
    # Notification options
    notify_on_completion: bool = Field(True, description="Notify when batch is complete")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for notifications")


class BatchProcessingResponse(BaseModel):
    """Response model for batch processing"""
    model_config = ConfigDict(extra="forbid")
    
    batch_id: UUID = Field(default_factory=uuid4, description="Unique batch ID")
    status: ProcessingStatus = Field(description="Batch status")
    
    # Batch results
    total_videos: int = Field(description="Total number of videos")
    completed_videos: int = Field(description="Number of completed videos")
    failed_videos: int = Field(description="Number of failed videos")
    
    # Results
    analysis_results: List[VideoAnalysisResponse] = Field(description="Analysis results")
    generation_results: List[ClipGenerationResponse] = Field(description="Generation results")
    
    # Processing info
    total_processing_time: float = Field(description="Total processing time in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class ProjectRequest(BaseModel):
    """Request model for project management"""
    model_config = ConfigDict(extra="forbid")
    
    name: str = Field(description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    
    # Project settings
    default_clip_duration: int = Field(30, ge=5, le=300, description="Default clip duration")
    default_quality: QualityLevel = Field(QualityLevel.HIGH, description="Default quality")
    default_platforms: List[PlatformType] = Field([PlatformType.YOUTUBE], description="Default platforms")
    
    # Collaboration
    collaborators: Optional[List[str]] = Field(None, description="Collaborator user IDs")
    is_public: bool = Field(False, description="Whether project is public")


class ProjectResponse(BaseModel):
    """Response model for project management"""
    model_config = ConfigDict(extra="forbid")
    
    project_id: UUID = Field(default_factory=uuid4, description="Unique project ID")
    name: str = Field(description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    
    # Project statistics
    total_videos: int = Field(0, description="Total videos in project")
    total_clips: int = Field(0, description="Total clips generated")
    total_views: int = Field(0, description="Total views across all clips")
    
    # Settings
    default_clip_duration: int = Field(description="Default clip duration")
    default_quality: QualityLevel = Field(description="Default quality")
    default_platforms: List[PlatformType] = Field(description="Default platforms")
    
    # Collaboration
    collaborators: List[str] = Field(description="Collaborator user IDs")
    is_public: bool = Field(description="Whether project is public")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AnalyticsRequest(BaseModel):
    """Request model for analytics"""
    model_config = ConfigDict(extra="forbid")
    
    project_id: Optional[UUID] = Field(None, description="Project ID to analyze")
    date_range: Optional[Dict[str, datetime]] = Field(None, description="Date range for analysis")
    metrics: List[str] = Field(["views", "engagement", "viral_score"], description="Metrics to analyze")
    
    # Analysis options
    group_by: Optional[str] = Field(None, description="Group results by field")
    include_predictions: bool = Field(False, description="Include predictive analytics")


class AnalyticsResponse(BaseModel):
    """Response model for analytics"""
    model_config = ConfigDict(extra="forbid")
    
    analytics_id: UUID = Field(default_factory=uuid4, description="Unique analytics ID")
    
    # Analytics data
    metrics: Dict[str, Any] = Field(description="Analytics metrics")
    trends: Dict[str, List[Dict[str, Any]]] = Field(description="Trend data")
    insights: List[str] = Field(description="AI-generated insights")
    recommendations: List[str] = Field(description="AI-generated recommendations")
    
    # Metadata
    date_range: Dict[str, datetime] = Field(description="Analysis date range")
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    model_config = ConfigDict(extra="forbid")
    
    status: str = Field(description="Service status")
    version: str = Field(description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # System info
    uptime: float = Field(description="Service uptime in seconds")
    memory_usage: float = Field(description="Memory usage percentage")
    cpu_usage: float = Field(description="CPU usage percentage")
    
    # Service dependencies
    dependencies: Dict[str, str] = Field(description="Dependency status")


class ErrorResponse(BaseModel):
    """Error response model"""
    model_config = ConfigDict(extra="forbid")
    
    error_code: str = Field(description="Error code")
    error_message: str = Field(description="Error message")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class Settings(BaseSettings):
    """Application settings"""
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    # API settings
    api_title: str = "OpusClip Improved API"
    api_version: str = "2.0.0"
    api_description: str = "Advanced video processing and AI-powered content creation"
    debug: bool = False
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    
    # Database settings
    database_url: str = "sqlite:///./opus_clip.db"
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis settings
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # AI settings
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    
    # Video processing settings
    max_video_size: int = 500 * 1024 * 1024  # 500MB
    max_video_duration: int = 3600  # 1 hour
    temp_dir: str = "./temp"
    output_dir: str = "./output"
    
    # Security settings
    secret_key: str = "your-secret-key-here"
    access_token_expire_minutes: int = 30
    algorithm: str = "HS256"
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Get application settings"""
    return Settings()






























