from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
API Models for HeyGen AI equivalent.
Request and response models for the REST API with LangChain integration.
"""

# =============================================================================
# Core Request/Response Models (Compatible with Enhanced Core)
# =============================================================================

class VideoRequest(BaseModel):
    """Core video generation request model."""
    script: str = Field(..., min_length=10, max_length=5000, 
                       description="Script text for the video")
    avatar_id: str = Field(..., description="ID of the avatar to use")
    voice_id: str = Field(..., description="ID of the voice to use")
    language: str = Field(default="en", description="Language code")
    resolution: str = Field(default="1080p", description="Video resolution")
    quality_preset: str = Field(default="medium", description="Quality preset (low/medium/high/ultra)")
    output_format: str = Field(default="mp4", description="Output video format")
    duration: Optional[int] = Field(None, ge=10, le=600, 
                                  description="Video duration in seconds (10-600)")
    background: Optional[str] = Field(None, description="Background image/video path")
    custom_settings: Optional[Dict] = Field(default_factory=dict, 
                                          description="Custom video settings")
    enable_expressions: bool = Field(default=True, description="Enable facial expressions")
    enable_effects: bool = Field(default=False, description="Enable video effects")

class VoiceGenerationRequest(BaseModel):
    """Core voice generation request model."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize")
    voice_id: str = Field(..., description="Voice ID to use")
    language: str = Field(default="en", description="Language code")
    quality: str = Field(default="medium", description="Audio quality (low/medium/high)")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")
    pitch: float = Field(default=1.0, ge=0.5, le=2.0, description="Pitch multiplier")
    emotion: Optional[str] = Field(None, description="Emotional tone")

class AvatarGenerationRequest(BaseModel):
    """Core avatar generation request model."""
    avatar_id: str = Field(..., description="Avatar ID to use")
    audio_path: str = Field(..., description="Path to audio file for lip-sync")
    resolution: str = Field(default="1080p", description="Output resolution")
    quality_preset: str = Field(default="medium", description="Quality preset")
    enable_expressions: bool = Field(default=True, description="Enable facial expressions")
    background: Optional[str] = Field(None, description="Background image/video path")
    custom_settings: Optional[Dict] = Field(default_factory=dict, description="Custom settings")

class VideoResponse(BaseModel):
    """Core video generation response model."""
    video_id: str = Field(..., description="Unique video ID")
    status: str = Field(..., description="Video generation status")
    output_url: Optional[str] = Field(None, description="URL to generated video")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    file_size: Optional[int] = Field(None, description="Video file size in bytes")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    metadata: Dict = Field(default_factory=dict, description="Video metadata")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    quality_metrics: Optional[Dict] = Field(None, description="Video quality metrics")
    processing_steps: Optional[List[str]] = Field(None, description="Processing steps completed")

# =============================================================================
# Legacy Models (Maintained for Backward Compatibility)
# =============================================================================

class VideoStatus(str, Enum):
    """Video generation status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class LanguageCode(str, Enum):
    """Supported language codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"

class VideoStyle(str, Enum):
    """Video style options."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    EDUCATIONAL = "educational"
    MARKETING = "marketing"
    ENTERTAINMENT = "entertainment"

class Resolution(str, Enum):
    """Video resolution options."""
    HD_720P = "720p"
    FULL_HD_1080P = "1080p"
    UHD_4K = "4k"

class OutputFormat(str, Enum):
    """Video output format options."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"

# Request Models
class CreateVideoRequest(BaseModel):
    """Request model for creating a video."""
    script: str = Field(..., min_length=10, max_length=5000, 
                       description="Script text for the video")
    avatar_id: str = Field(..., description="ID of the avatar to use")
    voice_id: str = Field(..., description="ID of the voice to use")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH, 
                                 description="Language code")
    style: VideoStyle = Field(default=VideoStyle.PROFESSIONAL, 
                            description="Video style")
    resolution: Resolution = Field(default=Resolution.FULL_HD_1080P, 
                                 description="Video resolution")
    output_format: OutputFormat = Field(default=OutputFormat.MP4, 
                                      description="Output video format")
    duration: Optional[int] = Field(None, ge=10, le=600, 
                                  description="Video duration in seconds (10-600)")
    background: Optional[HttpUrl] = Field(None, description="Background image/video URL")
    custom_settings: Optional[Dict] = Field(default_factory=dict, 
                                          description="Custom video settings")
    quality_preset: str = Field(default="medium", description="Quality preset")
    enable_expressions: bool = Field(default=True, description="Enable facial expressions")
    enable_effects: bool = Field(default=False, description="Enable video effects")


class BatchCreateVideoRequest(BaseModel):
    """Request model for batch video creation."""
    videos: List[CreateVideoRequest] = Field(..., min_items=1, max_items=10,
                                           description="List of video requests")


class GenerateScriptRequest(BaseModel):
    """Request model for script generation."""
    topic: str = Field(..., min_length=3, max_length=200, 
                      description="Topic for the script")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH, 
                                 description="Language code")
    style: VideoStyle = Field(default=VideoStyle.PROFESSIONAL, 
                            description="Script style")
    duration: str = Field(default="2 minutes", 
                         description="Target duration (e.g., '2 minutes', '30 seconds')")
    additional_context: Optional[str] = Field(None, 
                                            description="Additional context for script generation")


class OptimizeScriptRequest(BaseModel):
    """Request model for script optimization."""
    script: str = Field(..., min_length=10, max_length=5000, 
                       description="Script text to optimize")
    duration: str = Field(default="2 minutes", 
                         description="Target duration")
    style: VideoStyle = Field(default=VideoStyle.PROFESSIONAL, 
                            description="Script style")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH, 
                                 description="Language code")


class AnalyzeScriptRequest(BaseModel):
    """Request model for script analysis."""
    script: str = Field(..., min_length=10, max_length=5000, 
                       description="Script text to analyze")


class CloneVoiceRequest(BaseModel):
    """Request model for voice cloning."""
    voice_name: str = Field(..., min_length=2, max_length=50, 
                           description="Name for the cloned voice")
    audio_samples: List[HttpUrl] = Field(..., min_items=3, max_items=10,
                                       description="URLs to audio sample files")
    description: Optional[str] = Field(None, 
                                     description="Description of the voice")


class CreateAvatarRequest(BaseModel):
    """Request model for creating a custom avatar."""
    name: str = Field(..., min_length=2, max_length=50, 
                     description="Name for the avatar")
    image_url: HttpUrl = Field(..., description="URL to source image")
    style: VideoStyle = Field(default=VideoStyle.PROFESSIONAL, 
                            description="Avatar style")
    description: Optional[str] = Field(None, 
                                     description="Description of the avatar")


class TranslateScriptRequest(BaseModel):
    """Request model for script translation."""
    script: str = Field(..., min_length=10, max_length=5000, 
                       description="Script text to translate")
    target_language: LanguageCode = Field(..., description="Target language")
    source_language: LanguageCode = Field(default=LanguageCode.ENGLISH, 
                                        description="Source language")
    preserve_style: bool = Field(default=True, 
                               description="Preserve original style in translation")


class ChatRequest(BaseModel):
    """Request model for LangChain agent chat."""
    message: str = Field(..., min_length=1, max_length=2000, 
                        description="Message to send to the agent")
    context: Optional[str] = Field(None, 
                                 description="Additional context for the conversation")


class CreateKnowledgeBaseRequest(BaseModel):
    """Request model for creating a knowledge base."""
    name: str = Field(..., min_length=2, max_length=50, 
                     description="Name for the knowledge base")
    documents: List[str] = Field(..., min_items=1, max_items=100,
                               description="List of document texts")
    description: Optional[str] = Field(None, 
                                     description="Description of the knowledge base")


class SearchKnowledgeBaseRequest(BaseModel):
    """Request model for searching knowledge base."""
    query: str = Field(..., min_length=1, max_length=500, 
                      description="Search query")
    name: str = Field(default="scripts", 
                     description="Knowledge base name")
    max_results: int = Field(default=5, ge=1, le=20, 
                           description="Maximum number of results")


# Advanced Workflow Request Models
class CreateEducationalSeriesRequest(BaseModel):
    """Request model for creating educational video series."""
    topic: str = Field(..., min_length=3, max_length=200, 
                      description="Main topic for the series")
    series_length: int = Field(default=5, ge=2, le=20, 
                             description="Number of videos in the series")
    target_audience: Optional[str] = Field(default="students", 
                                         description="Target audience for the series")
    difficulty_level: Optional[str] = Field(default="intermediate", 
                                          description="Difficulty level (beginner/intermediate/advanced)")
    language: LanguageCode = Field(default=LanguageCode.ENGLISH, 
                                 description="Primary language for the series")


class CreateMarketingCampaignRequest(BaseModel):
    """Request model for creating marketing campaign."""
    product_info: Dict = Field(..., description="Product information dictionary")
    target_audience: str = Field(..., min_length=10, max_length=500, 
                               description="Target audience description")
    campaign_type: Optional[str] = Field(default="general", 
                                       description="Type of marketing campaign")
    budget_range: Optional[str] = Field(default="medium", 
                                      description="Budget range for the campaign")
    goals: Optional[List[str]] = Field(default_factory=list, 
                                     description="Campaign goals")


class CreateProductDemoRequest(BaseModel):
    """Request model for creating product demonstration."""
    product_info: Dict = Field(..., description="Product information dictionary")
    demo_type: Optional[str] = Field(default="feature_showcase", 
                                   description="Type of demonstration")
    target_users: Optional[str] = Field(default="general", 
                                      description="Target user group")
    focus_areas: Optional[List[str]] = Field(default_factory=list, 
                                           description="Specific features to highlight")


class CreateNewsSummaryRequest(BaseModel):
    """Request model for creating news summary."""
    news_topic: str = Field(..., min_length=5, max_length=200, 
                           description="News topic to summarize")
    target_languages: List[LanguageCode] = Field(default=[LanguageCode.ENGLISH], 
                                               description="Target languages for translation")
    summary_length: Optional[str] = Field(default="medium", 
                                        description="Length of summary (short/medium/long)")
    include_fact_checking: bool = Field(default=True, 
                                      description="Include fact-checking in the process")
    neutral_tone: bool = Field(default=True, 
                             description="Maintain neutral tone in summary")


# Response Models
class VideoResponse(BaseModel):
    """Response model for video generation."""
    video_id: str = Field(..., description="Unique video ID")
    status: VideoStatus = Field(..., description="Video generation status")
    output_url: Optional[HttpUrl] = Field(None, description="URL to generated video")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    file_size: Optional[int] = Field(None, description="Video file size in bytes")
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    metadata: Dict = Field(default_factory=dict, description="Video metadata")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class BatchVideoResponse(BaseModel):
    """Response model for batch video creation."""
    batch_id: str = Field(..., description="Batch operation ID")
    videos: List[VideoResponse] = Field(..., description="List of video responses")
    total_count: int = Field(..., description="Total number of videos")
    completed_count: int = Field(..., description="Number of completed videos")
    failed_count: int = Field(..., description="Number of failed videos")


class ScriptResponse(BaseModel):
    """Response model for script generation."""
    script_id: str = Field(..., description="Unique script ID")
    script: str = Field(..., description="Generated script text")
    word_count: int = Field(..., description="Number of words in script")
    estimated_duration: float = Field(..., description="Estimated speaking duration")
    language: LanguageCode = Field(..., description="Script language")
    style: VideoStyle = Field(..., description="Script style")
    created_at: str = Field(..., description="Creation timestamp")


class VoiceResponse(BaseModel):
    """Response model for voice information."""
    voice_id: str = Field(..., description="Unique voice ID")
    name: str = Field(..., description="Voice name")
    language: LanguageCode = Field(..., description="Voice language")
    accent: str = Field(..., description="Voice accent")
    gender: str = Field(..., description="Voice gender")
    style: str = Field(..., description="Voice style")
    sample_rate: int = Field(..., description="Audio sample rate")
    is_cloned: bool = Field(default=False, description="Whether voice is cloned")
    characteristics: Dict = Field(default_factory=dict, description="Voice characteristics")


class AvatarResponse(BaseModel):
    """Response model for avatar information."""
    avatar_id: str = Field(..., description="Unique avatar ID")
    name: str = Field(..., description="Avatar name")
    gender: str = Field(..., description="Avatar gender")
    style: str = Field(..., description="Avatar style")
    age_range: str = Field(..., description="Avatar age range")
    ethnicity: str = Field(..., description="Avatar ethnicity")
    image_url: HttpUrl = Field(..., description="URL to avatar image")
    is_custom: bool = Field(default=False, description="Whether avatar is custom")
    model_config: Dict = Field(default_factory=dict, description="Avatar model configuration")


class ScriptAnalysisResponse(BaseModel):
    """Response model for script analysis."""
    script_id: str = Field(..., description="Unique script ID")
    word_count: int = Field(..., description="Number of words")
    estimated_duration: float = Field(..., description="Estimated speaking duration")
    readability_score: float = Field(..., description="Readability score (0-100)")
    sentiment: Dict = Field(..., description="Sentiment analysis results")
    complexity: Dict = Field(..., description="Complexity analysis results")
    suggestions: List[str] = Field(..., description="Improvement suggestions")


class TranslationResponse(BaseModel):
    """Response model for script translation."""
    translation_id: str = Field(..., description="Unique translation ID")
    original_script: str = Field(..., description="Original script text")
    translated_script: str = Field(..., description="Translated script text")
    source_language: LanguageCode = Field(..., description="Source language")
    target_language: LanguageCode = Field(..., description="Target language")
    word_count: int = Field(..., description="Number of words in translated script")
    confidence_score: float = Field(..., description="Translation confidence score")


class ChatResponse(BaseModel):
    """Response model for LangChain agent chat."""
    response: str = Field(..., description="Agent response")
    timestamp: str = Field(..., description="Response timestamp")
    agent_used: str = Field(..., description="Type of agent used")


class KnowledgeBaseResponse(BaseModel):
    """Response model for knowledge base operations."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Operation message")
    document_count: Optional[int] = Field(None, description="Number of documents")
    timestamp: str = Field(..., description="Operation timestamp")


class SearchResponse(BaseModel):
    """Response model for knowledge base search."""
    query: str = Field(..., description="Search query")
    results: List[str] = Field(..., description="Search results")
    result_count: int = Field(..., description="Number of results")
    timestamp: str = Field(..., description="Search timestamp")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Overall system status")
    components: Dict[str, bool] = Field(..., description="Component health status")
    version: str = Field(..., description="System version")
    uptime: float = Field(..., description="System uptime in seconds")
    metadata: Optional[Dict] = Field(None, description="Additional metadata")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")


# Advanced Workflow Response Models
class EducationalSeriesResponse(BaseModel):
    """Response model for educational series creation."""
    workflow_type: str = Field(..., description="Type of workflow")
    status: str = Field(..., description="Workflow status")
    topic: str = Field(..., description="Series topic")
    series_length: int = Field(..., description="Number of episodes")
    episodes: List[Dict] = Field(..., description="List of episode information")
    series_metadata: Dict = Field(..., description="Series metadata")
    created_at: str = Field(..., description="Creation timestamp")


class MarketingCampaignResponse(BaseModel):
    """Response model for marketing campaign creation."""
    workflow_type: str = Field(..., description="Type of workflow")
    status: str = Field(..., description="Workflow status")
    product_info: Dict = Field(..., description="Product information")
    target_audience: str = Field(..., description="Target audience")
    campaign_scripts: List[Dict] = Field(..., description="Campaign script variants")
    brand_analysis: Dict = Field(..., description="Brand analysis results")
    audience_analysis: Dict = Field(..., description="Audience analysis results")
    created_at: str = Field(..., description="Creation timestamp")


class ProductDemoResponse(BaseModel):
    """Response model for product demonstration creation."""
    workflow_type: str = Field(..., description="Type of workflow")
    status: str = Field(..., description="Workflow status")
    product_info: Dict = Field(..., description="Product information")
    demo_script: str = Field(..., description="Demonstration script")
    product_analysis: Dict = Field(..., description="Product analysis results")
    feature_priority: List[Dict] = Field(..., description="Feature prioritization")
    benefit_mapping: Dict = Field(..., description="Feature to benefit mapping")
    cta_variations: List[str] = Field(..., description="Call-to-action variations")
    created_at: str = Field(..., description="Creation timestamp")


class NewsSummaryResponse(BaseModel):
    """Response model for news summary creation."""
    workflow_type: str = Field(..., description="Type of workflow")
    status: str = Field(..., description="Workflow status")
    news_topic: str = Field(..., description="News topic")
    video_script: str = Field(..., description="Video script")
    summary: str = Field(..., description="News summary")
    translations: Dict = Field(..., description="Translations by language")
    news_research: Dict = Field(..., description="Research results")
    fact_check_results: Dict = Field(..., description="Fact-checking results")
    created_at: str = Field(..., description="Creation timestamp") 