from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Types Module - Data Structures and Type Definitions

This module defines all the core data structures and types used throughout
the modular AI video workflow system.
"""



class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    EXTRACTING = "extracting"
    SUGGESTING = "suggesting"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ContentType(Enum):
    """Types of extracted content."""
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    NEWS = "news"
    PRODUCT = "product"
    LANDING_PAGE = "landing_page"
    SOCIAL_MEDIA = "social_media"
    UNKNOWN = "unknown"


class VideoFormat(Enum):
    """Supported video output formats."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"
    MKV = "mkv"


class AvatarType(Enum):
    """Types of avatars available."""
    REALISTIC = "realistic"
    CARTOON = "cartoon"
    ANIME = "anime"
    ABSTRACT = "abstract"


@dataclass
class ExtractedContent:
    """Content extracted from a web URL."""
    url: str
    title: Optional[str] = None
    summary: Optional[str] = None
    text: Optional[str] = None
    images: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    content_type: ContentType = ContentType.UNKNOWN
    language: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    author: Optional[str] = None
    published_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    extractor_used: Optional[str] = None
    extraction_time: Optional[float] = None
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'url': self.url,
            'title': self.title,
            'summary': self.summary,
            'text': self.text,
            'images': self.images,
            'links': self.links,
            'meta': self.meta,
            'content_type': self.content_type.value,
            'language': self.language,
            'keywords': self.keywords,
            'author': self.author,
            'published_date': self.published_date.isoformat() if self.published_date else None,
            'modified_date': self.modified_date.isoformat() if self.modified_date else None,
            'extractor_used': self.extractor_used,
            'extraction_time': self.extraction_time,
            'confidence_score': self.confidence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractedContent':
        """Create from dictionary."""
        # Handle date fields
        published_date = None
        if data.get('published_date'):
            try:
                published_date = datetime.fromisoformat(data['published_date'])
            except ValueError:
                pass
        
        modified_date = None
        if data.get('modified_date'):
            try:
                modified_date = datetime.fromisoformat(data['modified_date'])
            except ValueError:
                pass
        
        return cls(
            url=data['url'],
            title=data.get('title'),
            summary=data.get('summary'),
            text=data.get('text'),
            images=data.get('images', []),
            links=data.get('links', []),
            meta=data.get('meta', {}),
            content_type=ContentType(data.get('content_type', 'unknown')),
            language=data.get('language'),
            keywords=data.get('keywords', []),
            author=data.get('author'),
            published_date=published_date,
            modified_date=modified_date,
            extractor_used=data.get('extractor_used'),
            extraction_time=data.get('extraction_time'),
            confidence_score=data.get('confidence_score')
        )


@dataclass
class ContentSuggestions:
    """AI-generated suggestions for video content."""
    script: Optional[str] = None
    images: List[str] = field(default_factory=list)
    style: Optional[str] = None
    duration: Optional[int] = None  # in seconds
    tone: Optional[str] = None
    target_audience: Optional[str] = None
    call_to_action: Optional[str] = None
    background_music: Optional[str] = None
    transitions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None
    generation_time: Optional[float] = None
    engine_used: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'script': self.script,
            'images': self.images,
            'style': self.style,
            'duration': self.duration,
            'tone': self.tone,
            'target_audience': self.target_audience,
            'call_to_action': self.call_to_action,
            'background_music': self.background_music,
            'transitions': self.transitions,
            'effects': self.effects,
            'confidence_score': self.confidence_score,
            'generation_time': self.generation_time,
            'engine_used': self.engine_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContentSuggestions':
        """Create from dictionary."""
        return cls(
            script=data.get('script'),
            images=data.get('images', []),
            style=data.get('style'),
            duration=data.get('duration'),
            tone=data.get('tone'),
            target_audience=data.get('target_audience'),
            call_to_action=data.get('call_to_action'),
            background_music=data.get('background_music'),
            transitions=data.get('transitions', []),
            effects=data.get('effects', []),
            confidence_score=data.get('confidence_score'),
            generation_time=data.get('generation_time'),
            engine_used=data.get('engine_used')
        )


@dataclass
class VideoGenerationResult:
    """Result of video generation process."""
    success: bool
    video_url: Optional[str] = None
    video_path: Optional[Path] = None
    duration: Optional[float] = None  # in seconds
    file_size: Optional[int] = None  # in bytes
    format: VideoFormat = VideoFormat.MP4
    quality_score: Optional[float] = None
    generation_time: Optional[float] = None
    generator_used: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'video_url': self.video_url,
            'video_path': str(self.video_path) if self.video_path else None,
            'duration': self.duration,
            'file_size': self.file_size,
            'format': self.format.value,
            'quality_score': self.quality_score,
            'generation_time': self.generation_time,
            'generator_used': self.generator_used,
            'error_message': self.error_message,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoGenerationResult':
        """Create from dictionary."""
        video_path = None
        if data.get('video_path'):
            video_path = Path(data['video_path'])
        
        return cls(
            success=data['success'],
            video_url=data.get('video_url'),
            video_path=video_path,
            duration=data.get('duration'),
            file_size=data.get('file_size'),
            format=VideoFormat(data.get('format', 'mp4')),
            quality_score=data.get('quality_score'),
            generation_time=data.get('generation_time'),
            generator_used=data.get('generator_used'),
            error_message=data.get('error_message'),
            metadata=data.get('metadata', {})
        )


@dataclass
class WorkflowTimings:
    """Timing information for workflow stages."""
    extraction: Optional[float] = None
    suggestions: Optional[float] = None
    generation: Optional[float] = None
    total: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for metrics."""
        return {
            'extraction': self.extraction or 0.0,
            'suggestions': self.suggestions or 0.0,
            'generation': self.generation or 0.0,
            'total': self.total or 0.0
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowTimings':
        """Create from dictionary."""
        return cls(
            extraction=data.get('extraction'),
            suggestions=data.get('suggestions'),
            generation=data.get('generation'),
            total=data.get('total')
        )


@dataclass
class WorkflowState:
    """Complete state of a video generation workflow."""
    workflow_id: str
    source_url: str
    status: WorkflowStatus
    avatar: Optional[str] = None
    
    # Content and processing results
    content: Optional[ExtractedContent] = None
    suggestions: Optional[ContentSuggestions] = None
    video_url: Optional[str] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    timings: WorkflowTimings = field(default_factory=WorkflowTimings)
    
    # Performance tracking
    extractor_used: Optional[str] = None
    generator_used: Optional[str] = None
    
    # Error handling
    error: Optional[str] = None
    error_stage: Optional[str] = None
    
    # User customizations
    user_edits: Dict[str, Any] = field(default_factory=dict)
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'workflow_id': self.workflow_id,
            'source_url': self.source_url,
            'status': self.status.value,
            'avatar': self.avatar,
            'content': self.content.to_dict() if self.content else None,
            'suggestions': self.suggestions.to_dict() if self.suggestions else None,
            'video_url': self.video_url,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'timings': self.timings.to_dict(),
            'extractor_used': self.extractor_used,
            'generator_used': self.generator_used,
            'error': self.error,
            'error_stage': self.error_stage,
            'user_edits': self.user_edits,
            'config': self.config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Create from dictionary."""
        # Handle date fields
        created_at = datetime.fromisoformat(data['created_at'])
        updated_at = datetime.fromisoformat(data['updated_at'])
        
        # Handle nested objects
        content = None
        if data.get('content'):
            content = ExtractedContent.from_dict(data['content'])
        
        suggestions = None
        if data.get('suggestions'):
            suggestions = ContentSuggestions.from_dict(data['suggestions'])
        
        return cls(
            workflow_id=data['workflow_id'],
            source_url=data['source_url'],
            status=WorkflowStatus(data['status']),
            avatar=data.get('avatar'),
            content=content,
            suggestions=suggestions,
            video_url=data.get('video_url'),
            created_at=created_at,
            updated_at=updated_at,
            timings=WorkflowTimings.from_dict(data.get('timings', {})),
            extractor_used=data.get('extractor_used'),
            generator_used=data.get('generator_used'),
            error=data.get('error'),
            error_stage=data.get('error_stage'),
            user_edits=data.get('user_edits', {}),
            config=data.get('config', {})
        )


@dataclass
class WorkflowHooks:
    """Hooks for custom workflow behavior."""
    on_extraction_start: Optional[callable] = None
    on_extraction_complete: Optional[callable] = None
    on_suggestions_start: Optional[callable] = None
    on_suggestions_complete: Optional[callable] = None
    on_generation_start: Optional[callable] = None
    on_generation_complete: Optional[callable] = None
    on_workflow_complete: Optional[callable] = None
    on_workflow_failed: Optional[callable] = None


@dataclass
class PluginInfo:
    """Information about a plugin."""
    name: str
    version: str
    description: str
    author: str
    category: str
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    is_enabled: bool = True
    is_loaded: bool = False


@dataclass
class ComponentConfig:
    """Configuration for a component."""
    name: str
    type: str
    enabled: bool = True
    priority: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class SystemConfig:
    """System-wide configuration."""
    workflow: Dict[str, Any] = field(default_factory=dict)
    extractors: Dict[str, ComponentConfig] = field(default_factory=dict)
    suggestion_engines: Dict[str, ComponentConfig] = field(default_factory=dict)
    generators: Dict[str, ComponentConfig] = field(default_factory=dict)
    repositories: Dict[str, ComponentConfig] = field(default_factory=dict)
    metrics: Dict[str, ComponentConfig] = field(default_factory=dict)
    plugins: Dict[str, ComponentConfig] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict) 