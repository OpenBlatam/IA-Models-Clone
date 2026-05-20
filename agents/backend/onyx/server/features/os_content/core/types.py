from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Type definitions for OS Content UGC Video Generator
Centralized type definitions and data structures
"""


class ProcessingStatus(str, Enum):
    """Video processing status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class ContentType(str, Enum):
    """Content types"""
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"

class Language(str, Enum):
    """Supported languages"""
    SPANISH = "es"
    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"

@dataclass
class VideoRequest:
    """Video processing request"""
    id: str
    user_id: str
    title: str
    text_prompt: str
    image_urls: List[str] = field(default_factory=list)
    video_urls: List[str] = field(default_factory=list)
    language: Language = Language.SPANISH
    duration_per_image: float = 3.0
    resolution: tuple = (1080, 1920)
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VideoResponse:
    """Video processing response"""
    request_id: str
    video_url: str
    status: ProcessingStatus
    created_at: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    progress: Optional[float] = None
    estimated_duration: Optional[float] = None
    nlp_analysis: Optional[Dict[str, Any]] = None

@dataclass
class ProcessingTask:
    """Processing task definition"""
    id: str
    request: VideoRequest
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: Optional[float] = None
    retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

@dataclass
class CacheEntry:
    """Cache entry definition"""
    key: str
    value: Any
    ttl: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0

@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class FileInfo:
    """File information"""
    path: Union[str, Path]
    size: int
    content_type: str
    extension: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    checksum: Optional[str] = None

@dataclass
class NLPResult:
    """NLP analysis result"""
    text: str
    language: Language
    entities: List[Dict[str, Any]] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    sentiment: Optional[Dict[str, Any]] = None
    keywords: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    confidence: float = 0.0
    processing_time: float = 0.0

@dataclass
class CDNInfo:
    """CDN information"""
    url: str
    content_id: str
    content_type: ContentType
    size: int
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    cache_hit: bool = False

@dataclass
class LoadBalancerStats:
    """Load balancer statistics"""
    total_requests: int = 0
    failed_requests: int = 0
    healthy_backends: int = 0
    total_backends: int = 0
    average_response_time: float = 0.0
    algorithm: str = "round_robin"
    uptime: float = 0.0

@dataclass
class SystemHealth:
    """System health status"""
    status: str = "healthy"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    components: Dict[str, str] = field(default_factory=dict)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    errors: List[str] = field(default_factory=list)

# Type aliases for better readability
VideoPath = Union[str, Path]
CacheKey = str
TaskId = str
UserId = str
RequestId = str

# Validation functions
async def validate_video_request(request: VideoRequest) -> bool:
    """Validate video request"""
    if not request.id or not request.user_id or not request.title:
        return False
    
    if not request.text_prompt.strip():
        return False
    
    if not request.image_urls and not request.video_urls:
        return False
    
    if request.duration_per_image <= 0:
        return False
    
    return True

def validate_file_info(file_info: FileInfo) -> bool:
    """Validate file information"""
    if not file_info.path or file_info.size <= 0:
        return False
    
    if not file_info.content_type or not file_info.extension:
        return False
    
    return True

def validate_nlp_result(result: NLPResult) -> bool:
    """Validate NLP result"""
    if not result.text or not result.language:
        return False
    
    if result.confidence < 0 or result.confidence > 1:
        return False
    
    return True 