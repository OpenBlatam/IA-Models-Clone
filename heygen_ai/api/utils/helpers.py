from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import time
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from ..models.schemas import QualityLevel, LanguageCode, VideoStatus, ProcessingSettings
    import re
    import re
    import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Helper functions for HeyGen AI API
Provides utility functions with comprehensive type hints and Pydantic model integration.
"""



# Global variables for tracking
_start_time: float = time.time()
_system_version: str = "1.0.0"


def generate_video_id(user_id: str) -> str:
    """Generate unique video ID for user (pure function)"""
    timestamp: int = int(time.time())
    unique_suffix: str = str(uuid.uuid4())[:8]
    return f"video_{timestamp}_{user_id}_{unique_suffix}"


def calculate_estimated_duration(quality: Union[str, QualityLevel]) -> int:
    """Calculate estimated processing time in seconds (pure function)"""
    quality_str: str = quality.value if isinstance(quality, QualityLevel) else quality
    quality_durations: Dict[str, int] = {
        QualityLevel.LOW.value: 30,
        QualityLevel.MEDIUM.value: 60,
        QualityLevel.HIGH.value: 120
    }
    return quality_durations.get(quality_str, 60)


def get_system_version() -> str:
    """Get current system version (pure function)"""
    return _system_version


def get_uptime() -> Dict[str, Any]:
    """Get system uptime information (pure function)"""
    current_time: float = time.time()
    uptime_seconds: int = int(current_time - _start_time)
    
    return {
        "uptime_seconds": uptime_seconds,
        "uptime_formatted": str(timedelta(seconds=uptime_seconds)),
        "start_time": datetime.fromtimestamp(_start_time).isoformat(),
        "current_time": datetime.fromtimestamp(current_time).isoformat()
    }


def format_timestamp(timestamp: float) -> str:
    """Format timestamp to ISO format (pure function)"""
    return datetime.fromtimestamp(timestamp).isoformat()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations (pure function)"""
    # Remove or replace unsafe characters
    unsafe_chars: str = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        name: str
        ext: str
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:255-len(ext)-1] + '.' + ext if ext else name[:255]
    
    return filename


def create_output_directory(base_path: str, subdirectory: Optional[str] = None) -> Path:
    """Create output directory with proper structure (pure function)"""
    output_path: Path = Path(base_path)
    
    if subdirectory:
        output_path = output_path / subdirectory
    
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def generate_file_hash(content: bytes) -> str:
    """Generate SHA-256 hash for file content (pure function)"""
    return hashlib.sha256(content).hexdigest()


def parse_quality_settings(quality: Union[str, QualityLevel]) -> Dict[str, Any]:
    """Parse quality settings for video processing (pure function)"""
    quality_str: str = quality.value if isinstance(quality, QualityLevel) else quality
    quality_configs: Dict[str, Dict[str, Any]] = {
        QualityLevel.LOW.value: {
            "num_inference_steps": 20,
            "guidance_scale": 7.0,
            "fps": 8,
            "resolution": (512, 512)
        },
        QualityLevel.MEDIUM.value: {
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "fps": 12,
            "resolution": (768, 768)
        },
        QualityLevel.HIGH.value: {
            "num_inference_steps": 100,
            "guidance_scale": 8.0,
            "fps": 24,
            "resolution": (1024, 1024)
        }
    }
    
    return quality_configs.get(quality_str, quality_configs[QualityLevel.MEDIUM.value])


def validate_file_path(file_path: str) -> bool:
    """Validate if file path is safe and accessible (pure function)"""
    try:
        path: Path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False


def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes (pure function)"""
    try:
        size_bytes: int = Path(file_path).stat().st_size
        return size_bytes / (1024 * 1024)
    except Exception:
        return 0.0


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format (pure function)"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def calculate_progress(processing_time: float, estimated_duration: float) -> float:
    """Calculate processing progress percentage (pure function)"""
    if estimated_duration <= 0:
        return 0.0
    
    progress: float = min((processing_time / estimated_duration) * 100, 95.0)
    return round(progress, 1)


def generate_thumbnail_url(video_id: str) -> str:
    """Generate thumbnail URL for video (pure function)"""
    return f"/api/v1/videos/{video_id}/thumbnail"


def validate_video_id_format(video_id: str) -> bool:
    """Validate video ID format (pure function)"""
    pattern = re.compile(r'^video_\d+_[a-zA-Z0-9_-]+$')
    return bool(pattern.match(video_id))


def extract_video_metadata(file_path: str) -> Dict[str, Any]:
    """Extract video metadata (pure function)"""
    try:
        # In production, use ffmpeg or similar to extract metadata
        return {
            "duration": None,
            "resolution": None,
            "fps": None,
            "codec": None
        }
    except Exception:
        return {}


def calculate_processing_cost(quality: Union[str, QualityLevel], duration: float) -> float:
    """Calculate processing cost based on quality and duration (pure function)"""
    quality_str: str = quality.value if isinstance(quality, QualityLevel) else quality
    cost_per_minute: Dict[str, float] = {
        QualityLevel.LOW.value: 0.10,
        QualityLevel.MEDIUM.value: 0.25,
        QualityLevel.HIGH.value: 0.50
    }
    
    base_cost: float = cost_per_minute.get(quality_str, 0.25)
    return round(base_cost * (duration / 60), 2)


async def generate_request_id() -> str:
    """Generate unique request ID (pure function)"""
    timestamp: int = int(time.time() * 1000)
    random_suffix: str = str(uuid.uuid4())[:8]
    return f"req_{timestamp}_{random_suffix}"


def validate_email_format(email: str) -> bool:
    """Validate email format (pure function)"""
    pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    return bool(pattern.match(email))


def validate_username_format(username: str) -> bool:
    """Validate username format (pure function)"""
    pattern = re.compile(r'^[a-zA-Z0-9_-]{3,50}$')
    return bool(pattern.match(username))


async def generate_api_key() -> str:
    """Generate secure API key (pure function)"""
    return hashlib.sha256(uuid.uuid4().bytes).hexdigest()


def calculate_rate_limit_key(user_id: str, endpoint: str) -> str:
    """Calculate rate limit key (pure function)"""
    return f"rate_limit:{user_id}:{endpoint}"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format (pure function)"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes: float = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours: float = seconds / 3600
        return f"{hours:.1f}h"


def validate_script_length(script: str, min_length: int = 10, max_length: int = 1000) -> bool:
    """Validate script length (pure function)"""
    return min_length <= len(script.strip()) <= max_length


def extract_keywords_from_script(script: str) -> List[str]:
    """Extract keywords from script (pure function)"""
    # Simple keyword extraction - in production use NLP libraries
    words: List[str] = script.lower().split()
    stop_words: set[str] = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    keywords: List[str] = [word for word in words if word not in stop_words and len(word) > 3]
    return list(set(keywords))[:10]  # Return top 10 unique keywords


def calculate_script_complexity(script: str) -> str:
    """Calculate script complexity level (pure function)"""
    word_count: int = len(script.split())
    avg_word_length: float = sum(len(word) for word in script.split()) / word_count if word_count > 0 else 0
    
    if word_count < 50 and avg_word_length < 5:
        return "simple"
    elif word_count < 200 and avg_word_length < 7:
        return "medium"
    else:
        return "complex"


def create_processing_settings(quality: QualityLevel, custom_settings: Optional[ProcessingSettings] = None) -> ProcessingSettings:
    """Create processing settings based on quality and custom settings (pure function)"""
    if custom_settings:
        return custom_settings
    
    # Default settings based on quality
    default_settings: Dict[QualityLevel, ProcessingSettings] = {
        QualityLevel.LOW: ProcessingSettings(
            num_inference_steps=20,
            guidance_scale=7.0,
            fps=8,
            resolution=(512, 512)
        ),
        QualityLevel.MEDIUM: ProcessingSettings(
            num_inference_steps=50,
            guidance_scale=7.5,
            fps=12,
            resolution=(768, 768)
        ),
        QualityLevel.HIGH: ProcessingSettings(
            num_inference_steps=100,
            guidance_scale=8.0,
            fps=24,
            resolution=(1024, 1024)
        )
    }
    
    return default_settings.get(quality, default_settings[QualityLevel.MEDIUM])


def validate_video_parameters(
    script: str,
    voice_id: str,
    language: str,
    quality: str,
    duration: Optional[int] = None
) -> Tuple[bool, List[str]]:
    """Validate video generation parameters (pure function)"""
    errors: List[str] = []
    
    # Validate script
    if not validate_script_length(script):
        errors.append("Script length must be between 10 and 1000 characters")
    
    # Validate voice_id
    valid_voices: List[str] = ["Voice 1", "Voice 2", "Voice 3", "Voice 4", "Voice 5"]
    if voice_id not in valid_voices:
        errors.append(f"Voice ID must be one of: {', '.join(valid_voices)}")
    
    # Validate language
    valid_languages: List[str] = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]
    if language not in valid_languages:
        errors.append(f"Language must be one of: {', '.join(valid_languages)}")
    
    # Validate quality
    valid_qualities: List[str] = [QualityLevel.LOW.value, QualityLevel.MEDIUM.value, QualityLevel.HIGH.value]
    if quality not in valid_qualities:
        errors.append(f"Quality must be one of: {', '.join(valid_qualities)}")
    
    # Validate duration
    if duration is not None:
        if duration < 5 or duration > 300:
            errors.append("Duration must be between 5 and 300 seconds")
        if quality == QualityLevel.LOW.value and duration > 120:
            errors.append("Low quality videos cannot exceed 120 seconds")
    
    return len(errors) == 0, errors


def create_video_metadata(
    script: str,
    voice_id: str,
    language: str,
    quality: str,
    duration: Optional[int] = None
) -> Dict[str, Any]:
    """Create video metadata dictionary (pure function)"""
    return {
        "script_length": len(script),
        "script_complexity": calculate_script_complexity(script),
        "keywords": extract_keywords_from_script(script),
        "voice_id": voice_id,
        "language": language,
        "quality": quality,
        "duration": duration,
        "estimated_cost": calculate_processing_cost(quality, duration or 60),
        "created_at": datetime.utcnow().isoformat()
    }


def format_error_message(error_code: str, error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Format error message for API response (pure function)"""
    error_response: Dict[str, Any] = {
        "error_code": error_code,
        "error_type": error_type,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if details:
        error_response["details"] = details
    
    return error_response


def validate_pagination_parameters(page: int, page_size: int) -> Tuple[bool, List[str]]:
    """Validate pagination parameters (pure function)"""
    errors: List[str] = []
    
    if page < 1:
        errors.append("Page must be greater than 0")
    
    if page_size < 1 or page_size > 100:
        errors.append("Page size must be between 1 and 100")
    
    return len(errors) == 0, errors


def calculate_pagination_info(total_items: int, page: int, page_size: int) -> Dict[str, Any]:
    """Calculate pagination information (pure function)"""
    total_pages: int = (total_items + page_size - 1) // page_size
    offset: int = (page - 1) * page_size
    has_next: bool = page < total_pages
    has_prev: bool = page > 1
    
    return {
        "current_page": page,
        "page_size": page_size,
        "total_items": total_items,
        "total_pages": total_pages,
        "offset": offset,
        "has_next": has_next,
        "has_prev": has_prev,
        "next_page": page + 1 if has_next else None,
        "prev_page": page - 1 if has_prev else None
    }


# Named exports
__all__ = [
    "generate_video_id",
    "calculate_estimated_duration", 
    "get_system_version",
    "get_uptime",
    "format_timestamp",
    "sanitize_filename",
    "create_output_directory",
    "generate_file_hash",
    "parse_quality_settings",
    "validate_file_path",
    "get_file_size_mb",
    "format_file_size",
    "calculate_progress",
    "generate_thumbnail_url",
    "validate_video_id_format",
    "extract_video_metadata",
    "calculate_processing_cost",
    "generate_request_id",
    "validate_email_format",
    "validate_username_format",
    "generate_api_key",
    "calculate_rate_limit_key",
    "format_duration",
    "validate_script_length",
    "extract_keywords_from_script",
    "calculate_script_complexity",
    "create_processing_settings",
    "validate_video_parameters",
    "create_video_metadata",
    "format_error_message",
    "validate_pagination_parameters",
    "calculate_pagination_info"
] 