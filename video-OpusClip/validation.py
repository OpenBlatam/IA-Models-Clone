"""
Improved Validation Module

Comprehensive input validation and sanitization with:
- Early returns and guard clauses
- Security validation and sanitization
- Performance optimizations
- Structured error handling
- Caching for validation results
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union, Tuple
import re
import time
import hashlib
import structlog
from urllib.parse import urlparse, parse_qs
from functools import lru_cache

from .models.improved_models import (
    VideoClipRequest,
    VideoClipBatchRequest,
    ViralVideoRequest,
    LangChainRequest,
    ValidationResult
)

logger = structlog.get_logger("validation")

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

class ValidationConfig:
    """Configuration for validation rules and limits."""
    
    # URL validation
    MAX_URL_LENGTH = 500
    ALLOWED_URL_SCHEMES = {'http', 'https'}
    YOUTUBE_DOMAINS = {
        'youtube.com', 'www.youtube.com', 'youtu.be', 'm.youtube.com',
        'music.youtube.com', 'gaming.youtube.com'
    }
    
    # Security patterns
    MALICIOUS_PATTERNS = [
        'javascript:', 'data:', 'vbscript:', 'file://', 'ftp://',
        'eval(', 'exec(', 'system(', '<script', 'onload=', 'onerror=',
        'onclick=', 'onmouseover=', 'onfocus=', 'onblur=',
        'document.cookie', 'document.write', 'window.location',
        'alert(', 'confirm(', 'prompt(', 'setTimeout(', 'setInterval('
    ]
    
    # Content validation
    MAX_TITLE_LENGTH = 200
    MAX_DESCRIPTION_LENGTH = 5000
    MAX_TAGS_COUNT = 50
    MAX_TAG_LENGTH = 50
    
    # Processing limits
    MAX_BATCH_SIZE = 100
    MAX_CLIP_LENGTH = 600  # 10 minutes
    MIN_CLIP_LENGTH = 5    # 5 seconds
    MAX_VARIANTS = 50
    MIN_VARIANTS = 1
    
    # Language codes
    SUPPORTED_LANGUAGES = {
        'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko',
        'ar', 'hi', 'th', 'vi', 'tr', 'pl', 'nl', 'sv', 'da', 'no'
    }

# Global validation configuration
validation_config = ValidationConfig()

# =============================================================================
# URL VALIDATION AND SANITIZATION
# =============================================================================

@lru_cache(maxsize=1000)
def is_valid_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL with caching."""
    if not url or not isinstance(url, str):
        return False
    
    # Length check
    if len(url) > validation_config.MAX_URL_LENGTH:
        return False
    
    try:
        parsed = urlparse(url.strip())
        
        # Scheme validation
        if parsed.scheme not in validation_config.ALLOWED_URL_SCHEMES:
            return False
        
        # Domain validation
        if parsed.netloc.lower() not in validation_config.YOUTUBE_DOMAINS:
            return False
        
        # Path validation for different YouTube URL formats
        if parsed.netloc.lower() in {'youtube.com', 'www.youtube.com'}:
            # Standard YouTube URLs: /watch?v=VIDEO_ID
            if parsed.path == '/watch':
                query_params = parse_qs(parsed.query)
                return 'v' in query_params and len(query_params['v'][0]) == 11
            # Embed URLs: /embed/VIDEO_ID
            elif parsed.path.startswith('/embed/'):
                video_id = parsed.path.split('/embed/')[-1]
                return len(video_id) == 11 and video_id.replace('-', '').replace('_', '').isalnum()
            # Short URLs: /v/VIDEO_ID
            elif parsed.path.startswith('/v/'):
                video_id = parsed.path.split('/v/')[-1]
                return len(video_id) == 11 and video_id.replace('-', '').replace('_', '').isalnum()
        
        elif parsed.netloc.lower() == 'youtu.be':
            # Short YouTube URLs: youtu.be/VIDEO_ID
            video_id = parsed.path.lstrip('/')
            return len(video_id) == 11 and video_id.replace('-', '').replace('_', '').isalnum()
        
        return False
        
    except Exception as e:
        logger.warning("URL validation error", url=url, error=str(e))
        return False

def sanitize_youtube_url(url: str) -> Optional[str]:
    """Sanitize and validate YouTube URL with security checks."""
    if not url or not isinstance(url, str):
        return None
    
    # Early return for empty or invalid input
    url = url.strip()
    if not url:
        return None
    
    # Security validation - early return
    if contains_malicious_content(url):
        logger.warning("Malicious content detected in URL", url=url)
        return None
    
    # Basic format validation - early return
    if not is_valid_youtube_url(url):
        return None
    
    # Normalize URL
    try:
        parsed = urlparse(url)
        
        # Reconstruct clean URL
        if parsed.netloc.lower() in {'youtube.com', 'www.youtube.com'}:
            if parsed.path == '/watch':
                query_params = parse_qs(parsed.query)
                if 'v' in query_params:
                    video_id = query_params['v'][0]
                    return f"https://www.youtube.com/watch?v={video_id}"
            elif parsed.path.startswith('/embed/'):
                video_id = parsed.path.split('/embed/')[-1]
                return f"https://www.youtube.com/watch?v={video_id}"
            elif parsed.path.startswith('/v/'):
                video_id = parsed.path.split('/v/')[-1]
                return f"https://www.youtube.com/watch?v={video_id}"
        
        elif parsed.netloc.lower() == 'youtu.be':
            video_id = parsed.path.lstrip('/')
            return f"https://www.youtube.com/watch?v={video_id}"
        
        return url
        
    except Exception as e:
        logger.warning("URL sanitization error", url=url, error=str(e))
        return None

def contains_malicious_content(text: str) -> bool:
    """Check for malicious content patterns."""
    if not text or not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in validation_config.MALICIOUS_PATTERNS)

# =============================================================================
# CONTENT VALIDATION
# =============================================================================

def validate_text_content(
    text: str,
    field_name: str,
    max_length: Optional[int] = None,
    min_length: int = 1,
    allow_empty: bool = False
) -> ValidationResult:
    """Validate text content with comprehensive checks."""
    errors = []
    warnings = []
    
    # Early return for None
    if text is None:
        if allow_empty:
            return ValidationResult(is_valid=True, errors=[], warnings=[])
        else:
            return ValidationResult(
                is_valid=False,
                errors=[f"{field_name} is required"],
                warnings=[]
            )
    
    # Type validation - early return
    if not isinstance(text, str):
        return ValidationResult(
            is_valid=False,
            errors=[f"{field_name} must be a string"],
            warnings=[]
        )
    
    # Length validation
    text = text.strip()
    
    if not text and not allow_empty:
        return ValidationResult(
            is_valid=False,
            errors=[f"{field_name} cannot be empty"],
            warnings=[]
        )
    
    if len(text) < min_length:
        errors.append(f"{field_name} must be at least {min_length} characters long")
    
    if max_length and len(text) > max_length:
        errors.append(f"{field_name} cannot exceed {max_length} characters")
    
    # Security validation - early return
    if contains_malicious_content(text):
        return ValidationResult(
            is_valid=False,
            errors=[f"{field_name} contains potentially malicious content"],
            warnings=[]
        )
    
    # Content quality warnings
    if len(text) > max_length * 0.8 if max_length else False:
        warnings.append(f"{field_name} is quite long, consider shortening")
    
    if text.count('\n') > 10:
        warnings.append(f"{field_name} has many line breaks, consider formatting")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_language_code(language: str) -> ValidationResult:
    """Validate language code."""
    if not language or not isinstance(language, str):
        return ValidationResult(
            is_valid=False,
            errors=["Language code is required"],
            warnings=[]
        )
    
    language = language.strip().lower()
    
    if language not in validation_config.SUPPORTED_LANGUAGES:
        return ValidationResult(
            is_valid=False,
            errors=[f"Unsupported language code: {language}"],
            warnings=[]
        )
    
    return ValidationResult(is_valid=True, errors=[], warnings=[])

# =============================================================================
# REQUEST VALIDATION
# =============================================================================

def validate_video_request(request: VideoClipRequest) -> ValidationResult:
    """Validate video clip request with comprehensive checks."""
    errors = []
    warnings = []
    
    # Early return for None request
    if not request:
        return ValidationResult(
            is_valid=False,
            errors=["Request object is required"],
            warnings=[]
        )
    
    # URL validation - early return
    url_validation = validate_youtube_url(request.youtube_url)
    if not url_validation.is_valid:
        errors.extend(url_validation.errors)
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
    
    # Language validation - early return
    language_validation = validate_language_code(request.language)
    if not language_validation.is_valid:
        errors.extend(language_validation.errors)
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
    
    # Clip length validation
    if request.min_clip_length < validation_config.MIN_CLIP_LENGTH:
        errors.append(f"Minimum clip length must be at least {validation_config.MIN_CLIP_LENGTH} seconds")
    
    if request.max_clip_length > validation_config.MAX_CLIP_LENGTH:
        errors.append(f"Maximum clip length cannot exceed {validation_config.MAX_CLIP_LENGTH} seconds")
    
    if request.min_clip_length > request.max_clip_length:
        errors.append("Minimum clip length cannot be greater than maximum clip length")
    
    # Quality and format validation
    if request.quality not in ['low', 'medium', 'high', 'ultra']:
        errors.append("Invalid video quality specified")
    
    if request.format not in ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv']:
        errors.append("Invalid video format specified")
    
    # Priority validation
    if request.priority not in ['low', 'normal', 'high', 'urgent']:
        errors.append("Invalid priority level specified")
    
    # Custom parameters validation
    if request.custom_params:
        custom_validation = validate_custom_parameters(request.custom_params)
        if not custom_validation.is_valid:
            errors.extend(custom_validation.errors)
        warnings.extend(custom_validation.warnings)
    
    # Performance warnings
    if request.max_clip_length > 300:
        warnings.append("Long clip length may impact processing time")
    
    if request.quality == 'ultra':
        warnings.append("Ultra quality may require significant processing time")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_batch_request(request: VideoClipBatchRequest) -> ValidationResult:
    """Validate batch video request with comprehensive checks."""
    errors = []
    warnings = []
    
    # Early return for None request
    if not request:
        return ValidationResult(
            is_valid=False,
            errors=["Batch request object is required"],
            warnings=[]
        )
    
    # Batch size validation - early return
    if not request.requests or not isinstance(request.requests, list):
        return ValidationResult(
            is_valid=False,
            errors=["Requests list is required and must be a list"],
            warnings=[]
        )
    
    if len(request.requests) == 0:
        return ValidationResult(
            is_valid=False,
            errors=["Batch cannot be empty"],
            warnings=[]
        )
    
    if len(request.requests) > validation_config.MAX_BATCH_SIZE:
        return ValidationResult(
            is_valid=False,
            errors=[f"Batch size cannot exceed {validation_config.MAX_BATCH_SIZE}"],
            warnings=[]
        )
    
    # Validate each request in batch
    url_set = set()
    for i, req in enumerate(request.requests):
        if not req:
            errors.append(f"Request at index {i} is null")
            continue
        
        # Validate individual request
        req_validation = validate_video_request(req)
        if not req_validation.is_valid:
            errors.extend([f"Request {i}: {error}" for error in req_validation.errors])
        warnings.extend([f"Request {i}: {warning}" for warning in req_validation.warnings])
        
        # Check for duplicate URLs
        if req.youtube_url in url_set:
            errors.append(f"Duplicate YouTube URL at index {i}: {req.youtube_url}")
        else:
            url_set.add(req.youtube_url)
    
    # Batch configuration validation
    if request.max_workers < 1 or request.max_workers > 32:
        errors.append("Max workers must be between 1 and 32")
    
    if request.timeout < 30 or request.timeout > 3600:
        errors.append("Timeout must be between 30 and 3600 seconds")
    
    # Performance warnings
    if len(request.requests) > 50:
        warnings.append("Large batch size may impact processing time")
    
    if request.max_workers > 16:
        warnings.append("High worker count may impact system resources")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_viral_request(request: ViralVideoRequest) -> ValidationResult:
    """Validate viral video request with comprehensive checks."""
    errors = []
    warnings = []
    
    # Early return for None request
    if not request:
        return ValidationResult(
            is_valid=False,
            errors=["Request object is required"],
            warnings=[]
        )
    
    # URL validation - early return
    url_validation = validate_youtube_url(request.youtube_url)
    if not url_validation.is_valid:
        errors.extend(url_validation.errors)
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
    
    # Variants count validation - early return
    if request.n_variants < validation_config.MIN_VARIANTS:
        errors.append(f"Number of variants must be at least {validation_config.MIN_VARIANTS}")
    
    if request.n_variants > validation_config.MAX_VARIANTS:
        errors.append(f"Number of variants cannot exceed {validation_config.MAX_VARIANTS}")
    
    # Language validation - early return
    language_validation = validate_language_code(request.language)
    if not language_validation.is_valid:
        errors.extend(language_validation.errors)
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
    
    # Platform validation
    if request.platform not in ['youtube', 'tiktok', 'instagram', 'twitter', 'linkedin']:
        errors.append("Invalid platform specified")
    
    # Audience profile validation
    if request.audience_profile:
        profile_validation = validate_audience_profile(request.audience_profile)
        if not profile_validation.is_valid:
            errors.extend(profile_validation.errors)
        warnings.extend(profile_validation.warnings)
    
    # Performance warnings
    if request.n_variants > 20:
        warnings.append("High number of variants may impact processing time")
    
    if request.use_langchain and request.n_variants > 10:
        warnings.append("LangChain processing with many variants may be slow")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_langchain_request(request: LangChainRequest) -> ValidationResult:
    """Validate LangChain request with comprehensive checks."""
    errors = []
    warnings = []
    
    # Early return for None request
    if not request:
        return ValidationResult(
            is_valid=False,
            errors=["Request object is required"],
            warnings=[]
        )
    
    # URL validation - early return
    url_validation = validate_youtube_url(request.youtube_url)
    if not url_validation.is_valid:
        errors.extend(url_validation.errors)
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
    
    # Analysis type validation
    if request.analysis_type not in ['content', 'engagement', 'viral', 'optimization', 'comprehensive']:
        errors.append("Invalid analysis type specified")
    
    # Language validation - early return
    language_validation = validate_language_code(request.language)
    if not language_validation.is_valid:
        errors.extend(language_validation.errors)
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
    
    # Platform validation
    if request.platform not in ['youtube', 'tiktok', 'instagram', 'twitter', 'linkedin']:
        errors.append("Invalid platform specified")
    
    # Audience profile validation
    if request.audience_profile:
        profile_validation = validate_audience_profile(request.audience_profile)
        if not profile_validation.is_valid:
            errors.extend(profile_validation.errors)
        warnings.extend(profile_validation.warnings)
    
    # Performance warnings
    if request.analysis_type == 'comprehensive':
        warnings.append("Comprehensive analysis may take longer to process")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

# =============================================================================
# HELPER VALIDATION FUNCTIONS
# =============================================================================

def validate_youtube_url(url: str) -> ValidationResult:
    """Validate YouTube URL with detailed error messages."""
    if not url or not isinstance(url, str):
        return ValidationResult(
            is_valid=False,
            errors=["YouTube URL is required"],
            warnings=[]
        )
    
    url = url.strip()
    if not url:
        return ValidationResult(
            is_valid=False,
            errors=["YouTube URL cannot be empty"],
            warnings=[]
        )
    
    # Security check - early return
    if contains_malicious_content(url):
        return ValidationResult(
            is_valid=False,
            errors=["Potentially malicious URL detected"],
            warnings=[]
        )
    
    # Format validation
    if not is_valid_youtube_url(url):
        return ValidationResult(
            is_valid=False,
            errors=["Invalid YouTube URL format"],
            warnings=[]
        )
    
    return ValidationResult(is_valid=True, errors=[], warnings=[])

def validate_custom_parameters(params: Dict[str, Any]) -> ValidationResult:
    """Validate custom parameters."""
    errors = []
    warnings = []
    
    if not isinstance(params, dict):
        return ValidationResult(
            is_valid=False,
            errors=["Custom parameters must be a dictionary"],
            warnings=[]
        )
    
    # Check parameter count
    if len(params) > 20:
        warnings.append("Many custom parameters may impact processing")
    
    # Validate parameter names and values
    for key, value in params.items():
        if not isinstance(key, str) or len(key) > 50:
            errors.append(f"Invalid parameter name: {key}")
        
        if isinstance(value, str) and contains_malicious_content(value):
            errors.append(f"Parameter '{key}' contains potentially malicious content")
        
        if isinstance(value, str) and len(value) > 1000:
            warnings.append(f"Parameter '{key}' has a very long value")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_audience_profile(profile: Dict[str, Any]) -> ValidationResult:
    """Validate audience profile."""
    errors = []
    warnings = []
    
    if not isinstance(profile, dict):
        return ValidationResult(
            is_valid=False,
            errors=["Audience profile must be a dictionary"],
            warnings=[]
        )
    
    # Validate profile fields
    allowed_fields = {
        'age_range', 'interests', 'location', 'language', 'platform',
        'engagement_level', 'content_preferences', 'demographics'
    }
    
    for key in profile.keys():
        if key not in allowed_fields:
            warnings.append(f"Unknown audience profile field: {key}")
    
    # Validate specific fields
    if 'age_range' in profile:
        age_range = profile['age_range']
        if not isinstance(age_range, (list, tuple)) or len(age_range) != 2:
            errors.append("Age range must be a list/tuple with 2 elements")
        elif not all(isinstance(x, int) and 13 <= x <= 100 for x in age_range):
            errors.append("Age range values must be integers between 13 and 100")
    
    if 'interests' in profile:
        interests = profile['interests']
        if not isinstance(interests, list):
            errors.append("Interests must be a list")
        elif len(interests) > 20:
            warnings.append("Too many interests specified")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

# =============================================================================
# SYSTEM HEALTH VALIDATION
# =============================================================================

def validate_system_health() -> ValidationResult:
    """Validate system health for processing."""
    errors = []
    warnings = []
    
    try:
        import psutil
        
        # Memory check
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            errors.append("System memory usage is critically high")
        elif memory.percent > 80:
            warnings.append("System memory usage is high")
        
        # Disk space check
        disk = psutil.disk_usage('/')
        if disk.percent > 95:
            errors.append("Disk space is critically low")
        elif disk.percent > 85:
            warnings.append("Disk space is low")
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 95:
            warnings.append("CPU usage is very high")
        
    except ImportError:
        warnings.append("psutil not available, cannot check system health")
    except Exception as e:
        warnings.append(f"System health check failed: {str(e)}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

def validate_gpu_health() -> ValidationResult:
    """Validate GPU health for processing."""
    errors = []
    warnings = []
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # Check GPU memory
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                
                memory_usage = (memory_allocated + memory_reserved) / memory_total
                
                if memory_usage > 0.95:
                    errors.append(f"GPU {i} memory usage is critically high")
                elif memory_usage > 0.85:
                    warnings.append(f"GPU {i} memory usage is high")
        else:
            warnings.append("CUDA not available, GPU processing disabled")
    
    except ImportError:
        warnings.append("PyTorch not available, cannot check GPU health")
    except Exception as e:
        warnings.append(f"GPU health check failed: {str(e)}")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )

# =============================================================================
# VALIDATION CACHING
# =============================================================================

class ValidationCache:
    """Cache for validation results to improve performance."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.cache: Dict[str, Tuple[ValidationResult, float]] = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def _generate_key(self, data: Any) -> str:
        """Generate cache key for data."""
        if isinstance(data, dict):
            # Sort keys for consistent hashing
            sorted_data = {k: data[k] for k in sorted(data.keys())}
            data_str = str(sorted_data)
        else:
            data_str = str(data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get(self, data: Any) -> Optional[ValidationResult]:
        """Get cached validation result."""
        key = self._generate_key(data)
        
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return result
            else:
                # Expired, remove from cache
                del self.cache[key]
        
        return None
    
    def set(self, data: Any, result: ValidationResult) -> None:
        """Cache validation result."""
        key = self._generate_key(data)
        
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (result, time.time())
    
    def clear(self) -> None:
        """Clear validation cache."""
        self.cache.clear()

# Global validation cache
validation_cache = ValidationCache()

def validate_with_cache(validation_func, data: Any) -> ValidationResult:
    """Validate data with caching."""
    # Check cache first
    cached_result = validation_cache.get(data)
    if cached_result:
        return cached_result
    
    # Perform validation
    result = validation_func(data)
    
    # Cache result
    validation_cache.set(data, result)
    
    return result

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    'ValidationConfig',
    'validation_config',
    
    # URL validation
    'is_valid_youtube_url',
    'sanitize_youtube_url',
    'contains_malicious_content',
    
    # Content validation
    'validate_text_content',
    'validate_language_code',
    
    # Request validation
    'validate_video_request',
    'validate_batch_request',
    'validate_viral_request',
    'validate_langchain_request',
    
    # Helper validation
    'validate_youtube_url',
    'validate_custom_parameters',
    'validate_audience_profile',
    
    # System validation
    'validate_system_health',
    'validate_gpu_health',
    
    # Caching
    'ValidationCache',
    'validation_cache',
    'validate_with_cache'
]