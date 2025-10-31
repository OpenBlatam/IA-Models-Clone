"""
Video Processing Constants

Centralized configuration and constants for the video processing module.
"""

import re

# =============================================================================
# PERFORMANCE THRESHOLDS
# =============================================================================

SLOW_OPERATION_THRESHOLD = 0.1  # seconds
CACHE_SIZE_URLS = 128
CACHE_SIZE_LANGUAGES = 64
DEFAULT_MAX_WORKERS = 4

# =============================================================================
# DEPENDENCY FLAGS
# =============================================================================

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    sentry_sdk = None
    SENTRY_AVAILABLE = False

# =============================================================================
# REGEX PATTERNS
# =============================================================================

URL_PATTERN = re.compile(r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+')
LANGUAGE_PATTERN = re.compile(r'^[a-z]{2}(-[A-Z]{2})?$')

# =============================================================================
# VALIDATION RANGES
# =============================================================================

MIN_CLIP_LENGTH = 5  # seconds
MAX_CLIP_LENGTH = 300  # seconds
MIN_VIRAL_SCORE = 0.0
MAX_VIRAL_SCORE = 1.0

# =============================================================================
# DEFAULT VALUES
# =============================================================================

DEFAULT_LANGUAGE = "en"
DEFAULT_MAX_CLIP_LENGTH = 60
DEFAULT_MIN_CLIP_LENGTH = 15
DEFAULT_N_VARIANTS = 10

# =============================================================================
# FILE EXTENSIONS
# =============================================================================

SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.gif', '.webp']
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.aac', '.ogg']

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_MESSAGES = {
    'invalid_youtube_url': "Invalid YouTube URL: {url}",
    'invalid_language': "Invalid language code: {lang}",
    'invalid_clip_times': "Start time must be before end time",
    'invalid_viral_score': "Viral score must be between 0 and 1",
    'empty_caption': "Caption cannot be empty",
    'pandas_not_installed': "pandas is not installed",
    'duplicate_key': "Duplicate key found: {key}",
}

# =============================================================================
# METRICS NAMES
# =============================================================================

METRIC_NAMES = {
    'batch_operation': 'video_model_{operation}_total',
    'batch_duration': 'video_model_{operation}_duration',
    'validation_error': 'video_model_validation_error_total',
    'processing_error': 'video_model_processing_error_total',
}

# =============================================================================
# LOGGING LEVELS
# =============================================================================

LOG_LEVELS = {
    'debug': 10,
    'info': 20,
    'warning': 30,
    'error': 40,
    'critical': 50,
}

# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

CACHE_CONFIG = {
    'url_validation': {
        'maxsize': CACHE_SIZE_URLS,
        'ttl': 3600,  # 1 hour
    },
    'language_validation': {
        'maxsize': CACHE_SIZE_LANGUAGES,
        'ttl': 86400,  # 24 hours
    },
    'batch_operations': {
        'maxsize': 256,
        'ttl': 1800,  # 30 minutes
    },
} 