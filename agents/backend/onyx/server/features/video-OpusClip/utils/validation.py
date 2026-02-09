"""
Video Processing Validation

Cached and optimized validation functions for video processing.
"""

import re
from functools import lru_cache
from typing import Optional

from .constants import (
    CACHE_SIZE_URLS,
    CACHE_SIZE_LANGUAGES,
    URL_PATTERN,
    LANGUAGE_PATTERN,
    MIN_VIRAL_SCORE,
    MAX_VIRAL_SCORE,
    ERROR_MESSAGES
)

# =============================================================================
# URL VALIDATION
# =============================================================================

@lru_cache(maxsize=CACHE_SIZE_URLS)
def _validate_youtube_url(url: str) -> bool:
    """
    Cached URL validation using pre-compiled regex.
    
    Args:
        url: YouTube URL to validate
        
    Returns:
        bool: True if valid YouTube URL, False otherwise
    """
    return bool(URL_PATTERN.match(url))

def validate_youtube_url(url: str) -> None:
    """
    Validate YouTube URL and raise ValueError if invalid.
    
    Args:
        url: YouTube URL to validate
        
    Raises:
        ValueError: If URL is invalid
    """
    if not _validate_youtube_url(url):
        raise ValueError(ERROR_MESSAGES['invalid_youtube_url'].format(url=url))

# =============================================================================
# LANGUAGE VALIDATION
# =============================================================================

@lru_cache(maxsize=CACHE_SIZE_LANGUAGES)
def _validate_language(lang: str) -> bool:
    """
    Cached language validation using pre-compiled regex.
    
    Args:
        lang: Language code to validate (e.g., 'en', 'es-MX')
        
    Returns:
        bool: True if valid language code, False otherwise
    """
    return bool(LANGUAGE_PATTERN.match(lang))

def validate_language(lang: str) -> None:
    """
    Validate language code and raise ValueError if invalid.
    
    Args:
        lang: Language code to validate
        
    Raises:
        ValueError: If language code is invalid
    """
    if not _validate_language(lang):
        raise ValueError(ERROR_MESSAGES['invalid_language'].format(lang=lang))

# =============================================================================
# VIDEO CLIP VALIDATION
# =============================================================================

def _validate_video_clip_times(start: float, end: float) -> None:
    """
    Validate video clip time constraints.
    
    Args:
        start: Start time in seconds
        end: End time in seconds
        
    Raises:
        ValueError: If start >= end
    """
    if start >= end:
        raise ValueError(ERROR_MESSAGES['invalid_clip_times'])

def validate_video_clip_times(start: float, end: float) -> None:
    """
    Validate video clip time constraints with additional checks.
    
    Args:
        start: Start time in seconds
        end: End time in seconds
        
    Raises:
        ValueError: If times are invalid
    """
    if start < 0 or end < 0:
        raise ValueError("Start and end times must be non-negative")
    
    _validate_video_clip_times(start, end)

# =============================================================================
# VIRAL SCORE VALIDATION
# =============================================================================

def _validate_viral_score(score: float) -> None:
    """
    Validate viral score is within valid range.
    
    Args:
        score: Viral score to validate
        
    Raises:
        ValueError: If score is outside valid range
    """
    if not MIN_VIRAL_SCORE <= score <= MAX_VIRAL_SCORE:
        raise ValueError(ERROR_MESSAGES['invalid_viral_score'])

def validate_viral_score(score: float) -> None:
    """
    Validate viral score with type checking.
    
    Args:
        score: Viral score to validate
        
    Raises:
        ValueError: If score is invalid
        TypeError: If score is not a number
    """
    if not isinstance(score, (int, float)):
        raise TypeError("Viral score must be a number")
    
    _validate_viral_score(score)

# =============================================================================
# CAPTION VALIDATION
# =============================================================================

def _validate_caption(caption: str) -> None:
    """
    Validate caption is not empty.
    
    Args:
        caption: Caption text to validate
        
    Raises:
        ValueError: If caption is empty
    """
    if not caption.strip():
        raise ValueError(ERROR_MESSAGES['empty_caption'])

def validate_caption(caption: str) -> None:
    """
    Validate caption with type checking and length limits.
    
    Args:
        caption: Caption text to validate
        
    Raises:
        ValueError: If caption is invalid
        TypeError: If caption is not a string
    """
    if not isinstance(caption, str):
        raise TypeError("Caption must be a string")
    
    if len(caption) > 1000:  # Reasonable limit for captions
        raise ValueError("Caption too long (max 1000 characters)")
    
    _validate_caption(caption)

# =============================================================================
# COMPOSITE VALIDATION
# =============================================================================

def validate_video_request_data(
    youtube_url: str,
    language: str,
    max_clip_length: Optional[int] = None,
    min_clip_length: Optional[int] = None
) -> None:
    """
    Validate all video request data at once.
    
    Args:
        youtube_url: YouTube URL to validate
        language: Language code to validate
        max_clip_length: Maximum clip length (optional)
        min_clip_length: Minimum clip length (optional)
        
    Raises:
        ValueError: If any validation fails
    """
    validate_youtube_url(youtube_url)
    validate_language(language)
    
    if max_clip_length is not None:
        if not isinstance(max_clip_length, int) or max_clip_length <= 0:
            raise ValueError("max_clip_length must be a positive integer")
    
    if min_clip_length is not None:
        if not isinstance(min_clip_length, int) or min_clip_length <= 0:
            raise ValueError("min_clip_length must be a positive integer")
    
    if max_clip_length is not None and min_clip_length is not None:
        if min_clip_length >= max_clip_length:
            raise ValueError("min_clip_length must be less than max_clip_length")

def validate_viral_variant_data(
    start: float,
    end: float,
    caption: str,
    viral_score: float,
    variant_id: str
) -> None:
    """
    Validate all viral variant data at once.
    
    Args:
        start: Start time in seconds
        end: End time in seconds
        caption: Caption text
        viral_score: Viral score
        variant_id: Variant identifier
        
    Raises:
        ValueError: If any validation fails
    """
    validate_video_clip_times(start, end)
    validate_caption(caption)
    validate_viral_score(viral_score)
    
    if not isinstance(variant_id, str) or not variant_id.strip():
        raise ValueError("variant_id must be a non-empty string") 