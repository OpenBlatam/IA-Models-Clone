from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import re
from typing import List, Optional
import secrets
    import hashlib
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Instagram Captions API v14.0 - Validation Utilities
Utility functions for input validation and sanitization
"""


async def generate_request_id() -> str:
    """Generate unique request ID - pure function"""
    return f"v14-{secrets.token_urlsafe(8)}"

async def validate_api_key(api_key: str) -> bool:
    """Validate API key - pure function"""
    valid_keys = ["optimized-v14-key", "ultra-fast-key", "performance-key"]
    return api_key in valid_keys

def sanitize_content(content: str) -> str:
    """Sanitize content for security - pure function"""
    harmful_patterns = ['<script', 'javascript:', 'data:', 'vbscript:', 'onload=']
    for pattern in harmful_patterns:
        if pattern.lower() in content.lower():
            raise ValueError(f"Potentially harmful content detected: {pattern}")
    return content.strip()

def validate_style(style: str) -> bool:
    """Validate caption style - pure function"""
    valid_styles = ["casual", "professional", "inspirational", "playful"]
    return style in valid_styles

def validate_optimization_level(level: str) -> bool:
    """Validate optimization level - pure function"""
    valid_levels = ["ultra_fast", "balanced", "quality"]
    return level in valid_levels

def validate_hashtag_count(count: int) -> bool:
    """Validate hashtag count - pure function"""
    return 5 <= count <= 30

def sanitize_hashtags(hashtags: List[str]) -> List[str]:
    """Sanitize hashtags - pure function"""
    sanitized = []
    for hashtag in hashtags:
        # Remove special characters and ensure proper format
        clean_hashtag = re.sub(r'[^a-zA-Z0-9_]', '', hashtag)
        if clean_hashtag and not clean_hashtag.startswith('#'):
            clean_hashtag = f"#{clean_hashtag}"
        if clean_hashtag and len(clean_hashtag) > 1:
            sanitized.append(clean_hashtag.lower())
    return list(set(sanitized))  # Remove duplicates

def validate_batch_size(size: int) -> bool:
    """Validate batch size - pure function"""
    return 1 <= size <= 100

def generate_cache_key(content: str, style: str, hashtag_count: int) -> str:
    """Generate cache key - pure function"""
    key_data = f"{content}:{style}:{hashtag_count}"
    return hashlib.md5(key_data.encode()).hexdigest()

def validate_performance_thresholds(
    avg_response_time: float,
    cache_hit_rate: float,
    success_rate: float
) -> dict:
    """Validate performance thresholds - pure function"""
    thresholds = {
        "response_time_grade": "SLOW",
        "cache_grade": "POOR", 
        "success_grade": "POOR"
    }
    
    # Response time grading
    if avg_response_time < 0.015:
        thresholds["response_time_grade"] = "ULTRA_FAST"
    elif avg_response_time < 0.025:
        thresholds["response_time_grade"] = "FAST"
    elif avg_response_time < 0.050:
        thresholds["response_time_grade"] = "NORMAL"
    
    # Cache hit rate grading
    if cache_hit_rate >= 95:
        thresholds["cache_grade"] = "EXCELLENT"
    elif cache_hit_rate >= 80:
        thresholds["cache_grade"] = "GOOD"
    elif cache_hit_rate >= 60:
        thresholds["cache_grade"] = "FAIR"
    
    # Success rate grading
    if success_rate >= 99:
        thresholds["success_grade"] = "EXCELLENT"
    elif success_rate >= 95:
        thresholds["success_grade"] = "GOOD"
    elif success_rate >= 90:
        thresholds["success_grade"] = "FAIR"
    
    return thresholds 