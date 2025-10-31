from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union
from functools import wraps
from datetime import datetime, timezone
import logging
from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError, field_validator
import re
from .schemas import ErrorResponse
import inspect
from .error_handling import log_error
from typing import Any, List, Dict, Optional
"""
Utility functions for Instagram Captions API.

Pure functions and helpers for common operations, error handling, and performance optimization.
"""




logger = logging.getLogger(__name__)

T = TypeVar('T')

__all__ = [
    "create_error_response",
    "handle_api_errors",
    "validate_non_empty_string",
    "validate_list_not_empty",
    "validate_numeric_range",
    "generate_cache_key",
    "measure_execution_time",
    "timeout_operation",
    "serialize_for_cache",
    "deserialize_from_cache",
    "normalize_timezone_string",
    "extract_keywords_from_text",
    "calculate_improvement_percentage",
    "validate_caption_length",
    "sanitize_hashtags",
    "calculate_readability_score",
    "get_current_utc_timestamp",
    "batch_process_with_concurrency",
    "format_duration_human_readable",
    "truncate_text",
    "PerformanceMonitor",
    "log_performance_metrics",
]


def create_error_response(*, error_code: str, message: str, details: dict = None, request_id: str = None) -> dict:
    if not error_code or not message:
        log_error(module=__name__, function="create_error_response", parameters={"error_code": error_code, "message": message})
        return {"error_code": "INVALID_INPUT", "message": "error_code and message required", "details": {}, "request_id": request_id}
    return {
        "error_code": error_code,
        "message": message,
        "details": details or {},
        "request_id": request_id
    }


async def handle_api_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for standardized API error handling (RORO)."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except HTTPException as exc:
            raise
        except ValueError as e:
            logger.warning(f"Validation error in {func.__name__}: {e}")
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    error_code="VALIDATION_ERROR",
                    message=str(e)
                )
            )
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=create_error_response(
                    error_code="INTERNAL_ERROR",
                    message="An unexpected error occurred"
                )
            )
    return wrapper


def validate_non_empty_string(*, value: str, field_name: str) -> dict:
    if not value or not value.strip():
        log_error(module=__name__, function="validate_non_empty_string", parameters={"value": value, "field_name": field_name})
        return {"is_valid": False, "error": f"{field_name} cannot be empty"}
    return {"is_valid": True, "value": value.strip()}


def validate_list_not_empty(*, value: list, field_name: str) -> dict:
    if not value:
        log_error(module=__name__, function="validate_list_not_empty", parameters={"value": value, "field_name": field_name})
        return {"is_valid": False, "error": f"{field_name} cannot be empty"}
    return {"is_valid": True, "value": value}


def validate_numeric_range(*, value: float, min_val: float, max_val: float, field_name: str) -> dict:
    if value < min_val or value > max_val:
        log_error(module=__name__, function="validate_numeric_range", parameters={"value": value, "min_val": min_val, "max_val": max_val, "field_name": field_name})
        return {"is_valid": False, "error": f"{field_name} must be between {min_val} and {max_val}"}
    return {"is_valid": True, "value": value}


def generate_cache_key(*, args: tuple = (), kwargs: dict = None) -> dict:
    kwargs = kwargs or {}
    try:
        key_data = {"args": args, "kwargs": sorted(kwargs.items())}
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return {"cache_key": hashlib.md5(key_string.encode()).hexdigest()}
    except Exception as e:
        log_error(module=__name__, function="generate_cache_key", parameters={"args": args, "kwargs": kwargs}, context=str(e))
        return {"cache_key": None, "error": str(e)}


def measure_execution_time(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to measure and log function execution time (RORO)."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return {"result": result, "execution_time": execution_time}
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.warning(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper


async def timeout_operation(*, operation: Callable[..., Any], timeout_seconds: float, args: tuple = (), kwargs: dict = None) -> dict:
    """Execute operation with timeout (RORO)."""
    kwargs = kwargs or {}
    try:
        result = await asyncio.wait_for(
            operation(*args, **kwargs),
            timeout=timeout_seconds
        )
        return {"result": result, "timed_out": False}
    except asyncio.TimeoutError:
        return {"result": None, "timed_out": True, "error": f"Operation timed out after {timeout_seconds} seconds"}


def serialize_for_cache(*, data: Any) -> dict:
    if isinstance(data, BaseModel):
        return {"serialized": data.model_dump_json()}
    try:
        return {"serialized": json.dumps(data, default=str, sort_keys=True)}
    except TypeError as e:
        log_error(module=__name__, function="serialize_for_cache", parameters={"data": data}, context=str(e))
        return {"serialized": json.dumps({"error": "serialization_failed"})}


def deserialize_from_cache(*, data: str, model_class: type = None) -> dict:
    if not data:
        log_error(module=__name__, function="deserialize_from_cache", parameters={"data": data, "model_class": model_class})
        return {"deserialized": None, "error": "No data provided"}
    try:
        parsed = json.loads(data)
        if model_class and issubclass(model_class, BaseModel):
            return {"deserialized": model_class.model_validate(parsed)}
        return {"deserialized": parsed}
    except (json.JSONDecodeError, ValueError) as e:
        log_error(module=__name__, function="deserialize_from_cache", parameters={"data": data, "model_class": model_class}, context=str(e))
        return {"deserialized": None, "error": str(e)}


def normalize_timezone_string(*, timezone_str: str) -> dict:
    if not timezone_str:
        return {"timezone": "UTC"}
    normalized = timezone_str.strip().replace(" ", "_")
    timezone_mappings = {
        "est": "US/Eastern",
        "pst": "US/Pacific",
        "cst": "US/Central",
        "mst": "US/Mountain",
        "utc": "UTC",
        "gmt": "UTC"
    }
    tz = timezone_mappings.get(normalized.lower(), normalized)
    if not tz:
        log_error(module=__name__, function="normalize_timezone_string", parameters={"timezone_str": timezone_str})
        return {"timezone": "UTC"}
    return {"timezone": tz}


class ExtractKeywordsInput(BaseModel):
    text: str = Field(..., min_length=1)
    max_keywords: int = Field(default=10, ge=1, le=100)

def extract_keywords_from_text(*, input: ExtractKeywordsInput) -> dict:
    """Extract meaningful keywords from text with filtering (RORO, Pydantic)."""
    text = input.text
    max_keywords = input.max_keywords
    if not text:
        return {"keywords": []}
    words = re.findall(r'\b\w+\b', text.lower())
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    keywords = [word for word in words if len(word) > 3 and word not in stop_words]
    unique_keywords = []
    seen = set()
    for keyword in keywords:
        if keyword not in seen:
            unique_keywords.append(keyword)
            seen.add(keyword)
    return {"keywords": unique_keywords[:max_keywords]}

class ImprovementInput(BaseModel):
    original_score: float
    new_score: float

def calculate_improvement_percentage(*, input: ImprovementInput) -> dict:
    """Calculate improvement percentage (RORO, Pydantic)."""
    if input.original_score <= 0:
        pct = 100.0 if input.new_score > 0 else 0.0
    else:
        pct = ((input.new_score - input.original_score) / input.original_score) * 100
        pct = round(pct, 2)
    return {"improvement_percentage": pct}

class CaptionLengthInput(BaseModel):
    caption: str
    content_type: str = Field(default="post")

def validate_caption_length(*, input: CaptionLengthInput) -> dict:
    """Validate caption length based on content type (RORO, Pydantic)."""
    length_limits = {
        "post": 2200,
        "story": 500,
        "reel": 1000,
        "carousel": 2200
    }
    max_length = length_limits.get(input.content_type.lower(), 2200)
    is_valid = len(input.caption) <= max_length
    return {"is_valid": is_valid, "max_length": max_length}

class HashtagsInput(BaseModel):
    hashtags: List[str]

def sanitize_hashtags(*, input: HashtagsInput) -> dict:
    """Sanitize and validate hashtags (RORO, Pydantic)."""
    sanitized = []
    for hashtag in input.hashtags:
        clean_tag = re.sub(r'[^\w\-_]', '', hashtag.strip())
        if not clean_tag.startswith('#'):
            clean_tag = f"#{clean_tag}"
        tag_content = clean_tag[1:]
        if 1 <= len(tag_content) <= 30 and tag_content.replace('_', '').isalnum():
            sanitized.append(clean_tag.lower())
    unique_hashtags = []
    seen = set()
    for tag in sanitized:
        if tag not in seen:
            unique_hashtags.append(tag)
            seen.add(tag)
    return {"hashtags": unique_hashtags}

class ReadabilityInput(BaseModel):
    text: str

def calculate_readability_score(*, input: ReadabilityInput) -> dict:
    """Calculate simple readability score for Instagram content (RORO, Pydantic)."""
    text = input.text
    if not text:
        return {"readability_score": 0.0}
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?') + 1
    if not words or sentences == 0:
        return {"readability_score": 0.0}
    avg_words_per_sentence = len(words) / sentences
    avg_word_length = sum(len(word) for word in words) / len(words)
    sentence_score = max(0, 1 - abs(avg_words_per_sentence - 15) / 20)
    word_score = max(0, 1 - abs(avg_word_length - 5) / 5)
    score = round((sentence_score + word_score) / 2 * 100, 1)
    return {"readability_score": score}

def get_current_utc_timestamp() -> dict:
    """Get current UTC timestamp in ISO format (RORO, type hint)."""
    return {"timestamp": datetime.now(timezone.utc).isoformat()}

class BatchProcessInput(BaseModel):
    items: List[Any]
    max_concurrency: int = Field(default=5, ge=1, le=100)

def batch_process_with_concurrency(*, input: BatchProcessInput, processor: Callable[[Any], Any]) -> dict:
    """Process items in batches with controlled concurrency (RORO, Pydantic)."""
    async def process_batch(batch_items: List[Any]) -> List[Any]:
        tasks = [processor(item) for item in batch_items]
        return await asyncio.gather(*tasks, return_exceptions=True)
    batches = [input.items[i:i + input.max_concurrency] for i in range(0, len(input.items), input.max_concurrency)]
    all_results = []
    for batch in batches:
        batch_results = asyncio.run(process_batch(batch))
        all_results.extend(batch_results)
    return {"results": all_results}

class FormatDurationInput(BaseModel):
    seconds: float

def format_duration_human_readable(*, input: FormatDurationInput) -> dict:
    """Format duration in human-readable format (RORO, Pydantic)."""
    seconds = input.seconds
    if seconds < 1:
        return {"duration": f"{seconds * 1000:.1f}ms"}
    elif seconds < 60:
        return {"duration": f"{seconds:.1f}s"}
    elif seconds < 3600:
        minutes = seconds / 60
        return {"duration": f"{minutes:.1f}m"}
    else:
        hours = seconds / 3600
        return {"duration": f"{hours:.1f}h"}

class TruncateTextInput(BaseModel):
    text: str
    max_length: int
    suffix: str = Field(default="...")

def truncate_text(*, input: TruncateTextInput) -> dict:
    """Truncate text to specified length with suffix (RORO, Pydantic)."""
    if len(input.text) <= input.max_length:
        return {"text": input.text}
    return {"text": input.text[:input.max_length - len(input.suffix)] + input.suffix}


class PerformanceMonitor:
    """Simple performance monitoring utility."""
    
    def __init__(self) -> Any:
        self.metrics = {}
    
    def start_timer(self, operation: str) -> float:
        """Start timing an operation."""
        start_time = time.perf_counter()
        self.metrics[operation] = {"start_time": start_time}
        return start_time
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.metrics:
            return 0.0
        
        end_time = time.perf_counter()
        duration = end_time - self.metrics[operation]["start_time"]
        self.metrics[operation]["duration"] = duration
        self.metrics[operation]["end_time"] = end_time
        
        return duration
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self.metrics.copy()
    
    def reset(self) -> Any:
        """Reset all metrics."""
        self.metrics.clear()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def log_performance_metrics(operation: str):
    """Decorator to automatically log performance metrics."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = performance_monitor.start_timer(operation)
            
            try:
                result = await func(*args, **kwargs)
                duration = performance_monitor.end_timer(operation)
                logger.info(f"Operation '{operation}' completed in {format_duration_human_readable(duration)}")
                return result
            except Exception as e:
                duration = performance_monitor.end_timer(operation)
                logger.warning(f"Operation '{operation}' failed after {format_duration_human_readable(duration)}: {e}")
                raise
        
        return wrapper
    return decorator 