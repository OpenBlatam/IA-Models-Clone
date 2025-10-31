"""
Utility Functions
================

Common utility functions for the copywriting service.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from uuid import UUID, uuid4

import orjson
from pydantic import BaseModel


def generate_cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate a cache key from arguments"""
    # Convert all arguments to strings and sort them
    key_parts = []
    
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(orjson.dumps(arg, sort_keys=True).decode())
        else:
            key_parts.append(str(arg))
    
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (dict, list)):
            key_parts.append(f"{key}:{orjson.dumps(value, sort_keys=True).decode()}")
        else:
            key_parts.append(f"{key}:{value}")
    
    # Create hash of the combined key
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def serialize_model(model: BaseModel) -> Dict[str, Any]:
    """Serialize a Pydantic model to a dictionary"""
    return model.model_dump()


def deserialize_model(data: Dict[str, Any], model_class: type[BaseModel]) -> BaseModel:
    """Deserialize a dictionary to a Pydantic model"""
    return model_class.model_validate(data)


def format_processing_time(start_time: float) -> int:
    """Format processing time in milliseconds"""
    return int((time.time() - start_time) * 1000)


def is_valid_uuid(uuid_string: str) -> bool:
    """Check if a string is a valid UUID"""
    try:
        UUID(uuid_string)
        return True
    except ValueError:
        return False


def sanitize_error_message(message: str) -> str:
    """Sanitize error messages for security"""
    # Remove potentially sensitive information
    sensitive_patterns = [
        "password", "secret", "key", "token", "auth",
        "database", "connection", "sql", "query"
    ]
    
    sanitized = message.lower()
    for pattern in sensitive_patterns:
        if pattern in sanitized:
            return "An error occurred while processing your request"
    
    return message


def calculate_confidence_score(
    word_count: int,
    target_word_count: Optional[int] = None,
    content_quality: float = 0.8,
    relevance_score: float = 0.9
) -> float:
    """Calculate confidence score for generated content"""
    base_score = 0.5
    
    # Word count factor
    if target_word_count:
        word_ratio = min(word_count / target_word_count, target_word_count / word_count)
        word_factor = word_ratio * 0.2
    else:
        word_factor = 0.1
    
    # Content quality factor
    quality_factor = content_quality * 0.2
    
    # Relevance factor
    relevance_factor = relevance_score * 0.1
    
    confidence = base_score + word_factor + quality_factor + relevance_factor
    return min(max(confidence, 0.0), 1.0)


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text"""
    # Simple keyword extraction (in production, use NLP libraries)
    words = text.lower().split()
    
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should"
    }
    
    # Count word frequency
    word_count = {}
    for word in words:
        word = word.strip(".,!?;:")
        if len(word) > 2 and word not in stop_words:
            word_count[word] = word_count.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:max_keywords]]


def validate_content_length(content: str, min_length: int = 50, max_length: int = 5000) -> bool:
    """Validate content length"""
    return min_length <= len(content) <= max_length


def format_word_count(word_count: int) -> str:
    """Format word count for display"""
    if word_count < 1000:
        return str(word_count)
    elif word_count < 1000000:
        return f"{word_count / 1000:.1f}K"
    else:
        return f"{word_count / 1000000:.1f}M"


def calculate_readability_score(text: str) -> float:
    """Calculate a simple readability score (0-1)"""
    # Simple readability calculation based on sentence and word length
    sentences = text.split('.')
    words = text.split()
    
    if not sentences or not words:
        return 0.0
    
    avg_sentence_length = len(words) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Simple scoring (lower is more readable)
    score = (avg_sentence_length * 0.1) + (avg_word_length * 0.1)
    
    # Normalize to 0-1 scale (invert so higher is more readable)
    return max(0.0, min(1.0, 1.0 - (score / 10)))


def generate_content_hash(content: str) -> str:
    """Generate a hash for content to detect duplicates"""
    return hashlib.sha256(content.encode()).hexdigest()


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_timestamp(timestamp: datetime) -> str:
    """Format timestamp for display"""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse timestamp string to datetime"""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        return None


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity (0-1)"""
    # Simple Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def validate_email(email: str) -> bool:
    """Simple email validation"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def generate_random_string(length: int = 8) -> str:
    """Generate a random string of specified length"""
    import random
    import string
    
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely load JSON string"""
    try:
        return orjson.loads(json_str)
    except (orjson.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """Safely dump object to JSON string"""
    try:
        return orjson.dumps(obj).decode()
    except (TypeError, ValueError):
        return default


def retry_with_backoff(
    func: callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
):
    """Retry function with exponential backoff"""
    import asyncio
    
    async def async_retry(*args, **kwargs):
        delay = base_delay
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt == max_retries:
                    break
                
                await asyncio.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
        
        raise last_exception
    
    return async_retry


def measure_execution_time(func: callable):
    """Decorator to measure function execution time"""
    import functools
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result
        finally:
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
    
    return async_wrapper


# Performance monitoring utilities
class PerformanceTracker:
    """Track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration"""
        if operation not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        del self.start_times[operation]
        
        return duration
    
    def get_average_time(self, operation: str) -> float:
        """Get average execution time for an operation"""
        if operation not in self.metrics or not self.metrics[operation]:
            return 0.0
        
        return sum(self.metrics[operation]) / len(self.metrics[operation])
    
    def get_total_calls(self, operation: str) -> int:
        """Get total number of calls for an operation"""
        return len(self.metrics.get(operation, []))
    
    def reset_metrics(self) -> None:
        """Reset all metrics"""
        self.metrics.clear()
        self.start_times.clear()


# Global performance tracker instance
performance_tracker = PerformanceTracker()






























