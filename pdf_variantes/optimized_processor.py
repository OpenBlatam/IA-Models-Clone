"""
Ultra-Optimized PDF Processor
============================

High-performance PDF processing with functional patterns, early returns,
parallel processing, and memory optimization.

Key Optimizations:
- Pure functions with early returns
- Parallel processing with asyncio.gather
- Memory-efficient streaming
- Fallback strategies for robustness
- Functional composition patterns
- Minimal object creation

Author: TruthGPT Development Team
Version: 1.0.0 - Ultra-Optimized
License: MIT
"""

import asyncio
import fitz  # PyMuPDF
import hashlib
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Awaitable
from functools import wraps, partial
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ProcessingLimits:
    """Immutable processing limits for functional safety."""
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    max_pages: int = 100
    max_chars: int = 10000
    max_topics: int = 20
    preview_dpi: int = 120
    chunk_size: int = 16384

@dataclass(frozen=True)
class ProcessingResult:
    """Immutable result container."""
    success: bool
    data: Dict[str, Any]
    processing_time: float
    cache_hit: bool = False
    error: Optional[str] = None

# Global processing limits
LIMITS = ProcessingLimits()

def validate_file_input(file_bytes: bytes) -> Optional[str]:
    """Pure function for file validation with early returns."""
    if not file_bytes:
        return "Empty file provided"
    
    if len(file_bytes) > LIMITS.max_file_size:
        return f"File too large: {len(file_bytes)} bytes (max: {LIMITS.max_file_size})"
    
    # Basic PDF header check
    if not file_bytes.startswith(b'%PDF'):
        return "Invalid PDF file format"
    
    return None

def extract_text_pure(file_bytes: bytes, max_chars: int = LIMITS.max_chars) -> str:
    """Pure function for text extraction with early termination."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        total_chars = 0
        
        # Early termination on character limit
        for page_num in range(min(len(doc), LIMITS.max_pages)):
            page = doc[page_num]
            page_text = page.get_text("text")
            
            if not page_text.strip():
                continue
                
            text_parts.append(page_text)
            total_chars += len(page_text)
            
            if total_chars >= max_chars:
                break
        
        doc.close()
        return "".join(text_parts)[:max_chars]
        
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        return ""

def generate_preview_pure(file_bytes: bytes, page: int = 0, dpi: int = LIMITS.preview_dpi) -> bytes:
    """Pure function for preview generation."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        
        if page >= len(doc):
            page = 0
            
        page_obj = doc.load_page(page)
        pixmap = page_obj.get_pixmap(dpi=dpi)
        png_data = pixmap.tobytes("png")
        
        doc.close()
        return png_data
        
    except Exception as e:
        logger.error(f"Preview generation failed: {e}")
        return b""

def extract_topics_pure(file_bytes: bytes, max_pages: int = 10) -> List[str]:
    """Pure function for topic extraction with early termination."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        
        # Early termination on page limit
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            page_text = page.get_text("text")
            
            if page_text.strip():
                text_parts.append(page_text)
        
        doc.close()
        
        if not text_parts:
            return []
        
        # Simple keyword extraction (functional approach)
        full_text = " ".join(text_parts)
        words = full_text.lower().split()
        
        # Filter and count (functional style)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Count frequency functionally
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort and return top topics
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:LIMITS.max_topics]]
        
    except Exception as e:
        logger.error(f"Topic extraction failed: {e}")
        return []

def generate_file_hash(file_bytes: bytes) -> str:
    """Pure function to generate file hash."""
    return hashlib.md5(file_bytes).hexdigest()

# Cache implementation with functional patterns
_cache: Dict[str, Any] = {}
_cache_timestamps: Dict[str, float] = {}
_cache_max_size = 1024
_cache_ttl = 300

def get_cache_key(file_bytes: bytes, operation: str, params: str = "") -> str:
    """Pure function to generate cache key."""
    file_hash = generate_file_hash(file_bytes)
    return f"{operation}:{file_hash}:{params}"

def is_cache_valid(key: str) -> bool:
    """Pure function to check cache validity."""
    if key not in _cache:
        return False
    
    current_time = time.time()
    return current_time - _cache_timestamps[key] <= _cache_ttl

def get_from_cache(key: str) -> Optional[Any]:
    """Pure function to get from cache."""
    if not is_cache_valid(key):
        return None
    
    _cache_timestamps[key] = time.time()  # Update access time
    return _cache[key]

def set_cache(key: str, value: Any) -> None:
    """Function to set cache with cleanup."""
    # Cleanup expired entries
    current_time = time.time()
    expired_keys = [k for k, t in _cache_timestamps.items() if current_time - t > _cache_ttl]
    
    for k in expired_keys:
        _cache.pop(k, None)
        _cache_timestamps.pop(k, None)
    
    # LRU cleanup
    if len(_cache) >= _cache_max_size:
        sorted_items = sorted(_cache_timestamps.items(), key=lambda x: x[1])
        keys_to_remove = [k for k, _ in sorted_items[:len(_cache) - _cache_max_size + 1]]
        
        for k in keys_to_remove:
            _cache.pop(k, None)
            _cache_timestamps.pop(k, None)
    
    _cache[key] = value
    _cache_timestamps[key] = time.time()

# Async wrappers for CPU-bound operations
async def extract_text_async(file_bytes: bytes, max_chars: int = LIMITS.max_chars) -> str:
    """Async wrapper for text extraction with caching."""
    cache_key = get_cache_key(file_bytes, "text", str(max_chars))
    
    cached_result = get_from_cache(cache_key)
    if cached_result is not None:
        return cached_result
    
    text = await asyncio.to_thread(extract_text_pure, file_bytes, max_chars)
    set_cache(cache_key, text)
    
    return text

async def generate_preview_async(file_bytes: bytes, page: int = 0, dpi: int = LIMITS.preview_dpi) -> bytes:
    """Async wrapper for preview generation with caching."""
    cache_key = get_cache_key(file_bytes, "preview", f"{page}:{dpi}")
    
    cached_result = get_from_cache(cache_key)
    if cached_result is not None:
        return cached_result
    
    png_data = await asyncio.to_thread(generate_preview_pure, file_bytes, page, dpi)
    set_cache(cache_key, png_data)
    
    return png_data

async def extract_topics_async(file_bytes: bytes, max_pages: int = 10) -> List[str]:
    """Async wrapper for topic extraction with caching."""
    cache_key = get_cache_key(file_bytes, "topics", str(max_pages))
    
    cached_result = get_from_cache(cache_key)
    if cached_result is not None:
        return cached_result
    
    topics = await asyncio.to_thread(extract_topics_pure, file_bytes, max_pages)
    set_cache(cache_key, topics)
    
    return topics

# Functional composition patterns
def compose_functions(*functions: Callable) -> Callable:
    """Compose multiple functions into a single function."""
    def composed(x):
        result = x
        for func in reversed(functions):
            result = func(result)
        return result
    return composed

def with_fallback(fallback_value: Any):
    """Decorator to provide fallback values for functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Function {func.__name__} failed, using fallback: {e}")
                return fallback_value
        return wrapper
    return decorator

# Parallel processing with functional patterns
async def process_pdf_parallel(
    file_bytes: bytes,
    operations: List[str] = None,
    limits: ProcessingLimits = LIMITS
) -> ProcessingResult:
    """Process PDF with parallel operations and functional patterns."""
    
    start_time = time.time()
    
    # Early validation
    validation_error = validate_file_input(file_bytes)
    if validation_error:
        return ProcessingResult(
            success=False,
            data={},
            processing_time=time.time() - start_time,
            error=validation_error
        )
    
    # Default operations
    if operations is None:
        operations = ["text", "preview", "topics"]
    
    try:
        # Create task list based on requested operations
        tasks = []
        task_names = []
        
        if "text" in operations:
            tasks.append(extract_text_async(file_bytes, limits.max_chars))
            task_names.append("text")
        
        if "preview" in operations:
            tasks.append(generate_preview_async(file_bytes, 0, limits.preview_dpi))
            task_names.append("preview")
        
        if "topics" in operations:
            tasks.append(extract_topics_async(file_bytes, 10))
            task_names.append("topics")
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results functionally
        processed_data = {
            "file_size": len(file_bytes),
            "file_hash": generate_file_hash(file_bytes),
            "processing_time": time.time() - start_time
        }
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Operation {task_names[i]} failed: {result}")
                processed_data[task_names[i]] = None
            else:
                processed_data[task_names[i]] = result
        
        return ProcessingResult(
            success=True,
            data=processed_data,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        logger.error(f"Parallel processing failed: {e}")
        return ProcessingResult(
            success=False,
            data={},
            processing_time=time.time() - start_time,
            error=str(e)
        )

# Batch processing with functional patterns
async def process_multiple_pdfs(
    files: List[Tuple[str, bytes]],
    operations: List[str] = None,
    max_concurrent: int = 5
) -> List[ProcessingResult]:
    """Process multiple PDFs with controlled concurrency."""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single(filename: str, file_bytes: bytes) -> ProcessingResult:
        async with semaphore:
            return await process_pdf_parallel(file_bytes, operations)
    
    # Create tasks for all files
    tasks = [process_single(filename, file_bytes) for filename, file_bytes in files]
    
    # Execute with controlled concurrency
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(ProcessingResult(
                success=False,
                data={"filename": files[i][0]},
                processing_time=0,
                error=str(result)
            ))
        else:
            processed_results.append(result)
    
    return processed_results

# Performance monitoring with functional patterns
def monitor_performance(func_name: str):
    """Decorator for performance monitoring."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func_name} executed in {execution_time:.4f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func_name} failed after {execution_time:.4f}s: {e}")
                raise
        return wrapper
    return decorator

# Apply monitoring to main functions
extract_text_async = monitor_performance("extract_text_async")(extract_text_async)
generate_preview_async = monitor_performance("generate_preview_async")(generate_preview_async)
extract_topics_async = monitor_performance("extract_topics_async")(extract_topics_async)
process_pdf_parallel = monitor_performance("process_pdf_parallel")(process_pdf_parallel)

# Cache management functions
def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return {
        "cache_size": len(_cache),
        "cache_max_size": _cache_max_size,
        "cache_ttl": _cache_ttl,
        "hit_ratio": len(_cache) / max(len(_cache_timestamps), 1),
        "memory_usage": len(str(_cache))
    }

def clear_cache() -> None:
    """Clear all cache entries."""
    global _cache, _cache_timestamps
    _cache.clear()
    _cache_timestamps.clear()
    gc.collect()

def cleanup_cache() -> int:
    """Clean up expired cache entries and return count of removed items."""
    current_time = time.time()
    expired_keys = [k for k, t in _cache_timestamps.items() if current_time - t > _cache_ttl]
    
    for k in expired_keys:
        _cache.pop(k, None)
        _cache_timestamps.pop(k, None)
    
    return len(expired_keys)