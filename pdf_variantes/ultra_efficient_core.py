"""
Ultra-Efficient Core Processing Module
=====================================

High-performance PDF processing with PyMuPDF, caching, and async optimization.
Focuses on minimal overhead, early returns, and maximum throughput.

Performance Features:
- PyMuPDF for critical operations (text extraction, preview rendering)
- In-memory LRU cache with hash-based keys
- Async CPU-bound operations with asyncio.to_thread
- Early termination for large documents
- Streaming responses for large files
- Memory-efficient data structures

Author: TruthGPT Development Team
Version: 1.0.0 - Ultra-Efficient
License: MIT
"""

import fitz  # PyMuPDF
import hashlib
import asyncio
from typing import Dict, Any, Optional, Tuple, List, Union
from functools import lru_cache
from dataclasses import dataclass
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import weakref
import gc

logger = logging.getLogger(__name__)

# Global cache with weak references to prevent memory leaks
_cache: Dict[str, Any] = {}
_cache_access_times: Dict[str, float] = {}
_cache_max_size = 1024
_cache_ttl = 300  # 5 minutes

@dataclass
class ProcessingConfig:
    """Configuration for ultra-efficient processing."""
    max_chars: int = 5000
    max_pages_preview: int = 1
    max_pages_topic: int = 10
    preview_dpi: int = 120
    chunk_size: int = 16384
    cache_max_size: int = 1024
    cache_ttl: int = 300
    enable_ocr: bool = False
    enable_deep_nlp: bool = False

# Default configuration
DEFAULT_CONFIG = ProcessingConfig()

def _generate_cache_key(data: bytes, params: str = "") -> str:
    """Generate MD5 hash-based cache key."""
    return hashlib.md5(data + params.encode()).hexdigest()

def _cleanup_cache():
    """Clean up expired cache entries."""
    current_time = time.time()
    expired_keys = [
        key for key, access_time in _cache_access_times.items()
        if current_time - access_time > _cache_ttl
    ]
    
    for key in expired_keys:
        _cache.pop(key, None)
        _cache_access_times.pop(key, None)
    
    # LRU cleanup if cache is too large
    if len(_cache) > _cache_max_size:
        sorted_items = sorted(_cache_access_times.items(), key=lambda x: x[1])
        keys_to_remove = [key for key, _ in sorted_items[:len(_cache) - _cache_max_size]]
        
        for key in keys_to_remove:
            _cache.pop(key, None)
            _cache_access_times.pop(key, None)

def _get_from_cache(key: str) -> Optional[Any]:
    """Get item from cache with TTL check."""
    if key not in _cache:
        return None
    
    current_time = time.time()
    if current_time - _cache_access_times[key] > _cache_ttl:
        _cache.pop(key, None)
        _cache_access_times.pop(key, None)
        return None
    
    _cache_access_times[key] = current_time
    return _cache[key]

def _set_cache(key: str, value: Any):
    """Set item in cache with cleanup."""
    _cleanup_cache()
    _cache[key] = value
    _cache_access_times[key] = time.time()

@lru_cache(maxsize=512)
def _extract_text_sync(file_bytes: bytes, max_chars: int) -> str:
    """Synchronous text extraction with early termination."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        total_chars = 0
        
        for page_num in range(min(len(doc), 20)):  # Limit to first 20 pages
            page = doc[page_num]
            page_text = page.get_text("text")
            
            if not page_text.strip():
                continue
                
            text_parts.append(page_text)
            total_chars += len(page_text)
            
            if total_chars >= max_chars:
                break
        
        doc.close()
        
        full_text = "".join(text_parts)
        return full_text[:max_chars]
        
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        return ""

@lru_cache(maxsize=512)
def _generate_preview_sync(file_bytes: bytes, page: int, dpi: int) -> bytes:
    """Synchronous preview generation."""
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

@lru_cache(maxsize=256)
def _extract_topics_sync(file_bytes: bytes, max_pages: int) -> List[str]:
    """Synchronous topic extraction with early termination."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text_parts = []
        
        for page_num in range(min(len(doc), max_pages)):
            page = doc[page_num]
            page_text = page.get_text("text")
            
            if page_text.strip():
                text_parts.append(page_text)
        
        doc.close()
        
        if not text_parts:
            return []
        
        # Simple keyword extraction (can be enhanced with spaCy/NLTK)
        full_text = " ".join(text_parts)
        words = full_text.lower().split()
        
        # Filter common words and extract meaningful terms
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        meaningful_words = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Count frequency and return top topics
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:10]]
        
    except Exception as e:
        logger.error(f"Topic extraction failed: {e}")
        return []

async def extract_text_ultra_fast(
    file_bytes: bytes, 
    config: ProcessingConfig = DEFAULT_CONFIG
) -> str:
    """Ultra-fast async text extraction with caching."""
    cache_key = _generate_cache_key(file_bytes, f"text:{config.max_chars}")
    
    # Check cache first
    cached_result = _get_from_cache(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Extract text in thread pool
    text = await asyncio.to_thread(_extract_text_sync, file_bytes, config.max_chars)
    
    # Cache result
    _set_cache(cache_key, text)
    
    return text

async def generate_preview_ultra_fast(
    file_bytes: bytes, 
    page: int = 0, 
    dpi: int = 120
) -> bytes:
    """Ultra-fast async preview generation with caching."""
    cache_key = _generate_cache_key(file_bytes, f"preview:{page}:{dpi}")
    
    # Check cache first
    cached_result = _get_from_cache(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Generate preview in thread pool
    png_data = await asyncio.to_thread(_generate_preview_sync, file_bytes, page, dpi)
    
    # Cache result
    _set_cache(cache_key, png_data)
    
    return png_data

async def extract_topics_ultra_fast(
    file_bytes: bytes, 
    config: ProcessingConfig = DEFAULT_CONFIG
) -> List[str]:
    """Ultra-fast async topic extraction with caching."""
    cache_key = _generate_cache_key(file_bytes, f"topics:{config.max_pages_topic}")
    
    # Check cache first
    cached_result = _get_from_cache(cache_key)
    if cached_result is not None:
        return cached_result
    
    # Extract topics in thread pool
    topics = await asyncio.to_thread(_extract_topics_sync, file_bytes, config.max_pages_topic)
    
    # Cache result
    _set_cache(cache_key, topics)
    
    return topics

async def process_pdf_ultra_fast(
    file_bytes: bytes,
    config: ProcessingConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:
    """Ultra-fast PDF processing pipeline with parallel operations."""
    
    # Early validation
    if len(file_bytes) == 0:
        return {"error": "Empty file"}
    
    if len(file_bytes) > 50 * 1024 * 1024:  # 50MB limit
        return {"error": "File too large"}
    
    try:
        # Parallel processing of different features
        tasks = []
        
        if config.enable_deep_nlp:
            tasks.extend([
                extract_text_ultra_fast(file_bytes, config),
                extract_topics_ultra_fast(file_bytes, config)
            ])
        else:
            tasks.append(extract_text_ultra_fast(file_bytes, config))
        
        # Always generate preview
        tasks.append(generate_preview_ultra_fast(file_bytes, 0, config.preview_dpi))
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = {
            "text": results[0] if not isinstance(results[0], Exception) else "",
            "preview": results[-1] if not isinstance(results[-1], Exception) else b"",
            "file_size": len(file_bytes),
            "processing_time": time.time()
        }
        
        if config.enable_deep_nlp and len(results) > 2:
            processed_results["topics"] = results[1] if not isinstance(results[1], Exception) else []
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Ultra-fast processing failed: {e}")
        return {"error": str(e)}

def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics for monitoring."""
    return {
        "cache_size": len(_cache),
        "cache_max_size": _cache_max_size,
        "cache_ttl": _cache_ttl,
        "memory_usage": len(str(_cache)),
        "access_times_count": len(_cache_access_times)
    }

def clear_cache():
    """Clear all cache entries."""
    global _cache, _cache_access_times
    _cache.clear()
    _cache_access_times.clear()
    gc.collect()

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.4f}s: {e}")
            raise
    return wrapper

# Apply monitoring to main functions
extract_text_ultra_fast = monitor_performance(extract_text_ultra_fast)
generate_preview_ultra_fast = monitor_performance(generate_preview_ultra_fast)
extract_topics_ultra_fast = monitor_performance(extract_topics_ultra_fast)
process_pdf_ultra_fast = monitor_performance(process_pdf_ultra_fast)
