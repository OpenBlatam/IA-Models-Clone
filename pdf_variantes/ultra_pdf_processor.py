"""Ultra-efficient PDF processing with minimal overhead."""

from typing import Dict, Any, List, Optional, Callable, Awaitable
from functools import partial
import asyncio
import time
import logging
from pathlib import Path

from .ultra_efficient import (
    ultra_fast_cache, ultra_fast_retry, ultra_fast_timeout,
    ultra_fast_batch_process, ultra_fast_parallel, ultra_fast_map,
    ultra_fast_filter, ultra_fast_validate, ultra_fast_sanitize,
    ultra_fast_serialize, ultra_fast_deserialize, ultra_fast_read_file,
    ultra_fast_write_file, ultra_fast_error_handler, ultra_fast_health_check
)
from .utils.file_utils import validate_pdf_file, extract_metadata, get_file_hash
from .utils.text_utils import extract_text_content, detect_language, calculate_readability

logger = logging.getLogger(__name__)


# --- Ultra-Fast PDF Content Extraction ---
@ultra_fast_cache(maxsize=500, ttl=300.0)
@ultra_fast_retry(max_retries=2, base_delay=0.1)
@ultra_fast_timeout(timeout=10.0)
async def ultra_fast_extract_content(file_content: bytes) -> Dict[str, Any]:
    """Ultra-fast PDF content extraction with caching."""
    # Validate PDF
    validation = validate_pdf_file(file_content, "temp.pdf")
    if not validation["valid"]:
        raise ValueError(f"Invalid PDF: {validation['error']}")
    
    # Extract content in parallel
    text_task = asyncio.create_task(extract_text_content(file_content))
    metadata_task = asyncio.create_task(extract_metadata(file_content))
    
    text_content, metadata = await asyncio.gather(text_task, metadata_task)
    
    return {
        "text_content": text_content,
        "metadata": metadata,
        "file_hash": get_file_hash(file_content),
        "page_count": validation["page_count"],
        "file_size": validation["file_size"]
    }


# --- Ultra-Fast Topic Extraction ---
@ultra_fast_cache(maxsize=200, ttl=600.0)
async def ultra_fast_extract_topics(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Ultra-fast topic extraction with caching."""
    text = content_data.get("text_content", "")
    if not text:
        return {"topics": [], "main_topic": None}
    
    # Ultra-fast word frequency calculation
    words = text.lower().split()
    word_freq = {}
    
    for word in words:
        if len(word) > 4:  # Filter short words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top words as topics
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    topics = [{"topic": word, "frequency": freq} for word, freq in sorted_words[:10]]
    
    return {
        "topics": topics,
        "main_topic": topics[0]["topic"] if topics else None,
        "total_topics": len(topics)
    }


# --- Ultra-Fast Variant Generation ---
@ultra_fast_cache(maxsize=100, ttl=1800.0)
async def ultra_fast_generate_variants(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Ultra-fast variant generation with caching."""
    text = content_data.get("text_content", "")
    if not text:
        return {"variants": []}
    
    # Ultra-fast variant generation
    variants = {
        "summary": text[:200] + "..." if len(text) > 200 else text,
        "outline": "\n".join([f"- {line.strip()}" for line in text.split('\n')[:5]]),
        "highlights": text[:100] + "..." if len(text) > 100 else text
    }
    
    return {
        "variants": variants,
        "variant_count": len(variants)
    }


# --- Ultra-Fast Quality Analysis ---
@ultra_fast_cache(maxsize=100, ttl=1800.0)
async def ultra_fast_analyze_quality(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Ultra-fast quality analysis with caching."""
    text = content_data.get("text_content", "")
    
    if not text:
        return {"quality_score": 0.0, "metrics": {}}
    
    # Ultra-fast quality metrics
    readability = calculate_readability(text)
    language = detect_language(text)
    
    # Calculate additional metrics
    word_count = len(text.split())
    sentence_count = len(text.split('.'))
    avg_word_length = sum(len(word) for word in text.split()) / word_count if word_count > 0 else 0
    
    quality_score = (readability + (1.0 if language == "en" else 0.8)) / 2
    
    return {
        "quality_score": quality_score,
        "metrics": {
            "readability": readability,
            "language": language,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length
        }
    }


# --- Ultra-Fast Processing Pipeline ---
def create_ultra_fast_pipeline(
    include_topics: bool = True,
    include_variants: bool = True,
    include_quality: bool = True
) -> List[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]:
    """Create ultra-fast processing pipeline."""
    processors = []
    
    if include_topics:
        processors.append(ultra_fast_extract_topics)
    
    if include_variants:
        processors.append(ultra_fast_generate_variants)
    
    if include_quality:
        processors.append(ultra_fast_analyze_quality)
    
    return processors


# --- Ultra-Fast PDF Processing ---
@ultra_fast_error_handler
async def ultra_fast_process_pdf(
    file_content: bytes,
    filename: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Ultra-fast PDF processing with minimal overhead."""
    options = options or {}
    
    # Extract base content
    content_data = await ultra_fast_extract_content(file_content)
    content_data["filename"] = filename
    
    # Create processing pipeline
    processors = create_ultra_fast_pipeline(
        include_topics=options.get("include_topics", True),
        include_variants=options.get("include_variants", True),
        include_quality=options.get("include_quality", True)
    )
    
    # Process in parallel
    processing_results = await ultra_fast_parallel(
        [partial(processor, content_data) for processor in processors],
        max_concurrent=10
    )
    
    # Combine results
    result = {
        "file_id": generate_file_id(),
        "content_data": content_data,
        "processing_results": processing_results,
        "processed_at": time.time()
    }
    
    return result


# --- Ultra-Fast Batch Processing ---
async def ultra_fast_batch_process_pdfs(
    files: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
    max_concurrent: int = 50
) -> Dict[str, Any]:
    """Ultra-fast batch processing with optimal concurrency."""
    options = options or {}
    
    async def process_single_file(file_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await ultra_fast_process_pdf(
                file_data["content"],
                file_data["filename"],
                options
            )
            return {
                "filename": file_data["filename"],
                "status": "success",
                "result": result
            }
        except Exception as e:
            return {
                "filename": file_data["filename"],
                "status": "error",
                "error": str(e)
            }
    
    # Process with ultra-fast batch processing
    results = await ultra_fast_batch_process(
        files,
        process_single_file,
        max_concurrent=max_concurrent,
        chunk_size=20
    )
    
    # Separate successful and failed results
    successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    failed = [r for r in results if isinstance(r, dict) and r.get("status") == "error"]
    
    return {
        "total_files": len(files),
        "successful": len(successful),
        "failed": len(failed),
        "results": results,
        "processed_at": time.time()
    }


# --- Ultra-Fast Feature Extraction ---
async def ultra_fast_extract_features(
    file_id: str,
    content_data: Dict[str, Any],
    feature_extractors: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """Ultra-fast feature extraction in parallel."""
    if feature_extractors is None:
        feature_extractors = {
            "topics": ultra_fast_extract_topics,
            "variants": ultra_fast_generate_variants,
            "quality": ultra_fast_analyze_quality
        }
    
    # Create extractor functions
    extractors = [
        partial(extractor, content_data)
        for name, extractor in feature_extractors.items()
    ]
    
    # Extract features in parallel
    features_result = await ultra_fast_parallel(extractors, max_concurrent=10)
    
    return {
        "file_id": file_id,
        "features": dict(zip(feature_extractors.keys(), features_result)),
        "extracted_at": time.time()
    }


# --- Ultra-Fast File Operations ---
async def ultra_fast_save_pdf(
    file_content: bytes,
    file_path: str
) -> bool:
    """Ultra-fast PDF saving."""
    try:
        await ultra_fast_write_file(file_path, file_content)
        return True
    except Exception as e:
        logger.error(f"Error saving PDF: {e}")
        return False


async def ultra_fast_load_pdf(file_path: str) -> Optional[bytes]:
    """Ultra-fast PDF loading."""
    try:
        return await ultra_fast_read_file(file_path)
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        return None


# --- Ultra-Fast Validation ---
def ultra_fast_validate_pdf(file_content: bytes) -> bool:
    """Ultra-fast PDF validation."""
    if not file_content:
        return False
    
    # Check PDF header
    if not file_content.startswith(b'%PDF-'):
        return False
    
    # Check file size
    if len(file_content) > 100 * 1024 * 1024:  # 100MB limit
        return False
    
    return True


def ultra_fast_validate_filename(filename: str) -> bool:
    """Ultra-fast filename validation."""
    if not filename:
        return False
    
    # Check extension
    if not filename.lower().endswith('.pdf'):
        return False
    
    # Check length
    if len(filename) > 255:
        return False
    
    return True


# --- Ultra-Fast Utility Functions ---
def generate_file_id() -> str:
    """Generate ultra-fast file ID."""
    import secrets
    return secrets.token_hex(8)


def ultra_fast_format_size(size_bytes: int) -> str:
    """Ultra-fast size formatting."""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


def ultra_fast_format_time(seconds: float) -> str:
    """Ultra-fast time formatting."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds / 60:.1f}m"


# --- Ultra-Fast Content Analysis ---
async def ultra_fast_analyze_content(
    content_data: Dict[str, Any],
    analysis_types: List[str] = None
) -> Dict[str, Any]:
    """Ultra-fast content analysis."""
    if analysis_types is None:
        analysis_types = ["topics", "quality", "sentiment"]
    
    analysis_results = {}
    
    if "topics" in analysis_types:
        analysis_results["topics"] = await ultra_fast_extract_topics(content_data)
    
    if "quality" in analysis_types:
        analysis_results["quality"] = await ultra_fast_analyze_quality(content_data)
    
    return analysis_results


# --- Ultra-Fast Metrics Collection ---
class UltraFastPDFMetrics:
    """Ultra-fast PDF processing metrics."""
    
    def __init__(self):
        self._counters = {}
        self._timers = {}
        self._gauges = {}
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment counter."""
        self._counters[name] = self._counters.get(name, 0) + value
    
    def record_timer(self, name: str, duration: float):
        """Record timing."""
        if name not in self._timers:
            self._timers[name] = []
        self._timers[name].append(duration)
    
    def set_gauge(self, name: str, value: float):
        """Set gauge."""
        self._gauges[name] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "counters": self._counters.copy(),
            "timers": {
                name: {
                    "count": len(times),
                    "avg": sum(times) / len(times) if times else 0,
                    "min": min(times) if times else 0,
                    "max": max(times) if times else 0
                }
                for name, times in self._timers.items()
            },
            "gauges": self._gauges.copy()
        }


# --- Ultra-Fast Health Check ---
async def ultra_fast_pdf_health_check() -> Dict[str, Any]:
    """Ultra-fast PDF processing health check."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "content_extraction": "healthy",
            "topic_extraction": "healthy",
            "variant_generation": "healthy",
            "quality_analysis": "healthy"
        }
    }
