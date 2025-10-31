"""Enhanced PDF processing with functional patterns."""

from typing import Dict, Any, List, Optional, Callable, Awaitable
from functools import partial
import asyncio
import logging
from datetime import datetime

from .functional_utils import pipe, async_pipe, memoize_async, retry_with_backoff
from .async_services import with_timeout, parallel_extract_features
from .utils.file_utils import validate_pdf_file, extract_metadata, get_file_hash
from .utils.text_utils import extract_text_content, detect_language, calculate_readability

logger = logging.getLogger(__name__)


@memoize_async(maxsize=100)
async def extract_pdf_content(file_content: bytes) -> Dict[str, Any]:
    """Extract content from PDF with caching."""
    validation = validate_pdf_file(file_content, "temp.pdf")
    if not validation["valid"]:
        raise ValueError(f"Invalid PDF: {validation['error']}")
    
    text_content = extract_text_content(file_content)
    metadata = extract_metadata(file_content)
    
    return {
        "text_content": text_content,
        "metadata": metadata,
        "file_hash": get_file_hash(file_content),
        "page_count": validation["page_count"],
        "file_size": validation["file_size"]
    }


@retry_with_backoff(max_retries=3)
async def process_pdf_with_retry(
    file_content: bytes,
    filename: str,
    processors: List[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]
) -> Dict[str, Any]:
    """Process PDF with retry logic."""
    # Extract base content
    content_data = await extract_pdf_content(file_content)
    content_data["filename"] = filename
    
    # Apply processors in sequence
    current_data = content_data
    processing_results = []
    
    for i, processor in enumerate(processors):
        try:
            result = await processor(current_data)
            processing_results.append({
                "step": i + 1,
                "processor": processor.__name__,
                "success": True,
                "result": result
            })
            current_data.update(result)
        except Exception as e:
            processing_results.append({
                "step": i + 1,
                "processor": processor.__name__,
                "success": False,
                "error": str(e)
            })
            break
    
    return {
        "file_id": generate_file_id(),
        "content_data": current_data,
        "processing_results": processing_results,
        "processed_at": datetime.utcnow().isoformat()
    }


async def extract_topics_async(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract topics from content data."""
    text = content_data.get("text_content", "")
    if not text:
        return {"topics": [], "main_topic": None}
    
    # Simple topic extraction (replace with actual implementation)
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


async def generate_variants_async(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate variants from content data."""
    text = content_data.get("text_content", "")
    if not text:
        return {"variants": []}
    
    # Simple variant generation (replace with actual implementation)
    variants = {
        "summary": text[:200] + "..." if len(text) > 200 else text,
        "outline": "\n".join([f"- {line.strip()}" for line in text.split('\n')[:5]]),
        "highlights": text[:100] + "..." if len(text) > 100 else text
    }
    
    return {
        "variants": variants,
        "variant_count": len(variants)
    }


async def analyze_content_quality(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze content quality."""
    text = content_data.get("text_content", "")
    
    if not text:
        return {"quality_score": 0.0, "metrics": {}}
    
    readability = calculate_readability(text)
    language = detect_language(text)
    
    # Calculate additional quality metrics
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


def create_pdf_processing_pipeline(
    include_topics: bool = True,
    include_variants: bool = True,
    include_quality: bool = True
) -> List[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]:
    """Create a PDF processing pipeline."""
    processors = []
    
    if include_topics:
        processors.append(extract_topics_async)
    
    if include_variants:
        processors.append(generate_variants_async)
    
    if include_quality:
        processors.append(analyze_content_quality)
    
    return processors


async def process_pdf_complete(
    file_content: bytes,
    filename: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Complete PDF processing with all features."""
    options = options or {}
    
    # Create processing pipeline
    processors = create_pdf_processing_pipeline(
        include_topics=options.get("include_topics", True),
        include_variants=options.get("include_variants", True),
        include_quality=options.get("include_quality", True)
    )
    
    # Process with timeout
    result = await with_timeout(
        process_pdf_with_retry(file_content, filename, processors),
        timeout_seconds=options.get("timeout", 30.0)
    )
    
    return result


async def batch_process_pdfs(
    files: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
    max_concurrent: int = 5
) -> Dict[str, Any]:
    """Batch process multiple PDFs with concurrency control."""
    options = options or {}
    
    async def process_single_file(file_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await process_pdf_complete(
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
    
    # Process with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(file_data):
        async with semaphore:
            return await process_single_file(file_data)
    
    results = await asyncio.gather(
        *[process_with_semaphore(file_data) for file_data in files],
        return_exceptions=True
    )
    
    # Separate successful and failed results
    successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    failed = [r for r in results if isinstance(r, dict) and r.get("status") == "error"]
    
    return {
        "total_files": len(files),
        "successful": len(successful),
        "failed": len(failed),
        "results": results,
        "processed_at": datetime.utcnow().isoformat()
    }


def generate_file_id() -> str:
    """Generate a unique file ID."""
    import secrets
    return secrets.token_hex(16)


async def extract_features_parallel(
    file_id: str,
    content_data: Dict[str, Any],
    feature_extractors: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """Extract multiple features in parallel."""
    if feature_extractors is None:
        feature_extractors = {
            "topics": extract_topics_async,
            "variants": generate_variants_async,
            "quality": analyze_content_quality
        }
    
    # Create extractor functions that take file_id and content_data
    extractors = {
        name: partial(extractor, content_data)
        for name, extractor in feature_extractors.items()
    }
    
    return await parallel_extract_features(file_id, extractors)


def create_content_analyzer(
    analysis_types: List[str] = None
) -> Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]:
    """Create a content analyzer function."""
    if analysis_types is None:
        analysis_types = ["topics", "quality", "sentiment"]
    
    async def analyze_content(content_data: Dict[str, Any]) -> Dict[str, Any]:
        analysis_results = {}
        
        if "topics" in analysis_types:
            analysis_results["topics"] = await extract_topics_async(content_data)
        
        if "quality" in analysis_types:
            analysis_results["quality"] = await analyze_content_quality(content_data)
        
        return analysis_results
    
    return analyze_content


async def process_with_metrics(
    processor: Callable[[Any], Awaitable[Any]],
    input_data: Any,
    metrics_collector: Optional[Dict[str, Callable]] = None
) -> Any:
    """Process data with metrics collection."""
    if metrics_collector is None:
        metrics_collector = create_metrics_collector()
    
    start_time = time.time()
    
    try:
        result = await processor(input_data)
        
        # Record success metrics
        metrics_collector["increment_counter"]("processing_success")
        metrics_collector["record_timer"]("processing_time", time.time() - start_time)
        
        return result
        
    except Exception as e:
        # Record failure metrics
        metrics_collector["increment_counter"]("processing_failure")
        metrics_collector["record_timer"]("processing_time", time.time() - start_time)
        
        raise e
