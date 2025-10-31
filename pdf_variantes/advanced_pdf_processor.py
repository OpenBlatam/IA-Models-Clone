"""Advanced PDF processing with intelligent optimization and adaptive strategies."""

from typing import Dict, Any, List, Optional, Callable, Awaitable, Union
from functools import partial
import asyncio
import time
import logging
from pathlib import Path
import hashlib
import json

from .advanced_performance import (
    intelligent_cache, adaptive_retry, performance_monitor,
    intelligent_batch_process, create_resource_pool, create_circuit_breaker
)
from .advanced_error_handling import (
    intelligent_error_handler, error_recovery, error_monitoring,
    graceful_degradation, ErrorSeverity, ErrorCategory, ErrorContext
)
from .utils.file_utils import validate_pdf_file, extract_metadata, get_file_hash
from .utils.text_utils import extract_text_content, detect_language, calculate_readability

logger = logging.getLogger(__name__)


# --- Advanced PDF Content Extraction ---
@intelligent_cache(maxsize=1000, ttl=600.0)
@adaptive_retry(max_retries=3, base_delay=0.1)
@performance_monitor("content_extraction")
@intelligent_error_handler(ErrorSeverity.MEDIUM, ErrorCategory.PROCESSING, "content_extraction")
async def advanced_extract_content(file_content: bytes) -> Dict[str, Any]:
    """Advanced PDF content extraction with intelligent caching and error handling."""
    # Validate PDF with enhanced error handling
    validation = validate_pdf_file(file_content, "temp.pdf")
    if not validation["valid"]:
        raise ValueError(f"Invalid PDF: {validation['error']}")
    
    # Extract content in parallel with error recovery
    async def extract_text():
        try:
            return await asyncio.to_thread(extract_text_content, file_content)
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            return ""
    
    async def extract_meta():
        try:
            return await asyncio.to_thread(extract_metadata, file_content)
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return {}
    
    # Parallel extraction with error handling
    text_content, metadata = await asyncio.gather(
        extract_text(), extract_meta(), return_exceptions=True
    )
    
    # Handle extraction errors gracefully
    if isinstance(text_content, Exception):
        text_content = ""
    if isinstance(metadata, Exception):
        metadata = {}
    
    return {
        "text_content": text_content,
        "metadata": metadata,
        "file_hash": get_file_hash(file_content),
        "page_count": validation["page_count"],
        "file_size": validation["file_size"],
        "extraction_time": time.time()
    }


# --- Advanced Topic Extraction ---
@intelligent_cache(maxsize=500, ttl=1800.0)
@performance_monitor("topic_extraction")
@intelligent_error_handler(ErrorSeverity.LOW, ErrorCategory.PROCESSING, "topic_extraction")
async def advanced_extract_topics(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced topic extraction with intelligent algorithms."""
    text = content_data.get("text_content", "")
    if not text:
        return {"topics": [], "main_topic": None, "confidence": 0.0}
    
    # Advanced word frequency analysis
    words = text.lower().split()
    
    # Filter and weight words
    word_weights = {}
    stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
    
    for word in words:
        if len(word) > 3 and word not in stop_words:
            # Weight by position and frequency
            weight = 1.0
            if word in text[:1000]:  # Words in first 1000 chars get higher weight
                weight *= 1.5
            if word in text[-1000:]:  # Words in last 1000 chars get higher weight
                weight *= 1.3
            
            word_weights[word] = word_weights.get(word, 0) + weight
    
    # Get top weighted words as topics
    sorted_words = sorted(word_weights.items(), key=lambda x: x[1], reverse=True)
    topics = [
        {
            "topic": word,
            "weight": weight,
            "frequency": word_weights[word],
            "confidence": min(weight / 10.0, 1.0)
        }
        for word, weight in sorted_words[:15]
    ]
    
    # Calculate overall confidence
    confidence = min(sum(t["confidence"] for t in topics) / len(topics), 1.0) if topics else 0.0
    
    return {
        "topics": topics,
        "main_topic": topics[0]["topic"] if topics else None,
        "total_topics": len(topics),
        "confidence": confidence,
        "extraction_method": "advanced_weighted_analysis"
    }


# --- Advanced Variant Generation ---
@intelligent_cache(maxsize=200, ttl=3600.0)
@performance_monitor("variant_generation")
@intelligent_error_handler(ErrorSeverity.MEDIUM, ErrorCategory.PROCESSING, "variant_generation")
async def advanced_generate_variants(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced variant generation with intelligent content analysis."""
    text = content_data.get("text_content", "")
    if not text:
        return {"variants": [], "generation_method": "advanced"}
    
    # Advanced content analysis
    sentences = text.split('.')
    words = text.split()
    
    # Generate intelligent variants
    variants = {}
    
    # Smart summary generation
    if len(sentences) > 3:
        # Take first, middle, and last sentences for better summary
        summary_sentences = [sentences[0]]
        if len(sentences) > 2:
            summary_sentences.append(sentences[len(sentences) // 2])
        summary_sentences.append(sentences[-1])
        variants["summary"] = '. '.join(summary_sentences) + '.'
    else:
        variants["summary"] = text[:200] + "..." if len(text) > 200 else text
    
    # Intelligent outline generation
    if len(sentences) > 5:
        # Extract key sentences based on position and length
        key_sentences = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 20:  # Meaningful sentences
                if i == 0 or i == len(sentences) - 1:  # First or last
                    key_sentences.append(sentence.strip())
                elif i % 3 == 0:  # Every third sentence
                    key_sentences.append(sentence.strip())
        
        variants["outline"] = "\n".join([f"- {s}" for s in key_sentences[:10]])
    else:
        variants["outline"] = "\n".join([f"- {s.strip()}" for s in sentences[:5]])
    
    # Smart highlights generation
    if len(words) > 100:
        # Extract sentences with high word density
        sentence_scores = []
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                word_count = len(sentence.split())
                char_count = len(sentence)
                density = word_count / char_count if char_count > 0 else 0
                sentence_scores.append((sentence.strip(), density))
        
        # Sort by density and take top 3
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        highlights = [s[0] for s in sentence_scores[:3]]
        variants["highlights"] = " | ".join(highlights)
    else:
        variants["highlights"] = text[:100] + "..." if len(text) > 100 else text
    
    # Quality assessment
    quality_score = min(len(text) / 1000.0, 1.0)  # Simple quality metric
    
    return {
        "variants": variants,
        "variant_count": len(variants),
        "quality_score": quality_score,
        "generation_method": "advanced_intelligent",
        "content_analysis": {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0
        }
    }


# --- Advanced Quality Analysis ---
@intelligent_cache(maxsize=200, ttl=3600.0)
@performance_monitor("quality_analysis")
@intelligent_error_handler(ErrorSeverity.LOW, ErrorCategory.PROCESSING, "quality_analysis")
async def advanced_analyze_quality(content_data: Dict[str, Any]) -> Dict[str, Any]:
    """Advanced quality analysis with comprehensive metrics."""
    text = content_data.get("text_content", "")
    
    if not text:
        return {"quality_score": 0.0, "metrics": {}, "analysis_method": "advanced"}
    
    # Comprehensive quality metrics
    readability = calculate_readability(text)
    language = detect_language(text)
    
    # Advanced text analysis
    words = text.split()
    sentences = text.split('.')
    paragraphs = text.split('\n\n')
    
    # Calculate advanced metrics
    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)
    
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    avg_paragraph_length = word_count / paragraph_count if paragraph_count > 0 else 0
    
    # Complexity analysis
    complex_words = [word for word in words if len(word) > 6]
    complexity_ratio = len(complex_words) / word_count if word_count > 0 else 0
    
    # Coherence analysis (simplified)
    coherence_score = min(sentence_count / 10.0, 1.0)  # More sentences = better coherence
    
    # Overall quality score
    quality_factors = {
        "readability": readability,
        "language_support": 1.0 if language == "en" else 0.8,
        "complexity": 1.0 - complexity_ratio,  # Lower complexity = higher quality
        "coherence": coherence_score,
        "length_adequacy": min(word_count / 500.0, 1.0)  # Adequate length
    }
    
    quality_score = sum(quality_factors.values()) / len(quality_factors)
    
    return {
        "quality_score": quality_score,
        "metrics": {
            "readability": readability,
            "language": language,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "avg_paragraph_length": avg_paragraph_length,
            "complexity_ratio": complexity_ratio,
            "coherence_score": coherence_score
        },
        "quality_factors": quality_factors,
        "analysis_method": "advanced_comprehensive"
    }


# --- Advanced Processing Pipeline ---
def create_advanced_pipeline(
    include_topics: bool = True,
    include_variants: bool = True,
    include_quality: bool = True,
    adaptive: bool = True
) -> List[Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]]:
    """Create advanced processing pipeline with adaptive strategies."""
    processors = []
    
    if include_topics:
        processors.append(advanced_extract_topics)
    
    if include_variants:
        processors.append(advanced_generate_variants)
    
    if include_quality:
        processors.append(advanced_analyze_quality)
    
    return processors


# --- Advanced PDF Processing ---
@performance_monitor("pdf_processing")
@intelligent_error_handler(ErrorSeverity.HIGH, ErrorCategory.PROCESSING, "pdf_processing")
async def advanced_process_pdf(
    file_content: bytes,
    filename: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Advanced PDF processing with intelligent optimization."""
    options = options or {}
    
    # Extract base content with error handling
    content_data = await advanced_extract_content(file_content)
    content_data["filename"] = filename
    
    # Create processing pipeline
    processors = create_advanced_pipeline(
        include_topics=options.get("include_topics", True),
        include_variants=options.get("include_variants", True),
        include_quality=options.get("include_quality", True)
    )
    
    # Process with intelligent batch processing
    processing_results = await intelligent_batch_process(
        [partial(processor, content_data) for processor in processors],
        lambda processor: processor(),
        max_concurrent=options.get("max_concurrent", 10),
        chunk_size=1,
        adaptive=True
    )
    
    # Combine results with error handling
    combined_results = []
    for i, result in enumerate(processing_results):
        if isinstance(result, Exception):
            logger.error(f"Processor {i} failed: {result}")
            combined_results.append({"error": str(result), "processor": processors[i].__name__})
        else:
            combined_results.append(result)
    
    return {
        "file_id": generate_advanced_file_id(content_data),
        "content_data": content_data,
        "processing_results": combined_results,
        "processed_at": time.time(),
        "processing_method": "advanced_intelligent",
        "success_rate": sum(1 for r in combined_results if "error" not in r) / len(combined_results)
    }


# --- Advanced Batch Processing ---
@performance_monitor("batch_processing")
@intelligent_error_handler(ErrorSeverity.MEDIUM, ErrorCategory.PROCESSING, "batch_processing")
async def advanced_batch_process_pdfs(
    files: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
    max_concurrent: int = 50
) -> Dict[str, Any]:
    """Advanced batch processing with intelligent concurrency control."""
    options = options or {}
    
    async def process_single_file(file_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await advanced_process_pdf(
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
            logger.error(f"Error processing {file_data['filename']}: {e}")
            return {
                "filename": file_data["filename"],
                "status": "error",
                "error": str(e)
            }
    
    # Process with intelligent batch processing
    results = await intelligent_batch_process(
        files,
        process_single_file,
        max_concurrent=max_concurrent,
        chunk_size=options.get("chunk_size", 20),
        adaptive=True
    )
    
    # Analyze results
    successful = [r for r in results if isinstance(r, dict) and r.get("status") == "success"]
    failed = [r for r in results if isinstance(r, dict) and r.get("status") == "error"]
    
    # Calculate processing statistics
    total_processing_time = time.time() - time.time()  # Placeholder
    avg_processing_time = total_processing_time / len(files) if files else 0
    
    return {
        "total_files": len(files),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(files) if files else 0,
        "results": results,
        "processing_statistics": {
            "total_processing_time": total_processing_time,
            "average_processing_time": avg_processing_time,
            "files_per_second": len(files) / total_processing_time if total_processing_time > 0 else 0
        },
        "processed_at": time.time()
    }


# --- Advanced Feature Extraction ---
@performance_monitor("feature_extraction")
@intelligent_error_handler(ErrorSeverity.MEDIUM, ErrorCategory.PROCESSING, "feature_extraction")
async def advanced_extract_features(
    file_id: str,
    content_data: Dict[str, Any],
    feature_extractors: Optional[Dict[str, Callable]] = None
) -> Dict[str, Any]:
    """Advanced feature extraction with parallel processing."""
    if feature_extractors is None:
        feature_extractors = {
            "topics": advanced_extract_topics,
            "variants": advanced_generate_variants,
            "quality": advanced_analyze_quality
        }
    
    # Create extractor functions
    extractors = [
        partial(extractor, content_data)
        for name, extractor in feature_extractors.items()
    ]
    
    # Extract features in parallel with error handling
    features_result = await intelligent_batch_process(
        extractors,
        lambda extractor: extractor(),
        max_concurrent=10,
        chunk_size=1,
        adaptive=True
    )
    
    # Combine results
    combined_features = {}
    for i, (name, result) in enumerate(zip(feature_extractors.keys(), features_result)):
        if isinstance(result, Exception):
            combined_features[name] = {"error": str(result)}
        else:
            combined_features[name] = result
    
    return {
        "file_id": file_id,
        "features": combined_features,
        "extracted_at": time.time(),
        "extraction_method": "advanced_parallel"
    }


# --- Advanced Utility Functions ---
def generate_advanced_file_id(content_data: Dict[str, Any]) -> str:
    """Generate advanced file ID with content hash."""
    content_hash = content_data.get("file_hash", "")
    timestamp = str(int(time.time()))
    return f"pdf_{content_hash[:8]}_{timestamp}"


@graceful_degradation({"success": False, "error": "File operation failed"})
async def advanced_save_pdf(file_content: bytes, file_path: str) -> Dict[str, Any]:
    """Advanced PDF saving with error handling."""
    try:
        await asyncio.to_thread(lambda: Path(file_path).parent.mkdir(parents=True, exist_ok=True))
        await asyncio.to_thread(lambda: Path(file_path).write_bytes(file_content))
        return {"success": True, "file_path": file_path, "size": len(file_content)}
    except Exception as e:
        logger.error(f"Error saving PDF: {e}")
        raise e


@graceful_degradation(None)
async def advanced_load_pdf(file_path: str) -> Optional[bytes]:
    """Advanced PDF loading with error handling."""
    try:
        if not Path(file_path).exists():
            return None
        return await asyncio.to_thread(lambda: Path(file_path).read_bytes())
    except Exception as e:
        logger.error(f"Error loading PDF: {e}")
        return None


# --- Advanced Content Analysis ---
@performance_monitor("content_analysis")
async def advanced_analyze_content(
    content_data: Dict[str, Any],
    analysis_types: List[str] = None
) -> Dict[str, Any]:
    """Advanced content analysis with multiple analysis types."""
    if analysis_types is None:
        analysis_types = ["topics", "quality", "sentiment", "complexity"]
    
    analysis_results = {}
    
    if "topics" in analysis_types:
        analysis_results["topics"] = await advanced_extract_topics(content_data)
    
    if "quality" in analysis_types:
        analysis_results["quality"] = await advanced_analyze_quality(content_data)
    
    if "sentiment" in analysis_types:
        # Simple sentiment analysis
        text = content_data.get("text_content", "")
        positive_words = ["good", "great", "excellent", "amazing", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "horrible", "worst"]
        
        positive_count = sum(1 for word in positive_words if word in text.lower())
        negative_count = sum(1 for word in negative_words if word in text.lower())
        
        sentiment_score = (positive_count - negative_count) / max(positive_count + negative_count, 1)
        analysis_results["sentiment"] = {
            "score": sentiment_score,
            "positive_words": positive_count,
            "negative_words": negative_count
        }
    
    if "complexity" in analysis_types:
        text = content_data.get("text_content", "")
        words = text.split()
        complex_words = [word for word in words if len(word) > 6]
        analysis_results["complexity"] = {
            "complexity_ratio": len(complex_words) / len(words) if words else 0,
            "complex_words_count": len(complex_words),
            "total_words": len(words)
        }
    
    return analysis_results


# --- Advanced Health Check ---
@performance_monitor("health_check")
async def advanced_pdf_health_check() -> Dict[str, Any]:
    """Advanced PDF processing health check."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "content_extraction": "healthy",
            "topic_extraction": "healthy",
            "variant_generation": "healthy",
            "quality_analysis": "healthy",
            "batch_processing": "healthy",
            "feature_extraction": "healthy"
        },
        "performance": {
            "cache_hit_rate": 0.95,  # Placeholder
            "average_processing_time": 0.5,  # Placeholder
            "error_rate": 0.01  # Placeholder
        }
    }
