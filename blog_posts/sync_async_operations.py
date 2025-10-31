from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import json
import hashlib
from typing import List, Optional, Dict, Any, Union, Tuple
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from dataclasses import dataclass, field
from enum import Enum
from fastapi import (
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, validator
import structlog
from .functional_fastapi_components import (
from typing import Any, List, Dict, Optional
import logging
"""
🔄 Synchronous vs Asynchronous Operations in FastAPI
====================================================

Demonstrates proper use of:
- `def` for synchronous operations (CPU-bound, I/O-free)
- `async def` for asynchronous operations (I/O-bound, database, external APIs)
- Clear separation of concerns
- Performance optimization
- Best practices for FastAPI applications
"""


    FastAPI, APIRouter, Depends, HTTPException, status, 
    Request, Response, BackgroundTasks, Query, Path, Body
)

    TextAnalysisRequest, BatchAnalysisRequest, AnalysisUpdateRequest,
    AnalysisResponse, BatchAnalysisResponse, PaginatedResponse,
    AnalysisTypeEnum, OptimizationTierEnum, AnalysisStatusEnum
)

# ============================================================================
# Type Definitions
# ============================================================================

@dataclass
class ProcessingMetrics:
    """Metrics for operation processing."""
    operation_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def complete(self, success: bool = True, error_message: Optional[str] = None):
        """Complete the operation and calculate duration."""
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.success = success
        self.error_message = error_message

@dataclass
class CacheEntry:
    """Cache entry for storing results."""
    data: Any
    timestamp: datetime
    ttl_seconds: int
    access_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl_seconds
    
    def access(self) -> Any:
        """Increment access count."""
        self.access_count += 1

# ============================================================================
# Synchronous Operations (CPU-bound, I/O-free)
# ============================================================================

def validate_text_content(text: str) -> Tuple[bool, List[str], List[str]]:
    """
    Synchronous text validation (CPU-bound operation).
    
    Args:
        text: Text content to validate
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    
    if not text:
        errors.append("Text content cannot be empty")
        return False, errors, warnings
    
    cleaned = text.strip()
    if not cleaned:
        errors.append("Text content cannot be whitespace only")
        return False, errors, warnings
    
    if len(cleaned) > 10000:
        errors.append("Text content too long (max 10000 characters)")
        return False, errors, warnings
    
    if len(cleaned) < 10:
        warnings.append("Text content is very short, analysis may be less accurate")
    
    return True, errors, warnings

def calculate_text_statistics(text: str) -> Dict[str, Any]:
    """
    Synchronous text statistics calculation (CPU-bound operation).
    
    Args:
        text: Text content to analyze
        
    Returns:
        Dictionary with text statistics
    """
    if not text:
        return {}
    
    words = text.split()
    sentences = text.split('.')
    paragraphs = text.split('\n\n')
    
    return {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "paragraph_count": len([p for p in paragraphs if p.strip()]),
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
        "unique_words": len(set(word.lower() for word in words)),
        "reading_time_minutes": len(words) / 200,  # Average reading speed
        "complexity_score": calculate_complexity_score(text)
    }

def calculate_complexity_score(text: str) -> float:
    """
    Synchronous complexity score calculation (CPU-bound operation).
    
    Args:
        text: Text content to analyze
        
    Returns:
        Complexity score (0.0 to 1.0)
    """
    if not text:
        return 0.0
    
    words = text.split()
    if not words:
        return 0.0
    
    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    # Calculate sentence complexity
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    # Calculate unique word ratio
    unique_ratio = len(set(word.lower() for word in words)) / len(words)
    
    # Normalize scores
    word_score = min(avg_word_length / 10, 1.0)
    sentence_score = min(avg_sentence_length / 30, 1.0)
    vocabulary_score = unique_ratio
    
    # Weighted average
    complexity = (word_score * 0.3 + sentence_score * 0.4 + vocabulary_score * 0.3)
    
    return min(complexity, 1.0)

def generate_cache_key(text: str, analysis_type: str, optimization_tier: str) -> str:
    """
    Synchronous cache key generation (CPU-bound operation).
    
    Args:
        text: Text content
        analysis_type: Type of analysis
        optimization_tier: Optimization tier
        
    Returns:
        Cache key string
    """
    content = f"{text}:{analysis_type}:{optimization_tier}"
    hash_object = hashlib.sha256(content.encode())
    return f"analysis:{hash_object.hexdigest()}"

def calculate_processing_priority(
    optimization_tier: str,
    text_length: int,
    analysis_type: str
) -> int:
    """
    Synchronous priority calculation (CPU-bound operation).
    
    Args:
        optimization_tier: Optimization tier
        text_length: Length of text
        analysis_type: Type of analysis
        
    Returns:
        Priority score (1-10)
    """
    base_priority = 5
    
    # Adjust based on optimization tier
    tier_multipliers = {
        "basic": 0.8,
        "standard": 1.0,
        "advanced": 1.2,
        "ultra": 1.5
    }
    
    # Adjust based on text length
    length_factor = min(text_length / 1000, 2.0)
    
    # Adjust based on analysis type complexity
    complexity_factors = {
        "sentiment": 1.0,
        "quality": 1.2,
        "emotion": 1.3,
        "language": 0.8,
        "keywords": 1.1,
        "readability": 1.0,
        "entities": 1.4,
        "topics": 1.5
    }
    
    priority = int(
        base_priority * 
        tier_multipliers.get(optimization_tier, 1.0) * 
        length_factor * 
        complexity_factors.get(analysis_type, 1.0)
    )
    
    return max(1, min(10, priority))

def estimate_processing_time(
    text_length: int,
    analysis_type: str,
    optimization_tier: str
) -> float:
    """
    Synchronous processing time estimation (CPU-bound operation).
    
    Args:
        text_length: Length of text
        analysis_type: Type of analysis
        optimization_tier: Optimization tier
        
    Returns:
        Estimated processing time in milliseconds
    """
    base_time_ms = 100.0
    
    # Adjust based on text length
    length_factor = text_length / 1000
    
    # Adjust based on analysis type
    type_factors = {
        "sentiment": 1.0,
        "quality": 1.5,
        "emotion": 1.8,
        "language": 0.5,
        "keywords": 1.2,
        "readability": 1.3,
        "entities": 2.0,
        "topics": 2.5
    }
    
    # Adjust based on optimization tier
    tier_factors = {
        "basic": 1.5,
        "standard": 1.0,
        "advanced": 0.8,
        "ultra": 0.6
    }
    
    estimated_time = (
        base_time_ms * 
        length_factor * 
        type_factors.get(analysis_type, 1.0) * 
        tier_factors.get(optimization_tier, 1.0)
    )
    
    return max(50.0, estimated_time)

@lru_cache(maxsize=1000)
def get_analysis_config(analysis_type: str, optimization_tier: str) -> Dict[str, Any]:
    """
    Synchronous configuration retrieval with caching (CPU-bound operation).
    
    Args:
        analysis_type: Type of analysis
        optimization_tier: Optimization tier
        
    Returns:
        Configuration dictionary
    """
    # Simulate configuration lookup
    configs = {
        "sentiment": {
            "model_name": "distilbert-sentiment",
            "batch_size": 32,
            "max_length": 512,
            "confidence_threshold": 0.7
        },
        "quality": {
            "model_name": "quality-assessment",
            "batch_size": 16,
            "max_length": 1024,
            "confidence_threshold": 0.8
        },
        "emotion": {
            "model_name": "emotion-classifier",
            "batch_size": 24,
            "max_length": 512,
            "confidence_threshold": 0.75
        }
    }
    
    base_config = configs.get(analysis_type, configs["sentiment"])
    
    # Adjust based on optimization tier
    tier_adjustments = {
        "basic": {"batch_size": 8, "confidence_threshold": 0.6},
        "standard": {},
        "advanced": {"batch_size": 64, "confidence_threshold": 0.85},
        "ultra": {"batch_size": 128, "confidence_threshold": 0.9}
    }
    
    adjustments = tier_adjustments.get(optimization_tier, {})
    config = base_config.copy()
    config.update(adjustments)
    
    return config

def transform_analysis_to_response(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous data transformation (CPU-bound operation).
    
    Args:
        analysis_data: Raw analysis data
        
    Returns:
        Transformed response data
    """
    if not analysis_data:
        return {}
    
    # Transform database model to response format
    response_data = {
        "id": analysis_data.get("id"),
        "text_content": analysis_data.get("text_content"),
        "analysis_type": analysis_data.get("analysis_type"),
        "status": analysis_data.get("status"),
        "sentiment_score": analysis_data.get("sentiment_score"),
        "quality_score": analysis_data.get("quality_score"),
        "processing_time_ms": analysis_data.get("processing_time_ms"),
        "model_used": analysis_data.get("model_used"),
        "confidence_score": analysis_data.get("confidence_score"),
        "optimization_tier": analysis_data.get("optimization_tier"),
        "created_at": analysis_data.get("created_at"),
        "updated_at": analysis_data.get("updated_at"),
        "processed_at": analysis_data.get("processed_at"),
        "metadata": analysis_data.get("metadata", {})
    }
    
    # Add calculated fields
    if response_data.get("text_content"):
        stats = calculate_text_statistics(response_data["text_content"])
        response_data["text_statistics"] = stats
    
    return response_data

def calculate_batch_progress(
    completed_count: int,
    error_count: int,
    total_count: int
) -> Dict[str, Any]:
    """
    Synchronous batch progress calculation (CPU-bound operation).
    
    Args:
        completed_count: Number of completed items
        error_count: Number of failed items
        total_count: Total number of items
        
    Returns:
        Progress information dictionary
    """
    if total_count == 0:
        return {
            "progress_percentage": 0.0,
            "success_rate": 0.0,
            "error_rate": 0.0,
            "remaining_count": 0
        }
    
    progress_percentage = ((completed_count + error_count) / total_count) * 100
    success_rate = (completed_count / total_count) * 100
    error_rate = (error_count / total_count) * 100
    remaining_count = total_count - completed_count - error_count
    
    return {
        "progress_percentage": round(progress_percentage, 2),
        "success_rate": round(success_rate, 2),
        "error_rate": round(error_rate, 2),
        "remaining_count": remaining_count
    }

# ============================================================================
# Asynchronous Operations (I/O-bound, database, external APIs)
# ============================================================================

async def create_analysis_async(
    request: TextAnalysisRequest,
    db_manager: Any
) -> AnalysisResponse:
    """
    Asynchronous analysis creation (I/O-bound operation).
    
    Args:
        request: Analysis request
        db_manager: Database manager
        
    Returns:
        Created analysis response
    """
    # Validate input synchronously
    is_valid, errors, warnings = validate_text_content(request.text_content)
    if not is_valid:
        raise ValueError(f"Validation failed: {', '.join(errors)}")
    
    # Calculate priority synchronously
    priority = calculate_processing_priority(
        request.optimization_tier,
        len(request.text_content),
        request.analysis_type
    )
    
    # Get configuration synchronously
    config = get_analysis_config(request.analysis_type, request.optimization_tier)
    
    # Create analysis data
    analysis_data = {
        "text_content": request.text_content,
        "analysis_type": request.analysis_type,
        "optimization_tier": request.optimization_tier,
        "priority": priority,
        "config": config,
        "metadata": request.metadata
    }
    
    # Save to database asynchronously
    analysis = await db_manager.create_text_analysis(analysis_data)
    
    # Transform response synchronously
    response_data = transform_analysis_to_response(analysis)
    
    return AnalysisResponse(**response_data)

async def get_analysis_async(
    analysis_id: int,
    db_manager: Any
) -> Optional[AnalysisResponse]:
    """
    Asynchronous analysis retrieval (I/O-bound operation).
    
    Args:
        analysis_id: Analysis ID
        db_manager: Database manager
        
    Returns:
        Analysis response or None if not found
    """
    # Retrieve from database asynchronously
    analysis = await db_manager.get_text_analysis(analysis_id)
    
    if not analysis:
        return None
    
    # Transform response synchronously
    response_data = transform_analysis_to_response(analysis)
    
    return AnalysisResponse(**response_data)

async def update_analysis_async(
    analysis_id: int,
    update_data: Dict[str, Any],
    db_manager: Any
) -> Optional[AnalysisResponse]:
    """
    Asynchronous analysis update (I/O-bound operation).
    
    Args:
        analysis_id: Analysis ID
        update_data: Update data
        db_manager: Database manager
        
    Returns:
        Updated analysis response or None if not found
    """
    # Update in database asynchronously
    analysis = await db_manager.update_text_analysis(analysis_id, update_data)
    
    if not analysis:
        return None
    
    # Transform response synchronously
    response_data = transform_analysis_to_response(analysis)
    
    return AnalysisResponse(**response_data)

async def list_analyses_async(
    pagination: Dict[str, Any],
    filters: Dict[str, Any],
    db_manager: Any
) -> Tuple[List[AnalysisResponse], int]:
    """
    Asynchronous analysis listing (I/O-bound operation).
    
    Args:
        pagination: Pagination parameters
        filters: Filter parameters
        db_manager: Database manager
        
    Returns:
        Tuple of (analyses list, total count)
    """
    # Query database asynchronously
    analyses, total_count = await db_manager.list_text_analyses(
        limit=pagination.get("size", 20),
        offset=pagination.get("offset", 0),
        order_by=pagination.get("order_by", "created_at"),
        order_desc=pagination.get("order_desc", True),
        **filters
    )
    
    # Transform responses synchronously
    response_analyses = []
    for analysis in analyses:
        response_data = transform_analysis_to_response(analysis)
        response_analyses.append(AnalysisResponse(**response_data))
    
    return response_analyses, total_count

async def process_analysis_background_async(
    analysis_id: int,
    text_content: str,
    analysis_type: str,
    db_manager: Any
):
    """
    Asynchronous background analysis processing (I/O-bound operation).
    
    Args:
        analysis_id: Analysis ID
        text_content: Text content to analyze
        analysis_type: Type of analysis
        db_manager: Database manager
    """
    logger = structlog.get_logger("background_processor")
    
    try:
        logger.info(f"Starting background processing for analysis {analysis_id}")
        
        # Simulate async processing
        await asyncio.sleep(2)
        
        # Calculate statistics synchronously
        stats = calculate_text_statistics(text_content)
        
        # Estimate processing time synchronously
        estimated_time = estimate_processing_time(
            len(text_content),
            analysis_type,
            "standard"
        )
        
        # Update analysis asynchronously
        update_data = {
            "status": "completed",
            "sentiment_score": 0.5,
            "processing_time_ms": estimated_time,
            "model_used": "background-processor",
            "metadata": {"statistics": stats}
        }
        
        await db_manager.update_text_analysis(analysis_id, update_data)
        
        logger.info(f"Completed background processing for analysis {analysis_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed for analysis {analysis_id}: {e}")
        
        # Update with error asynchronously
        error_data = {
            "status": "error",
            "error_message": str(e)
        }
        
        await db_manager.update_text_analysis(analysis_id, error_data)

async def create_batch_analysis_async(
    request: BatchAnalysisRequest,
    db_manager: Any
) -> BatchAnalysisResponse:
    """
    Asynchronous batch analysis creation (I/O-bound operation).
    
    Args:
        request: Batch analysis request
        db_manager: Database manager
        
    Returns:
        Created batch response
    """
    # Validate texts synchronously
    for text in request.texts:
        is_valid, errors, warnings = validate_text_content(text)
        if not is_valid:
            raise ValueError(f"Invalid text in batch: {', '.join(errors)}")
    
    # Calculate batch statistics synchronously
    total_texts = len(request.texts)
    total_chars = sum(len(text) for text in request.texts)
    avg_text_length = total_chars / total_texts if total_texts > 0 else 0
    
    # Create batch data
    batch_data = {
        "batch_name": request.batch_name,
        "analysis_type": request.analysis_type,
        "optimization_tier": request.optimization_tier,
        "total_texts": total_texts,
        "priority": request.priority,
        "metadata": {
            "total_characters": total_chars,
            "average_text_length": avg_text_length
        }
    }
    
    # Save to database asynchronously
    batch = await db_manager.create_batch_analysis(batch_data)
    
    # Calculate initial progress synchronously
    progress = calculate_batch_progress(0, 0, total_texts)
    
    return BatchAnalysisResponse(
        id=batch.id,
        batch_name=batch.batch_name,
        analysis_type=batch.analysis_type,
        status=batch.status,
        total_texts=total_texts,
        completed_count=0,
        error_count=0,
        progress_percentage=progress["progress_percentage"],
        optimization_tier=batch.optimization_tier,
        priority=request.priority,
        created_at=batch.created_at,
        updated_at=batch.updated_at,
        metadata=batch.metadata
    )

async def process_batch_texts_async(
    batch_id: int,
    texts: List[str],
    analysis_type: str,
    db_manager: Any
):
    """
    Asynchronous batch text processing (I/O-bound operation).
    
    Args:
        batch_id: Batch ID
        texts: List of texts to process
        analysis_type: Type of analysis
        db_manager: Database manager
    """
    logger = structlog.get_logger("batch_processor")
    
    completed_count = 0
    error_count = 0
    
    logger.info(f"Starting batch processing for batch {batch_id}")
    
    # Process texts concurrently
    tasks = []
    for i, text in enumerate(texts):
        task = process_single_text_async(
            batch_id=batch_id,
            text_index=i,
            text=text,
            analysis_type=analysis_type,
            db_manager=db_manager
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count results
    for result in results:
        if isinstance(result, Exception):
            error_count += 1
        else:
            completed_count += 1
    
    # Update batch progress asynchronously
    await db_manager.update_batch_progress(batch_id, completed_count, error_count)
    
    logger.info(
        f"Completed batch {batch_id}: {completed_count} successful, {error_count} errors"
    )

async def process_single_text_async(
    batch_id: int,
    text_index: int,
    text: str,
    analysis_type: str,
    db_manager: Any
):
    """
    Asynchronous single text processing (I/O-bound operation).
    
    Args:
        batch_id: Batch ID
        text_index: Index of text in batch
        text: Text content
        analysis_type: Type of analysis
        db_manager: Database manager
    """
    try:
        # Create analysis request
        request = TextAnalysisRequest(
            text_content=text,
            analysis_type=analysis_type
        )
        
        # Process analysis asynchronously
        analysis_result = await create_analysis_async(request, db_manager)
        
        # Update with results asynchronously
        update_data = {
            "status": "completed",
            "sentiment_score": 0.5 + (text_index * 0.1),
            "processing_time_ms": 100.0 + text_index,
            "model_used": "batch-processor",
            "metadata": {"batch_id": batch_id, "text_index": text_index}
        }
        
        await db_manager.update_text_analysis(analysis_result.id, update_data)
        
    except Exception as e:
        logger = structlog.get_logger("batch_processor")
        logger.error(f"Error processing text {text_index} in batch {batch_id}: {e}")

# ============================================================================
# Cache Management (Mixed sync/async operations)
# ============================================================================

class AnalysisCache:
    """Cache manager for analysis results."""
    
    def __init__(self) -> Any:
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    def get_sync(self, key: str) -> Optional[Any]:
        """
        Synchronous cache retrieval (CPU-bound operation).
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        entry = self._cache.get(key)
        
        if not entry or entry.is_expired():
            if entry:
                del self._cache[key]
            return None
        
        entry.access()
        return entry.data
    
    async def get_async(self, key: str) -> Optional[Any]:
        """
        Asynchronous cache retrieval with cleanup (I/O-bound operation).
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found/expired
        """
        async with self._lock:
            return self.get_sync(key)
    
    def set_sync(self, key: str, data: Any, ttl_seconds: int = 3600):
        """
        Synchronous cache storage (CPU-bound operation).
        
        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Time to live in seconds
        """
        self._cache[key] = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds
        )
    
    async def set_async(self, key: str, data: Any, ttl_seconds: int = 3600):
        """
        Asynchronous cache storage (I/O-bound operation).
        
        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Time to live in seconds
        """
        async with self._lock:
            self.set_sync(key, data, ttl_seconds)
    
    def cleanup_sync(self) -> int:
        """
        Synchronous cache cleanup (CPU-bound operation).
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        return len(expired_keys)
    
    async def cleanup_async(self) -> int:
        """
        Asynchronous cache cleanup (I/O-bound operation).
        
        Returns:
            Number of entries removed
        """
        async with self._lock:
            return self.cleanup_sync()
    
    def get_stats_sync(self) -> Dict[str, Any]:
        """
        Synchronous cache statistics (CPU-bound operation).
        
        Returns:
            Cache statistics
        """
        total_entries = len(self._cache)
        expired_entries = sum(1 for entry in self._cache.values() if entry.is_expired())
        total_accesses = sum(entry.access_count for entry in self._cache.values())
        
        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "total_accesses": total_accesses,
            "average_accesses": total_accesses / total_entries if total_entries > 0 else 0
        }

# ============================================================================
# Route Handlers (Mixed sync/async operations)
# ============================================================================

async def create_analysis_handler(
    request: TextAnalysisRequest,
    background_tasks: BackgroundTasks,
    db_manager: Any,
    cache: AnalysisCache
) -> AnalysisResponse:
    """
    Route handler for creating analysis (mixed sync/async operations).
    
    Args:
        request: Analysis request
        background_tasks: Background task manager
        db_manager: Database manager
        cache: Cache manager
        
    Returns:
        Analysis response
    """
    # Generate cache key synchronously
    cache_key = generate_cache_key(
        request.text_content,
        request.analysis_type,
        request.optimization_tier
    )
    
    # Check cache asynchronously
    cached_result = await cache.get_async(cache_key)
    if cached_result:
        return AnalysisResponse(**cached_result)
    
    # Create analysis asynchronously
    analysis = await create_analysis_async(request, db_manager)
    
    # Cache result asynchronously
    response_data = analysis.model_dump()
    await cache.set_async(cache_key, response_data, ttl_seconds=1800)
    
    # Add background processing task
    background_tasks.add_task(
        process_analysis_background_async,
        analysis.id,
        request.text_content,
        request.analysis_type,
        db_manager
    )
    
    return analysis

async def get_analysis_handler(
    analysis_id: int,
    db_manager: Any,
    cache: AnalysisCache
) -> AnalysisResponse:
    """
    Route handler for getting analysis (mixed sync/async operations).
    
    Args:
        analysis_id: Analysis ID
        db_manager: Database manager
        cache: Cache manager
        
    Returns:
        Analysis response
    """
    # Generate cache key synchronously
    cache_key = f"analysis:{analysis_id}"
    
    # Check cache asynchronously
    cached_result = await cache.get_async(cache_key)
    if cached_result:
        return AnalysisResponse(**cached_result)
    
    # Get analysis asynchronously
    analysis = await get_analysis_async(analysis_id, db_manager)
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found"
        )
    
    # Cache result asynchronously
    response_data = analysis.model_dump()
    await cache.set_async(cache_key, response_data, ttl_seconds=3600)
    
    return analysis

async def list_analyses_handler(
    pagination: Dict[str, Any],
    filters: Dict[str, Any],
    db_manager: Any
) -> PaginatedResponse[AnalysisResponse]:
    """
    Route handler for listing analyses (mixed sync/async operations).
    
    Args:
        pagination: Pagination parameters
        filters: Filter parameters
        db_manager: Database manager
        
    Returns:
        Paginated analysis responses
    """
    # Validate pagination parameters synchronously
    page = pagination.get("page", 1)
    size = pagination.get("size", 20)
    
    if page < 1 or size < 1 or size > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid pagination parameters"
        )
    
    # Calculate offset synchronously
    offset = (page - 1) * size
    
    # Update pagination with calculated offset
    pagination["offset"] = offset
    
    # List analyses asynchronously
    analyses, total_count = await list_analyses_async(pagination, filters, db_manager)
    
    # Calculate pagination metadata synchronously
    total_pages = (total_count + size - 1) // size
    
    return PaginatedResponse(
        items=analyses,
        total=total_count,
        page=page,
        size=size,
        pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )

# ============================================================================
# Performance Monitoring (Mixed sync/async operations)
# ============================================================================

class PerformanceMonitor:
    """Performance monitoring for sync/async operations."""
    
    def __init__(self) -> Any:
        self._metrics: List[ProcessingMetrics] = []
        self._lock = asyncio.Lock()
    
    def start_operation_sync(self, operation_type: str) -> ProcessingMetrics:
        """
        Start operation monitoring synchronously.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            Processing metrics
        """
        metrics = ProcessingMetrics(
            operation_type=operation_type,
            start_time=datetime.now()
        )
        return metrics
    
    async def start_operation_async(self, operation_type: str) -> ProcessingMetrics:
        """
        Start operation monitoring asynchronously.
        
        Args:
            operation_type: Type of operation
            
        Returns:
            Processing metrics
        """
        async with self._lock:
            return self.start_operation_sync(operation_type)
    
    def complete_operation_sync(self, metrics: ProcessingMetrics, success: bool = True, error_message: Optional[str] = None):
        """
        Complete operation monitoring synchronously.
        
        Args:
            metrics: Processing metrics
            success: Operation success status
            error_message: Error message if failed
        """
        metrics.complete(success, error_message)
        self._metrics.append(metrics)
    
    async def complete_operation_async(self, metrics: ProcessingMetrics, success: bool = True, error_message: Optional[str] = None):
        """
        Complete operation monitoring asynchronously.
        
        Args:
            metrics: Processing metrics
            success: Operation success status
            error_message: Error message if failed
        """
        async with self._lock:
            self.complete_operation_sync(metrics, success, error_message)
    
    def get_stats_sync(self) -> Dict[str, Any]:
        """
        Get performance statistics synchronously.
        
        Returns:
            Performance statistics
        """
        if not self._metrics:
            return {}
        
        successful_ops = [m for m in self._metrics if m.success]
        failed_ops = [m for m in self._metrics if not m.success]
        
        total_duration = sum(m.duration_ms or 0 for m in self._metrics)
        avg_duration = total_duration / len(self._metrics) if self._metrics else 0
        
        return {
            "total_operations": len(self._metrics),
            "successful_operations": len(successful_ops),
            "failed_operations": len(failed_ops),
            "success_rate": len(successful_ops) / len(self._metrics) * 100,
            "average_duration_ms": avg_duration,
            "total_duration_ms": total_duration
        }
    
    async def get_stats_async(self) -> Dict[str, Any]:
        """
        Get performance statistics asynchronously.
        
        Returns:
            Performance statistics
        """
        async with self._lock:
            return self.get_stats_sync()

# ============================================================================
# Example Usage
# ============================================================================

async def example_sync_async_usage():
    """Example usage of sync/async operations."""
    
    # Initialize components
    cache = AnalysisCache()
    monitor = PerformanceMonitor()
    
    # Example text
    text = "This is a sample text for analysis. It contains multiple sentences and demonstrates various linguistic features."
    
    # Synchronous operations
    print("=== Synchronous Operations ===")
    
    # Text validation
    is_valid, errors, warnings = validate_text_content(text)
    print(f"Text validation: {is_valid}")
    if warnings:
        print(f"Warnings: {warnings}")
    
    # Text statistics
    stats = calculate_text_statistics(text)
    print(f"Text statistics: {stats}")
    
    # Cache key generation
    cache_key = generate_cache_key(text, "sentiment", "standard")
    print(f"Cache key: {cache_key}")
    
    # Priority calculation
    priority = calculate_processing_priority("standard", len(text), "sentiment")
    print(f"Processing priority: {priority}")
    
    # Asynchronous operations
    print("\n=== Asynchronous Operations ===")
    
    # Mock database manager
    class MockDBManager:
        async def create_text_analysis(self, data) -> Any:
            return {"id": 1, **data}
        
        async def get_text_analysis(self, analysis_id) -> Optional[Dict[str, Any]]:
            return {"id": analysis_id, "text_content": text, "status": "completed"}
    
    db_manager = MockDBManager()
    
    # Create analysis request
    request = TextAnalysisRequest(
        text_content=text,
        analysis_type=AnalysisTypeEnum.SENTIMENT,
        optimization_tier=OptimizationTierEnum.STANDARD
    )
    
    # Create analysis asynchronously
    analysis = await create_analysis_async(request, db_manager)
    print(f"Created analysis: {analysis.id}")
    
    # Cache operations
    await cache.set_async(cache_key, {"test": "data"}, ttl_seconds=60)
    cached_data = await cache.get_async(cache_key)
    print(f"Cached data: {cached_data}")
    
    # Performance monitoring
    metrics = await monitor.start_operation_async("example_operation")
    await asyncio.sleep(0.1)  # Simulate work
    await monitor.complete_operation_async(metrics, success=True)
    
    stats = await monitor.get_stats_async()
    print(f"Performance stats: {stats}")

match __name__:
    case "__main__":
    asyncio.run(example_sync_async_usage()) 