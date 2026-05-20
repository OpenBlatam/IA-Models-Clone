from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from .sync_async_operations import (
from .functional_fastapi_components import (
        import time
        import time
from typing import Any, List, Dict, Optional
import logging
"""
🧪 Test Suite for Synchronous vs Asynchronous Operations
=======================================================

Comprehensive tests for:
- Synchronous operations (CPU-bound)
- Asynchronous operations (I/O-bound)
- Mixed sync/async operations
- Performance monitoring
- Cache management
- Error handling
- Edge cases
"""


    # Synchronous functions
    validate_text_content,
    calculate_text_statistics,
    calculate_complexity_score,
    generate_cache_key,
    calculate_processing_priority,
    estimate_processing_time,
    get_analysis_config,
    transform_analysis_to_response,
    calculate_batch_progress,
    
    # Asynchronous functions
    create_analysis_async,
    get_analysis_async,
    update_analysis_async,
    list_analyses_async,
    process_analysis_background_async,
    create_batch_analysis_async,
    process_batch_texts_async,
    process_single_text_async,
    
    # Classes
    AnalysisCache,
    PerformanceMonitor,
    ProcessingMetrics,
    CacheEntry,
    
    # Route handlers
    create_analysis_handler,
    get_analysis_handler,
    list_analyses_handler
)

    TextAnalysisRequest,
    BatchAnalysisRequest,
    AnalysisResponse,
    BatchAnalysisResponse,
    PaginatedResponse,
    AnalysisTypeEnum,
    OptimizationTierEnum,
    AnalysisStatusEnum
)

# ============================================================================
# Test Data
# ============================================================================

SAMPLE_TEXT = """
This is a sample text for testing purposes. It contains multiple sentences 
and demonstrates various linguistic features that can be analyzed.

The text includes different types of content such as technical terms, 
everyday language, and complex sentence structures. This allows us to 
test various analysis functions effectively.
"""

SAMPLE_TEXT_SHORT = "Short text."
SAMPLE_TEXT_LONG = "A" * 15000
SAMPLE_TEXT_EMPTY = ""
SAMPLE_TEXT_WHITESPACE = "   \n\t   "

MOCK_ANALYSIS_DATA = {
    "id": 1,
    "text_content": SAMPLE_TEXT,
    "analysis_type": "sentiment",
    "status": "completed",
    "sentiment_score": 0.75,
    "quality_score": 0.85,
    "processing_time_ms": 150.0,
    "model_used": "test-model",
    "confidence_score": 0.9,
    "optimization_tier": "standard",
    "created_at": datetime.now(),
    "updated_at": datetime.now(),
    "processed_at": datetime.now(),
    "metadata": {"test": "data"}
}

# ============================================================================
# Synchronous Operations Tests
# ============================================================================

class TestSynchronousOperations:
    """Test suite for synchronous operations."""
    
    def test_validate_text_content_valid(self) -> bool:
        """Test text validation with valid content."""
        is_valid, errors, warnings = validate_text_content(SAMPLE_TEXT)
        
        assert is_valid is True
        assert len(errors) == 0
        assert len(warnings) == 0
    
    def test_validate_text_content_empty(self) -> bool:
        """Test text validation with empty content."""
        is_valid, errors, warnings = validate_text_content(SAMPLE_TEXT_EMPTY)
        
        assert is_valid is False
        assert len(errors) == 1
        assert "empty" in errors[0].lower()
        assert len(warnings) == 0
    
    def test_validate_text_content_whitespace(self) -> bool:
        """Test text validation with whitespace-only content."""
        is_valid, errors, warnings = validate_text_content(SAMPLE_TEXT_WHITESPACE)
        
        assert is_valid is False
        assert len(errors) == 1
        assert "whitespace" in errors[0].lower()
        assert len(warnings) == 0
    
    def test_validate_text_content_too_long(self) -> bool:
        """Test text validation with content that's too long."""
        is_valid, errors, warnings = validate_text_content(SAMPLE_TEXT_LONG)
        
        assert is_valid is False
        assert len(errors) == 1
        assert "too long" in errors[0].lower()
        assert len(warnings) == 0
    
    def test_validate_text_content_too_short(self) -> bool:
        """Test text validation with content that's too short."""
        is_valid, errors, warnings = validate_text_content(SAMPLE_TEXT_SHORT)
        
        assert is_valid is True
        assert len(errors) == 0
        assert len(warnings) == 1
        assert "short" in warnings[0].lower()
    
    def test_calculate_text_statistics(self) -> Any:
        """Test text statistics calculation."""
        stats = calculate_text_statistics(SAMPLE_TEXT)
        
        assert isinstance(stats, dict)
        assert "character_count" in stats
        assert "word_count" in stats
        assert "sentence_count" in stats
        assert "paragraph_count" in stats
        assert "average_word_length" in stats
        assert "unique_words" in stats
        assert "reading_time_minutes" in stats
        assert "complexity_score" in stats
        
        assert stats["character_count"] > 0
        assert stats["word_count"] > 0
        assert stats["sentence_count"] > 0
        assert stats["average_word_length"] > 0
        assert 0 <= stats["complexity_score"] <= 1
    
    def test_calculate_text_statistics_empty(self) -> Any:
        """Test text statistics calculation with empty text."""
        stats = calculate_text_statistics("")
        
        assert isinstance(stats, dict)
        assert len(stats) == 0
    
    def test_calculate_complexity_score(self) -> Any:
        """Test complexity score calculation."""
        score = calculate_complexity_score(SAMPLE_TEXT)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_calculate_complexity_score_empty(self) -> Any:
        """Test complexity score calculation with empty text."""
        score = calculate_complexity_score("")
        
        assert score == 0.0
    
    def test_generate_cache_key(self) -> Any:
        """Test cache key generation."""
        key = generate_cache_key("test text", "sentiment", "standard")
        
        assert isinstance(key, str)
        assert key.startswith("analysis:")
        assert len(key) > 20  # Should be a hash
    
    def test_generate_cache_key_consistent(self) -> Any:
        """Test that cache key generation is consistent."""
        key1 = generate_cache_key("test text", "sentiment", "standard")
        key2 = generate_cache_key("test text", "sentiment", "standard")
        
        assert key1 == key2
    
    def test_calculate_processing_priority(self) -> Any:
        """Test processing priority calculation."""
        priority = calculate_processing_priority("standard", 1000, "sentiment")
        
        assert isinstance(priority, int)
        assert 1 <= priority <= 10
    
    def test_calculate_processing_priority_different_tiers(self) -> Any:
        """Test processing priority with different optimization tiers."""
        basic_priority = calculate_processing_priority("basic", 1000, "sentiment")
        standard_priority = calculate_processing_priority("standard", 1000, "sentiment")
        advanced_priority = calculate_processing_priority("advanced", 1000, "sentiment")
        ultra_priority = calculate_processing_priority("ultra", 1000, "sentiment")
        
        # Ultra should have higher priority than basic
        assert ultra_priority >= basic_priority
    
    def test_estimate_processing_time(self) -> Any:
        """Test processing time estimation."""
        time_ms = estimate_processing_time(1000, "sentiment", "standard")
        
        assert isinstance(time_ms, float)
        assert time_ms > 0
    
    def test_get_analysis_config(self) -> Optional[Dict[str, Any]]:
        """Test analysis configuration retrieval."""
        config = get_analysis_config("sentiment", "standard")
        
        assert isinstance(config, dict)
        assert "model_name" in config
        assert "batch_size" in config
        assert "max_length" in config
        assert "confidence_threshold" in config
    
    def test_get_analysis_config_caching(self) -> Optional[Dict[str, Any]]:
        """Test that analysis config is cached."""
        # First call
        config1 = get_analysis_config("sentiment", "standard")
        # Second call should use cache
        config2 = get_analysis_config("sentiment", "standard")
        
        assert config1 == config2
    
    def test_transform_analysis_to_response(self) -> Any:
        """Test analysis data transformation."""
        response_data = transform_analysis_to_response(MOCK_ANALYSIS_DATA)
        
        assert isinstance(response_data, dict)
        assert "id" in response_data
        assert "text_content" in response_data
        assert "analysis_type" in response_data
        assert "text_statistics" in response_data
    
    def test_transform_analysis_to_response_empty(self) -> Any:
        """Test analysis data transformation with empty data."""
        response_data = transform_analysis_to_response({})
        
        assert isinstance(response_data, dict)
        assert len(response_data) == 0
    
    def test_calculate_batch_progress(self) -> Any:
        """Test batch progress calculation."""
        progress = calculate_batch_progress(5, 1, 10)
        
        assert isinstance(progress, dict)
        assert "progress_percentage" in progress
        assert "success_rate" in progress
        assert "error_rate" in progress
        assert "remaining_count" in progress
        
        assert progress["progress_percentage"] == 60.0
        assert progress["success_rate"] == 50.0
        assert progress["error_rate"] == 10.0
        assert progress["remaining_count"] == 4
    
    def test_calculate_batch_progress_empty(self) -> Any:
        """Test batch progress calculation with empty batch."""
        progress = calculate_batch_progress(0, 0, 0)
        
        assert progress["progress_percentage"] == 0.0
        assert progress["success_rate"] == 0.0
        assert progress["error_rate"] == 0.0
        assert progress["remaining_count"] == 0

# ============================================================================
# Asynchronous Operations Tests
# ============================================================================

class TestAsynchronousOperations:
    """Test suite for asynchronous operations."""
    
    @pytest.mark.asyncio
    async def test_create_analysis_async(self) -> Any:
        """Test asynchronous analysis creation."""
        # Mock database manager
        mock_db = AsyncMock()
        mock_db.create_text_analysis.return_value = MOCK_ANALYSIS_DATA
        
        request = TextAnalysisRequest(
            text_content=SAMPLE_TEXT,
            analysis_type=AnalysisTypeEnum.SENTIMENT,
            optimization_tier=OptimizationTierEnum.STANDARD
        )
        
        result = await create_analysis_async(request, mock_db)
        
        assert isinstance(result, AnalysisResponse)
        assert result.id == 1
        assert result.text_content == SAMPLE_TEXT
        assert result.analysis_type == "sentiment"
        
        # Verify database was called
        mock_db.create_text_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_analysis_async_invalid_text(self) -> Any:
        """Test asynchronous analysis creation with invalid text."""
        mock_db = AsyncMock()
        
        request = TextAnalysisRequest(
            text_content="",
            analysis_type=AnalysisTypeEnum.SENTIMENT,
            optimization_tier=OptimizationTierEnum.STANDARD
        )
        
        with pytest.raises(ValueError, match="Validation failed"):
            await create_analysis_async(request, mock_db)
    
    @pytest.mark.asyncio
    async def test_get_analysis_async_found(self) -> Optional[Dict[str, Any]]:
        """Test asynchronous analysis retrieval when found."""
        mock_db = AsyncMock()
        mock_db.get_text_analysis.return_value = MOCK_ANALYSIS_DATA
        
        result = await get_analysis_async(1, mock_db)
        
        assert isinstance(result, AnalysisResponse)
        assert result.id == 1
        
        mock_db.get_text_analysis.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_get_analysis_async_not_found(self) -> Optional[Dict[str, Any]]:
        """Test asynchronous analysis retrieval when not found."""
        mock_db = AsyncMock()
        mock_db.get_text_analysis.return_value = None
        
        result = await get_analysis_async(999, mock_db)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_analysis_async_success(self) -> Any:
        """Test asynchronous analysis update."""
        mock_db = AsyncMock()
        updated_data = MOCK_ANALYSIS_DATA.copy()
        updated_data["sentiment_score"] = 0.9
        mock_db.update_text_analysis.return_value = updated_data
        
        update_data = {"sentiment_score": 0.9}
        result = await update_analysis_async(1, update_data, mock_db)
        
        assert isinstance(result, AnalysisResponse)
        assert result.sentiment_score == 0.9
        
        mock_db.update_text_analysis.assert_called_once_with(1, update_data)
    
    @pytest.mark.asyncio
    async def test_update_analysis_async_not_found(self) -> Any:
        """Test asynchronous analysis update when not found."""
        mock_db = AsyncMock()
        mock_db.update_text_analysis.return_value = None
        
        result = await update_analysis_async(999, {"test": "data"}, mock_db)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_analyses_async(self) -> List[Any]:
        """Test asynchronous analysis listing."""
        mock_db = AsyncMock()
        mock_analyses = [MOCK_ANALYSIS_DATA]
        mock_db.list_text_analyses.return_value = (mock_analyses, 1)
        
        pagination = {"size": 20, "offset": 0, "order_by": "created_at", "order_desc": True}
        filters = {"analysis_type": "sentiment"}
        
        results, total_count = await list_analyses_async(pagination, filters, mock_db)
        
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], AnalysisResponse)
        assert total_count == 1
        
        mock_db.list_text_analyses.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_analysis_background_async(self) -> Any:
        """Test asynchronous background processing."""
        mock_db = AsyncMock()
        
        await process_analysis_background_async(1, SAMPLE_TEXT, "sentiment", mock_db)
        
        # Verify database update was called
        mock_db.update_text_analysis.assert_called_once()
        call_args = mock_db.update_text_analysis.call_args
        assert call_args[0][0] == 1  # analysis_id
        assert call_args[0][1]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_create_batch_analysis_async(self) -> Any:
        """Test asynchronous batch analysis creation."""
        mock_db = AsyncMock()
        mock_batch = {
            "id": 1,
            "batch_name": "test_batch",
            "analysis_type": "sentiment",
            "status": "pending",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "metadata": {}
        }
        mock_db.create_batch_analysis.return_value = Mock(**mock_batch)
        
        request = BatchAnalysisRequest(
            batch_name="test_batch",
            texts=[SAMPLE_TEXT, "Another text"],
            analysis_type=AnalysisTypeEnum.SENTIMENT,
            optimization_tier=OptimizationTierEnum.STANDARD,
            priority=5
        )
        
        result = await create_batch_analysis_async(request, mock_db)
        
        assert isinstance(result, BatchAnalysisResponse)
        assert result.batch_name == "test_batch"
        assert result.total_texts == 2
        assert result.completed_count == 0
        assert result.error_count == 0
    
    @pytest.mark.asyncio
    async def test_process_batch_texts_async(self) -> Any:
        """Test asynchronous batch text processing."""
        mock_db = AsyncMock()
        texts = ["Text 1", "Text 2", "Text 3"]
        
        await process_batch_texts_async(1, texts, "sentiment", mock_db)
        
        # Verify batch progress update was called
        mock_db.update_batch_progress.assert_called_once()
        call_args = mock_db.update_batch_progress.call_args
        assert call_args[0][0] == 1  # batch_id
        assert call_args[0][1] == 3  # completed_count
        assert call_args[0][2] == 0  # error_count

# ============================================================================
# Cache Management Tests
# ============================================================================

class TestAnalysisCache:
    """Test suite for cache management."""
    
    def test_cache_initialization(self) -> Any:
        """Test cache initialization."""
        cache = AnalysisCache()
        
        assert hasattr(cache, '_cache')
        assert hasattr(cache, '_lock')
        assert isinstance(cache._cache, dict)
    
    def test_cache_set_sync(self) -> Any:
        """Test synchronous cache storage."""
        cache = AnalysisCache()
        cache.set_sync("test_key", "test_data", ttl_seconds=60)
        
        assert "test_key" in cache._cache
        assert isinstance(cache._cache["test_key"], CacheEntry)
        assert cache._cache["test_key"].data == "test_data"
        assert cache._cache["test_key"].ttl_seconds == 60
    
    def test_cache_get_sync_found(self) -> Optional[Dict[str, Any]]:
        """Test synchronous cache retrieval when found."""
        cache = AnalysisCache()
        cache.set_sync("test_key", "test_data", ttl_seconds=60)
        
        result = cache.get_sync("test_key")
        
        assert result == "test_data"
        assert cache._cache["test_key"].access_count == 1
    
    def test_cache_get_sync_not_found(self) -> Optional[Dict[str, Any]]:
        """Test synchronous cache retrieval when not found."""
        cache = AnalysisCache()
        
        result = cache.get_sync("nonexistent_key")
        
        assert result is None
    
    def test_cache_get_sync_expired(self) -> Optional[Dict[str, Any]]:
        """Test synchronous cache retrieval when expired."""
        cache = AnalysisCache()
        cache.set_sync("test_key", "test_data", ttl_seconds=1)
        
        # Wait for expiration
        time.sleep(1.1)
        
        result = cache.get_sync("test_key")
        
        assert result is None
        assert "test_key" not in cache._cache  # Should be cleaned up
    
    @pytest.mark.asyncio
    async def test_cache_set_async(self) -> Any:
        """Test asynchronous cache storage."""
        cache = AnalysisCache()
        await cache.set_async("test_key", "test_data", ttl_seconds=60)
        
        assert "test_key" in cache._cache
        assert cache._cache["test_key"].data == "test_data"
    
    @pytest.mark.asyncio
    async def test_cache_get_async(self) -> Optional[Dict[str, Any]]:
        """Test asynchronous cache retrieval."""
        cache = AnalysisCache()
        await cache.set_async("test_key", "test_data", ttl_seconds=60)
        
        result = await cache.get_async("test_key")
        
        assert result == "test_data"
    
    def test_cache_cleanup_sync(self) -> Any:
        """Test synchronous cache cleanup."""
        cache = AnalysisCache()
        cache.set_sync("key1", "data1", ttl_seconds=1)
        cache.set_sync("key2", "data2", ttl_seconds=3600)  # Long TTL
        
        # Wait for first key to expire
        time.sleep(1.1)
        
        removed_count = cache.cleanup_sync()
        
        assert removed_count == 1
        assert "key1" not in cache._cache
        assert "key2" in cache._cache
    
    @pytest.mark.asyncio
    async def test_cache_cleanup_async(self) -> Any:
        """Test asynchronous cache cleanup."""
        cache = AnalysisCache()
        await cache.set_async("key1", "data1", ttl_seconds=1)
        await cache.set_async("key2", "data2", ttl_seconds=3600)
        
        # Wait for first key to expire
        await asyncio.sleep(1.1)
        
        removed_count = await cache.cleanup_async()
        
        assert removed_count == 1
    
    def test_cache_get_stats_sync(self) -> Optional[Dict[str, Any]]:
        """Test synchronous cache statistics."""
        cache = AnalysisCache()
        cache.set_sync("key1", "data1", ttl_seconds=60)
        cache.set_sync("key2", "data2", ttl_seconds=60)
        
        # Access one entry
        cache.get_sync("key1")
        cache.get_sync("key1")  # Access twice
        
        stats = cache.get_stats_sync()
        
        assert stats["total_entries"] == 2
        assert stats["active_entries"] == 2
        assert stats["total_accesses"] == 2
        assert stats["average_accesses"] == 1.0

# ============================================================================
# Performance Monitoring Tests
# ============================================================================

class TestPerformanceMonitor:
    """Test suite for performance monitoring."""
    
    def test_monitor_initialization(self) -> Any:
        """Test performance monitor initialization."""
        monitor = PerformanceMonitor()
        
        assert hasattr(monitor, '_metrics')
        assert hasattr(monitor, '_lock')
        assert isinstance(monitor._metrics, list)
    
    def test_start_operation_sync(self) -> Any:
        """Test synchronous operation start."""
        monitor = PerformanceMonitor()
        metrics = monitor.start_operation_sync("test_operation")
        
        assert isinstance(metrics, ProcessingMetrics)
        assert metrics.operation_type == "test_operation"
        assert metrics.start_time is not None
        assert metrics.end_time is None
        assert metrics.success is True
    
    @pytest.mark.asyncio
    async def test_start_operation_async(self) -> Any:
        """Test asynchronous operation start."""
        monitor = PerformanceMonitor()
        metrics = await monitor.start_operation_async("test_operation")
        
        assert isinstance(metrics, ProcessingMetrics)
        assert metrics.operation_type == "test_operation"
    
    def test_complete_operation_sync(self) -> Any:
        """Test synchronous operation completion."""
        monitor = PerformanceMonitor()
        metrics = monitor.start_operation_sync("test_operation")
        
        monitor.complete_operation_sync(metrics, success=True)
        
        assert metrics.end_time is not None
        assert metrics.duration_ms is not None
        assert metrics.duration_ms > 0
        assert metrics.success is True
        assert len(monitor._metrics) == 1
    
    def test_complete_operation_sync_with_error(self) -> Any:
        """Test synchronous operation completion with error."""
        monitor = PerformanceMonitor()
        metrics = monitor.start_operation_sync("test_operation")
        
        monitor.complete_operation_sync(metrics, success=False, error_message="Test error")
        
        assert metrics.success is False
        assert metrics.error_message == "Test error"
    
    @pytest.mark.asyncio
    async def test_complete_operation_async(self) -> Any:
        """Test asynchronous operation completion."""
        monitor = PerformanceMonitor()
        metrics = await monitor.start_operation_async("test_operation")
        
        await monitor.complete_operation_async(metrics, success=True)
        
        assert metrics.end_time is not None
        assert metrics.success is True
        assert len(monitor._metrics) == 1
    
    def test_get_stats_sync(self) -> Optional[Dict[str, Any]]:
        """Test synchronous statistics retrieval."""
        monitor = PerformanceMonitor()
        
        # Complete some operations
        metrics1 = monitor.start_operation_sync("op1")
        monitor.complete_operation_sync(metrics1, success=True)
        
        metrics2 = monitor.start_operation_sync("op2")
        monitor.complete_operation_sync(metrics2, success=False, error_message="Error")
        
        stats = monitor.get_stats_sync()
        
        assert stats["total_operations"] == 2
        assert stats["successful_operations"] == 1
        assert stats["failed_operations"] == 1
        assert stats["success_rate"] == 50.0
        assert stats["average_duration_ms"] > 0
    
    def test_get_stats_sync_empty(self) -> Optional[Dict[str, Any]]:
        """Test synchronous statistics retrieval with no operations."""
        monitor = PerformanceMonitor()
        stats = monitor.get_stats_sync()
        
        assert isinstance(stats, dict)
        assert len(stats) == 0
    
    @pytest.mark.asyncio
    async def test_get_stats_async(self) -> Optional[Dict[str, Any]]:
        """Test asynchronous statistics retrieval."""
        monitor = PerformanceMonitor()
        metrics = await monitor.start_operation_async("test_operation")
        await monitor.complete_operation_async(metrics, success=True)
        
        stats = await monitor.get_stats_async()
        
        assert stats["total_operations"] == 1
        assert stats["successful_operations"] == 1
        assert stats["success_rate"] == 100.0

# ============================================================================
# Route Handler Tests
# ============================================================================

class TestRouteHandlers:
    """Test suite for route handlers."""
    
    @pytest.mark.asyncio
    async def test_create_analysis_handler(self) -> Any:
        """Test analysis creation handler."""
        # Mock dependencies
        mock_background_tasks = Mock()
        mock_db = AsyncMock()
        mock_db.create_text_analysis.return_value = MOCK_ANALYSIS_DATA
        cache = AnalysisCache()
        
        request = TextAnalysisRequest(
            text_content=SAMPLE_TEXT,
            analysis_type=AnalysisTypeEnum.SENTIMENT,
            optimization_tier=OptimizationTierEnum.STANDARD
        )
        
        result = await create_analysis_handler(
            request, mock_background_tasks, mock_db, cache
        )
        
        assert isinstance(result, AnalysisResponse)
        assert result.text_content == SAMPLE_TEXT
        
        # Verify background task was added
        mock_background_tasks.add_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_analysis_handler_found(self) -> Optional[Dict[str, Any]]:
        """Test analysis retrieval handler when found."""
        mock_db = AsyncMock()
        mock_db.get_text_analysis.return_value = MOCK_ANALYSIS_DATA
        cache = AnalysisCache()
        
        result = await get_analysis_handler(1, mock_db, cache)
        
        assert isinstance(result, AnalysisResponse)
        assert result.id == 1
    
    @pytest.mark.asyncio
    async def test_get_analysis_handler_not_found(self) -> Optional[Dict[str, Any]]:
        """Test analysis retrieval handler when not found."""
        mock_db = AsyncMock()
        mock_db.get_text_analysis.return_value = None
        cache = AnalysisCache()
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await get_analysis_handler(999, mock_db, cache)
    
    @pytest.mark.asyncio
    async def test_list_analyses_handler(self) -> List[Any]:
        """Test analysis listing handler."""
        mock_db = AsyncMock()
        mock_analyses = [MOCK_ANALYSIS_DATA]
        mock_db.list_text_analyses.return_value = (mock_analyses, 1)
        
        pagination = {"page": 1, "size": 20}
        filters = {"analysis_type": "sentiment"}
        
        result = await list_analyses_handler(pagination, filters, mock_db)
        
        assert isinstance(result, PaginatedResponse)
        assert len(result.items) == 1
        assert result.total == 1
        assert result.page == 1
        assert result.size == 20
    
    @pytest.mark.asyncio
    async def test_list_analyses_handler_invalid_pagination(self) -> List[Any]:
        """Test analysis listing handler with invalid pagination."""
        mock_db = AsyncMock()
        
        pagination = {"page": 0, "size": 20}  # Invalid page
        filters = {}
        
        with pytest.raises(Exception):  # Should raise HTTPException
            await list_analyses_handler(pagination, filters, mock_db)

# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for sync/async operations."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self) -> Any:
        """Test complete analysis workflow."""
        # Initialize components
        cache = AnalysisCache()
        monitor = PerformanceMonitor()
        mock_db = AsyncMock()
        mock_db.create_text_analysis.return_value = MOCK_ANALYSIS_DATA
        mock_db.get_text_analysis.return_value = MOCK_ANALYSIS_DATA
        
        # Start monitoring
        metrics = await monitor.start_operation_async("full_workflow")
        
        # Create analysis request
        request = TextAnalysisRequest(
            text_content=SAMPLE_TEXT,
            analysis_type=AnalysisTypeEnum.SENTIMENT,
            optimization_tier=OptimizationTierEnum.STANDARD
        )
        
        # Validate text synchronously
        is_valid, errors, warnings = validate_text_content(request.text_content)
        assert is_valid is True
        
        # Calculate statistics synchronously
        stats = calculate_text_statistics(request.text_content)
        assert len(stats) > 0
        
        # Generate cache key synchronously
        cache_key = generate_cache_key(
            request.text_content,
            request.analysis_type,
            request.optimization_tier
        )
        
        # Check cache asynchronously
        cached_result = await cache.get_async(cache_key)
        assert cached_result is None  # Should not be cached yet
        
        # Create analysis asynchronously
        analysis = await create_analysis_async(request, mock_db)
        assert isinstance(analysis, AnalysisResponse)
        
        # Cache result asynchronously
        await cache.set_async(cache_key, analysis.model_dump(), ttl_seconds=1800)
        
        # Verify cache works
        cached_analysis = await cache.get_async(cache_key)
        assert cached_analysis is not None
        
        # Complete monitoring
        await monitor.complete_operation_async(metrics, success=True)
        
        # Check performance stats
        stats = await monitor.get_stats_async()
        assert stats["total_operations"] == 1
        assert stats["successful_operations"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self) -> Any:
        """Test batch processing workflow."""
        mock_db = AsyncMock()
        mock_batch = {
            "id": 1,
            "batch_name": "test_batch",
            "analysis_type": "sentiment",
            "status": "pending",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "metadata": {}
        }
        mock_db.create_batch_analysis.return_value = Mock(**mock_batch)
        
        # Create batch request
        request = BatchAnalysisRequest(
            batch_name="test_batch",
            texts=["Text 1", "Text 2", "Text 3"],
            analysis_type=AnalysisTypeEnum.SENTIMENT,
            optimization_tier=OptimizationTierEnum.STANDARD,
            priority=5
        )
        
        # Create batch asynchronously
        batch = await create_batch_analysis_async(request, mock_db)
        assert isinstance(batch, BatchAnalysisResponse)
        assert batch.total_texts == 3
        
        # Process batch texts asynchronously
        await process_batch_texts_async(1, request.texts, "sentiment", mock_db)
        
        # Verify batch progress was updated
        mock_db.update_batch_progress.assert_called_once()

# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for sync/async operations."""
    
    def test_sync_operations_performance(self) -> Any:
        """Test performance of synchronous operations."""
        
        start_time = time.time()
        
        # Run multiple synchronous operations
        for i in range(1000):
            validate_text_content(f"Test text {i}")
            calculate_text_statistics(f"Test text {i}")
            generate_cache_key(f"text{i}", "sentiment", "standard")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly (less than 1 second for 1000 operations)
        assert duration < 1.0
    
    @pytest.mark.asyncio
    async def test_async_operations_performance(self) -> Any:
        """Test performance of asynchronous operations."""
        
        mock_db = AsyncMock()
        mock_db.create_text_analysis.return_value = MOCK_ANALYSIS_DATA
        
        start_time = time.time()
        
        # Run multiple asynchronous operations concurrently
        tasks = []
        for i in range(100):
            request = TextAnalysisRequest(
                text_content=f"Test text {i}",
                analysis_type=AnalysisTypeEnum.SENTIMENT,
                optimization_tier=OptimizationTierEnum.STANDARD
            )
            task = create_analysis_async(request, mock_db)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert len(results) == 100
        assert all(isinstance(r, AnalysisResponse) for r in results)
        
        # Should complete quickly due to concurrency
        assert duration < 2.0

# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling in sync/async operations."""
    
    def test_sync_operations_with_invalid_input(self) -> Any:
        """Test synchronous operations with invalid input."""
        # Test with None input
        stats = calculate_text_statistics(None)
        assert stats == {}
        
        # Test with non-string input
        stats = calculate_text_statistics(123)
        assert stats == {}
        
        # Test with empty string
        score = calculate_complexity_score("")
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_async_operations_with_db_errors(self) -> Any:
        """Test asynchronous operations with database errors."""
        mock_db = AsyncMock()
        mock_db.create_text_analysis.side_effect = Exception("Database error")
        
        request = TextAnalysisRequest(
            text_content=SAMPLE_TEXT,
            analysis_type=AnalysisTypeEnum.SENTIMENT,
            optimization_tier=OptimizationTierEnum.STANDARD
        )
        
        with pytest.raises(Exception, match="Database error"):
            await create_analysis_async(request, mock_db)
    
    @pytest.mark.asyncio
    async def test_background_processing_error_handling(self) -> Any:
        """Test error handling in background processing."""
        mock_db = AsyncMock()
        mock_db.update_text_analysis.side_effect = Exception("Update error")
        
        # Should not raise exception, should handle gracefully
        await process_analysis_background_async(1, SAMPLE_TEXT, "sentiment", mock_db)
        
        # Should have attempted to update with error status
        mock_db.update_text_analysis.assert_called()

# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"]) 