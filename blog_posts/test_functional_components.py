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
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
from .functional_fastapi_components import (
        import time
        import time
        import time
from typing import Any, List, Dict, Optional
import logging
"""
🧪 Test Suite for Functional FastAPI Components
==============================================

Comprehensive testing of:
- Pydantic model validation
- Pure functions
- Functional decorators
- API handlers
- Error handling
- Type safety
"""


    # Pydantic Models
    TextAnalysisRequest, BatchAnalysisRequest, AnalysisUpdateRequest,
    PaginationRequest, AnalysisFilterRequest,
    AnalysisResponse, BatchAnalysisResponse, PaginatedResponse,
    HealthResponse, ErrorResponse, SuccessResponse,
    
    # Enums
    AnalysisTypeEnum, OptimizationTierEnum, AnalysisStatusEnum,
    
    # Data Classes
    RequestContext, ValidationResult, ProcessingResult,
    
    # Pure Functions
    validate_text_content, calculate_processing_priority,
    estimate_processing_time, calculate_batch_progress,
    generate_cache_key, transform_analysis_to_response,
    
    # Functional Decorators
    with_error_handling, with_validation, with_caching, with_logging,
    
    # Service Functions
    create_analysis_service, get_analysis_service,
    update_analysis_service, list_analyses_service, create_batch_service,
    
    # API Handlers
    create_analysis_handler, get_analysis_handler,
    update_analysis_handler, list_analyses_handler, create_batch_handler
)

# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_text_analysis_request():
    """Sample text analysis request."""
    return TextAnalysisRequest(
        text_content="This is a positive text for sentiment analysis.",
        analysis_type=AnalysisTypeEnum.SENTIMENT,
        optimization_tier=OptimizationTierEnum.STANDARD,
        metadata={"source": "test", "priority": "high"}
    )

@pytest.fixture
def sample_batch_analysis_request():
    """Sample batch analysis request."""
    return BatchAnalysisRequest(
        batch_name="Test Batch Analysis",
        texts=[
            "Great product, highly recommended!",
            "Disappointed with the service quality.",
            "Average experience, could be better."
        ],
        analysis_type=AnalysisTypeEnum.SENTIMENT,
        optimization_tier=OptimizationTierEnum.STANDARD,
        priority=7
    )

@pytest.fixture
def sample_analysis_update_request():
    """Sample analysis update request."""
    return AnalysisUpdateRequest(
        status=AnalysisStatusEnum.COMPLETED,
        sentiment_score=0.8,
        quality_score=0.95,
        processing_time_ms=150.5,
        model_used="test-model",
        confidence_score=0.92,
        metadata={"test": True}
    )

@pytest.fixture
def sample_pagination_request():
    """Sample pagination request."""
    return PaginationRequest(
        page=1,
        size=20,
        order_by="created_at",
        order_desc=True
    )

@pytest.fixture
def sample_analysis_filter_request():
    """Sample analysis filter request."""
    return AnalysisFilterRequest(
        analysis_type=AnalysisTypeEnum.SENTIMENT,
        status=AnalysisStatusEnum.COMPLETED,
        optimization_tier=OptimizationTierEnum.STANDARD,
        date_from=datetime.now() - timedelta(days=7),
        date_to=datetime.now(),
        min_sentiment_score=0.0,
        max_sentiment_score=1.0
    )

@pytest.fixture
def mock_db_manager():
    """Mock database manager."""
    manager = AsyncMock()
    manager.create_text_analysis = AsyncMock()
    manager.get_text_analysis = AsyncMock()
    manager.update_text_analysis = AsyncMock()
    manager.list_text_analyses = AsyncMock()
    manager.create_batch_analysis = AsyncMock()
    manager.update_batch_progress = AsyncMock()
    return manager

@pytest.fixture
def sample_analysis_model():
    """Sample analysis database model."""
    model = MagicMock()
    model.id = 1
    model.text_content = "Test text content"
    model.analysis_type = AnalysisTypeEnum.SENTIMENT
    model.status = AnalysisStatusEnum.COMPLETED
    model.sentiment_score = 0.8
    model.quality_score = 0.95
    model.processing_time_ms = 150.5
    model.model_used = "test-model"
    model.confidence_score = 0.92
    model.optimization_tier = OptimizationTierEnum.STANDARD
    model.created_at = datetime.now()
    model.updated_at = datetime.now()
    model.processed_at = datetime.now()
    model.metadata = {"test": True}
    return model

# ============================================================================
# Pydantic Model Tests
# ============================================================================

class TestTextAnalysisRequest:
    """Test TextAnalysisRequest model."""
    
    async def test_valid_request(self, sample_text_analysis_request) -> Any:
        """Test valid request creation."""
        assert sample_text_analysis_request.text_content == "This is a positive text for sentiment analysis."
        assert sample_text_analysis_request.analysis_type == AnalysisTypeEnum.SENTIMENT
        assert sample_text_analysis_request.optimization_tier == OptimizationTierEnum.STANDARD
        assert sample_text_analysis_request.metadata == {"source": "test", "priority": "high"}
    
    def test_empty_text_content(self) -> Any:
        """Test empty text content validation."""
        with pytest.raises(ValueError, match="Text content cannot be empty"):
            TextAnalysisRequest(
                text_content="",
                analysis_type=AnalysisTypeEnum.SENTIMENT
            )
    
    def test_whitespace_only_text_content(self) -> Any:
        """Test whitespace-only text content validation."""
        with pytest.raises(ValueError, match="Text content cannot be empty"):
            TextAnalysisRequest(
                text_content="   ",
                analysis_type=AnalysisTypeEnum.SENTIMENT
            )
    
    def test_text_content_too_long(self) -> Any:
        """Test text content length validation."""
        long_text = "x" * 10001
        with pytest.raises(ValueError):
            TextAnalysisRequest(
                text_content=long_text,
                analysis_type=AnalysisTypeEnum.SENTIMENT
            )
    
    def test_metadata_too_large(self) -> Any:
        """Test metadata size validation."""
        large_metadata = {"key": "x" * 1001}
        with pytest.raises(ValueError, match="Metadata too large"):
            TextAnalysisRequest(
                text_content="Test text",
                analysis_type=AnalysisTypeEnum.SENTIMENT,
                metadata=large_metadata
            )
    
    def test_invalid_metadata_key(self) -> Any:
        """Test metadata key validation."""
        invalid_metadata = {"x" * 51: "value"}
        with pytest.raises(ValueError, match="Invalid metadata key"):
            TextAnalysisRequest(
                text_content="Test text",
                analysis_type=AnalysisTypeEnum.SENTIMENT,
                metadata=invalid_metadata
            )
    
    def test_text_content_cleaning(self) -> Any:
        """Test text content cleaning."""
        request = TextAnalysisRequest(
            text_content="  Test text with spaces  ",
            analysis_type=AnalysisTypeEnum.SENTIMENT
        )
        assert request.text_content == "Test text with spaces"

class TestBatchAnalysisRequest:
    """Test BatchAnalysisRequest model."""
    
    async def test_valid_request(self, sample_batch_analysis_request) -> Any:
        """Test valid request creation."""
        assert sample_batch_analysis_request.batch_name == "Test Batch Analysis"
        assert len(sample_batch_analysis_request.texts) == 3
        assert sample_batch_analysis_request.analysis_type == AnalysisTypeEnum.SENTIMENT
        assert sample_batch_analysis_request.priority == 7
    
    def test_empty_texts_list(self) -> List[Any]:
        """Test empty texts list validation."""
        with pytest.raises(ValueError, match="At least one text must be provided"):
            BatchAnalysisRequest(
                batch_name="Test",
                texts=[],
                analysis_type=AnalysisTypeEnum.SENTIMENT
            )
    
    def test_empty_text_in_list(self) -> List[Any]:
        """Test empty text in list validation."""
        with pytest.raises(ValueError, match="Text content cannot be empty"):
            BatchAnalysisRequest(
                batch_name="Test",
                texts=["Valid text", ""],
                analysis_type=AnalysisTypeEnum.SENTIMENT
            )
    
    def test_too_many_texts(self) -> Any:
        """Test maximum texts validation."""
        texts = ["Text"] * 1001
        with pytest.raises(ValueError):
            BatchAnalysisRequest(
                batch_name="Test",
                texts=texts,
                analysis_type=AnalysisTypeEnum.SENTIMENT
            )
    
    def test_invalid_priority(self) -> Any:
        """Test priority validation."""
        with pytest.raises(ValueError):
            BatchAnalysisRequest(
                batch_name="Test",
                texts=["Valid text"],
                analysis_type=AnalysisTypeEnum.SENTIMENT,
                priority=11
            )

class TestAnalysisUpdateRequest:
    """Test AnalysisUpdateRequest model."""
    
    async def test_valid_request(self, sample_analysis_update_request) -> Any:
        """Test valid request creation."""
        assert sample_analysis_update_request.status == AnalysisStatusEnum.COMPLETED
        assert sample_analysis_update_request.sentiment_score == 0.8
        assert sample_analysis_update_request.quality_score == 0.95
    
    def test_invalid_sentiment_score(self) -> Any:
        """Test sentiment score validation."""
        with pytest.raises(ValueError):
            AnalysisUpdateRequest(sentiment_score=1.5)
    
    def test_invalid_quality_score(self) -> Any:
        """Test quality score validation."""
        with pytest.raises(ValueError):
            AnalysisUpdateRequest(quality_score=-0.1)
    
    def test_error_status_without_message(self) -> Any:
        """Test error status consistency."""
        with pytest.raises(ValueError, match="Error message required"):
            AnalysisUpdateRequest(
                status=AnalysisStatusEnum.ERROR
            )
    
    def test_completed_status_without_sentiment(self) -> Any:
        """Test completed status consistency."""
        with pytest.raises(ValueError, match="Sentiment score required"):
            AnalysisUpdateRequest(
                status=AnalysisStatusEnum.COMPLETED
            )

class TestPaginationRequest:
    """Test PaginationRequest model."""
    
    async def test_valid_request(self, sample_pagination_request) -> Any:
        """Test valid request creation."""
        assert sample_pagination_request.page == 1
        assert sample_pagination_request.size == 20
        assert sample_pagination_request.offset == 0
    
    def test_offset_calculation(self) -> Any:
        """Test offset calculation."""
        pagination = PaginationRequest(page=3, size=10)
        assert pagination.offset == 20
    
    def test_invalid_page(self) -> Any:
        """Test invalid page validation."""
        with pytest.raises(ValueError):
            PaginationRequest(page=0, size=10)
    
    def test_invalid_size(self) -> Any:
        """Test invalid size validation."""
        with pytest.raises(ValueError):
            PaginationRequest(page=1, size=101)

class TestAnalysisFilterRequest:
    """Test AnalysisFilterRequest model."""
    
    async def test_valid_request(self, sample_analysis_filter_request) -> Any:
        """Test valid request creation."""
        assert sample_analysis_filter_request.analysis_type == AnalysisTypeEnum.SENTIMENT
        assert sample_analysis_filter_request.status == AnalysisStatusEnum.COMPLETED
    
    def test_invalid_date_range(self) -> Any:
        """Test invalid date range validation."""
        with pytest.raises(ValueError, match="date_from must be before date_to"):
            AnalysisFilterRequest(
                date_from=datetime.now(),
                date_to=datetime.now() - timedelta(days=1)
            )
    
    def test_invalid_sentiment_range(self) -> Any:
        """Test invalid sentiment range validation."""
        with pytest.raises(ValueError, match="min_sentiment_score must be less than max_sentiment_score"):
            AnalysisFilterRequest(
                min_sentiment_score=0.8,
                max_sentiment_score=0.5
            )

# ============================================================================
# Response Model Tests
# ============================================================================

class TestAnalysisResponse:
    """Test AnalysisResponse model."""
    
    def test_valid_response(self, sample_analysis_model) -> Any:
        """Test valid response creation."""
        response = transform_analysis_to_response(sample_analysis_model)
        assert response.id == 1
        assert response.text_content == "Test text content"
        assert response.analysis_type == AnalysisTypeEnum.SENTIMENT
        assert response.status == AnalysisStatusEnum.COMPLETED
        assert response.sentiment_score == 0.8
        assert response.metadata == {"test": True}

class TestBatchAnalysisResponse:
    """Test BatchAnalysisResponse model."""
    
    def test_response_properties(self) -> Any:
        """Test response properties."""
        response = BatchAnalysisResponse(
            id=1,
            batch_name="Test",
            analysis_type=AnalysisTypeEnum.SENTIMENT,
            status=AnalysisStatusEnum.COMPLETED,
            total_texts=10,
            completed_count=8,
            error_count=1,
            progress_percentage=90.0,
            optimization_tier=OptimizationTierEnum.STANDARD,
            priority=5,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert response.is_completed is True
        assert response.success_rate == 80.0

class TestPaginatedResponse:
    """Test PaginatedResponse model."""
    
    def test_paginated_response_creation(self) -> Any:
        """Test paginated response creation."""
        items = [{"id": 1}, {"id": 2}]
        response = PaginatedResponse.create(
            items=items,
            total=50,
            page=2,
            size=20
        )
        
        assert response.items == items
        assert response.total == 50
        assert response.page == 2
        assert response.size == 20
        assert response.pages == 3
        assert response.has_next is True
        assert response.has_prev is True

# ============================================================================
# Pure Function Tests
# ============================================================================

class TestValidateTextContent:
    """Test validate_text_content function."""
    
    def test_valid_text(self) -> Any:
        """Test valid text validation."""
        result = validate_text_content("This is a valid text.")
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_empty_text(self) -> Any:
        """Test empty text validation."""
        result = validate_text_content("")
        assert result.is_valid is False
        assert "Text content cannot be empty" in result.errors[0]
    
    def test_whitespace_only_text(self) -> Any:
        """Test whitespace-only text validation."""
        result = validate_text_content("   ")
        assert result.is_valid is False
        assert "Text content cannot be whitespace only" in result.errors[0]
    
    def test_text_too_long(self) -> Any:
        """Test text length validation."""
        long_text = "x" * 10001
        result = validate_text_content(long_text)
        assert result.is_valid is False
        assert "Text content too long" in result.errors[0]
    
    def test_short_text_warning(self) -> Any:
        """Test short text warning."""
        result = validate_text_content("Short")
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "very short" in result.warnings[0]

class TestCalculateProcessingPriority:
    """Test calculate_processing_priority function."""
    
    def test_basic_tier_priority(self) -> Any:
        """Test basic tier priority calculation."""
        priority = calculate_processing_priority(
            OptimizationTierEnum.BASIC,
            1000,
            AnalysisTypeEnum.SENTIMENT
        )
        assert 1 <= priority <= 10
    
    def test_ultra_tier_priority(self) -> Any:
        """Test ultra tier priority calculation."""
        priority = calculate_processing_priority(
            OptimizationTierEnum.ULTRA,
            1000,
            AnalysisTypeEnum.SENTIMENT
        )
        assert priority > calculate_processing_priority(
            OptimizationTierEnum.BASIC,
            1000,
            AnalysisTypeEnum.SENTIMENT
        )
    
    def test_complex_analysis_priority(self) -> Any:
        """Test complex analysis priority calculation."""
        sentiment_priority = calculate_processing_priority(
            OptimizationTierEnum.STANDARD,
            1000,
            AnalysisTypeEnum.SENTIMENT
        )
        topics_priority = calculate_processing_priority(
            OptimizationTierEnum.STANDARD,
            1000,
            AnalysisTypeEnum.TOPICS
        )
        assert topics_priority > sentiment_priority

class TestEstimateProcessingTime:
    """Test estimate_processing_time function."""
    
    def test_basic_estimation(self) -> Any:
        """Test basic processing time estimation."""
        time_ms = estimate_processing_time(
            1000,
            AnalysisTypeEnum.SENTIMENT,
            OptimizationTierEnum.STANDARD
        )
        assert time_ms >= 50.0
    
    def test_complex_analysis_time(self) -> Any:
        """Test complex analysis time estimation."""
        sentiment_time = estimate_processing_time(
            1000,
            AnalysisTypeEnum.SENTIMENT,
            OptimizationTierEnum.STANDARD
        )
        topics_time = estimate_processing_time(
            1000,
            AnalysisTypeEnum.TOPICS,
            OptimizationTierEnum.STANDARD
        )
        assert topics_time > sentiment_time
    
    def test_optimization_tier_impact(self) -> Any:
        """Test optimization tier impact on time."""
        basic_time = estimate_processing_time(
            1000,
            AnalysisTypeEnum.SENTIMENT,
            OptimizationTierEnum.BASIC
        )
        ultra_time = estimate_processing_time(
            1000,
            AnalysisTypeEnum.SENTIMENT,
            OptimizationTierEnum.ULTRA
        )
        assert basic_time > ultra_time

class TestCalculateBatchProgress:
    """Test calculate_batch_progress function."""
    
    def test_empty_batch(self) -> Any:
        """Test empty batch progress."""
        progress = calculate_batch_progress(0, 0, 0)
        assert progress["progress_percentage"] == 0.0
        assert progress["success_rate"] == 0.0
        assert progress["error_rate"] == 0.0
        assert progress["remaining_count"] == 0
    
    def test_completed_batch(self) -> Any:
        """Test completed batch progress."""
        progress = calculate_batch_progress(10, 0, 10)
        assert progress["progress_percentage"] == 100.0
        assert progress["success_rate"] == 100.0
        assert progress["error_rate"] == 0.0
        assert progress["remaining_count"] == 0
    
    def test_partial_batch(self) -> Any:
        """Test partial batch progress."""
        progress = calculate_batch_progress(5, 2, 10)
        assert progress["progress_percentage"] == 70.0
        assert progress["success_rate"] == 50.0
        assert progress["error_rate"] == 20.0
        assert progress["remaining_count"] == 3

class TestGenerateCacheKey:
    """Test generate_cache_key function."""
    
    def test_deterministic_key(self) -> Any:
        """Test deterministic cache key generation."""
        key1 = generate_cache_key(
            "Test text",
            AnalysisTypeEnum.SENTIMENT,
            OptimizationTierEnum.STANDARD
        )
        key2 = generate_cache_key(
            "Test text",
            AnalysisTypeEnum.SENTIMENT,
            OptimizationTierEnum.STANDARD
        )
        assert key1 == key2
    
    def test_different_inputs_different_keys(self) -> Any:
        """Test different inputs produce different keys."""
        key1 = generate_cache_key(
            "Text 1",
            AnalysisTypeEnum.SENTIMENT,
            OptimizationTierEnum.STANDARD
        )
        key2 = generate_cache_key(
            "Text 2",
            AnalysisTypeEnum.SENTIMENT,
            OptimizationTierEnum.STANDARD
        )
        assert key1 != key2

# ============================================================================
# Functional Decorator Tests
# ============================================================================

class TestWithErrorHandling:
    """Test with_error_handling decorator."""
    
    @pytest.mark.asyncio
    async def test_successful_function(self) -> Any:
        """Test successful function execution."""
        @with_error_handling()
        async def test_func():
            
    """test_func function."""
return "success"
        
        result = await test_func()
        assert result.success is True
        assert result.data == "success"
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_failed_function(self) -> Any:
        """Test failed function execution."""
        @with_error_handling()
        async def test_func():
            
    """test_func function."""
raise ValueError("Test error")
        
        result = await test_func()
        assert result.success is False
        assert result.data is None
        assert "Test error" in result.error
    
    @pytest.mark.asyncio
    async def test_custom_error_handler(self) -> Any:
        """Test custom error handler."""
        def custom_handler(exc) -> Any:
            return {"custom": True, "error_type": type(exc).__name__}
        
        @with_error_handling(custom_handler)
        async def test_func():
            
    """test_func function."""
raise ValueError("Test error")
        
        result = await test_func()
        assert result.success is False
        assert result.metadata["custom"] is True
        assert result.metadata["error_type"] == "ValueError"

class TestWithValidation:
    """Test with_validation decorator."""
    
    @pytest.mark.asyncio
    async def test_valid_input(self) -> Any:
        """Test valid input validation."""
        def validator(input_data) -> Any:
            return ValidationResult(is_valid=True)
        
        @with_validation(validator)
        async def test_func(input_data) -> Any:
            return "success"
        
        result = await test_func("valid input")
        assert result.success is True
        assert result.data == "success"
    
    @pytest.mark.asyncio
    async def test_invalid_input(self) -> Any:
        """Test invalid input validation."""
        def validator(input_data) -> Any:
            return ValidationResult(
                is_valid=False,
                errors=["Invalid input"]
            )
        
        @with_validation(validator)
        async def test_func(input_data) -> Any:
            return "success"
        
        result = await test_func("invalid input")
        assert result.success is False
        assert "Validation failed" in result.error
        assert "Invalid input" in result.metadata["validation_errors"]

class TestWithCaching:
    """Test with_caching decorator."""
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self) -> Any:
        """Test caching behavior."""
        call_count = 0
        
        def cache_key_func(*args, **kwargs) -> Any:
            return f"test_key_{args[0]}"
        
        @with_caching(cache_key_func, ttl_seconds=3600)
        async def test_func(value) -> Any:
            nonlocal call_count
            call_count += 1
            return f"result_{value}"
        
        # First call
        result1 = await test_func("test")
        assert result1 == "result_test"
        assert call_count == 1
        
        # Second call (should be cached)
        result2 = await test_func("test")
        assert result2 == "result_test"
        assert call_count == 1  # Should not increment

class TestWithLogging:
    """Test with_logging decorator."""
    
    @pytest.mark.asyncio
    async def test_successful_logging(self) -> Any:
        """Test successful function logging."""
        @with_logging("test_logger")
        async def test_func():
            
    """test_func function."""
return "success"
        
        with patch('structlog.get_logger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log
            
            result = await test_func()
            
            assert result == "success"
            assert mock_log.info.call_count == 2  # Start and success
    
    @pytest.mark.asyncio
    async def test_failed_logging(self) -> Any:
        """Test failed function logging."""
        @with_logging("test_logger")
        async def test_func():
            
    """test_func function."""
raise ValueError("Test error")
        
        with patch('structlog.get_logger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log
            
            with pytest.raises(ValueError):
                await test_func()
            
            assert mock_log.error.call_count == 1

# ============================================================================
# Service Function Tests
# ============================================================================

class TestCreateAnalysisService:
    """Test create_analysis_service function."""
    
    @pytest.mark.asyncio
    async def test_successful_creation(self, sample_text_analysis_request, mock_db_manager, sample_analysis_model) -> Any:
        """Test successful analysis creation."""
        mock_db_manager.create_text_analysis.return_value = sample_analysis_model
        
        result = await create_analysis_service(sample_text_analysis_request, mock_db_manager)
        
        assert result.success is True
        assert result.data is not None
        mock_db_manager.create_text_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_validation_error(self, mock_db_manager) -> Any:
        """Test validation error handling."""
        invalid_request = TextAnalysisRequest(
            text_content="",
            analysis_type=AnalysisTypeEnum.SENTIMENT
        )
        
        result = await create_analysis_service(invalid_request, mock_db_manager)
        
        assert result.success is False
        assert "Invalid text content" in result.error

class TestGetAnalysisService:
    """Test get_analysis_service function."""
    
    @pytest.mark.asyncio
    async def test_successful_retrieval(self, mock_db_manager, sample_analysis_model) -> Any:
        """Test successful analysis retrieval."""
        mock_db_manager.get_text_analysis.return_value = sample_analysis_model
        
        result = await get_analysis_service(1, mock_db_manager)
        
        assert result.success is True
        assert result.data is not None
        mock_db_manager.get_text_analysis.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_not_found(self, mock_db_manager) -> Any:
        """Test not found handling."""
        mock_db_manager.get_text_analysis.return_value = None
        
        result = await get_analysis_service(999, mock_db_manager)
        
        assert result.success is False
        assert "not found" in result.error

class TestUpdateAnalysisService:
    """Test update_analysis_service function."""
    
    @pytest.mark.asyncio
    async def test_successful_update(self, sample_analysis_update_request, mock_db_manager, sample_analysis_model) -> Any:
        """Test successful analysis update."""
        mock_db_manager.update_text_analysis.return_value = sample_analysis_model
        
        result = await update_analysis_service(1, sample_analysis_update_request, mock_db_manager)
        
        assert result.success is True
        assert result.data is not None
        mock_db_manager.update_text_analysis.assert_called_once()

class TestListAnalysesService:
    """Test list_analyses_service function."""
    
    @pytest.mark.asyncio
    async def test_successful_listing(self, sample_pagination_request, sample_analysis_filter_request, mock_db_manager, sample_analysis_model) -> List[Any]:
        """Test successful analyses listing."""
        mock_db_manager.list_text_analyses.return_value = [sample_analysis_model]
        
        result = await list_analyses_service(sample_pagination_request, sample_analysis_filter_request, mock_db_manager)
        
        assert result.success is True
        assert result.data is not None
        assert len(result.data.items) == 1

# ============================================================================
# API Handler Tests
# ============================================================================

class TestCreateAnalysisHandler:
    """Test create_analysis_handler function."""
    
    @pytest.mark.asyncio
    async def test_successful_handler(self, sample_text_analysis_request, mock_db_manager, sample_analysis_model) -> Any:
        """Test successful handler execution."""
        mock_db_manager.create_text_analysis.return_value = sample_analysis_model
        
        result = await create_analysis_handler(sample_text_analysis_request, mock_db_manager)
        
        assert isinstance(result, AnalysisResponse)
        assert result.id == 1
    
    @pytest.mark.asyncio
    async def test_handler_error(self, sample_text_analysis_request, mock_db_manager) -> Any:
        """Test handler error handling."""
        mock_db_manager.create_text_analysis.side_effect = ValueError("Database error")
        
        with pytest.raises(Exception):
            await create_analysis_handler(sample_text_analysis_request, mock_db_manager)

class TestGetAnalysisHandler:
    """Test get_analysis_handler function."""
    
    @pytest.mark.asyncio
    async def test_successful_handler(self, mock_db_manager, sample_analysis_model) -> Any:
        """Test successful handler execution."""
        mock_db_manager.get_text_analysis.return_value = sample_analysis_model
        
        result = await get_analysis_handler(1, mock_db_manager)
        
        assert isinstance(result, AnalysisResponse)
        assert result.id == 1
    
    @pytest.mark.asyncio
    async def test_not_found_handler(self, mock_db_manager) -> Any:
        """Test not found handler."""
        mock_db_manager.get_text_analysis.return_value = None
        
        with pytest.raises(Exception):
            await get_analysis_handler(999, mock_db_manager)

# ============================================================================
# Integration Tests
# ============================================================================

class TestFunctionalIntegration:
    """Integration tests for functional components."""
    
    @pytest.mark.asyncio
    async def test_full_analysis_workflow(self, mock_db_manager, sample_analysis_model) -> Any:
        """Test full analysis workflow."""
        # Create request
        request = TextAnalysisRequest(
            text_content="Integration test text",
            analysis_type=AnalysisTypeEnum.SENTIMENT,
            optimization_tier=OptimizationTierEnum.STANDARD
        )
        
        # Validate
        validation_result = validate_text_content(request.text_content)
        assert validation_result.is_valid is True
        
        # Calculate priority
        priority = calculate_processing_priority(
            request.optimization_tier,
            len(request.text_content),
            request.analysis_type
        )
        assert 1 <= priority <= 10
        
        # Estimate time
        estimated_time = estimate_processing_time(
            len(request.text_content),
            request.analysis_type,
            request.optimization_tier
        )
        assert estimated_time > 0
        
        # Create analysis
        mock_db_manager.create_text_analysis.return_value = sample_analysis_model
        result = await create_analysis_service(request, mock_db_manager)
        
        assert result.success is True
        assert result.data is not None
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, mock_db_manager) -> Any:
        """Test batch processing workflow."""
        # Create batch request
        request = BatchAnalysisRequest(
            batch_name="Integration Test Batch",
            texts=["Text 1", "Text 2", "Text 3"],
            analysis_type=AnalysisTypeEnum.SENTIMENT
        )
        
        # Calculate progress
        progress = calculate_batch_progress(1, 0, 3)
        assert progress["progress_percentage"] == 33.33
        
        # Create batch
        mock_batch = MagicMock()
        mock_batch.id = 1
        mock_batch.batch_name = request.batch_name
        mock_batch.analysis_type = request.analysis_type
        mock_batch.status = AnalysisStatusEnum.PENDING
        mock_batch.optimization_tier = request.optimization_tier
        mock_batch.created_at = datetime.now()
        mock_batch.updated_at = datetime.now()
        
        mock_db_manager.create_batch_analysis.return_value = mock_batch
        
        result = await create_batch_service(request, mock_db_manager)
        
        assert result.success is True
        assert result.data is not None

# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for functional components."""
    
    def test_validation_performance(self) -> Any:
        """Test validation performance."""
        
        start_time = time.time()
        for _ in range(1000):
            validate_text_content("Test text for performance validation")
        end_time = time.time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 1.0
    
    def test_priority_calculation_performance(self) -> Any:
        """Test priority calculation performance."""
        
        start_time = time.time()
        for _ in range(1000):
            calculate_processing_priority(
                OptimizationTierEnum.STANDARD,
                1000,
                AnalysisTypeEnum.SENTIMENT
            )
        end_time = time.time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 1.0
    
    def test_cache_key_generation_performance(self) -> Any:
        """Test cache key generation performance."""
        
        start_time = time.time()
        for _ in range(1000):
            generate_cache_key(
                "Test text for cache key generation",
                AnalysisTypeEnum.SENTIMENT,
                OptimizationTierEnum.STANDARD
            )
        end_time = time.time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 1.0

# ============================================================================
# Test Utilities
# ============================================================================

def test_example_functional_usage():
    """Test example functional usage."""
    # This would test the example_functional_usage function
    # Implementation depends on the actual function content
    pass

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 