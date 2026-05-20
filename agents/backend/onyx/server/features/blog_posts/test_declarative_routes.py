from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException, status
from .declarative_routes import (
from .functional_fastapi_components import (
        from fastapi import APIRouter
        from fastapi import APIRouter
        from fastapi import APIRouter
        import time
        import time
from typing import Any, List, Dict, Optional
import logging
"""
🧪 Test Suite for Declarative Route Definitions
==============================================

Comprehensive testing of:
- Route handlers with clear return type annotations
- Response wrapper decorators
- Request logging and performance monitoring
- Dependency injection
- Error handling
- OpenAPI documentation generation
"""


    # Route Classes
    AnalysisRoutes, BatchRoutes, HealthRoutes,
    
    # Response Types
    RouteResponse, AnalysisListResponse, AnalysisDetailResponse,
    BatchDetailResponse, HealthCheckResponse, ErrorDetailResponse,
    
    # Decorators
    with_response_wrapper, with_request_logging, with_performance_monitoring,
    
    # Dependencies
    get_db_manager, get_current_user, get_request_id,
    
    # Type Annotations
    AnalysisID, BatchID, PageNumber, PageSize, OrderBy, OrderDesc,
    DBManager, AuthToken, BackgroundTaskManager,
    
    # Application Factory
    create_analysis_router, create_app
)

    # Pydantic Models
    TextAnalysisRequest, BatchAnalysisRequest, AnalysisUpdateRequest,
    PaginationRequest, AnalysisFilterRequest,
    AnalysisResponse, BatchAnalysisResponse, PaginatedResponse,
    HealthResponse, ErrorResponse, SuccessResponse,
    
    # Enums
    AnalysisTypeEnum, OptimizationTierEnum, AnalysisStatusEnum,
    
    # Service Functions
    create_analysis_service, get_analysis_service,
    update_analysis_service, list_analyses_service, create_batch_service
)

# ============================================================================
# Test Fixtures
# ============================================================================

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
def mock_auth_token():
    """Mock authentication token."""
    token = MagicMock()
    token.credentials = "mock_token"
    return token

@pytest.fixture
def mock_request():
    """Mock FastAPI request."""
    request = MagicMock()
    request.method = "GET"
    request.url = "http://localhost:8000/api/v1/analyses"
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    request.headers = {"user-agent": "test-agent", "X-Request-ID": "test-request-id"}
    return request

@pytest.fixture
def mock_background_tasks():
    """Mock background tasks."""
    tasks = MagicMock()
    tasks.add_task = MagicMock()
    return tasks

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
def sample_analysis_response():
    """Sample analysis response."""
    return AnalysisResponse(
        id=1,
        text_content="Test text content",
        analysis_type=AnalysisTypeEnum.SENTIMENT,
        status=AnalysisStatusEnum.COMPLETED,
        sentiment_score=0.8,
        quality_score=0.95,
        processing_time_ms=150.5,
        model_used="test-model",
        confidence_score=0.92,
        optimization_tier=OptimizationTierEnum.STANDARD,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        processed_at=datetime.now(),
        metadata={"test": True}
    )

@pytest.fixture
def sample_batch_response():
    """Sample batch response."""
    return BatchAnalysisResponse(
        id=1,
        batch_name="Test Batch",
        analysis_type=AnalysisTypeEnum.SENTIMENT,
        status=AnalysisStatusEnum.COMPLETED,
        total_texts=10,
        completed_count=8,
        error_count=1,
        progress_percentage=90.0,
        optimization_tier=OptimizationTierEnum.STANDARD,
        priority=5,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={"test": True}
    )

@pytest.fixture
def fastapi_app():
    """Create FastAPI test application."""
    return create_app()

@pytest.fixture
def test_client(fastapi_app) -> Any:
    """Create test client."""
    return TestClient(fastapi_app)

# ============================================================================
# Response Type Tests
# ============================================================================

class TestRouteResponse:
    """Test route response types."""
    
    def test_route_response_creation(self) -> Any:
        """Test basic route response creation."""
        response = RouteResponse(
            success=True,
            data={"test": "data"},
            message="Test message",
            request_id="test-id"
        )
        
        assert response.success is True
        assert response.data == {"test": "data"}
        assert response.message == "Test message"
        assert response.request_id == "test-id"
        assert isinstance(response.timestamp, datetime)
    
    def test_analysis_detail_response(self, sample_analysis_response) -> Any:
        """Test analysis detail response."""
        response = AnalysisDetailResponse(
            success=True,
            data=sample_analysis_response,
            message="Analysis retrieved successfully",
            request_id="test-id"
        )
        
        assert response.success is True
        assert response.data == sample_analysis_response
        assert response.message == "Analysis retrieved successfully"
    
    def test_analysis_list_response(self, sample_analysis_response) -> List[Any]:
        """Test analysis list response."""
        paginated_data = PaginatedResponse.create(
            items=[sample_analysis_response],
            total=1,
            page=1,
            size=20
        )
        
        response = AnalysisListResponse(
            success=True,
            data=paginated_data,
            message="Analyses retrieved successfully",
            request_id="test-id"
        )
        
        assert response.success is True
        assert response.data == paginated_data
        assert len(response.data.items) == 1
    
    def test_batch_detail_response(self, sample_batch_response) -> Any:
        """Test batch detail response."""
        response = BatchDetailResponse(
            success=True,
            data=sample_batch_response,
            message="Batch retrieved successfully",
            request_id="test-id"
        )
        
        assert response.success is True
        assert response.data == sample_batch_response
        assert response.message == "Batch retrieved successfully"
    
    def test_health_check_response(self) -> Any:
        """Test health check response."""
        health_data = HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            uptime_seconds=1.0,
            database={"status": "healthy"},
            performance={"response_time_ms": 1000.0}
        )
        
        response = HealthCheckResponse(
            success=True,
            data=health_data,
            message="Health check completed",
            request_id="test-id"
        )
        
        assert response.success is True
        assert response.data == health_data
        assert response.data.status == "healthy"

# ============================================================================
# Decorator Tests
# ============================================================================

class TestWithResponseWrapper:
    """Test response wrapper decorator."""
    
    @pytest.mark.asyncio
    async def test_successful_response_wrapping(self) -> Any:
        """Test successful response wrapping."""
        @with_response_wrapper(AnalysisDetailResponse)
        async def test_func():
            
    """test_func function."""
return {"id": 1, "text": "test"}
        
        result = await test_func()
        
        assert isinstance(result, AnalysisDetailResponse)
        assert result.success is True
        assert result.data == {"id": 1, "text": "test"}
        assert result.message == "Operation completed successfully"
    
    @pytest.mark.asyncio
    async def test_already_wrapped_response(self) -> Any:
        """Test response that's already wrapped."""
        @with_response_wrapper(AnalysisDetailResponse)
        async def test_func():
            
    """test_func function."""
return AnalysisDetailResponse(
                success=True,
                data={"id": 1},
                message="Custom message"
            )
        
        result = await test_func()
        
        assert isinstance(result, AnalysisDetailResponse)
        assert result.success is True
        assert result.data == {"id": 1}
        assert result.message == "Custom message"
    
    @pytest.mark.asyncio
    async async def test_http_exception_passthrough(self) -> Any:
        """Test HTTP exception passthrough."""
        @with_response_wrapper(AnalysisDetailResponse)
        async def test_func():
            
    """test_func function."""
raise HTTPException(status_code=404, detail="Not found")
        
        with pytest.raises(HTTPException) as exc_info:
            await test_func()
        
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Not found"
    
    @pytest.mark.asyncio
    async def test_general_exception_wrapping(self) -> Any:
        """Test general exception wrapping."""
        @with_response_wrapper(AnalysisDetailResponse)
        async def test_func():
            
    """test_func function."""
raise ValueError("Test error")
        
        result = await test_func()
        
        assert isinstance(result, ErrorDetailResponse)
        assert result.success is False
        assert result.data.error == "Test error"
        assert result.data.error_code == "INTERNAL_ERROR"

class TestWithRequestLogging:
    """Test request logging decorator."""
    
    @pytest.mark.asyncio
    async async def test_successful_request_logging(self, mock_request) -> Any:
        """Test successful request logging."""
        @with_request_logging()
        async def test_func(request: Mock):
            
    """test_func function."""
return "success"
        
        with patch('structlog.get_logger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log
            
            result = await test_func(mock_request)
            
            assert result == "success"
            assert mock_log.info.call_count == 2  # Start and success
    
    @pytest.mark.asyncio
    async async def test_failed_request_logging(self, mock_request) -> Any:
        """Test failed request logging."""
        @with_request_logging()
        async def test_func(request: Mock):
            
    """test_func function."""
raise ValueError("Test error")
        
        with patch('structlog.get_logger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log
            
            with pytest.raises(ValueError):
                await test_func(mock_request)
            
            assert mock_log.error.call_count == 1

class TestWithPerformanceMonitoring:
    """Test performance monitoring decorator."""
    
    @pytest.mark.asyncio
    async def test_successful_performance_monitoring(self) -> Any:
        """Test successful performance monitoring."""
        @with_performance_monitoring()
        async def test_func():
            
    """test_func function."""
await asyncio.sleep(0.1)  # Simulate work
            return "success"
        
        with patch('structlog.get_logger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log
            
            result = await test_func()
            
            assert result == "success"
            assert mock_log.info.call_count == 1
            call_args = mock_log.info.call_args[1]
            assert call_args["function"] == "test_func"
            assert call_args["success"] is True
            assert call_args["processing_time_seconds"] > 0
    
    @pytest.mark.asyncio
    async def test_failed_performance_monitoring(self) -> Any:
        """Test failed performance monitoring."""
        @with_performance_monitoring()
        async def test_func():
            
    """test_func function."""
await asyncio.sleep(0.1)  # Simulate work
            raise ValueError("Test error")
        
        with patch('structlog.get_logger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log
            
            with pytest.raises(ValueError):
                await test_func()
            
            assert mock_log.error.call_count == 1
            call_args = mock_log.error.call_args[1]
            assert call_args["function"] == "test_func"
            assert call_args["success"] is False
            assert call_args["processing_time_seconds"] > 0

# ============================================================================
# Dependency Tests
# ============================================================================

class TestDependencies:
    """Test dependency functions."""
    
    @pytest.mark.asyncio
    async def test_get_db_manager(self) -> Optional[Dict[str, Any]]:
        """Test database manager dependency."""
        with patch('declarative_routes.SQLAlchemy2Manager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            
            result = await get_db_manager()
            
            assert result == mock_manager
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, mock_auth_token) -> Optional[Dict[str, Any]]:
        """Test current user dependency."""
        result = await get_current_user(mock_auth_token)
        
        assert isinstance(result, dict)
        assert result["id"] == 1
        assert result["email"] == "user@example.com"
        assert result["role"] == "user"
    
    @pytest.mark.asyncio
    async async def test_get_request_id_with_header(self, mock_request) -> Optional[Dict[str, Any]]:
        """Test request ID dependency with header."""
        result = await get_request_id(mock_request)
        
        assert result == "test-request-id"
    
    @pytest.mark.asyncio
    async async def test_get_request_id_without_header(self, mock_request) -> Optional[Dict[str, Any]]:
        """Test request ID dependency without header."""
        mock_request.headers = {}
        
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = "generated-id"
            result = await get_request_id(mock_request)
            
            assert result == "generated-id"

# ============================================================================
# Route Handler Tests
# ============================================================================

class TestAnalysisRoutes:
    """Test analysis routes."""
    
    @pytest.fixture
    def router(self) -> Any:
        """Create router for testing."""
        router = APIRouter()
        return AnalysisRoutes(router)
    
    @pytest.mark.asyncio
    async def test_create_analysis_success(self, router, sample_text_analysis_request, 
                                         mock_db_manager, sample_analysis_response) -> Any:
        """Test successful analysis creation."""
        # Mock service response
        with patch('declarative_routes.create_analysis_service') as mock_service:
            mock_service.return_value.success = True
            mock_service.return_value.data = sample_analysis_response
            
            # Mock dependencies
            with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
                with patch('declarative_routes.get_current_user', return_value={"id": 1}):
                    with patch('declarative_routes.get_request_id', return_value="test-id"):
                        with patch('declarative_routes.BackgroundTasks') as mock_bg_tasks:
                            mock_bg_tasks.return_value.add_task = MagicMock()
                            
                            # Call the route handler
                            result = await router._register_routes.create_analysis(
                                request=sample_text_analysis_request,
                                background_tasks=mock_bg_tasks.return_value,
                                db_manager=mock_db_manager,
                                current_user={"id": 1},
                                request_id="test-id"
                            )
                            
                            assert isinstance(result, AnalysisDetailResponse)
                            assert result.success is True
                            assert result.data == sample_analysis_response
                            assert result.message == "Analysis created successfully"
    
    @pytest.mark.asyncio
    async def test_create_analysis_validation_error(self, router, mock_db_manager) -> Any:
        """Test analysis creation with validation error."""
        invalid_request = TextAnalysisRequest(
            text_content="",  # Invalid empty content
            analysis_type=AnalysisTypeEnum.SENTIMENT
        )
        
        with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
            with patch('declarative_routes.get_current_user', return_value={"id": 1}):
                with patch('declarative_routes.get_request_id', return_value="test-id"):
                    with patch('declarative_routes.BackgroundTasks') as mock_bg_tasks:
                        mock_bg_tasks.return_value.add_task = MagicMock()
                        
                        with pytest.raises(HTTPException) as exc_info:
                            await router._register_routes.create_analysis(
                                request=invalid_request,
                                background_tasks=mock_bg_tasks.return_value,
                                db_manager=mock_db_manager,
                                current_user={"id": 1},
                                request_id="test-id"
                            )
                        
                        assert exc_info.value.status_code == 400
                        assert "Validation failed" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_get_analysis_success(self, router, mock_db_manager, sample_analysis_response) -> Optional[Dict[str, Any]]:
        """Test successful analysis retrieval."""
        with patch('declarative_routes.get_analysis_service') as mock_service:
            mock_service.return_value.success = True
            mock_service.return_value.data = sample_analysis_response
            
            with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
                with patch('declarative_routes.get_current_user', return_value={"id": 1}):
                    with patch('declarative_routes.get_request_id', return_value="test-id"):
                        result = await router._register_routes.get_analysis(
                            analysis_id=1,
                            db_manager=mock_db_manager,
                            current_user={"id": 1},
                            request_id="test-id"
                        )
                        
                        assert isinstance(result, AnalysisDetailResponse)
                        assert result.success is True
                        assert result.data == sample_analysis_response
    
    @pytest.mark.asyncio
    async def test_get_analysis_not_found(self, router, mock_db_manager) -> Optional[Dict[str, Any]]:
        """Test analysis retrieval with not found."""
        with patch('declarative_routes.get_analysis_service') as mock_service:
            mock_service.return_value.success = False
            mock_service.return_value.error = "Analysis not found"
            
            with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
                with patch('declarative_routes.get_current_user', return_value={"id": 1}):
                    with patch('declarative_routes.get_request_id', return_value="test-id"):
                        with pytest.raises(HTTPException) as exc_info:
                            await router._register_routes.get_analysis(
                                analysis_id=999,
                                db_manager=mock_db_manager,
                                current_user={"id": 1},
                                request_id="test-id"
                            )
                        
                        assert exc_info.value.status_code == 404
                        assert "not found" in str(exc_info.value.detail)

class TestBatchRoutes:
    """Test batch routes."""
    
    @pytest.fixture
    def router(self) -> Any:
        """Create router for testing."""
        router = APIRouter()
        return BatchRoutes(router)
    
    @pytest.mark.asyncio
    async def test_create_batch_success(self, router, sample_batch_analysis_request,
                                      mock_db_manager, sample_batch_response) -> Any:
        """Test successful batch creation."""
        with patch('declarative_routes.create_batch_service') as mock_service:
            mock_service.return_value.success = True
            mock_service.return_value.data = sample_batch_response
            
            with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
                with patch('declarative_routes.get_current_user', return_value={"id": 1}):
                    with patch('declarative_routes.get_request_id', return_value="test-id"):
                        with patch('declarative_routes.BackgroundTasks') as mock_bg_tasks:
                            mock_bg_tasks.return_value.add_task = MagicMock()
                            
                            result = await router._register_routes.create_batch(
                                request=sample_batch_analysis_request,
                                background_tasks=mock_bg_tasks.return_value,
                                db_manager=mock_db_manager,
                                current_user={"id": 1},
                                request_id="test-id"
                            )
                            
                            assert isinstance(result, BatchDetailResponse)
                            assert result.success is True
                            assert result.data == sample_batch_response

class TestHealthRoutes:
    """Test health routes."""
    
    @pytest.fixture
    def router(self) -> Any:
        """Create router for testing."""
        router = APIRouter()
        return HealthRoutes(router)
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, router, mock_db_manager) -> Any:
        """Test successful health check."""
        with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
            with patch('declarative_routes.get_request_id', return_value="test-id"):
                result = await router._register_routes.health_check(
                    db_manager=mock_db_manager,
                    request_id="test-id"
                )
                
                assert isinstance(result, HealthCheckResponse)
                assert result.success is True
                assert result.data.status == "healthy"
                assert result.message == "Health check completed"
    
    @pytest.mark.asyncio
    async def test_detailed_health_check(self, router, mock_db_manager) -> Any:
        """Test detailed health check."""
        with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
            with patch('declarative_routes.get_request_id', return_value="test-id"):
                with patch.object(router, '_check_database', return_value={"status": "healthy", "error": None}):
                    with patch.object(router, '_check_cache', return_value={"status": "healthy", "error": None}):
                        with patch.object(router, '_check_memory', return_value={"status": "healthy", "error": None}):
                            with patch.object(router, '_check_disk', return_value={"status": "healthy", "error": None}):
                                result = await router._register_routes.detailed_health_check(
                                    db_manager=mock_db_manager,
                                    request_id="test-id"
                                )
                                
                                assert isinstance(result, HealthCheckResponse)
                                assert result.success is True
                                assert result.data.status == "healthy"

# ============================================================================
# Integration Tests
# ============================================================================

class TestRouteIntegration:
    """Integration tests for routes."""
    
    @pytest.mark.asyncio
    async def test_analysis_workflow(self, mock_db_manager, sample_analysis_response) -> Any:
        """Test complete analysis workflow."""
        # Mock all dependencies
        with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
            with patch('declarative_routes.get_current_user', return_value={"id": 1}):
                with patch('declarative_routes.get_request_id', return_value="test-id"):
                    with patch('declarative_routes.create_analysis_service') as mock_create:
                        mock_create.return_value.success = True
                        mock_create.return_value.data = sample_analysis_response
                        
                        with patch('declarative_routes.BackgroundTasks') as mock_bg_tasks:
                            mock_bg_tasks.return_value.add_task = MagicMock()
                            
                            # Create request
                            request = TextAnalysisRequest(
                                text_content="Integration test text",
                                analysis_type=AnalysisTypeEnum.SENTIMENT,
                                optimization_tier=OptimizationTierEnum.STANDARD
                            )
                            
                            # Test creation
                            router = AnalysisRoutes(MagicMock())
                            result = await router._register_routes.create_analysis(
                                request=request,
                                background_tasks=mock_bg_tasks.return_value,
                                db_manager=mock_db_manager,
                                current_user={"id": 1},
                                request_id="test-id"
                            )
                            
                            assert result.success is True
                            assert result.data == sample_analysis_response
                            
                            # Verify background task was added
                            mock_bg_tasks.return_value.add_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_workflow(self, mock_db_manager, sample_batch_response) -> Any:
        """Test complete batch workflow."""
        with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
            with patch('declarative_routes.get_current_user', return_value={"id": 1}):
                with patch('declarative_routes.get_request_id', return_value="test-id"):
                    with patch('declarative_routes.create_batch_service') as mock_create:
                        mock_create.return_value.success = True
                        mock_create.return_value.data = sample_batch_response
                        
                        with patch('declarative_routes.BackgroundTasks') as mock_bg_tasks:
                            mock_bg_tasks.return_value.add_task = MagicMock()
                            
                            # Create batch request
                            request = BatchAnalysisRequest(
                                batch_name="Integration Test Batch",
                                texts=["Text 1", "Text 2", "Text 3"],
                                analysis_type=AnalysisTypeEnum.SENTIMENT
                            )
                            
                            # Test creation
                            router = BatchRoutes(MagicMock())
                            result = await router._register_routes.create_batch(
                                request=request,
                                background_tasks=mock_bg_tasks.return_value,
                                db_manager=mock_db_manager,
                                current_user={"id": 1},
                                request_id="test-id"
                            )
                            
                            assert result.success is True
                            assert result.data == sample_batch_response
                            
                            # Verify background task was added
                            mock_bg_tasks.return_value.add_task.assert_called_once()

# ============================================================================
# Application Tests
# ============================================================================

class TestApplication:
    """Test FastAPI application."""
    
    def test_create_analysis_router(self) -> Any:
        """Test analysis router creation."""
        router = create_analysis_router()
        
        assert router is not None
        assert router.prefix == "/api/v1"
        assert "Analysis" in router.tags
    
    def test_create_app(self) -> Any:
        """Test application creation."""
        app = create_app()
        
        assert app is not None
        assert app.title == "Text Analysis API"
        assert app.version == "1.0.0"
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"
    
    def test_global_exception_handler(self, fastapi_app) -> Any:
        """Test global exception handler."""
        @fastapi_app.get("/test-error")
        async def test_error():
            
    """test_error function."""
raise ValueError("Test error")
        
        client = TestClient(fastapi_app)
        
        with patch('structlog.get_logger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log
            
            response = client.get("/test-error")
            
            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
            assert "Internal server error" in data["error"]
            
            # Verify logging
            mock_log.error.assert_called_once()

# ============================================================================
# OpenAPI Documentation Tests
# ============================================================================

class TestOpenAPIDocumentation:
    """Test OpenAPI documentation generation."""
    
    async def test_openapi_schema_generation(self, fastapi_app) -> Any:
        """Test OpenAPI schema generation."""
        client = TestClient(fastapi_app)
        
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        # Check basic schema structure
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check API info
        assert schema["info"]["title"] == "Text Analysis API"
        assert schema["info"]["version"] == "1.0.0"
        
        # Check paths exist
        paths = schema["paths"]
        assert "/api/v1/analyses" in paths
        assert "/api/v1/batches" in paths
        assert "/api/v1/health" in paths
    
    def test_route_documentation(self, fastapi_app) -> Any:
        """Test route documentation."""
        client = TestClient(fastapi_app)
        
        response = client.get("/openapi.json")
        schema = response.json()
        
        # Check analysis routes
        analyses_path = schema["paths"]["/api/v1/analyses"]
        
        # POST endpoint
        assert "post" in analyses_path
        post_operation = analyses_path["post"]
        assert post_operation["summary"] == "Create Text Analysis"
        assert "Create a new text analysis" in post_operation["description"]
        assert post_operation["tags"] == ["Analysis"]
        
        # GET endpoint
        assert "get" in analyses_path
        get_operation = analyses_path["get"]
        assert get_operation["summary"] == "List Analyses"
        assert "Retrieve paginated list" in get_operation["description"]
    
    def test_response_model_documentation(self, fastapi_app) -> Any:
        """Test response model documentation."""
        client = TestClient(fastapi_app)
        
        response = client.get("/openapi.json")
        schema = response.json()
        
        # Check response models
        components = schema.get("components", {})
        schemas = components.get("schemas", {})
        
        # Check AnalysisDetailResponse
        assert "AnalysisDetailResponse" in schemas
        analysis_response = schemas["AnalysisDetailResponse"]
        assert analysis_response["type"] == "object"
        assert "success" in analysis_response["properties"]
        assert "data" in analysis_response["properties"]
        assert "message" in analysis_response["properties"]

# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for routes."""
    
    @pytest.mark.asyncio
    async def test_route_performance(self, mock_db_manager, sample_analysis_response) -> Any:
        """Test route performance."""
        
        with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
            with patch('declarative_routes.get_current_user', return_value={"id": 1}):
                with patch('declarative_routes.get_request_id', return_value="test-id"):
                    with patch('declarative_routes.get_analysis_service') as mock_service:
                        mock_service.return_value.success = True
                        mock_service.return_value.data = sample_analysis_response
                        
                        router = AnalysisRoutes(MagicMock())
                        
                        start_time = time.time()
                        
                        # Execute multiple requests
                        for _ in range(10):
                            result = await router._register_routes.get_analysis(
                                analysis_id=1,
                                db_manager=mock_db_manager,
                                current_user={"id": 1},
                                request_id="test-id"
                            )
                            assert result.success is True
                        
                        end_time = time.time()
                        total_time = end_time - start_time
                        
                        # Should complete in reasonable time
                        assert total_time < 1.0  # Less than 1 second for 10 requests
    
    def test_decorator_performance(self) -> Any:
        """Test decorator performance."""
        
        @with_response_wrapper(AnalysisDetailResponse)
        @with_request_logging()
        @with_performance_monitoring()
        async def test_func():
            
    """test_func function."""
return {"test": "data"}
        
        start_time = time.time()
        
        # Execute multiple times
        for _ in range(100):
            asyncio.run(test_func())
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete in reasonable time
        assert total_time < 5.0  # Less than 5 seconds for 100 executions

# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling in routes."""
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, mock_db_manager) -> Any:
        """Test database error handling."""
        with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
            with patch('declarative_routes.get_current_user', return_value={"id": 1}):
                with patch('declarative_routes.get_request_id', return_value="test-id"):
                    with patch('declarative_routes.get_analysis_service') as mock_service:
                        mock_service.side_effect = Exception("Database connection failed")
                        
                        router = AnalysisRoutes(MagicMock())
                        
                        with pytest.raises(HTTPException) as exc_info:
                            await router._register_routes.get_analysis(
                                analysis_id=1,
                                db_manager=mock_db_manager,
                                current_user={"id": 1},
                                request_id="test-id"
                            )
                        
                        assert exc_info.value.status_code == 500
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, mock_db_manager) -> Any:
        """Test validation error handling."""
        with patch('declarative_routes.get_db_manager', return_value=mock_db_manager):
            with patch('declarative_routes.get_current_user', return_value={"id": 1}):
                with patch('declarative_routes.get_request_id', return_value="test-id"):
                    with patch('declarative_routes.BackgroundTasks') as mock_bg_tasks:
                        mock_bg_tasks.return_value.add_task = MagicMock()
                        
                        router = AnalysisRoutes(MagicMock())
                        
                        # Invalid request
                        invalid_request = TextAnalysisRequest(
                            text_content="",  # Empty content
                            analysis_type=AnalysisTypeEnum.SENTIMENT
                        )
                        
                        with pytest.raises(HTTPException) as exc_info:
                            await router._register_routes.create_analysis(
                                request=invalid_request,
                                background_tasks=mock_bg_tasks.return_value,
                                db_manager=mock_db_manager,
                                current_user={"id": 1},
                                request_id="test-id"
                            )
                        
                        assert exc_info.value.status_code == 400
                        assert "Validation failed" in str(exc_info.value.detail)

# ============================================================================
# Test Utilities
# ============================================================================

def test_example_usage():
    """Test example usage function."""
    # This would test the example_usage function
    # Implementation depends on the actual function content
    pass

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 