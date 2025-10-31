"""
AI Integration System - Integration Engine Tests
Comprehensive test suite for the integration engine
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from ..integration_engine import (
    AIIntegrationEngine,
    IntegrationRequest,
    IntegrationResult,
    ContentType,
    IntegrationStatus,
    PlatformConnector
)

class MockConnector(PlatformConnector):
    """Mock connector for testing"""
    
    def __init__(self, config, should_fail=False, should_authenticate=True):
        super().__init__(config)
        self.should_fail = should_fail
        self.should_authenticate = should_authenticate
    
    async def authenticate(self) -> bool:
        return self.should_authenticate
    
    async def create_content(self, content_data):
        if self.should_fail:
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="mock",
                status=IntegrationStatus.FAILED,
                error_message="Mock failure"
            )
        else:
            return IntegrationResult(
                request_id=content_data.get("content_id", "unknown"),
                platform="mock",
                status=IntegrationStatus.COMPLETED,
                external_id="mock_id_123"
            )
    
    async def update_content(self, external_id, content_data):
        return IntegrationResult(
            request_id=content_data.get("content_id", "unknown"),
            platform="mock",
            status=IntegrationStatus.COMPLETED,
            external_id=external_id
        )
    
    async def delete_content(self, external_id):
        return IntegrationResult(
            request_id="unknown",
            platform="mock",
            status=IntegrationStatus.COMPLETED,
            external_id=external_id
        )
    
    async def get_content_status(self, external_id):
        return IntegrationResult(
            request_id="unknown",
            platform="mock",
            status=IntegrationStatus.COMPLETED,
            external_id=external_id
        )

@pytest.fixture
def integration_engine():
    """Create a fresh integration engine for each test"""
    return AIIntegrationEngine()

@pytest.fixture
def mock_connector():
    """Create a mock connector"""
    return MockConnector({"test": "config"})

@pytest.fixture
def failing_connector():
    """Create a failing mock connector"""
    return MockConnector({"test": "config"}, should_fail=True)

@pytest.fixture
def auth_failing_connector():
    """Create an authentication failing mock connector"""
    return MockConnector({"test": "config"}, should_authenticate=False)

@pytest.fixture
def sample_integration_request():
    """Create a sample integration request"""
    return IntegrationRequest(
        content_id="test_content_001",
        content_type=ContentType.BLOG_POST,
        content_data={
            "title": "Test Blog Post",
            "content": "This is a test blog post content.",
            "author": "Test Author",
            "tags": ["test", "blog"]
        },
        target_platforms=["mock"],
        priority=1,
        max_retries=3
    )

class TestAIIntegrationEngine:
    """Test cases for AIIntegrationEngine"""
    
    def test_engine_initialization(self, integration_engine):
        """Test engine initialization"""
        assert integration_engine.connectors == {}
        assert integration_engine.integration_queue == []
        assert integration_engine.results == {}
        assert integration_engine.is_running == False
    
    def test_register_connector(self, integration_engine, mock_connector):
        """Test connector registration"""
        integration_engine.register_connector("mock", mock_connector)
        
        assert "mock" in integration_engine.connectors
        assert integration_engine.connectors["mock"] == mock_connector
    
    def test_get_available_platforms(self, integration_engine, mock_connector):
        """Test getting available platforms"""
        integration_engine.register_connector("mock", mock_connector)
        integration_engine.register_connector("test", mock_connector)
        
        platforms = integration_engine.get_available_platforms()
        assert "mock" in platforms
        assert "test" in platforms
        assert len(platforms) == 2
    
    @pytest.mark.asyncio
    async def test_add_integration_request(self, integration_engine, sample_integration_request):
        """Test adding integration request to queue"""
        await integration_engine.add_integration_request(sample_integration_request)
        
        assert len(integration_engine.integration_queue) == 1
        assert integration_engine.integration_queue[0] == sample_integration_request
    
    @pytest.mark.asyncio
    async def test_process_single_request_success(self, integration_engine, mock_connector, sample_integration_request):
        """Test processing a single successful request"""
        integration_engine.register_connector("mock", mock_connector)
        
        await integration_engine.process_single_request(sample_integration_request)
        
        assert sample_integration_request.content_id in integration_engine.results
        results = integration_engine.results[sample_integration_request.content_id]
        assert len(results) == 1
        assert results[0].status == IntegrationStatus.COMPLETED
        assert results[0].external_id == "mock_id_123"
    
    @pytest.mark.asyncio
    async def test_process_single_request_failure(self, integration_engine, failing_connector, sample_integration_request):
        """Test processing a single failing request"""
        integration_engine.register_connector("mock", failing_connector)
        
        await integration_engine.process_single_request(sample_integration_request)
        
        assert sample_integration_request.content_id in integration_engine.results
        results = integration_engine.results[sample_integration_request.content_id]
        assert len(results) == 1
        assert results[0].status == IntegrationStatus.FAILED
        assert "Mock failure" in results[0].error_message
    
    @pytest.mark.asyncio
    async def test_process_single_request_auth_failure(self, integration_engine, auth_failing_connector, sample_integration_request):
        """Test processing request with authentication failure"""
        integration_engine.register_connector("mock", auth_failing_connector)
        
        await integration_engine.process_single_request(sample_integration_request)
        
        assert sample_integration_request.content_id in integration_engine.results
        results = integration_engine.results[sample_integration_request.content_id]
        assert len(results) == 0  # No results due to auth failure
    
    @pytest.mark.asyncio
    async def test_process_single_request_unknown_platform(self, integration_engine, sample_integration_request):
        """Test processing request with unknown platform"""
        await integration_engine.process_single_request(sample_integration_request)
        
        assert sample_integration_request.content_id in integration_engine.results
        results = integration_engine.results[sample_integration_request.content_id]
        assert len(results) == 0  # No results due to unknown platform
    
    @pytest.mark.asyncio
    async def test_process_integration_queue(self, integration_engine, mock_connector):
        """Test processing integration queue"""
        integration_engine.register_connector("mock", mock_connector)
        
        # Add multiple requests
        for i in range(3):
            request = IntegrationRequest(
                content_id=f"test_content_{i:03d}",
                content_type=ContentType.BLOG_POST,
                content_data={"title": f"Test {i}", "content": f"Content {i}"},
                target_platforms=["mock"],
                priority=1
            )
            await integration_engine.add_integration_request(request)
        
        assert len(integration_engine.integration_queue) == 3
        
        # Process queue
        await integration_engine.process_integration_queue()
        
        assert len(integration_engine.integration_queue) == 0
        assert len(integration_engine.results) == 3
    
    @pytest.mark.asyncio
    async def test_get_integration_status(self, integration_engine, mock_connector, sample_integration_request):
        """Test getting integration status"""
        integration_engine.register_connector("mock", mock_connector)
        
        await integration_engine.process_single_request(sample_integration_request)
        
        status = await integration_engine.get_integration_status(sample_integration_request.content_id)
        
        assert status["content_id"] == sample_integration_request.content_id
        assert status["overall_status"] == "completed"
        assert len(status["results"]) == 1
    
    @pytest.mark.asyncio
    async def test_get_integration_status_not_found(self, integration_engine):
        """Test getting status for non-existent integration"""
        status = await integration_engine.get_integration_status("non_existent")
        
        assert status["status"] == "not_found"
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self, integration_engine, mock_connector):
        """Test successful connection test"""
        integration_engine.register_connector("mock", mock_connector)
        
        result = await integration_engine.test_connection("mock")
        assert result == True
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self, integration_engine, auth_failing_connector):
        """Test failed connection test"""
        integration_engine.register_connector("mock", auth_failing_connector)
        
        result = await integration_engine.test_connection("mock")
        assert result == False
    
    @pytest.mark.asyncio
    async def test_test_connection_unknown_platform(self, integration_engine):
        """Test connection test for unknown platform"""
        result = await integration_engine.test_connection("unknown")
        assert result == False
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, integration_engine, failing_connector):
        """Test retry mechanism for failed requests"""
        integration_engine.register_connector("mock", failing_connector)
        
        request = IntegrationRequest(
            content_id="retry_test",
            content_type=ContentType.BLOG_POST,
            content_data={"title": "Retry Test", "content": "Test content"},
            target_platforms=["mock"],
            priority=1,
            max_retries=2
        )
        
        await integration_engine.process_single_request(request)
        
        # Should be in queue for retry
        assert len(integration_engine.integration_queue) == 1
        assert integration_engine.integration_queue[0].retry_count == 1
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, integration_engine, failing_connector):
        """Test max retries exceeded"""
        integration_engine.register_connector("mock", failing_connector)
        
        request = IntegrationRequest(
            content_id="max_retry_test",
            content_type=ContentType.BLOG_POST,
            content_data={"title": "Max Retry Test", "content": "Test content"},
            target_platforms=["mock"],
            priority=1,
            max_retries=1,
            retry_count=1  # Already at max retries
        )
        
        await integration_engine.process_single_request(request)
        
        # Should not be in queue for retry
        assert len(integration_engine.integration_queue) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_platforms(self, integration_engine):
        """Test processing request with multiple platforms"""
        mock_connector1 = MockConnector({"test": "config1"})
        mock_connector2 = MockConnector({"test": "config2"})
        
        integration_engine.register_connector("mock1", mock_connector1)
        integration_engine.register_connector("mock2", mock_connector2)
        
        request = IntegrationRequest(
            content_id="multi_platform_test",
            content_type=ContentType.BLOG_POST,
            content_data={"title": "Multi Platform Test", "content": "Test content"},
            target_platforms=["mock1", "mock2"],
            priority=1
        )
        
        await integration_engine.process_single_request(request)
        
        results = integration_engine.results[request.content_id]
        assert len(results) == 2
        assert all(result.status == IntegrationStatus.COMPLETED for result in results)
    
    @pytest.mark.asyncio
    async def test_mixed_platform_results(self, integration_engine):
        """Test processing with mixed success/failure results"""
        success_connector = MockConnector({"test": "success"})
        failure_connector = MockConnector({"test": "failure"}, should_fail=True)
        
        integration_engine.register_connector("success", success_connector)
        integration_engine.register_connector("failure", failure_connector)
        
        request = IntegrationRequest(
            content_id="mixed_results_test",
            content_type=ContentType.BLOG_POST,
            content_data={"title": "Mixed Results Test", "content": "Test content"},
            target_platforms=["success", "failure"],
            priority=1
        )
        
        await integration_engine.process_single_request(request)
        
        results = integration_engine.results[request.content_id]
        assert len(results) == 2
        
        success_results = [r for r in results if r.status == IntegrationStatus.COMPLETED]
        failure_results = [r for r in results if r.status == IntegrationStatus.FAILED]
        
        assert len(success_results) == 1
        assert len(failure_results) == 1

class TestIntegrationRequest:
    """Test cases for IntegrationRequest"""
    
    def test_integration_request_creation(self):
        """Test creating integration request"""
        request = IntegrationRequest(
            content_id="test_001",
            content_type=ContentType.BLOG_POST,
            content_data={"title": "Test", "content": "Content"},
            target_platforms=["platform1", "platform2"],
            priority=2,
            max_retries=5
        )
        
        assert request.content_id == "test_001"
        assert request.content_type == ContentType.BLOG_POST
        assert request.content_data == {"title": "Test", "content": "Content"}
        assert request.target_platforms == ["platform1", "platform2"]
        assert request.priority == 2
        assert request.max_retries == 5
        assert request.retry_count == 0
        assert request.metadata is None

class TestIntegrationResult:
    """Test cases for IntegrationResult"""
    
    def test_integration_result_creation(self):
        """Test creating integration result"""
        result = IntegrationResult(
            request_id="test_001",
            platform="test_platform",
            status=IntegrationStatus.COMPLETED,
            external_id="ext_123",
            response_data={"id": "ext_123", "status": "created"}
        )
        
        assert result.request_id == "test_001"
        assert result.platform == "test_platform"
        assert result.status == IntegrationStatus.COMPLETED
        assert result.external_id == "ext_123"
        assert result.response_data == {"id": "ext_123", "status": "created"}
        assert result.error_message is None
        assert result.timestamp is not None

class TestContentType:
    """Test cases for ContentType enum"""
    
    def test_content_type_values(self):
        """Test content type enum values"""
        assert ContentType.BLOG_POST == "blog_post"
        assert ContentType.EMAIL_CAMPAIGN == "email_campaign"
        assert ContentType.SOCIAL_MEDIA_POST == "social_media_post"
        assert ContentType.PRODUCT_DESCRIPTION == "product_description"
        assert ContentType.LANDING_PAGE == "landing_page"
        assert ContentType.DOCUMENT == "document"
        assert ContentType.PRESENTATION == "presentation"

class TestIntegrationStatus:
    """Test cases for IntegrationStatus enum"""
    
    def test_integration_status_values(self):
        """Test integration status enum values"""
        assert IntegrationStatus.PENDING == "pending"
        assert IntegrationStatus.IN_PROGRESS == "in_progress"
        assert IntegrationStatus.COMPLETED == "completed"
        assert IntegrationStatus.FAILED == "failed"
        assert IntegrationStatus.RETRY == "retry"

# Integration tests
class TestIntegrationEngineIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_full_integration_workflow(self, integration_engine):
        """Test complete integration workflow"""
        # Setup connectors
        connector1 = MockConnector({"config": "1"})
        connector2 = MockConnector({"config": "2"})
        
        integration_engine.register_connector("platform1", connector1)
        integration_engine.register_connector("platform2", connector2)
        
        # Create request
        request = IntegrationRequest(
            content_id="full_workflow_test",
            content_type=ContentType.BLOG_POST,
            content_data={
                "title": "Full Workflow Test",
                "content": "This is a comprehensive test of the integration workflow.",
                "author": "Test Author",
                "tags": ["test", "integration", "workflow"]
            },
            target_platforms=["platform1", "platform2"],
            priority=1,
            metadata={"test": True, "timestamp": datetime.utcnow().isoformat()}
        )
        
        # Add to queue
        await integration_engine.add_integration_request(request)
        assert len(integration_engine.integration_queue) == 1
        
        # Process queue
        await integration_engine.process_integration_queue()
        assert len(integration_engine.integration_queue) == 0
        
        # Check results
        status = await integration_engine.get_integration_status(request.content_id)
        assert status["overall_status"] == "completed"
        assert len(status["results"]) == 2
        
        # Verify both platforms were processed
        platforms_processed = [result["platform"] for result in status["results"]]
        assert "platform1" in platforms_processed
        assert "platform2" in platforms_processed
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, integration_engine):
        """Test error handling and recovery mechanisms"""
        # Setup mixed connectors
        success_connector = MockConnector({"config": "success"})
        failure_connector = MockConnector({"config": "failure"}, should_fail=True)
        auth_fail_connector = MockConnector({"config": "auth_fail"}, should_authenticate=False)
        
        integration_engine.register_connector("success", success_connector)
        integration_engine.register_connector("failure", failure_connector)
        integration_engine.register_connector("auth_fail", auth_fail_connector)
        
        # Create request with all platforms
        request = IntegrationRequest(
            content_id="error_handling_test",
            content_type=ContentType.EMAIL_CAMPAIGN,
            content_data={
                "title": "Error Handling Test",
                "content": "Testing error handling mechanisms.",
                "subject": "Test Email"
            },
            target_platforms=["success", "failure", "auth_fail"],
            priority=1
        )
        
        # Process request
        await integration_engine.process_single_request(request)
        
        # Check results
        results = integration_engine.results[request.content_id]
        assert len(results) == 2  # Only success and failure, auth_fail should be skipped
        
        # Verify result types
        success_results = [r for r in results if r.status == IntegrationStatus.COMPLETED]
        failure_results = [r for r in results if r.status == IntegrationStatus.FAILED]
        
        assert len(success_results) == 1
        assert len(failure_results) == 1
        assert success_results[0].platform == "success"
        assert failure_results[0].platform == "failure"



























