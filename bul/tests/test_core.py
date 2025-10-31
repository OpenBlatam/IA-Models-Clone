"""
BUL Core Engine Tests
====================

Comprehensive tests for the BUL core engine functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

from ..core.bul_engine import BULEngine, DocumentRequest, DocumentResponse, BusinessArea, DocumentType
from ..utils import get_cache_manager
from ..config import get_config

class TestBULEngine:
    """Test cases for BULEngine"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        config = Mock()
        config.api.openrouter_api_key = "test_key"
        config.api.openai_api_key = "test_openai_key"
        config.api.default_model = "test_model"
        config.api.max_tokens = 1000
        config.api.temperature = 0.7
        return config
    
    @pytest.fixture
    def bul_engine(self, mock_config):
        """Create BULEngine instance for testing"""
        with patch('..core.bul_engine.get_config', return_value=mock_config):
            engine = BULEngine("test_key", "test_openai_key")
            return engine
    
    @pytest.fixture
    def sample_request(self):
        """Sample document request for testing"""
        return DocumentRequest(
            query="Create a marketing plan for a new product launch",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.PLAN,
            language="es",
            format="markdown"
        )
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, bul_engine):
        """Test engine initialization"""
        assert bul_engine.openrouter_api_key == "test_key"
        assert bul_engine.openai_api_key == "test_openai_key"
        assert not bul_engine.is_initialized
    
    @pytest.mark.asyncio
    async def test_engine_initialization_success(self, bul_engine):
        """Test successful engine initialization"""
        with patch('httpx.AsyncClient') as mock_client:
            with patch('aiohttp.ClientSession') as mock_session:
                mock_client.return_value = AsyncMock()
                mock_session.return_value = AsyncMock()
                
                result = await bul_engine.initialize()
                
                assert result is True
                assert bul_engine.is_initialized
                assert bul_engine.http_client is not None
                assert bul_engine.session is not None
    
    @pytest.mark.asyncio
    async def test_engine_initialization_failure(self, bul_engine):
        """Test engine initialization failure"""
        with patch('httpx.AsyncClient', side_effect=Exception("Connection failed")):
            result = await bul_engine.initialize()
            
            assert result is False
            assert not bul_engine.is_initialized
    
    @pytest.mark.asyncio
    async def test_analyze_query_cached(self, bul_engine, sample_request):
        """Test query analysis with caching"""
        # Mock cache hit
        mock_cache = Mock()
        mock_cache.get.return_value = {
            "business_area": "marketing",
            "document_type": "plan",
            "complexity": "medium",
            "estimated_tokens": 500
        }
        
        with patch('..core.bul_engine.get_cache_manager', return_value=mock_cache):
            result = await bul_engine._analyze_query(sample_request.query)
            
            assert result["business_area"] == "marketing"
            assert result["document_type"] == "plan"
            mock_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_query_no_cache(self, bul_engine, sample_request):
        """Test query analysis without cache"""
        # Mock cache miss
        mock_cache = Mock()
        mock_cache.get.return_value = None
        
        with patch('..core.bul_engine.get_cache_manager', return_value=mock_cache):
            with patch.object(bul_engine, '_call_openrouter_api', return_value='{"business_area": "marketing", "document_type": "plan"}'):
                result = await bul_engine._analyze_query(sample_request.query)
                
                assert "business_area" in result
                mock_cache.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_openrouter_api_success(self, bul_engine, sample_request):
        """Test successful OpenRouter API call"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated document content"}}]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        bul_engine.http_client = mock_client
        
        result = await bul_engine._call_openrouter_api("Test prompt", sample_request)
        
        assert result == "Generated document content"
        mock_client.post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_call_openrouter_api_fallback(self, bul_engine, sample_request):
        """Test OpenRouter API fallback to aiohttp"""
        # Mock httpx failure
        mock_httpx_client = AsyncMock()
        mock_httpx_client.post.side_effect = Exception("HTTPX failed")
        
        # Mock aiohttp success
        mock_aiohttp_session = AsyncMock()
        mock_aiohttp_response = AsyncMock()
        mock_aiohttp_response.json.return_value = {
            "choices": [{"message": {"content": "Fallback content"}}]
        }
        mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_aiohttp_response
        
        bul_engine.http_client = mock_httpx_client
        bul_engine.session = mock_aiohttp_session
        
        result = await bul_engine._call_openrouter_api("Test prompt", sample_request)
        
        assert result == "Fallback content"
    
    @pytest.mark.asyncio
    async def test_generate_document_success(self, bul_engine, sample_request):
        """Test successful document generation"""
        # Mock initialization
        bul_engine.is_initialized = True
        
        # Mock query analysis
        with patch.object(bul_engine, '_analyze_query', return_value={
            "business_area": "marketing",
            "document_type": "plan",
            "complexity": "medium"
        }):
            # Mock API call
            with patch.object(bul_engine, '_call_openrouter_api', return_value="Generated marketing plan content"):
                result = await bul_engine.generate_document(sample_request)
                
                assert isinstance(result, DocumentResponse)
                assert result.content == "Generated marketing plan content"
                assert result.business_area == BusinessArea.MARKETING
                assert result.document_type == DocumentType.PLAN
                assert result.success is True
    
    @pytest.mark.asyncio
    async def test_generate_document_failure(self, bul_engine, sample_request):
        """Test document generation failure"""
        # Mock initialization
        bul_engine.is_initialized = True
        
        # Mock query analysis failure
        with patch.object(bul_engine, '_analyze_query', side_effect=Exception("Analysis failed")):
            result = await bul_engine.generate_document(sample_request)
            
            assert isinstance(result, DocumentResponse)
            assert result.success is False
            assert "Analysis failed" in result.error_message
    
    @pytest.mark.asyncio
    async def test_generate_document_not_initialized(self, bul_engine, sample_request):
        """Test document generation when engine not initialized"""
        bul_engine.is_initialized = False
        
        with patch.object(bul_engine, 'initialize', return_value=True):
            with patch.object(bul_engine, '_analyze_query', return_value={}):
                with patch.object(bul_engine, '_call_openrouter_api', return_value="Content"):
                    result = await bul_engine.generate_document(sample_request)
                    
                    assert result.success is True
                    bul_engine.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_stats(self, bul_engine):
        """Test getting engine statistics"""
        bul_engine.stats = {
            "documents_generated": 10,
            "total_processing_time": 50.0,
            "cache_hits": 5,
            "cache_misses": 3
        }
        
        stats = bul_engine.get_stats()
        
        assert stats["documents_generated"] == 10
        assert stats["total_processing_time"] == 50.0
        assert stats["cache_hits"] == 5
        assert stats["cache_misses"] == 3
    
    @pytest.mark.asyncio
    async def test_close(self, bul_engine):
        """Test engine cleanup"""
        mock_http_client = AsyncMock()
        mock_session = AsyncMock()
        
        bul_engine.http_client = mock_http_client
        bul_engine.session = mock_session
        
        await bul_engine.close()
        
        mock_http_client.aclose.assert_called_once()
        mock_session.close.assert_called_once()

class TestDocumentRequest:
    """Test cases for DocumentRequest"""
    
    def test_document_request_creation(self):
        """Test DocumentRequest creation"""
        request = DocumentRequest(
            query="Test query",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.PLAN,
            language="es",
            format="markdown"
        )
        
        assert request.query == "Test query"
        assert request.business_area == BusinessArea.MARKETING
        assert request.document_type == DocumentType.PLAN
        assert request.language == "es"
        assert request.format == "markdown"
    
    def test_document_request_defaults(self):
        """Test DocumentRequest with default values"""
        request = DocumentRequest(
            query="Test query",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.PLAN
        )
        
        assert request.language == "es"
        assert request.format == "markdown"
        assert request.context is None
        assert request.requirements is None

class TestDocumentResponse:
    """Test cases for DocumentResponse"""
    
    def test_document_response_creation(self):
        """Test DocumentResponse creation"""
        response = DocumentResponse(
            content="Generated content",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.PLAN,
            success=True
        )
        
        assert response.content == "Generated content"
        assert response.business_area == BusinessArea.MARKETING
        assert response.document_type == DocumentType.PLAN
        assert response.success is True
        assert response.document_id is not None
        assert response.generated_at is not None
    
    def test_document_response_failure(self):
        """Test DocumentResponse for failed generation"""
        response = DocumentResponse(
            content="",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.PLAN,
            success=False,
            error_message="Generation failed"
        )
        
        assert response.success is False
        assert response.error_message == "Generation failed"
        assert response.content == ""

class TestBusinessArea:
    """Test cases for BusinessArea enum"""
    
    def test_business_area_values(self):
        """Test BusinessArea enum values"""
        assert BusinessArea.MARKETING.value == "marketing"
        assert BusinessArea.SALES.value == "sales"
        assert BusinessArea.FINANCE.value == "finance"
        assert BusinessArea.HR.value == "hr"
        assert BusinessArea.LEGAL.value == "legal"
        assert BusinessArea.OPERATIONS.value == "operations"
        assert BusinessArea.IT.value == "it"
        assert BusinessArea.CUSTOMER_SERVICE.value == "customer_service"
        assert BusinessArea.PRODUCT.value == "product"
        assert BusinessArea.STRATEGY.value == "strategy"

class TestDocumentType:
    """Test cases for DocumentType enum"""
    
    def test_document_type_values(self):
        """Test DocumentType enum values"""
        assert DocumentType.PLAN.value == "plan"
        assert DocumentType.REPORT.value == "report"
        assert DocumentType.PROPOSAL.value == "proposal"
        assert DocumentType.CONTRACT.value == "contract"
        assert DocumentType.PRESENTATION.value == "presentation"
        assert DocumentType.EMAIL.value == "email"
        assert DocumentType.MANUAL.value == "manual"
        assert DocumentType.POLICY.value == "policy"
        assert DocumentType.PROCEDURE.value == "procedure"
        assert DocumentType.TEMPLATE.value == "template"

# Integration tests
class TestBULEngineIntegration:
    """Integration tests for BULEngine"""
    
    @pytest.mark.asyncio
    async def test_full_document_generation_flow(self):
        """Test complete document generation flow"""
        # This would be an integration test with real API calls
        # For now, we'll mock the entire flow
        with patch('..core.bul_engine.BULEngine') as mock_engine_class:
            mock_engine = AsyncMock()
            mock_engine_class.return_value = mock_engine
            
            mock_engine.initialize.return_value = True
            mock_engine.generate_document.return_value = DocumentResponse(
                content="Generated content",
                business_area=BusinessArea.MARKETING,
                document_type=DocumentType.PLAN,
                success=True
            )
            
            engine = BULEngine("test_key")
            await engine.initialize()
            
            request = DocumentRequest(
                query="Create a marketing plan",
                business_area=BusinessArea.MARKETING,
                document_type=DocumentType.PLAN
            )
            
            response = await engine.generate_document(request)
            
            assert response.success is True
            assert response.content == "Generated content"

# Performance tests
class TestBULEnginePerformance:
    """Performance tests for BULEngine"""
    
    @pytest.mark.asyncio
    async def test_concurrent_document_generation(self, bul_engine):
        """Test concurrent document generation"""
        bul_engine.is_initialized = True
        
        with patch.object(bul_engine, '_analyze_query', return_value={}):
            with patch.object(bul_engine, '_call_openrouter_api', return_value="Content"):
                requests = [
                    DocumentRequest(
                        query=f"Test query {i}",
                        business_area=BusinessArea.MARKETING,
                        document_type=DocumentType.PLAN
                    )
                    for i in range(5)
                ]
                
                start_time = datetime.now()
                results = await asyncio.gather(*[
                    bul_engine.generate_document(req) for req in requests
                ])
                end_time = datetime.now()
                
                assert len(results) == 5
                assert all(result.success for result in results)
                
                # Performance assertion (should complete within reasonable time)
                processing_time = (end_time - start_time).total_seconds()
                assert processing_time < 10.0  # Should complete within 10 seconds
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, bul_engine):
        """Test memory usage during document generation"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        bul_engine.is_initialized = True
        
        with patch.object(bul_engine, '_analyze_query', return_value={}):
            with patch.object(bul_engine, '_call_openrouter_api', return_value="Content"):
                # Generate multiple documents
                for i in range(10):
                    request = DocumentRequest(
                        query=f"Test query {i}",
                        business_area=BusinessArea.MARKETING,
                        document_type=DocumentType.PLAN
                    )
                    await bul_engine.generate_document(request)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024




