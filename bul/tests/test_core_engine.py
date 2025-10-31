"""
Test suite for BUL Core Engine
==============================

Comprehensive tests for the BUL document generation engine.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json

from core.bul_engine import BULEngine, DocumentRequest, DocumentResponse, BusinessArea, DocumentType


class TestBULEngine:
    """Test cases for BULEngine"""
    
    @pytest.fixture
    async def engine(self):
        """Create a test engine instance"""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test_key',
            'OPENAI_API_KEY': 'test_openai_key'
        }):
            engine = BULEngine('test_key', 'test_openai_key')
            await engine.initialize()
            return engine
    
    @pytest.fixture
    def sample_request(self):
        """Create a sample document request"""
        return DocumentRequest(
            query="Create a marketing strategy for a tech startup",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.MARKETING_STRATEGY,
            company_name="TechStartup Inc",
            industry="Technology",
            company_size="Small",
            target_audience="Tech-savvy consumers",
            language="en",
            format="markdown"
        )
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.is_initialized is True
        assert engine.openrouter_api_key == 'test_key'
        assert engine.openai_api_key == 'test_openai_key'
        assert engine.http_client is not None
        assert engine.session is not None
    
    @pytest.mark.asyncio
    async def test_analyze_query(self, engine):
        """Test query analysis functionality"""
        query = "I need a business plan for my restaurant"
        analysis = await engine._analyze_query(query)
        
        assert 'business_area' in analysis
        assert 'document_type' in analysis
        assert 'confidence' in analysis
        assert 'keywords_found' in analysis
        assert analysis['confidence'] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_query_marketing(self, engine):
        """Test marketing query analysis"""
        query = "Create a marketing campaign for our new product launch"
        analysis = await engine._analyze_query(query)
        
        assert analysis['business_area'] == 'marketing'
        assert analysis['document_type'] == 'marketing_strategy'
    
    @pytest.mark.asyncio
    async def test_analyze_query_sales(self, engine):
        """Test sales query analysis"""
        query = "I need a sales proposal for enterprise clients"
        analysis = await engine._analyze_query(query)
        
        assert analysis['business_area'] == 'sales'
        assert analysis['document_type'] == 'sales_proposal'
    
    @pytest.mark.asyncio
    async def test_create_prompt(self, engine, sample_request):
        """Test prompt creation"""
        prompt = engine._create_prompt(sample_request)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 100
        assert sample_request.query in prompt
        assert sample_request.business_area.value in prompt
        assert sample_request.document_type.value in prompt
        assert sample_request.company_name in prompt
    
    @pytest.mark.asyncio
    async def test_generate_title(self, engine, sample_request):
        """Test title generation"""
        content = "# Marketing Strategy for TechStartup Inc\n\nThis document outlines..."
        title = engine._generate_title(sample_request, content)
        
        assert isinstance(title, str)
        assert len(title) > 0
        assert len(title) <= 100
    
    @pytest.mark.asyncio
    async def test_generate_summary(self, engine):
        """Test summary generation"""
        content = "This is a comprehensive marketing strategy document that outlines various approaches to market our new product. It includes detailed analysis of target markets, competitive positioning, pricing strategies, and promotional activities."
        summary = engine._generate_summary(content)
        
        assert isinstance(summary, str)
        assert len(summary) <= 203  # 200 chars + "..."
        assert "marketing strategy" in summary.lower()
    
    @pytest.mark.asyncio
    async def test_update_stats(self, engine):
        """Test statistics update"""
        initial_docs = engine.stats["documents_generated"]
        
        response = DocumentResponse(
            id="test_id",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.MARKETING_STRATEGY,
            processing_time=5.0,
            confidence_score=0.9
        )
        
        engine._update_stats(response)
        
        assert engine.stats["documents_generated"] == initial_docs + 1
        assert engine.stats["total_processing_time"] > 0
        assert BusinessArea.MARKETING.value in engine.stats["business_areas_used"]
        assert DocumentType.MARKETING_STRATEGY.value in engine.stats["document_types_generated"]
    
    @pytest.mark.asyncio
    async def test_get_stats(self, engine):
        """Test statistics retrieval"""
        stats = await engine.get_stats()
        
        assert isinstance(stats, dict)
        assert "documents_generated" in stats
        assert "total_processing_time" in stats
        assert "average_processing_time" in stats
        assert "average_confidence" in stats
        assert "business_areas_used" in stats
        assert "document_types_generated" in stats
        assert "is_initialized" in stats
    
    @pytest.mark.asyncio
    async def test_call_openrouter_api_success(self, engine, sample_request):
        """Test successful OpenRouter API call"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Generated document content"}}]
        }
        
        with patch.object(engine.http_client, 'post', return_value=mock_response):
            prompt = engine._create_prompt(sample_request)
            content = await engine._call_openrouter_api(prompt, sample_request)
            
            assert content == "Generated document content"
            assert engine.stats["api_calls_successful"] > 0
    
    @pytest.mark.asyncio
    async def test_call_openrouter_api_failure(self, engine, sample_request):
        """Test OpenRouter API failure with fallback"""
        # Mock HTTP error
        with patch.object(engine.http_client, 'post', side_effect=Exception("API Error")):
            # Mock aiohttp fallback also failing
            with patch.object(engine.session, 'post', side_effect=Exception("Fallback Error")):
                # Mock OpenAI fallback
                with patch.object(engine, '_fallback_to_openai', return_value="Fallback content"):
                    prompt = engine._create_prompt(sample_request)
                    content = await engine._call_openrouter_api(prompt, sample_request)
                    
                    assert content == "Fallback content"
                    assert engine.stats["api_calls_failed"] > 0
                    assert engine.stats["fallback_usage"] > 0
    
    @pytest.mark.asyncio
    async def test_fallback_to_openai(self, engine):
        """Test OpenAI fallback functionality"""
        mock_llm = Mock()
        mock_chain = Mock()
        mock_chain.arun.return_value = "OpenAI generated content"
        
        with patch.object(engine, 'llm', mock_llm):
            with patch('core.bul_engine.LLMChain', return_value=mock_chain):
                with patch('core.bul_engine.PromptTemplate'):
                    content = await engine._fallback_to_openai("Test prompt")
                    assert content == "OpenAI generated content"
    
    @pytest.mark.asyncio
    async def test_generate_document_success(self, engine, sample_request):
        """Test successful document generation"""
        # Mock the API call
        with patch.object(engine, '_call_openrouter_api', return_value="Generated document content"):
            with patch.object(engine, '_analyze_query', return_value={
                "business_area": "marketing",
                "document_type": "marketing_strategy",
                "confidence": 0.9
            }):
                response = await engine.generate_document(sample_request)
                
                assert isinstance(response, DocumentResponse)
                assert response.content == "Generated document content"
                assert response.business_area == BusinessArea.MARKETING
                assert response.document_type == DocumentType.MARKETING_STRATEGY
                assert response.confidence_score == 0.9
                assert response.processing_time > 0
                assert response.word_count > 0
    
    @pytest.mark.asyncio
    async def test_generate_document_with_retry(self, engine, sample_request):
        """Test document generation with retry logic"""
        call_count = 0
        
        async def mock_api_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:  # Fail first 2 times
                raise Exception("API Error")
            return "Generated content after retry"
        
        with patch.object(engine, '_call_openrouter_api', side_effect=mock_api_call):
            with patch.object(engine, '_analyze_query', return_value={
                "business_area": "marketing",
                "document_type": "marketing_strategy",
                "confidence": 0.9
            }):
                response = await engine.generate_document(sample_request)
                
                assert response.content == "Generated content after retry"
                assert call_count == 3  # Should have retried 3 times
    
    @pytest.mark.asyncio
    async def test_generate_document_cache_hit(self, engine, sample_request):
        """Test document generation with cache hit"""
        # Mock cache hit
        cached_response = {
            "id": "cached_id",
            "content": "Cached content",
            "business_area": "marketing",
            "document_type": "marketing_strategy",
            "processing_time": 0.1,
            "confidence_score": 0.9,
            "word_count": 100,
            "created_at": datetime.now(),
            "metadata": {}
        }
        
        with patch.object(engine.cache, 'get', return_value=cached_response):
            response = await engine.generate_document(sample_request)
            
            assert response.content == "Cached content"
            assert engine.stats["cache_hits"] > 0
    
    @pytest.mark.asyncio
    async def test_generate_document_cache_miss(self, engine, sample_request):
        """Test document generation with cache miss"""
        # Mock cache miss
        with patch.object(engine.cache, 'get', return_value=None):
            with patch.object(engine.cache, 'set'):
                with patch.object(engine, '_call_openrouter_api', return_value="Generated content"):
                    with patch.object(engine, '_analyze_query', return_value={
                        "business_area": "marketing",
                        "document_type": "marketing_strategy",
                        "confidence": 0.9
                    }):
                        response = await engine.generate_document(sample_request)
                        
                        assert response.content == "Generated content"
                        assert engine.stats["cache_misses"] > 0
    
    @pytest.mark.asyncio
    async def test_performance_metrics_update(self, engine):
        """Test performance metrics update"""
        initial_avg = engine.performance_metrics["avg_response_time"]
        
        # Update with a processing time
        engine._update_performance_metrics(5.0)
        
        assert len(engine.performance_metrics["response_times"]) == 1
        assert engine.performance_metrics["avg_response_time"] == 5.0
        assert engine.performance_metrics["max_response_time"] == 5.0
        assert engine.performance_metrics["min_response_time"] == 5.0
        
        # Update with another time
        engine._update_performance_metrics(3.0)
        
        assert len(engine.performance_metrics["response_times"]) == 2
        assert engine.performance_metrics["avg_response_time"] == 4.0
        assert engine.performance_metrics["max_response_time"] == 5.0
        assert engine.performance_metrics["min_response_time"] == 3.0
    
    @pytest.mark.asyncio
    async def test_engine_close(self, engine):
        """Test engine cleanup"""
        await engine.close()
        
        # Verify cleanup (in real implementation, check if connections are closed)
        assert engine.session is None or engine.session.closed
        assert engine.http_client is None or engine.http_client.is_closed
    
    @pytest.mark.asyncio
    async def test_error_handling(self, engine, sample_request):
        """Test error handling in document generation"""
        with patch.object(engine, '_analyze_query', side_effect=Exception("Analysis error")):
            with pytest.raises(Exception, match="Analysis error"):
                await engine.generate_document(sample_request)
            
            assert engine.stats["error_count"] > 0
            assert engine.stats["last_error"] == "Analysis error"


class TestDocumentRequest:
    """Test cases for DocumentRequest"""
    
    def test_document_request_creation(self):
        """Test DocumentRequest creation"""
        request = DocumentRequest(
            query="Test query",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.MARKETING_STRATEGY,
            company_name="Test Company",
            language="en"
        )
        
        assert request.query == "Test query"
        assert request.business_area == BusinessArea.MARKETING
        assert request.document_type == DocumentType.MARKETING_STRATEGY
        assert request.company_name == "Test Company"
        assert request.language == "en"
        assert request.id is not None
        assert isinstance(request.created_at, datetime)
    
    def test_document_request_defaults(self):
        """Test DocumentRequest default values"""
        request = DocumentRequest()
        
        assert request.query == ""
        assert request.business_area == BusinessArea.STRATEGY
        assert request.document_type == DocumentType.BUSINESS_PLAN
        assert request.language == "es"
        assert request.format == "markdown"
        assert request.id is not None
        assert isinstance(request.created_at, datetime)


class TestDocumentResponse:
    """Test cases for DocumentResponse"""
    
    def test_document_response_creation(self):
        """Test DocumentResponse creation"""
        response = DocumentResponse(
            id="test_id",
            content="Test content",
            title="Test Title",
            business_area=BusinessArea.MARKETING,
            document_type=DocumentType.MARKETING_STRATEGY,
            word_count=100,
            processing_time=5.0,
            confidence_score=0.9
        )
        
        assert response.id == "test_id"
        assert response.content == "Test content"
        assert response.title == "Test Title"
        assert response.business_area == BusinessArea.MARKETING
        assert response.document_type == DocumentType.MARKETING_STRATEGY
        assert response.word_count == 100
        assert response.processing_time == 5.0
        assert response.confidence_score == 0.9
        assert isinstance(response.created_at, datetime)
    
    def test_document_response_defaults(self):
        """Test DocumentResponse default values"""
        response = DocumentResponse()
        
        assert response.id == ""
        assert response.content == ""
        assert response.title == ""
        assert response.business_area == BusinessArea.STRATEGY
        assert response.document_type == DocumentType.BUSINESS_PLAN
        assert response.word_count == 0
        assert response.processing_time == 0.0
        assert response.confidence_score == 0.0
        assert isinstance(response.created_at, datetime)


class TestBusinessArea:
    """Test cases for BusinessArea enum"""
    
    def test_business_area_values(self):
        """Test BusinessArea enum values"""
        assert BusinessArea.MARKETING.value == "marketing"
        assert BusinessArea.SALES.value == "sales"
        assert BusinessArea.OPERATIONS.value == "operations"
        assert BusinessArea.HR.value == "hr"
        assert BusinessArea.FINANCE.value == "finance"
        assert BusinessArea.LEGAL.value == "legal"
        assert BusinessArea.TECHNICAL.value == "technical"
        assert BusinessArea.CONTENT.value == "content"
        assert BusinessArea.STRATEGY.value == "strategy"
        assert BusinessArea.CUSTOMER_SERVICE.value == "customer_service"


class TestDocumentType:
    """Test cases for DocumentType enum"""
    
    def test_document_type_values(self):
        """Test DocumentType enum values"""
        assert DocumentType.BUSINESS_PLAN.value == "business_plan"
        assert DocumentType.MARKETING_STRATEGY.value == "marketing_strategy"
        assert DocumentType.SALES_PROPOSAL.value == "sales_proposal"
        assert DocumentType.OPERATIONAL_MANUAL.value == "operational_manual"
        assert DocumentType.HR_POLICY.value == "hr_policy"
        assert DocumentType.FINANCIAL_REPORT.value == "financial_report"
        assert DocumentType.LEGAL_CONTRACT.value == "legal_contract"
        assert DocumentType.TECHNICAL_SPECIFICATION.value == "technical_specification"
        assert DocumentType.CONTENT_STRATEGY.value == "content_strategy"
        assert DocumentType.STRATEGIC_PLAN.value == "strategic_plan"
        assert DocumentType.CUSTOMER_SERVICE_GUIDE.value == "customer_service_guide"


# Integration tests
class TestBULEngineIntegration:
    """Integration tests for BULEngine"""
    
    @pytest.mark.asyncio
    async def test_full_document_generation_flow(self):
        """Test complete document generation flow"""
        with patch.dict('os.environ', {
            'OPENROUTER_API_KEY': 'test_key',
            'OPENAI_API_KEY': 'test_openai_key'
        }):
            engine = BULEngine('test_key', 'test_openai_key')
            await engine.initialize()
            
            request = DocumentRequest(
                query="Create a comprehensive business plan for a new restaurant",
                business_area=BusinessArea.STRATEGY,
                document_type=DocumentType.BUSINESS_PLAN,
                company_name="Delicious Bites",
                industry="Food & Beverage",
                company_size="Small",
                language="en"
            )
            
            # Mock the entire flow
            with patch.object(engine, '_call_openrouter_api', return_value="Complete business plan content"):
                response = await engine.generate_document(request)
                
                assert isinstance(response, DocumentResponse)
                assert response.content == "Complete business plan content"
                assert response.business_area == BusinessArea.STRATEGY
                assert response.document_type == DocumentType.BUSINESS_PLAN
                assert response.processing_time > 0
                assert response.confidence_score > 0
                assert response.word_count > 0
                
                # Verify stats were updated
                stats = await engine.get_stats()
                assert stats["documents_generated"] > 0
                assert stats["total_processing_time"] > 0
                
            await engine.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
















