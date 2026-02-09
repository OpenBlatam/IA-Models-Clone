"""
Tests for Perplexity Query Processing System
============================================

Unit tests for Perplexity components.
"""

import pytest
from datetime import datetime
from core.perplexity import (
    PerplexityProcessor,
    PerplexityService,
    QueryType,
    QueryTypeDetector,
    ResponseFormatter,
    CitationManager,
    PerplexityValidator,
    PerplexityCache,
    PerplexityMetrics
)


class TestQueryTypeDetector:
    """Tests for query type detection."""
    
    def test_weather_detection(self):
        detector = QueryTypeDetector()
        assert detector.detect("What is the weather today?") == QueryType.WEATHER
    
    def test_news_detection(self):
        detector = QueryTypeDetector()
        assert detector.detect("What are the latest news?") == QueryType.RECENT_NEWS
    
    def test_coding_detection(self):
        detector = QueryTypeDetector()
        assert detector.detect("How to code in Python?") == QueryType.CODING
    
    def test_url_lookup_detection(self):
        detector = QueryTypeDetector()
        assert detector.detect("https://example.com") == QueryType.URL_LOOKUP
    
    def test_general_fallback(self):
        detector = QueryTypeDetector()
        assert detector.detect("Random question") == QueryType.GENERAL


class TestCitationManager:
    """Tests for citation management."""
    
    def test_add_citations(self):
        manager = CitationManager()
        from core.perplexity.types import SearchResult
        
        search_results = [
            SearchResult(
                index=1,
                title="Python Guide",
                url="https://python.org",
                snippet="Python is a programming language"
            )
        ]
        
        text = "Python is a programming language."
        result = manager.add_citations(text, search_results)
        assert "[1]" in result
    
    def test_citation_normalization_url_lookup(self):
        manager = CitationManager()
        text = "Content[1][2][3]."
        result = manager.normalize_citations(text, "url_lookup")
        assert "[1]" in result
        assert "[2]" not in result
        assert "[3]" not in result
    
    def test_citation_removal_translation(self):
        manager = CitationManager()
        text = "Translated text[1][2]."
        result = manager.normalize_citations(text, "translation")
        assert "[1]" not in result
        assert "[2]" not in result


class TestResponseFormatter:
    """Tests for response formatting."""
    
    def test_latex_normalization(self):
        formatter = ResponseFormatter()
        text = "Formula: $E=mc^2$"
        result = formatter._normalize_latex(text)
        assert "$" not in result
        assert "\\(" in result or "\\[" in result
    
    def test_no_leading_header(self):
        formatter = ResponseFormatter()
        text = "## Header\n\nContent"
        result = formatter._ensure_no_leading_header(text)
        assert not result.startswith("##")
    
    def test_no_ending_question(self):
        formatter = ResponseFormatter()
        text = "This is a question?"
        result = formatter._ensure_no_ending_question(text)
        assert not result.endswith("?")


class TestPerplexityValidator:
    """Tests for response validation."""
    
    def test_validate_no_leading_header(self):
        validator = PerplexityValidator()
        answer = "## Header\n\nContent"
        is_valid, issues = validator.validate(answer)
        assert not is_valid
        assert any(issue.rule == "no_leading_header" for issue in issues)
    
    def test_validate_citation_format(self):
        validator = PerplexityValidator()
        answer = "Text [1] [2]."  # Space before citation
        is_valid, issues = validator.validate(answer)
        assert not is_valid
        assert any(issue.rule == "citation_no_space" for issue in issues)
    
    def test_validate_latex_format(self):
        validator = PerplexityValidator()
        answer = "Formula: $x^2$"
        is_valid, issues = validator.validate(answer)
        assert not is_valid
        assert any(issue.rule == "latex_no_dollar" for issue in issues)


class TestPerplexityCache:
    """Tests for caching."""
    
    def test_cache_set_get(self):
        cache = PerplexityCache(ttl_seconds=3600)
        from core.perplexity.types import ProcessedQuery, QueryType, SearchResult
        
        processed = ProcessedQuery(
            original_query="test",
            query_type=QueryType.GENERAL,
            search_results=[]
        )
        
        cache.set("test", [], processed, "answer")
        result = cache.get("test", [])
        assert result == "answer"
    
    def test_cache_expiration(self):
        cache = PerplexityCache(ttl_seconds=0)  # Immediate expiration
        from core.perplexity.types import ProcessedQuery, QueryType
        
        processed = ProcessedQuery(
            original_query="test",
            query_type=QueryType.GENERAL,
            search_results=[]
        )
        
        cache.set("test", [], processed, "answer")
        result = cache.get("test", [])
        assert result is None  # Expired


class TestPerplexityProcessor:
    """Tests for main processor."""
    
    @pytest.mark.asyncio
    async def test_process_query(self):
        processor = PerplexityProcessor(enable_cache=False, enable_metrics=False)
        processed = processor.process_query("What is Python?", [])
        assert processed.query_type == QueryType.GENERAL
        assert processed.original_query == "What is Python?"
    
    @pytest.mark.asyncio
    async def test_generate_answer_no_llm(self):
        processor = PerplexityProcessor(enable_cache=False, enable_metrics=False)
        processed = processor.process_query("test", [])
        answer = await processor.generate_answer(processed, None)
        assert isinstance(answer, str)
        assert len(answer) > 0


class TestPerplexityService:
    """Tests for service layer."""
    
    @pytest.mark.asyncio
    async def test_answer_query(self):
        service = PerplexityService(
            enable_cache=False,
            enable_metrics=False,
            enable_validation=False
        )
        result = await service.answer_query("test query", [])
        assert 'query' in result
        assert 'answer' in result
        assert 'query_type' in result
    
    def test_get_cache_stats(self):
        service = PerplexityService(enable_cache=True)
        stats = service.get_cache_stats()
        assert 'enabled' in stats
        assert stats['enabled'] is True
    
    def test_get_metrics(self):
        service = PerplexityService(enable_metrics=True)
        metrics = service.get_metrics()
        assert 'total_queries' in metrics
        assert metrics['total_queries'] == 0  # Initially empty




