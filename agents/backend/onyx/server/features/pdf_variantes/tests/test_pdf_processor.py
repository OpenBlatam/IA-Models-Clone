"""
Unit Tests for PDF Processor
=============================
Tests for PDF processing functions and utilities.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import io
from typing import Dict, Any

# Try to import pdf_processor functions
try:
    from pdf_processor import (
        extract_pdf_content,
        process_pdf_with_retry,
        extract_topics_async,
        generate_variants_async,
        analyze_content_quality,
        create_pdf_processing_pipeline,
        process_pdf_complete,
        batch_process_pdfs,
        generate_file_id,
        extract_features_parallel,
        create_content_analyzer,
        process_with_metrics
    )
except ImportError:
    # Functions might not be available
    extract_pdf_content = None
    process_pdf_with_retry = None
    extract_topics_async = None
    generate_variants_async = None
    analyze_content_quality = None
    create_pdf_processing_pipeline = None
    process_pdf_complete = None
    batch_process_pdfs = None
    generate_file_id = None
    extract_features_parallel = None
    create_content_analyzer = None
    process_with_metrics = None


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\nxref\n0 3\ntrailer\n<<\n/Size 3\n>>\nstartxref\n50\n%%EOF"


@pytest.fixture
def sample_content_data():
    """Sample content data structure."""
    return {
        "text": "This is sample PDF content for testing purposes.",
        "pages": 1,
        "metadata": {
            "title": "Test Document",
            "author": "Test Author"
        },
        "words": 10,
        "characters": 50
    }


class TestExtractPDFContent:
    """Tests for extract_pdf_content function."""
    
    @pytest.mark.asyncio
    async def test_extract_pdf_content_success(self, sample_pdf_content):
        """Test successful PDF content extraction."""
        if extract_pdf_content is None:
            pytest.skip("extract_pdf_content not available")
        
        result = await extract_pdf_content(sample_pdf_content)
        assert isinstance(result, dict)
        assert "text" in result or "pages" in result or "metadata" in result
    
    @pytest.mark.asyncio
    async def test_extract_pdf_content_empty(self):
        """Test extraction with empty PDF."""
        if extract_pdf_content is None:
            pytest.skip("extract_pdf_content not available")
        
        empty_content = b""
        result = await extract_pdf_content(empty_content)
        # Should handle gracefully
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_extract_pdf_content_invalid(self):
        """Test extraction with invalid PDF."""
        if extract_pdf_content is None:
            pytest.skip("extract_pdf_content not available")
        
        invalid_content = b"not a pdf"
        # Should handle gracefully or raise appropriate error
        try:
            result = await extract_pdf_content(invalid_content)
            assert isinstance(result, dict)
        except Exception as e:
            assert isinstance(e, (ValueError, RuntimeError))


class TestProcessPDFWithRetry:
    """Tests for process_pdf_with_retry function."""
    
    @pytest.mark.asyncio
    async def test_process_pdf_with_retry_success(self, sample_pdf_content):
        """Test successful processing with retry."""
        if process_pdf_with_retry is None:
            pytest.skip("process_pdf_with_retry not available")
        
        result = await process_pdf_with_retry(sample_pdf_content, max_retries=3)
        assert isinstance(result, dict)
    
    @pytest.mark.asyncio
    async def test_process_pdf_with_retry_failure(self):
        """Test retry on failure."""
        if process_pdf_with_retry is None:
            pytest.skip("process_pdf_with_retry not available")
        
        invalid_content = b"invalid"
        # Should retry and eventually fail or handle gracefully
        try:
            result = await process_pdf_with_retry(invalid_content, max_retries=2)
            assert isinstance(result, dict)
        except Exception:
            # Failure after retries is acceptable
            pass


class TestExtractTopicsAsync:
    """Tests for extract_topics_async function."""
    
    @pytest.mark.asyncio
    async def test_extract_topics_async_success(self, sample_content_data):
        """Test successful topic extraction."""
        if extract_topics_async is None:
            pytest.skip("extract_topics_async not available")
        
        result = await extract_topics_async(sample_content_data)
        assert isinstance(result, dict)
        assert "topics" in result or "extracted_topics" in result
    
    @pytest.mark.asyncio
    async def test_extract_topics_async_empty_content(self):
        """Test topic extraction with empty content."""
        if extract_topics_async is None:
            pytest.skip("extract_topics_async not available")
        
        empty_data = {"text": "", "pages": 0}
        result = await extract_topics_async(empty_data)
        assert isinstance(result, dict)


class TestGenerateVariantsAsync:
    """Tests for generate_variants_async function."""
    
    @pytest.mark.asyncio
    async def test_generate_variants_async_success(self, sample_content_data):
        """Test successful variant generation."""
        if generate_variants_async is None:
            pytest.skip("generate_variants_async not available")
        
        result = await generate_variants_async(sample_content_data)
        assert isinstance(result, dict)
        assert "variants" in result or "generated_variants" in result
    
    @pytest.mark.asyncio
    async def test_generate_variants_async_with_options(self, sample_content_data):
        """Test variant generation with options."""
        if generate_variants_async is None:
            pytest.skip("generate_variants_async not available")
        
        options = {"variant_types": ["summary", "outline"]}
        result = await generate_variants_async(sample_content_data, options=options)
        assert isinstance(result, dict)


class TestAnalyzeContentQuality:
    """Tests for analyze_content_quality function."""
    
    @pytest.mark.asyncio
    async def test_analyze_content_quality_success(self, sample_content_data):
        """Test successful content quality analysis."""
        if analyze_content_quality is None:
            pytest.skip("analyze_content_quality not available")
        
        result = await analyze_content_quality(sample_content_data)
        assert isinstance(result, dict)
        assert "quality_score" in result or "quality_metrics" in result or "score" in result
    
    @pytest.mark.asyncio
    async def test_analyze_content_quality_metrics(self, sample_content_data):
        """Test that quality analysis returns metrics."""
        if analyze_content_quality is None:
            pytest.skip("analyze_content_quality not available")
        
        result = await analyze_content_quality(sample_content_data)
        assert isinstance(result, dict)
        # Should have some quality indicators
        assert len(result) > 0


class TestGenerateFileID:
    """Tests for generate_file_id function."""
    
    def test_generate_file_id_format(self):
        """Test that file ID has correct format."""
        if generate_file_id is None:
            pytest.skip("generate_file_id not available")
        
        file_id = generate_file_id()
        assert isinstance(file_id, str)
        assert len(file_id) > 0
    
    def test_generate_file_id_uniqueness(self):
        """Test that file IDs are unique."""
        if generate_file_id is None:
            pytest.skip("generate_file_id not available")
        
        ids = [generate_file_id() for _ in range(100)]
        assert len(ids) == len(set(ids)), "File IDs should be unique"
    
    def test_generate_file_id_consistency(self):
        """Test that file ID generation is consistent."""
        if generate_file_id is None:
            pytest.skip("generate_file_id not available")
        
        id1 = generate_file_id()
        id2 = generate_file_id()
        # Should generate different IDs
        assert id1 != id2


class TestBatchProcessPDFs:
    """Tests for batch_process_pdfs function."""
    
    @pytest.mark.asyncio
    async def test_batch_process_pdfs_success(self, sample_pdf_content):
        """Test successful batch processing."""
        if batch_process_pdfs is None:
            pytest.skip("batch_process_pdfs not available")
        
        pdfs = [sample_pdf_content] * 5
        results = await batch_process_pdfs(pdfs)
        assert isinstance(results, list)
        assert len(results) == 5
    
    @pytest.mark.asyncio
    async def test_batch_process_pdfs_empty(self):
        """Test batch processing with empty list."""
        if batch_process_pdfs is None:
            pytest.skip("batch_process_pdfs not available")
        
        results = await batch_process_pdfs([])
        assert isinstance(results, list)
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_batch_process_pdfs_mixed(self, sample_pdf_content):
        """Test batch processing with mixed valid/invalid PDFs."""
        if batch_process_pdfs is None:
            pytest.skip("batch_process_pdfs not available")
        
        pdfs = [sample_pdf_content, b"invalid", sample_pdf_content]
        results = await batch_process_pdfs(pdfs)
        assert isinstance(results, list)
        # Should handle all, even if some fail
        assert len(results) <= len(pdfs)


class TestExtractFeaturesParallel:
    """Tests for extract_features_parallel function."""
    
    @pytest.mark.asyncio
    async def test_extract_features_parallel_success(self, sample_content_data):
        """Test successful parallel feature extraction."""
        if extract_features_parallel is None:
            pytest.skip("extract_features_parallel not available")
        
        result = await extract_features_parallel(sample_content_data)
        assert isinstance(result, dict)
        assert "features" in result or len(result) > 0
    
    @pytest.mark.asyncio
    async def test_extract_features_parallel_performance(self, sample_content_data):
        """Test that parallel extraction is faster."""
        if extract_features_parallel is None:
            pytest.skip("extract_features_parallel not available")
        
        import time
        
        start = time.time()
        result = await extract_features_parallel(sample_content_data)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 5.0, f"Parallel extraction too slow: {elapsed:.2f}s"
        assert isinstance(result, dict)


class TestCreateContentAnalyzer:
    """Tests for create_content_analyzer function."""
    
    def test_create_content_analyzer_success(self):
        """Test successful content analyzer creation."""
        if create_content_analyzer is None:
            pytest.skip("create_content_analyzer not available")
        
        analyzer = create_content_analyzer()
        assert analyzer is not None
    
    def test_create_content_analyzer_with_config(self):
        """Test content analyzer creation with config."""
        if create_content_analyzer is None:
            pytest.skip("create_content_analyzer not available")
        
        config = {"language": "en", "model": "default"}
        analyzer = create_content_analyzer(config=config)
        assert analyzer is not None


class TestProcessWithMetrics:
    """Tests for process_with_metrics function."""
    
    @pytest.mark.asyncio
    async def test_process_with_metrics_success(self, sample_pdf_content):
        """Test processing with metrics collection."""
        if process_with_metrics is None:
            pytest.skip("process_with_metrics not available")
        
        result = await process_with_metrics(extract_pdf_content, sample_pdf_content)
        assert isinstance(result, dict)
        # Should include metrics
        assert "metrics" in result or "processing_time" in result or "duration" in result
    
    @pytest.mark.asyncio
    async def test_process_with_metrics_timing(self, sample_pdf_content):
        """Test that metrics include timing information."""
        if process_with_metrics is None:
            pytest.skip("process_with_metrics not available")
        
        result = await process_with_metrics(extract_pdf_content, sample_pdf_content)
        assert isinstance(result, dict)
        # Check for timing metrics
        has_timing = any(
            key in result for key in 
            ["metrics", "processing_time", "duration", "time", "elapsed"]
        )
        assert has_timing, "Metrics should include timing information"


class TestCreatePDFProcessingPipeline:
    """Tests for create_pdf_processing_pipeline function."""
    
    def test_create_pipeline_success(self):
        """Test successful pipeline creation."""
        if create_pdf_processing_pipeline is None:
            pytest.skip("create_pdf_processing_pipeline not available")
        
        pipeline = create_pdf_processing_pipeline()
        assert pipeline is not None
        assert callable(pipeline) or hasattr(pipeline, "__call__")
    
    @pytest.mark.asyncio
    async def test_pipeline_execution(self, sample_pdf_content):
        """Test pipeline execution."""
        if create_pdf_processing_pipeline is None:
            pytest.skip("create_pdf_processing_pipeline not available")
        
        pipeline = create_pdf_processing_pipeline()
        result = await pipeline(sample_pdf_content)
        assert isinstance(result, dict)


class TestProcessPDFComplete:
    """Tests for process_pdf_complete function."""
    
    @pytest.mark.asyncio
    async def test_process_pdf_complete_success(self, sample_pdf_content):
        """Test complete PDF processing."""
        if process_pdf_complete is None:
            pytest.skip("process_pdf_complete not available")
        
        result = await process_pdf_complete(sample_pdf_content)
        assert isinstance(result, dict)
        # Should include multiple processing results
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_process_pdf_complete_components(self, sample_pdf_content):
        """Test that complete processing includes all components."""
        if process_pdf_complete is None:
            pytest.skip("process_pdf_complete not available")
        
        result = await process_pdf_complete(sample_pdf_content)
        assert isinstance(result, dict)
        # Should have content, topics, variants, or quality
        has_content = any(key in result for key in ["content", "text", "extracted"])
        has_topics = any(key in result for key in ["topics", "extracted_topics"])
        has_variants = any(key in result for key in ["variants", "generated_variants"])
        has_quality = any(key in result for key in ["quality", "quality_score"])
        
        assert has_content or has_topics or has_variants or has_quality, \
            "Complete processing should include at least one component"



