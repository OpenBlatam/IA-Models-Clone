"""
Performance Tests for PDF Variantes
====================================
Tests to ensure performance requirements are met.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile
import io

# Performance thresholds (in seconds)
PERFORMANCE_THRESHOLDS = {
    "filename_validation": 0.001,  # 1ms
    "file_extension_validation": 0.001,
    "integer_range_validation": 0.001,
    "email_validation": 0.001,
    "uuid_validation": 0.001,
    "pdf_validation": 0.1,  # 100ms
    "metadata_extraction": 0.5,  # 500ms
    "variant_generation": 5.0,  # 5 seconds
}


class TestValidationPerformance:
    """Performance tests for validation functions."""
    
    def test_filename_validation_performance(self):
        """Test that filename validation is fast."""
        try:
            from utils.validation import validate_filename
        except ImportError:
            pytest.skip("validate_filename not available")
        
        start = time.time()
        for _ in range(1000):
            validate_filename("test.pdf")
        elapsed = time.time() - start
        
        avg_time = elapsed / 1000
        assert avg_time < PERFORMANCE_THRESHOLDS["filename_validation"], \
            f"Filename validation too slow: {avg_time:.4f}s (threshold: {PERFORMANCE_THRESHOLDS['filename_validation']}s)"
    
    def test_file_extension_validation_performance(self):
        """Test that file extension validation is fast."""
        try:
            from utils.validation import validate_file_extension
        except ImportError:
            pytest.skip("validate_file_extension not available")
        
        start = time.time()
        for _ in range(1000):
            validate_file_extension("test.pdf", [".pdf", ".txt"])
        elapsed = time.time() - start
        
        avg_time = elapsed / 1000
        assert avg_time < PERFORMANCE_THRESHOLDS["file_extension_validation"], \
            f"File extension validation too slow: {avg_time:.4f}s"
    
    def test_integer_range_validation_performance(self):
        """Test that integer range validation is fast."""
        try:
            from utils.validation import validate_integer_range
        except ImportError:
            pytest.skip("validate_integer_range not available")
        
        start = time.time()
        for _ in range(1000):
            validate_integer_range(5, min_value=1, max_value=10)
        elapsed = time.time() - start
        
        avg_time = elapsed / 1000
        assert avg_time < PERFORMANCE_THRESHOLDS["integer_range_validation"], \
            f"Integer range validation too slow: {avg_time:.4f}s"
    
    def test_email_validation_performance(self):
        """Test that email validation is fast."""
        try:
            from utils.validation import validate_email
        except ImportError:
            pytest.skip("validate_email not available")
        
        start = time.time()
        for _ in range(1000):
            validate_email("test@example.com")
        elapsed = time.time() - start
        
        avg_time = elapsed / 1000
        assert avg_time < PERFORMANCE_THRESHOLDS["email_validation"], \
            f"Email validation too slow: {avg_time:.4f}s"
    
    def test_uuid_validation_performance(self):
        """Test that UUID validation is fast."""
        try:
            from utils.validation import validate_uuid
        except ImportError:
            pytest.skip("validate_uuid not available")
        
        uuid = "123e4567-e89b-12d3-a456-426614174000"
        start = time.time()
        for _ in range(1000):
            validate_uuid(uuid)
        elapsed = time.time() - start
        
        avg_time = elapsed / 1000
        assert avg_time < PERFORMANCE_THRESHOLDS["uuid_validation"], \
            f"UUID validation too slow: {avg_time:.4f}s"


class TestPDFProcessingPerformance:
    """Performance tests for PDF processing."""
    
    @pytest.mark.asyncio
    async def test_pdf_validation_performance(self):
        """Test that PDF validation meets performance requirements."""
        try:
            from utils.file_helpers import validate_pdf_file
        except ImportError:
            pytest.skip("validate_pdf_file not available")
        
        # Minimal valid PDF
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\ntrailer\n<<\n/Size 1\n>>\nstartxref\n10\n%%EOF"
        
        start = time.time()
        result = validate_pdf_file(pdf_content, "test.pdf")
        elapsed = time.time() - start
        
        assert elapsed < PERFORMANCE_THRESHOLDS["pdf_validation"], \
            f"PDF validation too slow: {elapsed:.4f}s (threshold: {PERFORMANCE_THRESHOLDS['pdf_validation']}s)"
        assert "valid" in result
    
    @pytest.mark.asyncio
    async def test_metadata_extraction_performance(self):
        """Test that metadata extraction meets performance requirements."""
        try:
            from utils.file_helpers import extract_metadata
        except ImportError:
            pytest.skip("extract_metadata not available")
        
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\nxref\n0 3\ntrailer\n<<\n/Size 3\n>>\nstartxref\n50\n%%EOF"
        
        start = time.time()
        metadata = extract_metadata(pdf_content)
        elapsed = time.time() - start
        
        assert elapsed < PERFORMANCE_THRESHOLDS["metadata_extraction"], \
            f"Metadata extraction too slow: {elapsed:.4f}s (threshold: {PERFORMANCE_THRESHOLDS['metadata_extraction']}s)"
        assert isinstance(metadata, dict)


class TestConcurrentOperations:
    """Tests for concurrent operation performance."""
    
    @pytest.mark.asyncio
    async def test_concurrent_filename_validation(self):
        """Test concurrent filename validation."""
        try:
            from utils.validation import validate_filename
        except ImportError:
            pytest.skip("validate_filename not available")
        
        filenames = [f"test_{i}.pdf" for i in range(100)]
        
        start = time.time()
        results = await asyncio.gather(*[
            asyncio.to_thread(validate_filename, filename)
            for filename in filenames
        ])
        elapsed = time.time() - start
        
        # All should be valid
        assert all(valid for valid, _ in results)
        # Should complete in reasonable time
        assert elapsed < 1.0, f"Concurrent validation too slow: {elapsed:.4f}s"
    
    @pytest.mark.asyncio
    async def test_concurrent_file_extension_validation(self):
        """Test concurrent file extension validation."""
        try:
            from utils.validation import validate_file_extension
        except ImportError:
            pytest.skip("validate_file_extension not available")
        
        test_cases = [
            ("test.pdf", [".pdf", ".txt"]),
            ("document.txt", [".pdf", ".txt"]),
            ("file.doc", [".pdf", ".txt"]),
        ] * 50  # 150 total operations
        
        start = time.time()
        results = await asyncio.gather(*[
            asyncio.to_thread(validate_file_extension, filename, extensions)
            for filename, extensions in test_cases
        ])
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 1.0, f"Concurrent extension validation too slow: {elapsed:.4f}s"


class TestMemoryUsage:
    """Tests for memory usage."""
    
    def test_filename_validation_memory(self):
        """Test that filename validation doesn't leak memory."""
        try:
            from utils.validation import validate_filename
        except ImportError:
            pytest.skip("validate_filename not available")
        
        import gc
        import sys
        
        # Clear any existing references
        gc.collect()
        initial_count = len(gc.get_objects())
        
        # Run many validations
        for _ in range(10000):
            validate_filename("test.pdf")
        
        # Force garbage collection
        gc.collect()
        final_count = len(gc.get_objects())
        
        # Memory growth should be minimal (allow some overhead)
        growth = final_count - initial_count
        assert growth < 1000, f"Potential memory leak: {growth} objects created"


class TestScalability:
    """Tests for scalability."""
    
    def test_validation_scales_linearly(self):
        """Test that validation scales linearly with input size."""
        try:
            from utils.validation import validate_filename
        except ImportError:
            pytest.skip("validate_filename not available")
        
        sizes = [10, 50, 100, 500, 1000]
        times = []
        
        for size in sizes:
            start = time.time()
            for _ in range(size):
                validate_filename("test.pdf")
            elapsed = time.time() - start
            times.append(elapsed / size)  # Average time per operation
        
        # Times should be relatively consistent (within 2x)
        max_time = max(times)
        min_time = min(times)
        ratio = max_time / min_time if min_time > 0 else 1
        
        assert ratio < 2.0, \
            f"Validation doesn't scale linearly: ratio {ratio:.2f} (times: {times})"



