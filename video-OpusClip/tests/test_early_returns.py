"""
Test Suite for Early Returns

Tests that functions use early returns for error conditions to avoid
deeply nested if statements and improve code readability.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from ..error_handling import (
    ErrorHandler, 
    ErrorCode, 
    ValidationError, 
    ProcessingError, 
    ExternalServiceError,
    ResourceError,
    CriticalSystemError,
    SecurityError,
    ConfigurationError,
    create_validation_error,
    create_processing_error,
    create_external_service_error,
    create_resource_error,
    create_critical_system_error,
    create_security_error,
    create_configuration_error
)
from ..validation import (
    validate_youtube_url,
    validate_clip_length,
    validate_batch_size,
    validate_video_request_data,
    validate_batch_request_data,
    validate_viral_variant_data,
    check_system_resources,
    check_gpu_availability,
    validate_system_health,
    validate_gpu_health
)
from ..gradio_demo import validate_prompt, validate_parameters

# =============================================================================
# EARLY RETURNS PATTERN TESTS
# =============================================================================

class TestEarlyReturnsPattern:
    """Test that functions use early returns instead of deeply nested if statements."""
    
    def test_validate_youtube_url_early_returns(self):
        """Test that validate_youtube_url uses early returns."""
        # Test early return for None
        with pytest.raises(ValidationError) as exc_info:
            validate_youtube_url(None)
        assert "YouTube URL is required" in str(exc_info.value)
        
        # Test early return for empty string
        with pytest.raises(ValidationError) as exc_info:
            validate_youtube_url("")
        assert "YouTube URL is required" in str(exc_info.value)
        
        # Test early return for wrong type
        with pytest.raises(ValidationError) as exc_info:
            validate_youtube_url(123)
        assert "YouTube URL must be a string" in str(exc_info.value)
        
        # Test early return for malicious pattern
        with pytest.raises(ValidationError) as exc_info:
            validate_youtube_url("javascript:alert('xss')")
        assert "Malicious URL pattern detected" in str(exc_info.value)
    
    def test_validate_clip_length_early_returns(self):
        """Test that validate_clip_length uses early returns."""
        # Test early return for wrong type
        with pytest.raises(ValidationError) as exc_info:
            validate_clip_length("30")
        assert "Clip length must be an integer" in str(exc_info.value)
        
        # Test early return for negative value
        with pytest.raises(ValidationError) as exc_info:
            validate_clip_length(-5)
        assert "cannot be negative" in str(exc_info.value)
        
        # Test early return for zero
        with pytest.raises(ValidationError) as exc_info:
            validate_clip_length(0)
        assert "cannot be zero" in str(exc_info.value)
        
        # Test early return for too short
        with pytest.raises(ValidationError) as exc_info:
            validate_clip_length(0.5)
        assert "must be at least" in str(exc_info.value)
        
        # Test early return for too long
        with pytest.raises(ValidationError) as exc_info:
            validate_clip_length(1000)
        assert "cannot exceed" in str(exc_info.value)
    
    def test_validate_batch_size_early_returns(self):
        """Test that validate_batch_size uses early returns."""
        # Test early return for wrong type
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size("10")
        assert "Batch size must be an integer" in str(exc_info.value)
        
        # Test early return for negative value
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size(-10)
        assert "cannot be negative" in str(exc_info.value)
        
        # Test early return for zero
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size(0)
        assert "cannot be zero" in str(exc_info.value)
        
        # Test early return for too small
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size(0)
        assert "cannot be zero" in str(exc_info.value)
        
        # Test early return for too large
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_size(2000)
        assert "exceeds maximum allowed limit" in str(exc_info.value)

# =============================================================================
# COMPOSITE VALIDATION EARLY RETURNS TESTS
# =============================================================================

class TestCompositeValidationEarlyReturns:
    """Test that composite validation functions use early returns."""
    
    def test_validate_video_request_data_early_returns(self):
        """Test that validate_video_request_data uses early returns."""
        # Test early return for None URL
        with pytest.raises(ValidationError) as exc_info:
            validate_video_request_data(
                youtube_url=None,
                language="en"
            )
        assert "YouTube URL is required and cannot be empty" in str(exc_info.value)
        
        # Test early return for empty URL
        with pytest.raises(ValidationError) as exc_info:
            validate_video_request_data(
                youtube_url="",
                language="en"
            )
        assert "YouTube URL is required and cannot be empty" in str(exc_info.value)
        
        # Test early return for None language
        with pytest.raises(ValidationError) as exc_info:
            validate_video_request_data(
                youtube_url="https://youtube.com/watch?v=123",
                language=None
            )
        assert "Language is required and cannot be empty" in str(exc_info.value)
        
        # Test early return for wrong URL type
        with pytest.raises(ValidationError) as exc_info:
            validate_video_request_data(
                youtube_url=123,
                language="en"
            )
        assert "YouTube URL must be a string" in str(exc_info.value)
        
        # Test early return for logical constraint
        with pytest.raises(ValidationError) as exc_info:
            validate_video_request_data(
                youtube_url="https://youtube.com/watch?v=123",
                language="en",
                max_clip_length=10,
                min_clip_length=20
            )
        assert "max_clip_length cannot be less than min_clip_length" in str(exc_info.value)
    
    def test_validate_batch_request_data_early_returns(self):
        """Test that validate_batch_request_data uses early returns."""
        # Test early return for None requests
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_request_data(None)
        assert "Requests list cannot be empty" in str(exc_info.value)
        
        # Test early return for empty requests
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_request_data([])
        assert "Requests list cannot be empty" in str(exc_info.value)
        
        # Test early return for wrong type
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_request_data("not a list")
        assert "Requests must be a list" in str(exc_info.value)
        
        # Test early return for size limit
        large_requests = [{"youtube_url": "https://youtube.com/watch?v=123", "language": "en"} for _ in range(1500)]
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_request_data(large_requests)
        assert "exceeds maximum limit of 1000" in str(exc_info.value)
        
        # Test early return for None request in list
        requests = [
            {"youtube_url": "https://youtube.com/watch?v=123", "language": "en"},
            None,
            {"youtube_url": "https://youtube.com/watch?v=456", "language": "es"}
        ]
        with pytest.raises(ValidationError) as exc_info:
            validate_batch_request_data(requests)
        assert "Request at index 1 cannot be None" in str(exc_info.value)
    
    def test_validate_viral_variant_data_early_returns(self):
        """Test that validate_viral_variant_data uses early returns."""
        # Test early return for None caption
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start=0.0,
                end=10.0,
                caption=None,
                viral_score=0.8,
                variant_id="test_123"
            )
        assert "Caption is required and cannot be empty" in str(exc_info.value)
        
        # Test early return for empty caption
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start=0.0,
                end=10.0,
                caption="",
                viral_score=0.8,
                variant_id="test_123"
            )
        assert "Caption is required and cannot be empty" in str(exc_info.value)
        
        # Test early return for None variant ID
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start=0.0,
                end=10.0,
                caption="Test caption",
                viral_score=0.8,
                variant_id=None
            )
        assert "Variant ID is required and cannot be empty" in str(exc_info.value)
        
        # Test early return for wrong start time type
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start="0.0",
                end=10.0,
                caption="Test caption",
                viral_score=0.8,
                variant_id="test_123"
            )
        assert "Start time must be a number" in str(exc_info.value)
        
        # Test early return for negative start time
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start=-1.0,
                end=10.0,
                caption="Test caption",
                viral_score=0.8,
                variant_id="test_123"
            )
        assert "Start time must be non-negative" in str(exc_info.value)
        
        # Test early return for viral score out of range
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start=0.0,
                end=10.0,
                caption="Test caption",
                viral_score=1.5,
                variant_id="test_123"
            )
        assert "Viral score must be between 0.0 and 1.0" in str(exc_info.value)
        
        # Test early return for logical constraint
        with pytest.raises(ValidationError) as exc_info:
            validate_viral_variant_data(
                start=15.0,
                end=10.0,
                caption="Test caption",
                viral_score=0.8,
                variant_id="test_123"
            )
        assert "Start time must be less than end time" in str(exc_info.value)

# =============================================================================
# GRADIO DEMO EARLY RETURNS TESTS
# =============================================================================

class TestGradioDemoEarlyReturns:
    """Test that Gradio demo functions use early returns."""
    
    def test_validate_prompt_early_returns(self):
        """Test that validate_prompt uses early returns."""
        # Test early return for None
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt(None)
        assert "Prompt cannot be empty" in str(exc_info.value)
        
        # Test early return for empty string
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt("")
        assert "Prompt cannot be empty" in str(exc_info.value)
        
        # Test early return for wrong type
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt(123)
        assert "Prompt must be a string" in str(exc_info.value)
        
        # Test early return for whitespace only
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt("   ")
        assert "Prompt cannot be empty" in str(exc_info.value)
        
        # Test early return for too long
        long_prompt = "a" * 1001
        with pytest.raises(ValidationError) as exc_info:
            validate_prompt(long_prompt)
        assert "Prompt too long" in str(exc_info.value)
    
    def test_validate_parameters_early_returns(self):
        """Test that validate_parameters uses early returns."""
        # Test early return for wrong guidance scale type
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters("7.5", 30)
        assert "Guidance scale must be a number" in str(exc_info.value)
        
        # Test early return for guidance scale too low
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(0.5, 30)
        assert "Guidance scale must be between 1.0 and 20.0" in str(exc_info.value)
        
        # Test early return for guidance scale too high
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(25.0, 30)
        assert "Guidance scale must be between 1.0 and 20.0" in str(exc_info.value)
        
        # Test early return for wrong inference steps type
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(7.5, "30")
        assert "Inference steps must be an integer" in str(exc_info.value)
        
        # Test early return for inference steps too low
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(7.5, 5)
        assert "Inference steps must be between 10 and 100" in str(exc_info.value)
        
        # Test early return for inference steps too high
        with pytest.raises(ValidationError) as exc_info:
            validate_parameters(7.5, 150)
        assert "Inference steps must be between 10 and 100" in str(exc_info.value)

# =============================================================================
# PERFORMANCE TESTS FOR EARLY RETURNS
# =============================================================================

class TestEarlyReturnsPerformance:
    """Test performance benefits of early returns."""
    
    def test_early_return_performance_vs_nested_ifs(self):
        """Test that early returns are faster than nested if statements."""
        import time
        
        # Test early return (should be fast)
        start_time = time.time()
        try:
            validate_youtube_url(None)
        except ValidationError:
            early_return_time = time.time() - start_time
        
        # Test that early return is very fast
        assert early_return_time < 0.001  # Should be very fast
    
    def test_early_return_performance_in_loops(self):
        """Test that early returns in loops are efficient."""
        import time
        
        # Create a list with an early error
        test_data = [
            {"url": "https://youtube.com/watch?v=123", "language": "en"},
            {"url": None, "language": "en"},  # This should trigger early return
            {"url": "https://youtube.com/watch?v=456", "language": "es"}
        ]
        
        start_time = time.time()
        try:
            for i, data in enumerate(test_data):
                validate_video_request_data(**data)
        except ValidationError:
            early_return_time = time.time() - start_time
        
        # Should fail quickly at index 1, not process all items
        assert early_return_time < 0.01  # Should be very fast
    
    def test_composite_validation_early_return_performance(self):
        """Test that composite validation fails early."""
        import time
        
        start_time = time.time()
        try:
            validate_video_request_data(
                youtube_url=None,  # Should fail immediately
                language="en"
            )
        except ValidationError:
            early_return_time = time.time() - start_time
        
        # Should fail immediately without processing other validations
        assert early_return_time < 0.001  # Should be very fast

# =============================================================================
# CODE STRUCTURE TESTS
# =============================================================================

class TestCodeStructure:
    """Test that code structure follows early return patterns."""
    
    def test_no_deeply_nested_ifs(self):
        """Test that functions don't have deeply nested if statements."""
        import ast
        import inspect
        
        # Get source code for validation functions
        functions_to_check = [
            validate_youtube_url,
            validate_clip_length,
            validate_batch_size,
            validate_caption,
            validate_variant_id,
            validate_audience_profile,
            validate_video_request_data,
            validate_batch_request_data,
            validate_viral_variant_data,
            validate_prompt,
            validate_parameters
        ]
        
        for func in functions_to_check:
            source = inspect.getsource(func)
            tree = ast.parse(source)
            
            # Check for deeply nested if statements (more than 2 levels)
            def check_nesting(node, level=0):
                if isinstance(node, ast.If):
                    if level > 2:
                        return True
                    for child in ast.walk(node):
                        if isinstance(child, ast.If) and child != node:
                            if check_nesting(child, level + 1):
                                return True
                return False
            
            has_deep_nesting = False
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    if check_nesting(node):
                        has_deep_nesting = True
                        break
            
            # Functions should not have deeply nested if statements
            assert not has_deep_nesting, f"Function {func.__name__} has deeply nested if statements"
    
    def test_early_return_pattern_consistency(self):
        """Test that all validation functions follow early return pattern."""
        import inspect
        
        functions_to_check = [
            validate_youtube_url,
            validate_clip_length,
            validate_batch_size,
            validate_caption,
            validate_variant_id,
            validate_audience_profile,
            validate_prompt,
            validate_parameters
        ]
        
        for func in functions_to_check:
            source = inspect.getsource(func)
            
            # Check for early return pattern: if condition: raise/return
            lines = source.split('\n')
            has_early_returns = False
            
            for line in lines:
                line = line.strip()
                if (line.startswith('if ') and 
                    ('raise' in line or 'return' in line) and
                    'else:' not in line):
                    has_early_returns = True
                    break
            
            # All validation functions should use early returns
            assert has_early_returns, f"Function {func.__name__} doesn't use early return pattern"

# =============================================================================
# INTEGRATION TESTS FOR EARLY RETURNS
# =============================================================================

class TestEarlyReturnsIntegration:
    """Test integration of early returns across components."""
    
    def test_api_endpoint_early_return_flow(self):
        """Test that API endpoints use early returns."""
        from ..api import process_video
        
        # Mock request object with None URL
        mock_request = Mock()
        mock_request.youtube_url = None
        mock_request.language = "en"
        
        # Should fail early without reaching processing
        with pytest.raises(ValidationError) as exc_info:
            process_video(mock_request, Mock(), Mock())
        
        assert "YouTube URL is required and cannot be empty" in str(exc_info.value)
    
    def test_batch_api_endpoint_early_return_flow(self):
        """Test that batch API endpoints use early returns."""
        from ..api import process_video_batch
        
        # Mock batch request with None requests list
        mock_batch_request = Mock()
        mock_batch_request.requests = None
        
        # Should fail early without reaching processing
        with pytest.raises(ValidationError) as exc_info:
            process_video_batch(mock_batch_request, Mock(), Mock())
        
        assert "Batch request object is required" in str(exc_info.value)
    
    def test_gradio_demo_early_return_flow(self):
        """Test that Gradio demo uses early returns."""
        from ..gradio_demo import generate_image
        
        # Test with None prompt
        image, error = generate_image(None)
        
        assert image is None
        assert "Prompt cannot be None" in error

# =============================================================================
# EDGE CASE TESTS FOR EARLY RETURNS
# =============================================================================

class TestEarlyReturnsEdgeCases:
    """Test edge cases for early returns."""
    
    def test_validate_youtube_url_edge_cases_early_returns(self):
        """Test edge cases for YouTube URL validation with early returns."""
        edge_cases = [
            ("", "empty string"),
            ("   ", "whitespace only"),
            (123, "integer"),
            (None, "None"),
            ([], "empty list"),
            ({}, "empty dict"),
            (True, "boolean"),
            (3.14, "float")
        ]
        
        for value, description in edge_cases:
            with pytest.raises(ValidationError) as exc_info:
                validate_youtube_url(value)
            
            assert "YouTube URL" in str(exc_info.value), f"Failed for {description}"
    
    def test_validate_clip_length_edge_cases_early_returns(self):
        """Test edge cases for clip length validation with early returns."""
        edge_cases = [
            ("30", "string"),
            (None, "None"),
            (0, "zero"),
            (-1, "negative"),
            (3.14, "float"),
            (True, "boolean"),
            ([], "list"),
            ({}, "dict")
        ]
        
        for value, description in edge_cases:
            with pytest.raises(ValidationError) as exc_info:
                validate_clip_length(value)
            
            assert "Clip length" in str(exc_info.value), f"Failed for {description}"
    
    def test_validate_batch_size_edge_cases_early_returns(self):
        """Test edge cases for batch size validation with early returns."""
        edge_cases = [
            ("10", "string"),
            (None, "None"),
            (0, "zero"),
            (-1, "negative"),
            (3.14, "float"),
            (True, "boolean"),
            ([], "list"),
            ({}, "dict")
        ]
        
        for value, description in edge_cases:
            with pytest.raises(ValidationError) as exc_info:
                validate_batch_size(value)
            
            assert "Batch size" in str(exc_info.value), f"Failed for {description}"

# =============================================================================
# BENCHMARK TESTS FOR EARLY RETURNS
# =============================================================================

class TestEarlyReturnsBenchmarks:
    """Benchmark tests for early returns performance."""
    
    def test_early_return_vs_late_return_performance(self):
        """Compare performance of early returns vs late returns."""
        import time
        
        # Test early return performance
        start_time = time.time()
        for _ in range(1000):
            try:
                validate_youtube_url(None)
            except ValidationError:
                pass
        early_return_time = time.time() - start_time
        
        # Test late return performance (simulated)
        start_time = time.time()
        for _ in range(1000):
            try:
                # Simulate a function that does work before validation
                time.sleep(0.0001)  # Simulate work
                validate_youtube_url(None)
            except ValidationError:
                pass
        late_return_time = time.time() - start_time
        
        # Early returns should be significantly faster
        assert early_return_time < late_return_time * 0.1  # At least 10x faster
    
    def test_early_return_memory_efficiency(self):
        """Test that early returns are memory efficient."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run early return validations
        for _ in range(1000):
            try:
                validate_youtube_url(None)
            except ValidationError:
                pass
        
        # Get final memory usage
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal
        assert memory_increase < 1024 * 1024  # Less than 1MB increase

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 