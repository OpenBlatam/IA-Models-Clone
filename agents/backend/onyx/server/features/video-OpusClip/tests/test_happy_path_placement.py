"""
Test Suite for Happy Path Placement

Tests that functions place the happy path (success case) last for improved
readability and maintainability.
"""

import pytest
import ast
import inspect
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
    ConfigurationError
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
from ..gradio_demo import validate_prompt, validate_parameters, generate_image

# =============================================================================
# HAPPY PATH PLACEMENT TESTS
# =============================================================================

class TestHappyPathPlacement:
    """Test that functions place the happy path last for improved readability."""
    
    def test_validate_youtube_url_happy_path_last(self):
        """Test that validate_youtube_url places happy path last."""
        source = inspect.getsource(validate_youtube_url)
        lines = source.split('\n')
        
        # Find the happy path comment
        happy_path_line = None
        for i, line in enumerate(lines):
            if "HAPPY PATH" in line:
                happy_path_line = i
                break
        
        # Happy path should be near the end (last few lines)
        assert happy_path_line is not None, "Happy path comment not found"
        assert happy_path_line >= len(lines) - 3, "Happy path should be at the end"
    
    def test_validate_clip_length_happy_path_last(self):
        """Test that validate_clip_length places happy path last."""
        source = inspect.getsource(validate_clip_length)
        lines = source.split('\n')
        
        # Find the happy path comment
        happy_path_line = None
        for i, line in enumerate(lines):
            if "HAPPY PATH" in line:
                happy_path_line = i
                break
        
        # Happy path should be near the end (last few lines)
        assert happy_path_line is not None, "Happy path comment not found"
        assert happy_path_line >= len(lines) - 3, "Happy path should be at the end"
    
    def test_validate_batch_size_happy_path_last(self):
        """Test that validate_batch_size places happy path last."""
        source = inspect.getsource(validate_batch_size)
        lines = source.split('\n')
        
        # Find the happy path comment
        happy_path_line = None
        for i, line in enumerate(lines):
            if "HAPPY PATH" in line:
                happy_path_line = i
                break
        
        # Happy path should be near the end (last few lines)
        assert happy_path_line is not None, "Happy path comment not found"
        assert happy_path_line >= len(lines) - 3, "Happy path should be at the end"
    
    def test_validate_video_request_data_happy_path_last(self):
        """Test that validate_video_request_data places happy path last."""
        source = inspect.getsource(validate_video_request_data)
        lines = source.split('\n')
        
        # Find the happy path comment
        happy_path_line = None
        for i, line in enumerate(lines):
            if "HAPPY PATH" in line:
                happy_path_line = i
                break
        
        # Happy path should be near the end (last few lines)
        assert happy_path_line is not None, "Happy path comment not found"
        assert happy_path_line >= len(lines) - 5, "Happy path should be at the end"
    
    def test_validate_batch_request_data_happy_path_last(self):
        """Test that validate_batch_request_data places happy path last."""
        source = inspect.getsource(validate_batch_request_data)
        lines = source.split('\n')
        
        # Find the happy path comment
        happy_path_line = None
        for i, line in enumerate(lines):
            if "HAPPY PATH" in line:
                happy_path_line = i
                break
        
        # Happy path should be near the end (last few lines)
        assert happy_path_line is not None, "Happy path comment not found"
        assert happy_path_line >= len(lines) - 3, "Happy path should be at the end"
    
    def test_validate_viral_variant_data_happy_path_last(self):
        """Test that validate_viral_variant_data places happy path last."""
        source = inspect.getsource(validate_viral_variant_data)
        lines = source.split('\n')
        
        # Find the happy path comment
        happy_path_line = None
        for i, line in enumerate(lines):
            if "HAPPY PATH" in line:
                happy_path_line = i
                break
        
        # Happy path should be near the end (last few lines)
        assert happy_path_line is not None, "Happy path comment not found"
        assert happy_path_line >= len(lines) - 3, "Happy path should be at the end"
    
    def test_validate_prompt_happy_path_last(self):
        """Test that validate_prompt places happy path last."""
        source = inspect.getsource(validate_prompt)
        lines = source.split('\n')
        
        # Find the happy path comment
        happy_path_line = None
        for i, line in enumerate(lines):
            if "HAPPY PATH" in line:
                happy_path_line = i
                break
        
        # Happy path should be near the end (last few lines)
        assert happy_path_line is not None, "Happy path comment not found"
        assert happy_path_line >= len(lines) - 3, "Happy path should be at the end"
    
    def test_validate_parameters_happy_path_last(self):
        """Test that validate_parameters places happy path last."""
        source = inspect.getsource(validate_parameters)
        lines = source.split('\n')
        
        # Find the happy path comment
        happy_path_line = None
        for i, line in enumerate(lines):
            if "HAPPY PATH" in line:
                happy_path_line = i
                break
        
        # Happy path should be near the end (last few lines)
        assert happy_path_line is not None, "Happy path comment not found"
        assert happy_path_line >= len(lines) - 3, "Happy path should be at the end"

# =============================================================================
# API ENDPOINT HAPPY PATH TESTS
# =============================================================================

class TestAPIEndpointHappyPath:
    """Test that API endpoints place the happy path last."""
    
    def test_process_video_happy_path_last(self):
        """Test that process_video endpoint places happy path last."""
        from ..api import process_video
        
        source = inspect.getsource(process_video)
        lines = source.split('\n')
        
        # Find the happy path comment
        happy_path_line = None
        for i, line in enumerate(lines):
            if "HAPPY PATH" in line:
                happy_path_line = i
                break
        
        # Happy path should be near the end (last few lines)
        assert happy_path_line is not None, "Happy path comment not found"
        assert happy_path_line >= len(lines) - 10, "Happy path should be at the end"
    
    def test_process_video_batch_happy_path_last(self):
        """Test that process_video_batch endpoint places happy path last."""
        from ..api import process_video_batch
        
        source = inspect.getsource(process_video_batch)
        lines = source.split('\n')
        
        # Find the happy path comment
        happy_path_line = None
        for i, line in enumerate(lines):
            if "HAPPY PATH" in line:
                happy_path_line = i
                break
        
        # Happy path should be near the end (last few lines)
        assert happy_path_line is not None, "Happy path comment not found"
        assert happy_path_line >= len(lines) - 10, "Happy path should be at the end"
    
    def test_process_viral_variants_happy_path_last(self):
        """Test that process_viral_variants endpoint places happy path last."""
        from ..api import process_viral_variants
        
        source = inspect.getsource(process_viral_variants)
        lines = source.split('\n')
        
        # Find the happy path comment
        happy_path_line = None
        for i, line in enumerate(lines):
            if "HAPPY PATH" in line:
                happy_path_line = i
                break
        
        # Happy path should be near the end (last few lines)
        assert happy_path_line is not None, "Happy path comment not found"
        assert happy_path_line >= len(lines) - 10, "Happy path should be at the end"

# =============================================================================
# GRADIO DEMO HAPPY PATH TESTS
# =============================================================================

class TestGradioDemoHappyPath:
    """Test that Gradio demo functions place the happy path last."""
    
    def test_generate_image_happy_path_last(self):
        """Test that generate_image function places happy path last."""
        source = inspect.getsource(generate_image)
        lines = source.split('\n')
        
        # Find the happy path comment
        happy_path_line = None
        for i, line in enumerate(lines):
            if "HAPPY PATH" in line:
                happy_path_line = i
                break
        
        # Happy path should be near the end (last few lines)
        assert happy_path_line is not None, "Happy path comment not found"
        assert happy_path_line >= len(lines) - 15, "Happy path should be at the end"

# =============================================================================
# CODE STRUCTURE ANALYSIS TESTS
# =============================================================================

class TestCodeStructureAnalysis:
    """Test code structure for happy path placement patterns."""
    
    def test_error_handling_before_happy_path(self):
        """Test that error handling comes before happy path in all functions."""
        functions_to_check = [
            validate_youtube_url,
            validate_clip_length,
            validate_batch_size,
            validate_video_request_data,
            validate_batch_request_data,
            validate_viral_variant_data,
            validate_prompt,
            validate_parameters
        ]
        
        for func in functions_to_check:
            source = inspect.getsource(func)
            lines = source.split('\n')
            
            # Find error handling and happy path comments
            error_handling_lines = []
            happy_path_line = None
            
            for i, line in enumerate(lines):
                if "ERROR HANDLING" in line:
                    error_handling_lines.append(i)
                elif "HAPPY PATH" in line:
                    happy_path_line = i
                    break
            
            # All error handling should come before happy path
            if happy_path_line is not None and error_handling_lines:
                for error_line in error_handling_lines:
                    assert error_line < happy_path_line, f"Error handling should come before happy path in {func.__name__}"
    
    def test_comment_pattern_consistency(self):
        """Test that all functions use consistent comment patterns."""
        functions_to_check = [
            validate_youtube_url,
            validate_clip_length,
            validate_batch_size,
            validate_video_request_data,
            validate_batch_request_data,
            validate_viral_variant_data,
            validate_prompt,
            validate_parameters
        ]
        
        for func in functions_to_check:
            source = inspect.getsource(func)
            
            # Check for consistent comment patterns
            assert "# ERROR HANDLING:" in source, f"Function {func.__name__} should have ERROR HANDLING comments"
            assert "# HAPPY PATH:" in source, f"Function {func.__name__} should have HAPPY PATH comment"
            
            # Check that comments are properly formatted
            lines = source.split('\n')
            for line in lines:
                if "ERROR HANDLING:" in line or "HAPPY PATH:" in line:
                    assert line.strip().startswith("#"), f"Comment should start with # in {func.__name__}"
    
    def test_function_structure_pattern(self):
        """Test that functions follow the error handling -> happy path pattern."""
        functions_to_check = [
            validate_youtube_url,
            validate_clip_length,
            validate_batch_size,
            validate_video_request_data,
            validate_batch_request_data,
            validate_viral_variant_data,
            validate_prompt,
            validate_parameters
        ]
        
        for func in functions_to_check:
            source = inspect.getsource(func)
            lines = source.split('\n')
            
            # Find the structure: error handling first, then happy path
            error_handling_found = False
            happy_path_found = False
            
            for line in lines:
                if "ERROR HANDLING:" in line:
                    error_handling_found = True
                elif "HAPPY PATH:" in line:
                    happy_path_found = True
                    # Happy path should come after error handling
                    assert error_handling_found, f"Happy path should come after error handling in {func.__name__}"
            
            # Both patterns should be present
            assert error_handling_found, f"Function {func.__name__} should have error handling"
            assert happy_path_found, f"Function {func.__name__} should have happy path"

# =============================================================================
# FUNCTIONAL TESTS FOR HAPPY PATH
# =============================================================================

class TestHappyPathFunctionality:
    """Test that happy path actually works correctly."""
    
    def test_validate_youtube_url_happy_path_works(self):
        """Test that validate_youtube_url happy path works with valid input."""
        # This should not raise an exception (happy path)
        validate_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
    
    def test_validate_clip_length_happy_path_works(self):
        """Test that validate_clip_length happy path works with valid input."""
        # This should not raise an exception (happy path)
        validate_clip_length(30)
    
    def test_validate_batch_size_happy_path_works(self):
        """Test that validate_batch_size happy path works with valid input."""
        # This should not raise an exception (happy path)
        validate_batch_size(10)
    
    def test_validate_video_request_data_happy_path_works(self):
        """Test that validate_video_request_data happy path works with valid input."""
        # This should not raise an exception (happy path)
        validate_video_request_data(
            youtube_url="https://youtube.com/watch?v=dQw4w9WgXcQ",
            language="en"
        )
    
    def test_validate_prompt_happy_path_works(self):
        """Test that validate_prompt happy path works with valid input."""
        # This should not raise an exception (happy path)
        validate_prompt("A beautiful sunset over the ocean")
    
    def test_validate_parameters_happy_path_works(self):
        """Test that validate_parameters happy path works with valid input."""
        # This should not raise an exception (happy path)
        validate_parameters(7.5, 30)

# =============================================================================
# PERFORMANCE TESTS FOR HAPPY PATH
# =============================================================================

class TestHappyPathPerformance:
    """Test performance of happy path execution."""
    
    def test_happy_path_performance_vs_error_path(self):
        """Test that happy path is faster than error path."""
        import time
        
        # Test happy path (should be fast)
        start_time = time.time()
        validate_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
        happy_path_time = time.time() - start_time
        
        # Test error path (should be slower due to exception handling)
        start_time = time.time()
        try:
            validate_youtube_url(None)
        except ValidationError:
            error_path_time = time.time() - start_time
        
        # Happy path should be faster than error path
        assert happy_path_time < error_path_time, "Happy path should be faster than error path"
    
    def test_happy_path_performance_consistency(self):
        """Test that happy path performance is consistent."""
        import time
        
        times = []
        for _ in range(10):
            start_time = time.time()
            validate_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
            times.append(time.time() - start_time)
        
        # Performance should be consistent (low variance)
        avg_time = sum(times) / len(times)
        variance = sum((t - avg_time) ** 2 for t in times) / len(times)
        
        # Variance should be low (consistent performance)
        assert variance < 0.0001, "Happy path performance should be consistent"

# =============================================================================
# READABILITY TESTS
# =============================================================================

class TestHappyPathReadability:
    """Test that happy path placement improves readability."""
    
    def test_function_readability_score(self):
        """Test that functions with happy path last are more readable."""
        functions_to_check = [
            validate_youtube_url,
            validate_clip_length,
            validate_batch_size,
            validate_video_request_data,
            validate_batch_request_data,
            validate_viral_variant_data,
            validate_prompt,
            validate_parameters
        ]
        
        for func in functions_to_check:
            source = inspect.getsource(func)
            lines = source.split('\n')
            
            # Calculate readability metrics
            total_lines = len(lines)
            error_handling_lines = sum(1 for line in lines if "ERROR HANDLING" in line)
            happy_path_lines = sum(1 for line in lines if "HAPPY PATH" in line)
            
            # Functions should have reasonable structure
            assert total_lines > 0, f"Function {func.__name__} should have content"
            assert error_handling_lines > 0, f"Function {func.__name__} should have error handling"
            assert happy_path_lines > 0, f"Function {func.__name__} should have happy path"
            
            # Error handling should be more extensive than happy path
            assert error_handling_lines >= happy_path_lines, f"Error handling should be more extensive than happy path in {func.__name__}"
    
    def test_comment_clarity(self):
        """Test that comments clearly indicate happy path placement."""
        functions_to_check = [
            validate_youtube_url,
            validate_clip_length,
            validate_batch_size,
            validate_video_request_data,
            validate_batch_request_data,
            validate_viral_variant_data,
            validate_prompt,
            validate_parameters
        ]
        
        for func in functions_to_check:
            source = inspect.getsource(func)
            
            # Check for clear happy path comment
            assert "# HAPPY PATH:" in source, f"Function {func.__name__} should have clear happy path comment"
            
            # Check that happy path comment is descriptive
            lines = source.split('\n')
            for line in lines:
                if "HAPPY PATH:" in line:
                    assert len(line.strip()) > len("# HAPPY PATH:"), f"Happy path comment should be descriptive in {func.__name__}"

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestHappyPathIntegration:
    """Test integration of happy path placement across components."""
    
    def test_validation_chain_happy_path(self):
        """Test that validation chain follows happy path pattern."""
        # Test that composite validation follows the pattern
        validate_video_request_data(
            youtube_url="https://youtube.com/watch?v=dQw4w9WgXcQ",
            language="en"
        )
        
        # Test that individual validations follow the pattern
        validate_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
        validate_language_code("en")
    
    def test_api_endpoint_happy_path_flow(self):
        """Test that API endpoints follow happy path pattern."""
        # Mock the necessary components
        with patch('video_opusclip.api.validate_system_health'), \
             patch('video_opusclip.api.validate_gpu_health'), \
             patch('video_opusclip.api.validate_video_request_data'), \
             patch('video_opusclip.api.validate_and_sanitize_url'):
            
            # The happy path should be the last part of the function
            # This test verifies the structure, not the actual execution
            from ..api import process_video
            
            source = inspect.getsource(process_video)
            assert "# HAPPY PATH:" in source, "API endpoint should have happy path comment"
    
    def test_gradio_demo_happy_path_flow(self):
        """Test that Gradio demo follows happy path pattern."""
        # Mock the necessary components
        with patch('video_opusclip.gradio_demo.pipe'), \
             patch('video_opusclip.gradio_demo.device', 'cpu'):
            
            # The happy path should be the last part of the function
            source = inspect.getsource(generate_image)
            assert "# HAPPY PATH:" in source, "Gradio demo should have happy path comment"

# =============================================================================
# DOCUMENTATION TESTS
# =============================================================================

class TestHappyPathDocumentation:
    """Test that happy path placement is well documented."""
    
    def test_function_docstrings_mention_happy_path(self):
        """Test that function docstrings mention happy path pattern."""
        functions_to_check = [
            validate_youtube_url,
            validate_clip_length,
            validate_batch_size,
            validate_video_request_data,
            validate_batch_request_data,
            validate_viral_variant_data,
            validate_prompt,
            validate_parameters
        ]
        
        for func in functions_to_check:
            docstring = func.__doc__
            assert docstring is not None, f"Function {func.__name__} should have a docstring"
            
            # Docstring should mention validation or error handling
            assert any(word in docstring.lower() for word in ['validate', 'error', 'raise']), \
                f"Function {func.__name__} docstring should mention validation or error handling"
    
    def test_comment_consistency(self):
        """Test that comments are consistent across all functions."""
        functions_to_check = [
            validate_youtube_url,
            validate_clip_length,
            validate_batch_size,
            validate_video_request_data,
            validate_batch_request_data,
            validate_viral_variant_data,
            validate_prompt,
            validate_parameters
        ]
        
        for func in functions_to_check:
            source = inspect.getsource(func)
            
            # Check for consistent comment formatting
            lines = source.split('\n')
            for line in lines:
                if "ERROR HANDLING:" in line:
                    assert line.strip().startswith("# ERROR HANDLING:"), \
                        f"Error handling comment should be properly formatted in {func.__name__}"
                elif "HAPPY PATH:" in line:
                    assert line.strip().startswith("# HAPPY PATH:"), \
                        f"Happy path comment should be properly formatted in {func.__name__}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 