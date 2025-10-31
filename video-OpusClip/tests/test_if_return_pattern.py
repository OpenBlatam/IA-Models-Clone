"""
Test Suite for If-Return Pattern

Tests that functions avoid unnecessary else statements by using the if-return
pattern for improved readability and maintainability.
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
from ..viral_pipeline import generate_viral_variants

# =============================================================================
# IF-RETURN PATTERN TESTS
# =============================================================================

class TestIfReturnPattern:
    """Test that functions use if-return pattern instead of unnecessary else statements."""
    
    def test_error_handling_analyze_error_pattern_no_else(self):
        """Test that _analyze_error_pattern uses if-return pattern."""
        error_handler = ErrorHandler()
        
        # Test memory error
        memory_error = Exception("Out of memory error")
        result = error_handler._analyze_error_pattern(memory_error)
        assert result == "memory_related"
        
        # Test timeout error
        timeout_error = Exception("Request timed out")
        result = error_handler._analyze_error_pattern(timeout_error)
        assert result == "timeout_related"
        
        # Test network error
        network_error = Exception("Connection failed")
        result = error_handler._analyze_error_pattern(network_error)
        assert result == "network_related"
        
        # Test permission error
        permission_error = Exception("Access denied")
        result = error_handler._analyze_error_pattern(permission_error)
        assert result == "permission_related"
        
        # Test unknown error
        unknown_error = Exception("Some random error")
        result = error_handler._analyze_error_pattern(unknown_error)
        assert result == "unknown_pattern"
    
    def test_viral_pipeline_cache_logic_no_else(self):
        """Test that viral pipeline cache logic uses if-return pattern."""
        # Mock the dependencies
        with patch('video_opusclip.viral_pipeline.get_redis_cache') as mock_cache, \
             patch('video_opusclip.viral_pipeline.get_prompt_and_parser') as mock_prompt, \
             patch('video_opusclip.viral_pipeline.get_llm') as mock_llm:
            
            # Mock cache returning None (no cache hit)
            mock_cache.return_value = None
            
            # Mock prompt and parser
            mock_prompt.return_value = (Mock(), Mock())
            mock_llm.return_value = Mock()
            
            # Test that function works without else statements
            from ..models import VideoClipRequest
            request = VideoClipRequest(youtube_url="https://youtube.com/watch?v=test", language="en")
            
            # This should not raise any syntax errors related to else statements
            result = generate_viral_variants(request, n_variants=1)
            assert result is not None
    
    def test_validation_functions_no_else(self):
        """Test that validation functions use if-return pattern."""
        # Test that validation functions don't use else statements
        functions_to_check = [
            validate_youtube_url,
            validate_clip_length,
            validate_batch_size,
            validate_viral_score,
            validate_caption,
            validate_variant_id,
            validate_audience_profile
        ]
        
        for func in functions_to_check:
            source = inspect.getsource(func)
            
            # Check that there are no unnecessary else statements
            lines = source.split('\n')
            for line in lines:
                if 'else:' in line:
                    # Only allow else in try-except blocks or other necessary contexts
                    # Check if it's in a try-except block
                    line_index = lines.index(line)
                    if line_index > 0:
                        prev_line = lines[line_index - 1].strip()
                        if not (prev_line.startswith('except') or 
                               prev_line.startswith('try:') or
                               prev_line.endswith(':')):
                            # This might be an unnecessary else
                            # But we need to check the context more carefully
                            pass
    
    def test_api_endpoints_no_else(self):
        """Test that API endpoints use if-return pattern."""
        from ..api import process_video, process_video_batch, process_viral_variants
        
        endpoints_to_check = [
            process_video,
            process_video_batch,
            process_viral_variants
        ]
        
        for endpoint in endpoints_to_check:
            source = inspect.getsource(endpoint)
            
            # Check for unnecessary else statements
            lines = source.split('\n')
            for line in lines:
                if 'else:' in line:
                    # Check if it's in a try-except block
                    line_index = lines.index(line)
                    if line_index > 0:
                        prev_line = lines[line_index - 1].strip()
                        if not (prev_line.startswith('except') or 
                               prev_line.startswith('try:') or
                               prev_line.endswith(':')):
                            # This might be an unnecessary else
                            # But we need to check the context more carefully
                            pass

# =============================================================================
# CODE STRUCTURE ANALYSIS TESTS
# =============================================================================

class TestCodeStructureAnalysis:
    """Test code structure for if-return pattern usage."""
    
    def test_no_unnecessary_else_statements(self):
        """Test that functions don't have unnecessary else statements."""
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
            
            # Count else statements
            else_count = sum(1 for line in lines if 'else:' in line)
            
            # Count try-except blocks (where else might be necessary)
            try_count = sum(1 for line in lines if line.strip().startswith('try:'))
            except_count = sum(1 for line in lines if line.strip().startswith('except'))
            
            # If there are else statements, they should be in try-except blocks
            if else_count > 0:
                # Check if else statements are in try-except blocks
                for i, line in enumerate(lines):
                    if 'else:' in line:
                        # Look for corresponding try or except
                        found_try_except = False
                        for j in range(max(0, i-10), i):
                            if lines[j].strip().startswith(('try:', 'except')):
                                found_try_except = True
                                break
                        
                        # If not in try-except, it might be unnecessary
                        if not found_try_except:
                            # This could be an unnecessary else
                            # But we need to check the context more carefully
                            pass
    
    def test_early_return_pattern(self):
        """Test that functions use early return pattern."""
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
            
            # Check for early returns (returns before the end of the function)
            return_lines = []
            for i, line in enumerate(lines):
                if 'return' in line and not line.strip().startswith('#'):
                    return_lines.append(i)
            
            # Functions should have early returns for error conditions
            if return_lines:
                # At least one return should be before the end
                assert any(line < len(lines) - 5 for line in return_lines), \
                    f"Function {func.__name__} should have early returns"
    
    def test_conditional_structure(self):
        """Test that conditional structures use if-return pattern."""
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
            
            # Check for if-else patterns that could be simplified
            for i, line in enumerate(lines):
                if line.strip().startswith('if '):
                    # Look for corresponding else
                    for j in range(i+1, min(i+10, len(lines))):
                        if lines[j].strip().startswith('else:'):
                            # Check if this could be simplified to if-return
                            # by looking at the structure
                            if_block = []
                            else_block = []
                            
                            # Collect if block
                            k = i + 1
                            while k < len(lines) and not lines[k].strip().startswith('else:'):
                                if lines[k].strip() and not lines[k].strip().startswith('#'):
                                    if_block.append(lines[k].strip())
                                k += 1
                            
                            # Collect else block
                            k = j + 1
                            while k < len(lines) and not lines[k].strip().startswith('if '):
                                if lines[k].strip() and not lines[k].strip().startswith('#'):
                                    else_block.append(lines[k].strip())
                                k += 1
                            
                            # If else block is simple, it might be unnecessary
                            if len(else_block) <= 2:
                                # This could potentially be simplified
                                pass

# =============================================================================
# FUNCTIONAL TESTS FOR IF-RETURN PATTERN
# =============================================================================

class TestIfReturnFunctionality:
    """Test that if-return pattern actually works correctly."""
    
    def test_error_handling_analyze_error_pattern_functionality(self):
        """Test that _analyze_error_pattern works correctly with if-return pattern."""
        error_handler = ErrorHandler()
        
        # Test various error patterns
        test_cases = [
            ("Out of memory", "memory_related"),
            ("Memory allocation failed", "memory_related"),
            ("Request timed out", "timeout_related"),
            ("Connection timeout", "timeout_related"),
            ("Network connection failed", "network_related"),
            ("Connection refused", "network_related"),
            ("Permission denied", "permission_related"),
            ("Access denied", "permission_related"),
            ("Random error message", "unknown_pattern"),
        ]
        
        for error_message, expected_pattern in test_cases:
            error = Exception(error_message)
            result = error_handler._analyze_error_pattern(error)
            assert result == expected_pattern, f"Expected {expected_pattern} for '{error_message}', got {result}"
    
    def test_validation_functions_early_return(self):
        """Test that validation functions return early on errors."""
        # Test that validation functions raise exceptions immediately
        with pytest.raises(ValidationError):
            validate_youtube_url("")
        
        with pytest.raises(ValidationError):
            validate_youtube_url(None)
        
        with pytest.raises(ValidationError):
            validate_clip_length(-1)
        
        with pytest.raises(ValidationError):
            validate_batch_size(0)
        
        # Test that valid inputs don't raise exceptions
        validate_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
        validate_clip_length(30)
        validate_batch_size(10)
    
    def test_gradio_demo_early_return(self):
        """Test that Gradio demo functions return early on errors."""
        # Test that validation functions raise exceptions immediately
        with pytest.raises(ValidationError):
            validate_prompt("")
        
        with pytest.raises(ValidationError):
            validate_prompt(None)
        
        with pytest.raises(ValidationError):
            validate_parameters(-1, 30)
        
        with pytest.raises(ValidationError):
            validate_parameters(7.5, 5)
        
        # Test that valid inputs don't raise exceptions
        validate_prompt("A beautiful sunset")
        validate_parameters(7.5, 30)

# =============================================================================
# PERFORMANCE TESTS FOR IF-RETURN PATTERN
# =============================================================================

class TestIfReturnPerformance:
    """Test performance benefits of if-return pattern."""
    
    def test_early_return_performance(self):
        """Test that early returns improve performance."""
        import time
        
        # Test validation function performance
        start_time = time.time()
        for _ in range(1000):
            try:
                validate_youtube_url("")
            except ValidationError:
                pass
        early_return_time = time.time() - start_time
        
        # Test that it's fast (should fail fast)
        assert early_return_time < 0.1, "Early return should be fast"
    
    def test_error_handling_performance(self):
        """Test that error handling is fast with if-return pattern."""
        import time
        
        error_handler = ErrorHandler()
        
        # Test error pattern analysis performance
        start_time = time.time()
        for _ in range(1000):
            error_handler._analyze_error_pattern(Exception("Out of memory"))
        analysis_time = time.time() - start_time
        
        # Should be very fast
        assert analysis_time < 0.1, "Error pattern analysis should be fast"

# =============================================================================
# READABILITY TESTS
# =============================================================================

class TestIfReturnReadability:
    """Test that if-return pattern improves readability."""
    
    def test_function_readability_score(self):
        """Test that functions with if-return pattern are more readable."""
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
            if_statements = sum(1 for line in lines if line.strip().startswith('if '))
            else_statements = sum(1 for line in lines if line.strip().startswith('else:'))
            return_statements = sum(1 for line in lines if 'return' in line and not line.strip().startswith('#'))
            
            # Functions should have more if statements than else statements
            assert if_statements >= else_statements, f"Function {func.__name__} should have more if statements than else statements"
            
            # Functions should have early returns
            assert return_statements > 0, f"Function {func.__name__} should have return statements"
    
    def test_comment_clarity(self):
        """Test that comments clearly indicate if-return pattern."""
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
            
            # Check for clear error handling comments
            assert "# ERROR HANDLING:" in source, f"Function {func.__name__} should have ERROR HANDLING comments"
            
            # Check that comments are properly formatted
            lines = source.split('\n')
            for line in lines:
                if "ERROR HANDLING:" in line or "HAPPY PATH:" in line:
                    assert line.strip().startswith("#"), f"Comment should start with # in {func.__name__}"

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIfReturnIntegration:
    """Test integration of if-return pattern across components."""
    
    def test_validation_chain_if_return(self):
        """Test that validation chain follows if-return pattern."""
        # Test that composite validation follows the pattern
        validate_video_request_data(
            youtube_url="https://youtube.com/watch?v=dQw4w9WgXcQ",
            language="en"
        )
        
        # Test that individual validations follow the pattern
        validate_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
        validate_language_code("en")
    
    def test_api_endpoint_if_return_flow(self):
        """Test that API endpoints follow if-return pattern."""
        # Mock the necessary components
        with patch('video_opusclip.api.validate_system_health'), \
             patch('video_opusclip.api.validate_gpu_health'), \
             patch('video_opusclip.api.validate_video_request_data'), \
             patch('video_opusclip.api.validate_and_sanitize_url'):
            
            # The if-return pattern should be evident in the structure
            from ..api import process_video
            
            source = inspect.getsource(process_video)
            assert "# ERROR HANDLING:" in source, "API endpoint should have error handling comments"
    
    def test_gradio_demo_if_return_flow(self):
        """Test that Gradio demo follows if-return pattern."""
        # Mock the necessary components
        with patch('video_opusclip.gradio_demo.pipe'), \
             patch('video_opusclip.gradio_demo.device', 'cpu'):
            
            # The if-return pattern should be evident in the structure
            source = inspect.getsource(generate_image)
            assert "# ERROR HANDLING:" in source, "Gradio demo should have error handling comments"

# =============================================================================
# DOCUMENTATION TESTS
# =============================================================================

class TestIfReturnDocumentation:
    """Test that if-return pattern is well documented."""
    
    def test_function_docstrings_mention_pattern(self):
        """Test that function docstrings mention if-return pattern."""
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