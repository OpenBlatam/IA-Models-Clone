"""
Test suite for guard clauses implementation.

Tests that preconditions and invalid states are handled early in all functions.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation import (
    validate_youtube_url,
    validate_language_code,
    validate_clip_length,
    validate_viral_score,
    validate_caption,
    validate_variant_id,
    validate_audience_profile,
    validate_batch_size,
    validate_video_request_data,
    validate_viral_variant_data,
    validate_batch_request_data,
    validate_system_health,
    validate_gpu_health,
    check_system_resources,
    check_gpu_availability
)

from error_handling import (
    ValidationError,
    ProcessingError,
    ResourceError,
    SecurityError,
    ConfigurationError,
    CriticalSystemError
)

from gradio_demo import (
    validate_prompt,
    validate_parameters,
    generate_image
)

# =============================================================================
# TEST DATA
# =============================================================================

VALID_YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://youtu.be/dQw4w9WgXcQ",
    "https://www.youtube.com/embed/dQw4w9WgXcQ"
]

INVALID_YOUTUBE_URLS = [
    None,
    "",
    "not_a_url",
    "https://vimeo.com/123456",
    "javascript:alert('xss')",
    "data:text/html,<script>alert('xss')</script>",
    "file:///etc/passwd",
    "ftp://malicious.com/file",
    "eval(alert('xss'))",
    "exec(system('rm -rf /'))"
]

VALID_LANGUAGE_CODES = [
    "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh",
    "en-US", "en-GB", "es-MX", "es-ES", "fr-FR", "de-DE", "it-IT"
]

INVALID_LANGUAGE_CODES = [
    None,
    "",
    "invalid",
    "123",
    "EN",
    "english",
    "xx-YY"
]

VALID_CLIP_LENGTHS = [1, 30, 60, 300, 600]
INVALID_CLIP_LENGTHS = [-1, 0, 601, 86401, "30", 30.5]

VALID_VIRAL_SCORES = [0.0, 0.5, 1.0]
INVALID_VIRAL_SCORES = [-0.1, 1.1, "0.5", None]

VALID_CAPTIONS = [
    "A great video about AI",
    "Short caption",
    "A" * 1000  # Max length
]

INVALID_CAPTIONS = [
    None,
    "",
    "   ",
    "A" * 1001  # Too long
]

VALID_VARIANT_IDS = [
    "variant_1",
    "variant_abc123",
    "variant-with-dashes"
]

INVALID_VARIANT_IDS = [
    None,
    "",
    "variant with spaces",
    "variant.with.dots",
    "variant/with/slashes"
]

VALID_AUDIENCE_PROFILES = [
    {"age": "18-25", "interests": ["tech", "gaming"]},
    {"location": "US", "language": "en"},
    {}
]

INVALID_AUDIENCE_PROFILES = [
    None,
    "not_a_dict",
    {"invalid_key": "value"},
    {"age": 123},  # Should be string
    {"interests": "not_a_list"}  # Should be list
]

VALID_BATCH_SIZES = [1, 10, 50, 100]
INVALID_BATCH_SIZES = [0, 101, -1, "10", 10.5]

VALID_PROMPTS = [
    "A beautiful landscape",
    "Short prompt",
    "A" * 1000  # Max length
]

INVALID_PROMPTS = [
    None,
    "",
    "   ",
    "A" * 1001  # Too long
]

# =============================================================================
# VALIDATION FUNCTION TESTS
# =============================================================================

class TestYouTubeURLValidation:
    """Test guard clauses in YouTube URL validation."""
    
    def test_valid_youtube_urls(self):
        """Test that valid YouTube URLs pass validation."""
        for url in VALID_YOUTUBE_URLS:
            # Should not raise any exception
            validate_youtube_url(url)
    
    def test_none_url(self):
        """Test guard clause for None URL."""
        with pytest.raises(ValidationError, match="YouTube URL is required"):
            validate_youtube_url(None)
    
    def test_empty_url(self):
        """Test guard clause for empty URL."""
        with pytest.raises(ValidationError, match="YouTube URL is required"):
            validate_youtube_url("")
    
    def test_wrong_data_type(self):
        """Test guard clause for wrong data type."""
        with pytest.raises(ValidationError, match="YouTube URL must be a string"):
            validate_youtube_url(123)
    
    def test_url_too_long(self):
        """Test guard clause for URL too long."""
        long_url = "https://www.youtube.com/watch?v=" + "a" * 2000
        with pytest.raises(ValidationError, match="YouTube URL too long"):
            validate_youtube_url(long_url)
    
    def test_malicious_patterns(self):
        """Test guard clauses for malicious patterns."""
        for url in INVALID_YOUTUBE_URLS[4:]:  # Skip None, empty, etc.
            with pytest.raises(ValidationError, match="Malicious URL pattern detected"):
                validate_youtube_url(url)
    
    def test_invalid_youtube_format(self):
        """Test guard clause for invalid YouTube format."""
        with pytest.raises(ValidationError, match="Invalid YouTube URL format"):
            validate_youtube_url("https://vimeo.com/123456")

class TestLanguageCodeValidation:
    """Test guard clauses in language code validation."""
    
    def test_valid_language_codes(self):
        """Test that valid language codes pass validation."""
        for lang in VALID_LANGUAGE_CODES:
            validate_language_code(lang)
    
    def test_none_language(self):
        """Test guard clause for None language."""
        with pytest.raises(ValidationError, match="Language code is required"):
            validate_language_code(None)
    
    def test_empty_language(self):
        """Test guard clause for empty language."""
        with pytest.raises(ValidationError, match="Language code is required"):
            validate_language_code("")
    
    def test_wrong_data_type(self):
        """Test guard clause for wrong data type."""
        with pytest.raises(ValidationError, match="Language code must be a string"):
            validate_language_code(123)
    
    def test_invalid_language_codes(self):
        """Test guard clause for invalid language codes."""
        for lang in INVALID_LANGUAGE_CODES[3:]:  # Skip None, empty
            with pytest.raises(ValidationError, match="Invalid language code"):
                validate_language_code(lang)

class TestClipLengthValidation:
    """Test guard clauses in clip length validation."""
    
    def test_valid_clip_lengths(self):
        """Test that valid clip lengths pass validation."""
        for length in VALID_CLIP_LENGTHS:
            validate_clip_length(length)
    
    def test_wrong_data_type(self):
        """Test guard clause for wrong data type."""
        with pytest.raises(ValidationError, match="Clip length must be an integer"):
            validate_clip_length("30")
    
    def test_negative_length(self):
        """Test guard clause for negative length."""
        with pytest.raises(ValidationError, match="Clip length cannot be negative"):
            validate_clip_length(-1)
    
    def test_zero_length(self):
        """Test guard clause for zero length."""
        with pytest.raises(ValidationError, match="Clip length cannot be zero"):
            validate_clip_length(0)
    
    def test_too_short(self):
        """Test guard clause for too short length."""
        with pytest.raises(ValidationError, match="Clip length must be at least 1 seconds"):
            validate_clip_length(0, min_length=1)
    
    def test_too_long(self):
        """Test guard clause for too long length."""
        with pytest.raises(ValidationError, match="Clip length cannot exceed 600 seconds"):
            validate_clip_length(601, max_length=600)
    
    def test_unrealistic_value(self):
        """Test guard clause for unrealistic value."""
        with pytest.raises(ValidationError, match="Clip length exceeds maximum allowed duration"):
            validate_clip_length(86401)

class TestViralScoreValidation:
    """Test guard clauses in viral score validation."""
    
    def test_valid_viral_scores(self):
        """Test that valid viral scores pass validation."""
        for score in VALID_VIRAL_SCORES:
            validate_viral_score(score)
    
    def test_wrong_data_type(self):
        """Test guard clause for wrong data type."""
        with pytest.raises(ValidationError, match="Viral score must be a number"):
            validate_viral_score("0.5")
    
    def test_out_of_range(self):
        """Test guard clause for out of range values."""
        with pytest.raises(ValidationError, match="Viral score must be between 0.0 and 1.0"):
            validate_viral_score(-0.1)
        
        with pytest.raises(ValidationError, match="Viral score must be between 0.0 and 1.0"):
            validate_viral_score(1.1)

class TestCaptionValidation:
    """Test guard clauses in caption validation."""
    
    def test_valid_captions(self):
        """Test that valid captions pass validation."""
        for caption in VALID_CAPTIONS:
            validate_caption(caption)
    
    def test_none_caption(self):
        """Test guard clause for None caption."""
        with pytest.raises(ValidationError, match="Caption is required"):
            validate_caption(None)
    
    def test_empty_caption(self):
        """Test guard clause for empty caption."""
        with pytest.raises(ValidationError, match="Caption is required"):
            validate_caption("")
    
    def test_wrong_data_type(self):
        """Test guard clause for wrong data type."""
        with pytest.raises(ValidationError, match="Caption must be a string"):
            validate_caption(123)
    
    def test_empty_after_stripping(self):
        """Test guard clause for empty after stripping."""
        with pytest.raises(ValidationError, match="Caption cannot be empty"):
            validate_caption("   ")
    
    def test_too_long(self):
        """Test guard clause for too long caption."""
        long_caption = "A" * 1001
        with pytest.raises(ValidationError, match="Caption too long"):
            validate_caption(long_caption)

class TestVariantIDValidation:
    """Test guard clauses in variant ID validation."""
    
    def test_valid_variant_ids(self):
        """Test that valid variant IDs pass validation."""
        for variant_id in VALID_VARIANT_IDS:
            validate_variant_id(variant_id)
    
    def test_none_variant_id(self):
        """Test guard clause for None variant ID."""
        with pytest.raises(ValidationError, match="Variant ID is required"):
            validate_variant_id(None)
    
    def test_empty_variant_id(self):
        """Test guard clause for empty variant ID."""
        with pytest.raises(ValidationError, match="Variant ID is required"):
            validate_variant_id("")
    
    def test_wrong_data_type(self):
        """Test guard clause for wrong data type."""
        with pytest.raises(ValidationError, match="Variant ID must be a string"):
            validate_variant_id(123)
    
    def test_invalid_variant_ids(self):
        """Test guard clause for invalid variant IDs."""
        for variant_id in INVALID_VARIANT_IDS[2:]:  # Skip None, empty
            with pytest.raises(ValidationError, match="Invalid variant ID"):
                validate_variant_id(variant_id)

class TestAudienceProfileValidation:
    """Test guard clauses in audience profile validation."""
    
    def test_valid_audience_profiles(self):
        """Test that valid audience profiles pass validation."""
        for profile in VALID_AUDIENCE_PROFILES:
            validate_audience_profile(profile)
    
    def test_none_profile(self):
        """Test guard clause for None profile."""
        with pytest.raises(ValidationError, match="Audience profile must be a dictionary"):
            validate_audience_profile(None)
    
    def test_wrong_data_type(self):
        """Test guard clause for wrong data type."""
        with pytest.raises(ValidationError, match="Audience profile must be a dictionary"):
            validate_audience_profile("not_a_dict")
    
    def test_invalid_profile_keys(self):
        """Test guard clause for invalid profile keys."""
        with pytest.raises(ValidationError, match="Invalid audience profile key"):
            validate_audience_profile({"invalid_key": "value"})
    
    def test_invalid_age_type(self):
        """Test guard clause for invalid age type."""
        with pytest.raises(ValidationError, match="Age must be a string"):
            validate_audience_profile({"age": 123})
    
    def test_invalid_interests_type(self):
        """Test guard clause for invalid interests type."""
        with pytest.raises(ValidationError, match="Interests must be a list"):
            validate_audience_profile({"interests": "not_a_list"})

class TestBatchSizeValidation:
    """Test guard clauses in batch size validation."""
    
    def test_valid_batch_sizes(self):
        """Test that valid batch sizes pass validation."""
        for size in VALID_BATCH_SIZES:
            validate_batch_size(size)
    
    def test_wrong_data_type(self):
        """Test guard clause for wrong data type."""
        with pytest.raises(ValidationError, match="Batch size must be an integer"):
            validate_batch_size("10")
    
    def test_zero_size(self):
        """Test guard clause for zero size."""
        with pytest.raises(ValidationError, match="Batch size must be at least 1"):
            validate_batch_size(0)
    
    def test_negative_size(self):
        """Test guard clause for negative size."""
        with pytest.raises(ValidationError, match="Batch size cannot be negative"):
            validate_batch_size(-1)
    
    def test_too_large(self):
        """Test guard clause for too large size."""
        with pytest.raises(ValidationError, match="Batch size cannot exceed 100"):
            validate_batch_size(101, max_size=100)

# =============================================================================
# GRADIO DEMO TESTS
# =============================================================================

class TestPromptValidation:
    """Test guard clauses in prompt validation."""
    
    def test_valid_prompts(self):
        """Test that valid prompts pass validation."""
        for prompt in VALID_PROMPTS:
            validate_prompt(prompt)
    
    def test_none_prompt(self):
        """Test guard clause for None prompt."""
        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            validate_prompt(None)
    
    def test_empty_prompt(self):
        """Test guard clause for empty prompt."""
        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            validate_prompt("")
    
    def test_wrong_data_type(self):
        """Test guard clause for wrong data type."""
        with pytest.raises(ValidationError, match="Prompt must be a string"):
            validate_prompt(123)
    
    def test_empty_after_stripping(self):
        """Test guard clause for empty after stripping."""
        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            validate_prompt("   ")
    
    def test_too_long(self):
        """Test guard clause for too long prompt."""
        long_prompt = "A" * 1001
        with pytest.raises(ValidationError, match="Prompt too long"):
            validate_prompt(long_prompt)

class TestParameterValidation:
    """Test guard clauses in parameter validation."""
    
    def test_valid_parameters(self):
        """Test that valid parameters pass validation."""
        validate_parameters(7.5, 30)
    
    def test_wrong_guidance_scale_type(self):
        """Test guard clause for wrong guidance scale type."""
        with pytest.raises(ValidationError, match="Guidance scale must be a number"):
            validate_parameters("7.5", 30)
    
    def test_guidance_scale_out_of_range(self):
        """Test guard clause for guidance scale out of range."""
        with pytest.raises(ValidationError, match="Guidance scale must be between 1.0 and 20.0"):
            validate_parameters(0.5, 30)
        
        with pytest.raises(ValidationError, match="Guidance scale must be between 1.0 and 20.0"):
            validate_parameters(25.0, 30)
    
    def test_wrong_inference_steps_type(self):
        """Test guard clause for wrong inference steps type."""
        with pytest.raises(ValidationError, match="Inference steps must be an integer"):
            validate_parameters(7.5, "30")
    
    def test_inference_steps_out_of_range(self):
        """Test guard clause for inference steps out of range."""
        with pytest.raises(ValidationError, match="Inference steps must be between 10 and 100"):
            validate_parameters(7.5, 5)
        
        with pytest.raises(ValidationError, match="Inference steps must be between 10 and 100"):
            validate_parameters(7.5, 150)

class TestImageGeneration:
    """Test guard clauses in image generation."""
    
    @patch('gradio_demo.pipe')
    @patch('gradio_demo.device')
    @patch('gradio_demo.logger')
    def test_none_prompt(self, mock_logger, mock_device, mock_pipe):
        """Test guard clause for None prompt."""
        mock_device.return_value = "cpu"
        mock_pipe.return_value = Mock()
        
        result, error = generate_image(None)
        
        assert result is None
        assert "Prompt cannot be None" in error
        mock_logger.warning.assert_called()
    
    @patch('gradio_demo.pipe')
    @patch('gradio_demo.device')
    @patch('gradio_demo.logger')
    def test_empty_prompt(self, mock_logger, mock_device, mock_pipe):
        """Test guard clause for empty prompt."""
        mock_device.return_value = "cpu"
        mock_pipe.return_value = Mock()
        
        result, error = generate_image("")
        
        assert result is None
        assert "Prompt cannot be empty" in error
        mock_logger.warning.assert_called()
    
    @patch('gradio_demo.pipe')
    @patch('gradio_demo.device')
    @patch('gradio_demo.logger')
    def test_wrong_prompt_type(self, mock_logger, mock_device, mock_pipe):
        """Test guard clause for wrong prompt type."""
        mock_device.return_value = "cpu"
        mock_pipe.return_value = Mock()
        
        result, error = generate_image(123)
        
        assert result is None
        assert "Prompt must be a string" in error
        mock_logger.warning.assert_called()
    
    @patch('gradio_demo.pipe')
    @patch('gradio_demo.device')
    @patch('gradio_demo.logger')
    def test_guidance_scale_out_of_range(self, mock_logger, mock_device, mock_pipe):
        """Test guard clause for guidance scale out of range."""
        mock_device.return_value = "cpu"
        mock_pipe.return_value = Mock()
        
        result, error = generate_image("test prompt", guidance_scale=25.0)
        
        assert result is None
        assert "Guidance scale must be between 1.0 and 20.0" in error
        mock_logger.warning.assert_called()
    
    @patch('gradio_demo.pipe')
    @patch('gradio_demo.device')
    @patch('gradio_demo.logger')
    def test_inference_steps_out_of_range(self, mock_logger, mock_device, mock_pipe):
        """Test guard clause for inference steps out of range."""
        mock_device.return_value = "cpu"
        mock_pipe.return_value = Mock()
        
        result, error = generate_image("test prompt", num_inference_steps=5)
        
        assert result is None
        assert "Inference steps must be between 10 and 100" in error
        mock_logger.warning.assert_called()
    
    @patch('gradio_demo.pipe')
    @patch('gradio_demo.device')
    @patch('gradio_demo.logger')
    def test_prompt_too_long(self, mock_logger, mock_device, mock_pipe):
        """Test guard clause for prompt too long."""
        mock_device.return_value = "cpu"
        mock_pipe.return_value = Mock()
        
        long_prompt = "A" * 1001
        result, error = generate_image(long_prompt)
        
        assert result is None
        assert "Prompt too long" in error
        mock_logger.warning.assert_called()
    
    @patch('gradio_demo.pipe')
    @patch('gradio_demo.device')
    @patch('gradio_demo.logger')
    def test_pipeline_not_available(self, mock_logger, mock_device, mock_pipe):
        """Test guard clause for pipeline not available."""
        mock_device.return_value = "cpu"
        mock_pipe.return_value = None
        
        result, error = generate_image("test prompt")
        
        assert result is None
        assert "Image generation pipeline not available" in error
        mock_logger.error.assert_called()

# =============================================================================
# SYSTEM HEALTH TESTS
# =============================================================================

class TestSystemHealthValidation:
    """Test guard clauses in system health validation."""
    
    @patch('validation.psutil')
    def test_memory_critical(self, mock_psutil):
        """Test guard clause for critical memory usage."""
        mock_virtual_memory = Mock()
        mock_virtual_memory.percent = 95.0
        mock_psutil.virtual_memory.return_value = mock_virtual_memory
        
        with pytest.raises(CriticalSystemError, match="Critical memory usage"):
            validate_system_health()
    
    @patch('validation.psutil')
    def test_disk_critical(self, mock_psutil):
        """Test guard clause for critical disk usage."""
        mock_virtual_memory = Mock()
        mock_virtual_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_virtual_memory
        
        mock_disk_usage = Mock()
        mock_disk_usage.percent = 95.0
        mock_psutil.disk_usage.return_value = mock_disk_usage
        
        with pytest.raises(CriticalSystemError, match="Critical disk space"):
            validate_system_health()
    
    @patch('validation.psutil')
    def test_cpu_critical(self, mock_psutil):
        """Test guard clause for critical CPU usage."""
        mock_virtual_memory = Mock()
        mock_virtual_memory.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_virtual_memory
        
        mock_disk_usage = Mock()
        mock_disk_usage.percent = 50.0
        mock_psutil.disk_usage.return_value = mock_disk_usage
        
        mock_cpu_percent = Mock()
        mock_cpu_percent.return_value = 95.0
        mock_psutil.cpu_percent = mock_cpu_percent
        
        with pytest.raises(ResourceError, match="High CPU usage"):
            validate_system_health()

class TestGPUHealthValidation:
    """Test guard clauses in GPU health validation."""
    
    @patch('validation.torch')
    def test_gpu_not_available(self, mock_torch):
        """Test guard clause for GPU not available."""
        mock_torch.cuda.is_available.return_value = False
        
        result = check_gpu_availability()
        
        assert result["available"] is False
        assert "GPU not available" in result["status"]
    
    @patch('validation.torch')
    def test_gpu_memory_critical(self, mock_torch):
        """Test guard clause for critical GPU memory."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 9000000000  # 9GB
        mock_torch.cuda.get_device_properties.return_value.total_memory = 10000000000  # 10GB
        
        result = check_gpu_availability()
        
        assert result["available"] is True
        assert result["memory_critical"] is True

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestGuardClauseIntegration:
    """Test integration of guard clauses across multiple functions."""
    
    def test_video_request_validation_flow(self):
        """Test complete flow of video request validation with guard clauses."""
        # Test with valid data
        try:
            validate_video_request_data(
                youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                language="en",
                max_clip_length=60,
                min_clip_length=10,
                audience_profile={"age": "18-25"}
            )
        except ValidationError:
            pytest.fail("Valid video request should not raise ValidationError")
    
    def test_video_request_validation_invalid_url(self):
        """Test guard clause flow with invalid URL."""
        with pytest.raises(ValidationError, match="Invalid YouTube URL format"):
            validate_video_request_data(
                youtube_url="invalid_url",
                language="en"
            )
    
    def test_video_request_validation_invalid_language(self):
        """Test guard clause flow with invalid language."""
        with pytest.raises(ValidationError, match="Invalid language code"):
            validate_video_request_data(
                youtube_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                language="invalid_lang"
            )
    
    def test_batch_request_validation_flow(self):
        """Test complete flow of batch request validation with guard clauses."""
        requests = [
            {"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "language": "en"},
            {"youtube_url": "https://www.youtube.com/watch?v=abc123", "language": "es"}
        ]
        
        try:
            validate_batch_request_data(requests, batch_size=2)
        except ValidationError:
            pytest.fail("Valid batch request should not raise ValidationError")
    
    def test_batch_request_validation_empty_batch(self):
        """Test guard clause flow with empty batch."""
        with pytest.raises(ValidationError, match="Batch cannot be empty"):
            validate_batch_request_data([], batch_size=0)
    
    def test_batch_request_validation_too_large(self):
        """Test guard clause flow with too large batch."""
        requests = [{"youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "language": "en"}] * 101
        
        with pytest.raises(ValidationError, match="Batch size exceeds maximum limit"):
            validate_batch_request_data(requests, batch_size=101)

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestGuardClausePerformance:
    """Test performance benefits of guard clauses."""
    
    def test_early_return_performance(self):
        """Test that guard clauses provide early returns for better performance."""
        import time
        
        # Test with invalid input (should return early)
        start_time = time.perf_counter()
        try:
            validate_youtube_url(None)
        except ValidationError:
            pass
        early_return_time = time.perf_counter() - start_time
        
        # Test with valid input (goes through full validation)
        start_time = time.perf_counter()
        try:
            validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        except ValidationError:
            pass
        full_validation_time = time.perf_counter() - start_time
        
        # Early return should be faster
        assert early_return_time < full_validation_time
    
    def test_guard_clause_order_performance(self):
        """Test that guard clauses are ordered for optimal performance."""
        import time
        
        # Test with None (first guard clause)
        start_time = time.perf_counter()
        try:
            validate_language_code(None)
        except ValidationError:
            pass
        none_time = time.perf_counter() - start_time
        
        # Test with wrong type (second guard clause)
        start_time = time.perf_counter()
        try:
            validate_language_code(123)
        except ValidationError:
            pass
        wrong_type_time = time.perf_counter() - start_time
        
        # Test with invalid format (third guard clause)
        start_time = time.perf_counter()
        try:
            validate_language_code("invalid")
        except ValidationError:
            pass
        invalid_format_time = time.perf_counter() - start_time
        
        # All should be fast due to early returns
        assert none_time < 0.001
        assert wrong_type_time < 0.001
        assert invalid_format_time < 0.001

# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestGuardClauseEdgeCases:
    """Test edge cases in guard clause implementation."""
    
    def test_boundary_values(self):
        """Test guard clauses with boundary values."""
        # Test exact boundary values
        validate_clip_length(1, min_length=1, max_length=600)
        validate_clip_length(600, min_length=1, max_length=600)
        
        # Test just outside boundaries
        with pytest.raises(ValidationError):
            validate_clip_length(0, min_length=1, max_length=600)
        
        with pytest.raises(ValidationError):
            validate_clip_length(601, min_length=1, max_length=600)
    
    def test_unicode_inputs(self):
        """Test guard clauses with unicode inputs."""
        # Test unicode in URLs
        unicode_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&title=🎵🎶"
        validate_youtube_url(unicode_url)
        
        # Test unicode in captions
        unicode_caption = "A video about 🎵 music and 🎶 songs"
        validate_caption(unicode_caption)
        
        # Test unicode in prompts
        unicode_prompt = "Generate an image of 🎨 art"
        validate_prompt(unicode_prompt)
    
    def test_whitespace_handling(self):
        """Test guard clauses with various whitespace scenarios."""
        # Test whitespace-only inputs
        with pytest.raises(ValidationError, match="Caption cannot be empty"):
            validate_caption("   \t\n  ")
        
        with pytest.raises(ValidationError, match="Prompt cannot be empty"):
            validate_prompt("   \t\n  ")
        
        # Test whitespace around valid inputs
        validate_caption("  valid caption  ")
        validate_prompt("  valid prompt  ")
    
    def test_extreme_values(self):
        """Test guard clauses with extreme values."""
        # Test extremely long strings
        long_string = "A" * 10000
        with pytest.raises(ValidationError, match="YouTube URL too long"):
            validate_youtube_url(long_string)
        
        # Test extremely large numbers
        with pytest.raises(ValidationError, match="Clip length exceeds maximum allowed duration"):
            validate_clip_length(999999999)
        
        # Test extremely small numbers
        with pytest.raises(ValidationError, match="Viral score must be between 0.0 and 1.0"):
            validate_viral_score(-999999999.0)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 