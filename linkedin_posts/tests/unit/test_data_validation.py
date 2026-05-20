"""
Data Validation Tests for LinkedIn Posts
=======================================

Comprehensive data validation tests for LinkedIn posts including input validation,
business rules, data integrity, and constraint checking.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import re

# Mock data structures
class MockValidationResult:
    def __init__(self, is_valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.sanitized_data = None

class MockSanitizationResult:
    def __init__(self, original_data: str, sanitized_data: str, changes_made: List[str]):
        self.original_data = original_data
        self.sanitized_data = sanitized_data
        self.changes_made = changes_made

class TestDataValidation:
    """Test data validation and sanitization"""
    
    @pytest.fixture
    def mock_validation_service(self):
        """Mock validation service"""
        service = AsyncMock()
        
        # Mock content validation
        service.validate_content.return_value = MockValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Consider adding more hashtags"]
        )
        
        # Mock user input validation
        service.validate_user_input.return_value = MockValidationResult(
            is_valid=True,
            errors=[],
            warnings=[]
        )
        
        # Mock security validation
        service.validate_security.return_value = MockValidationResult(
            is_valid=True,
            errors=[],
            warnings=[]
        )
        
        return service
    
    @pytest.fixture
    def mock_sanitization_service(self):
        """Mock sanitization service"""
        service = AsyncMock()
        
        # Mock content sanitization
        service.sanitize_content.return_value = MockSanitizationResult(
            original_data="<script>alert('xss')</script>Content",
            sanitized_data="Content",
            changes_made=["removed_script_tags", "removed_html_tags"]
        )
        
        # Mock user data sanitization
        service.sanitize_user_data.return_value = MockSanitizationResult(
            original_data="User's data with special chars: & < >",
            sanitized_data="User's data with special chars: &amp; &lt; &gt;",
            changes_made=["escaped_html_entities"]
        )
        
        return service
    
    @pytest.fixture
    def mock_repository(self):
        """Mock repository for validation tests"""
        repo = AsyncMock()
        
        # Mock validation rules
        repo.get_validation_rules.return_value = {
            "content": {
                "max_length": 3000,
                "min_length": 10,
                "allowed_tags": ["b", "i", "strong", "em"],
                "forbidden_patterns": ["<script>", "javascript:"]
            },
            "user_data": {
                "max_length": 100,
                "allowed_characters": "a-zA-Z0-9\\s",
                "forbidden_patterns": ["<script>", "javascript:"]
            }
        }
        
        return repo
    
    @pytest.fixture
    def post_service(self, mock_repository, mock_validation_service, mock_sanitization_service):
        """Post service with mocked dependencies"""
        from services.post_service import PostService
        
        service = PostService(
            repository=mock_repository,
            ai_service=AsyncMock(),
            cache_service=AsyncMock(),
            validation_service=mock_validation_service,
            sanitization_service=mock_sanitization_service
        )
        return service
    
    async def test_content_validation_workflow(self, post_service, mock_validation_service):
        """Test complete content validation workflow"""
        # Arrange
        post_data = {
            "content": "This is a test post content with #hashtags",
            "title": "Test Post Title",
            "tags": ["test", "validation"]
        }
        
        # Act
        validation_result = await post_service.validate_post_data(post_data)
        
        # Assert
        assert validation_result is not None
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0
        mock_validation_service.validate_content.assert_called_once()
    
    async def test_xss_prevention(self, post_service, mock_sanitization_service):
        """Test XSS prevention in content"""
        # Arrange
        malicious_content = "<script>alert('xss')</script>This is content"
        
        # Act
        sanitized_content = await post_service.sanitize_content(malicious_content)
        
        # Assert
        assert sanitized_content is not None
        assert "<script>" not in sanitized_content.sanitized_data
        assert "alert('xss')" not in sanitized_content.sanitized_data
        mock_sanitization_service.sanitize_content.assert_called_once()
    
    async def test_sql_injection_prevention(self, post_service, mock_validation_service):
        """Test SQL injection prevention"""
        # Arrange
        malicious_input = "'; DROP TABLE posts; --"
        
        # Act
        validation_result = await post_service.validate_user_input(malicious_input)
        
        # Assert
        assert validation_result is not None
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0
        mock_validation_service.validate_user_input.assert_called_once()
    
    async def test_content_length_validation(self, post_service, mock_validation_service):
        """Test content length validation"""
        # Arrange
        short_content = "Short"
        long_content = "A" * 4000  # Exceeds LinkedIn limit
        
        # Act
        short_validation = await post_service.validate_content_length(short_content)
        long_validation = await post_service.validate_content_length(long_content)
        
        # Assert
        assert short_validation.is_valid is False
        assert long_validation.is_valid is False
        assert "minimum length" in short_validation.errors[0]
        assert "maximum length" in long_validation.errors[0]
    
    async def test_hashtag_validation(self, post_service, mock_validation_service):
        """Test hashtag validation"""
        # Arrange
        valid_hashtags = ["#technology", "#innovation", "#leadership"]
        invalid_hashtags = ["#", "#123", "#a" * 50]  # Invalid hashtags
        
        # Act
        valid_result = await post_service.validate_hashtags(valid_hashtags)
        invalid_result = await post_service.validate_hashtags(invalid_hashtags)
        
        # Assert
        assert valid_result.is_valid is True
        assert invalid_result.is_valid is False
        assert len(invalid_result.errors) > 0
    
    async def test_url_validation(self, post_service, mock_validation_service):
        """Test URL validation in content"""
        # Arrange
        valid_urls = [
            "https://example.com",
            "http://linkedin.com/posts/123",
            "https://www.google.com"
        ]
        invalid_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "ftp://malicious.com"
        ]
        
        # Act
        valid_result = await post_service.validate_urls(valid_urls)
        invalid_result = await post_service.validate_urls(invalid_urls)
        
        # Assert
        assert valid_result.is_valid is True
        assert invalid_result.is_valid is False
        assert len(invalid_result.errors) > 0
    
    async def test_character_encoding_validation(self, post_service, mock_validation_service):
        """Test character encoding validation"""
        # Arrange
        content_with_unicode = "Post with unicode: 🚀 📈 💼"
        content_with_special_chars = "Post with special chars: & < > \" '"
        
        # Act
        unicode_result = await post_service.validate_character_encoding(content_with_unicode)
        special_chars_result = await post_service.validate_character_encoding(content_with_special_chars)
        
        # Assert
        assert unicode_result.is_valid is True
        assert special_chars_result.is_valid is True
    
    async def test_file_upload_validation(self, post_service, mock_validation_service):
        """Test file upload validation"""
        # Arrange
        valid_file = {
            "name": "image.jpg",
            "type": "image/jpeg",
            "size": 1024 * 1024,  # 1MB
            "content": b"fake_image_data"
        }
        invalid_file = {
            "name": "script.exe",
            "type": "application/x-executable",
            "size": 10 * 1024 * 1024,  # 10MB
            "content": b"malicious_executable"
        }
        
        # Act
        valid_result = await post_service.validate_file_upload(valid_file)
        invalid_result = await post_service.validate_file_upload(invalid_file)
        
        # Assert
        assert valid_result.is_valid is True
        assert invalid_result.is_valid is False
        assert len(invalid_result.errors) > 0
    
    async def test_data_sanitization_workflow(self, post_service, mock_sanitization_service):
        """Test complete data sanitization workflow"""
        # Arrange
        raw_data = {
            "content": "<script>alert('xss')</script>Content",
            "title": "Title with <b>HTML</b>",
            "description": "Description with & < > characters"
        }
        
        # Act
        sanitized_data = await post_service.sanitize_post_data(raw_data)
        
        # Assert
        assert sanitized_data is not None
        assert "<script>" not in sanitized_data["content"]
        assert "alert('xss')" not in sanitized_data["content"]
        mock_sanitization_service.sanitize_content.assert_called()
    
    async def test_input_whitelist_validation(self, post_service, mock_validation_service):
        """Test input whitelist validation"""
        # Arrange
        allowed_tags = ["b", "i", "strong", "em"]
        content_with_allowed_tags = "<b>Bold</b> and <i>italic</i> text"
        content_with_forbidden_tags = "<script>alert('xss')</script><b>Bold</b>"
        
        # Act
        allowed_result = await post_service.validate_html_tags(content_with_allowed_tags, allowed_tags)
        forbidden_result = await post_service.validate_html_tags(content_with_forbidden_tags, allowed_tags)
        
        # Assert
        assert allowed_result.is_valid is True
        assert forbidden_result.is_valid is False
        assert len(forbidden_result.errors) > 0
    
    async def test_rate_limiting_validation(self, post_service, mock_validation_service):
        """Test rate limiting validation"""
        # Arrange
        user_id = "user_123"
        current_time = datetime.now()
        
        # Mock recent posts
        recent_posts = [
            {"created_at": current_time - timedelta(minutes=5)},
            {"created_at": current_time - timedelta(minutes=3)},
            {"created_at": current_time - timedelta(minutes=1)}
        ]
        
        # Act
        rate_limit_result = await post_service.validate_rate_limit(user_id, recent_posts)
        
        # Assert
        assert rate_limit_result is not None
        assert "can_post" in rate_limit_result
        assert "next_allowed_time" in rate_limit_result
    
    async def test_content_quality_validation(self, post_service, mock_validation_service):
        """Test content quality validation"""
        # Arrange
        high_quality_content = "Professional post about industry trends with valuable insights and actionable advice."
        low_quality_content = "Check this out! Amazing stuff! Click here!"
        
        # Act
        high_quality_result = await post_service.validate_content_quality(high_quality_content)
        low_quality_result = await post_service.validate_content_quality(low_quality_content)
        
        # Assert
        assert high_quality_result.is_valid is True
        assert low_quality_result.is_valid is False
        assert len(low_quality_result.warnings) > 0
    
    async def test_validation_error_handling(self, post_service, mock_validation_service):
        """Test validation error handling"""
        # Arrange
        mock_validation_service.validate_content.side_effect = Exception("Validation service error")
        
        # Act & Assert
        with pytest.raises(Exception):
            await post_service.validate_post_data({"content": "test"})

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
