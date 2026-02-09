"""
Tests para ContentValidator
"""

import pytest
from services.content_validator import ContentValidator, ValidationResult
from core.models import GeneratedContent, Platform, ContentType


def test_validate_instagram_post():
    """Test validación de post de Instagram"""
    validator = ContentValidator()
    
    content = GeneratedContent(
        content_id="test-1",
        identity_profile_id="identity-1",
        platform=Platform.INSTAGRAM,
        content_type=ContentType.POST,
        content="💪 Este es un post de prueba con hashtags #fitness #motivation",
        hashtags=["fitness", "motivation"]
    )
    
    result = validator.validate(content)
    
    assert isinstance(result, ValidationResult)
    assert result.score >= 0.0
    assert result.score <= 1.0


def test_validate_too_long_content():
    """Test validación de contenido muy largo"""
    validator = ContentValidator()
    
    # Contenido muy largo (más de 2200 caracteres)
    long_content = "a" * 2500
    
    content = GeneratedContent(
        content_id="test-2",
        identity_profile_id="identity-1",
        platform=Platform.INSTAGRAM,
        content_type=ContentType.POST,
        content=long_content,
        hashtags=[]
    )
    
    result = validator.validate(content)
    
    assert not result.is_valid or result.score < 0.7
    assert len(result.issues) > 0


def test_validate_no_hashtags():
    """Test validación sin hashtags"""
    validator = ContentValidator()
    
    content = GeneratedContent(
        content_id="test-3",
        identity_profile_id="identity-1",
        platform=Platform.INSTAGRAM,
        content_type=ContentType.POST,
        content="Contenido sin hashtags",
        hashtags=[]
    )
    
    result = validator.validate(content)
    
    # Debe tener warning sobre falta de hashtags
    assert len(result.warnings) > 0 or len(result.suggestions) > 0




