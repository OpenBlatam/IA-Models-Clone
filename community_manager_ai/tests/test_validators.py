"""
Tests for Validators
====================
"""

import pytest
from community_manager_ai.utils.validators import (
    validate_platform,
    validate_platforms,
    validate_content_length,
    validate_scheduled_time
)
from datetime import datetime, timedelta


def test_validate_platform():
    """Test validar plataforma"""
    assert validate_platform("facebook") is True
    assert validate_platform("instagram") is True
    assert validate_platform("twitter") is True
    assert validate_platform("invalid") is False


def test_validate_platforms():
    """Test validar lista de plataformas"""
    is_valid, error = validate_platforms(["facebook", "twitter"])
    assert is_valid is True
    assert error is None
    
    is_valid, error = validate_platforms(["invalid"])
    assert is_valid is False
    assert error is not None


def test_validate_content_length():
    """Test validar longitud de contenido"""
    # Contenido válido para Twitter
    content = "Short tweet"
    is_valid, error = validate_content_length(content, "twitter")
    assert is_valid is True
    
    # Contenido muy largo para Twitter
    content = "x" * 300
    is_valid, error = validate_content_length(content, "twitter")
    assert is_valid is False
    assert "excede" in error.lower()


def test_validate_scheduled_time():
    """Test validar fecha programada"""
    # Fecha futura (válida)
    future_time = datetime.now() + timedelta(hours=1)
    is_valid, error = validate_scheduled_time(future_time)
    assert is_valid is True
    
    # Fecha pasada (inválida)
    past_time = datetime.now() - timedelta(hours=1)
    is_valid, error = validate_scheduled_time(past_time)
    assert is_valid is False
    assert "futuro" in error.lower()




