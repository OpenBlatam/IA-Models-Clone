"""
Tests para URL validator
"""

import pytest
from infrastructure.utils.url_validator import URLValidator


def test_validate_valid_url():
    """Test validación de URL válida"""
    is_valid, error = URLValidator.validate("https://example.com")
    assert is_valid is True
    assert error is None


def test_validate_invalid_scheme():
    """Test validación de scheme inválido"""
    is_valid, error = URLValidator.validate("ftp://example.com")
    assert is_valid is False
    assert "Scheme no permitido" in error


def test_validate_empty_url():
    """Test validación de URL vacía"""
    is_valid, error = URLValidator.validate("")
    assert is_valid is False
    assert "vacía" in error.lower()


def test_validate_localhost():
    """Test validación bloquea localhost"""
    is_valid, error = URLValidator.validate("http://localhost")
    assert is_valid is False
    assert "local" in error.lower()


def test_normalize_url():
    """Test normalización de URL"""
    url = URLValidator.normalize("example.com")
    assert url.startswith("https://")
    
    url = URLValidator.normalize("https://example.com")
    assert url == "https://example.com"








