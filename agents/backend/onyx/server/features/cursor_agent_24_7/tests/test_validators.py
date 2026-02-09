"""
Tests for Validators
====================

Tests para el sistema de validación.
"""

import pytest
from ..core.validators import CommandValidator, InputValidator, ValidationResult


def test_command_validator():
    """Test validador de comandos"""
    validator = CommandValidator()
    
    # Comando válido
    result = validator.validate("print('hello')")
    assert result.valid is True
    
    # Comando peligroso
    result = validator.validate("__import__('os').system('rm -rf /')")
    assert result.valid is False
    assert len(result.errors) > 0
    
    # Comando muy largo
    result = validator.validate("x" * 20000)
    assert result.valid is False


def test_sanitize():
    """Test sanitización"""
    validator = CommandValidator()
    
    # Sanitizar comando con caracteres peligrosos
    sanitized = validator.sanitize("test\x00\r\n")
    assert "\x00" not in sanitized
    assert "\r" not in sanitized


def test_input_validator():
    """Test validador de entrada"""
    # Email válido
    assert InputValidator.validate_email("user@example.com") is True
    assert InputValidator.validate_email("invalid") is False
    
    # URL válida
    assert InputValidator.validate_url("https://example.com") is True
    assert InputValidator.validate_url("not-a-url") is False
    
    # JSON válido
    assert InputValidator.validate_json('{"key": "value"}') is True
    assert InputValidator.validate_json("invalid json") is False
    
    # Longitud
    assert InputValidator.validate_length("test", min_len=1, max_len=10) is True
    assert InputValidator.validate_length("", min_len=1) is False



