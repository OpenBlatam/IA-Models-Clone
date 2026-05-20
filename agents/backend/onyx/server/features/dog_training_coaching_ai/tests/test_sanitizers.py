"""
Tests for Sanitizers
=====================
"""

import pytest
from ...utils.sanitizers import (
    sanitize_text,
    sanitize_dog_breed,
    sanitize_dog_age,
    sanitize_list
)


def test_sanitize_text():
    """Test text sanitization."""
    assert sanitize_text("  hello  world  ") == "hello world"
    assert sanitize_text("hello\nworld") == "hello world"
    assert sanitize_text("", max_length=10) == ""
    assert sanitize_text("a" * 20, max_length=10) == "a" * 10


def test_sanitize_dog_breed():
    """Test dog breed sanitization."""
    assert sanitize_dog_breed("golden retriever") == "Golden Retriever"
    assert sanitize_dog_breed("german-shepherd") == "German-Shepherd"
    assert sanitize_dog_breed(None) is None
    assert sanitize_dog_breed("") is None


def test_sanitize_dog_age():
    """Test dog age sanitization."""
    assert sanitize_dog_age("  2 years  ") == "2 years"
    assert sanitize_dog_age(None) is None
    assert sanitize_dog_age("") is None


def test_sanitize_list():
    """Test list sanitization."""
    assert sanitize_list(["  item1  ", "item2", ""]) == ["item1", "item2"]
    assert sanitize_list(None) == []
    assert sanitize_list([]) == []

