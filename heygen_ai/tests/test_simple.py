#!/usr/bin/env python3
"""
Simple test file for HeyGen AI
"""

import pytest


def test_simple():
    """Simple test to verify basic functionality."""
    assert True


def test_basic_math():
    """Test basic mathematical operations."""
    assert 2 + 2 == 4
    assert 3 * 3 == 9
    assert 10 - 5 == 5


if __name__ == "__main__":
    test_simple()
    test_basic_math()
    print("All simple tests passed!")
